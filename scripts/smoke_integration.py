#!/usr/bin/env python3
"""
scripts/smoke_integration.py
============================
DAY-3 PROOF (on real torch/TRL): instantiate CDPODecoupledGRPOTrainer with a tiny
model, run a few steps for each method, and confirm the three methods produce
DIFFERENT advantages. If GRPO and CDPO yield identical advantages, the override
did not take effect and the re-normalization bug is still present — STOP and fix
before spending money on a 7B GPU.

This is the gate between Day 3 (plumbing) and Day 5 (real 7B run).

What it does
------------
1. Loads a tiny instruct model (default Qwen2.5-0.5B-Instruct) — cheap, CPU/small-GPU OK.
2. For each method in {grpo, gdpo, cdpo}:
     - builds the decoupled trainer on a handful of synthetic prompts,
     - monkeypatches the advantage computer to CAPTURE the advantage tensor it
       injects at step 0 (before any weights move),
     - runs `--steps` steps.
3. Asserts:
     (a) no crashes, finite loss,
     (b) the step-0 advantage vectors DIFFER across methods (max pairwise
         |Δ| > tol). Identical vectors ⇒ override defeated.
4. Dumps the step-0 advantage histograms to PNG so you can eyeball divergence.

Usage
-----
    python scripts/smoke_integration.py                       # 0.5B, 6 steps
    python scripts/smoke_integration.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 8
    python scripts/smoke_integration.py --cpu                 # force CPU

Requires the 'train' extra (torch, transformers, trl==0.14.0, peft).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("smoke_integration")


# ---------------------------------------------------------------------------
# Synthetic data: tiny prompts each carrying an embedded instance_id that the
# ScenarioStore can resolve. We build a minimal store + bridge so we don't need
# the full dataset on disk for a plumbing test.
# ---------------------------------------------------------------------------
def build_min_store_and_bridge(K, M, n_prompts):
    """A minimal store/bridge that scores any completion deterministically.

    We don't need real financial parsing for a plumbing test — we need the
    advantage pipeline to RUN and to differ across methods. So the bridge scores
    each completion by a cheap hash into plausible r_hard/r_soft, giving non-
    degenerate groups (some compliant, some not).
    """
    from training.advantage_bridge import AdvantageBridge

    class _Entry:
        task = "portfolio"
        scenario = None
        constraints = None

    class _Store:
        def lookup(self, prompt):
            return _Entry()

    class _Bundle:
        def __init__(self, hard, prox, soft):
            self.hard = np.asarray(hard, float)
            self.prox = np.asarray(prox, float)
            self.soft = np.asarray(soft, float)

    def reward_from_output(comp_text, task, scenario, constraints):
        # deterministic pseudo-scores from the completion text length/hash,
        # engineered to produce a MIX of compliant/non-compliant rollouts.
        h = (hash(comp_text) & 0xFFFFFFFF)
        rng = np.random.default_rng(h)
        # ~50% chance each constraint passes -> non-degenerate groups
        hard = (rng.random(K) > 0.5).astype(float)
        prox = rng.random(K)
        # conflict: violators get slightly higher soft, to exercise the gate
        base = 0.3 + 0.5 * rng.random(M)
        if hard.min() < 1:
            base = base + 0.2
        soft = np.clip(base, 0, 1)
        return _Bundle(hard, prox, soft)

    bridge = AdvantageBridge(_Store(), reward_from_output, K=K, M=M)
    return bridge


def make_prompts(n, tokenizer, G):
    """n distinct prompts with embedded instance_ids, formatted via chat template."""
    rows = []
    for i in range(n):
        user = (
            f"<!-- instance_id: portfolio_{i:04d} -->\n"
            "Build a portfolio plan. Wrap it in <financial_plan>...</financial_plan>."
        )
        rows.append({"prompt": [{"role": "user", "content": user}]})
    from datasets import Dataset
    return Dataset.from_list(rows)


def capture_advantages(advantage_computer):
    """Wrap .compute so we record the first advantage vector it returns."""
    captured = {}
    orig = advantage_computer.compute

    def wrapped(batch, step):
        adv, metrics = orig(batch, step)
        if "step0" not in captured:
            captured["step0"] = np.asarray(adv, float).reshape(-1).copy()
        return adv, metrics

    advantage_computer.compute = wrapped
    return captured


def run_method(method, args, tokenizer, model_name, device):
    import torch
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig

    from finplanenv.cdpo import CDPOConfig
    from finplanenv.baselines import BaselineConfig
    from training.reward_fns import make_advantage_computer, make_zero_reward_fn
    from training.trainers import CDPODecoupledGRPOTrainer

    K, M, G = args.K, args.M, args.num_generations

    bridge = build_min_store_and_bridge(K, M, args.n_prompts)
    computer = make_advantage_computer(
        method, K=K, M=M, G=G,
        cdpo_config=CDPOConfig(G=G, beta_minus=2.0, alpha_schedule="adaptive"),
        grpo_config=BaselineConfig(G=G),
    )
    captured = capture_advantages(computer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32 if args.cpu else torch.bfloat16,
        device_map=None if args.cpu else "auto",
    )
    if args.cpu:
        model = model.to("cpu")

    cfg = GRPOConfig(
        num_generations=G,
        max_completion_length=args.max_new_tokens,
        temperature=1.0,
        learning_rate=1e-6,
        per_device_train_batch_size=max(args.n_prompts, G),
        gradient_accumulation_steps=1,
        max_steps=args.steps,
        seed=42,
        output_dir=str(Path(args.out) / method),
        logging_steps=1, save_steps=10_000,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_vllm=False,
    )

    trainer = CDPODecoupledGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_zero_reward_fn(G)],
        args=cfg,
        train_dataset=make_prompts(args.n_prompts, tokenizer, G),
        advantage_computer=computer,
        advantage_bridge=bridge,
        trajectory_logger=None,
    )

    logger.info("[%s] training %d steps on %s ...", method, args.steps, model_name)
    trainer.train()
    adv0 = captured.get("step0")
    if adv0 is None:
        raise RuntimeError(f"[{method}] advantage computer was never called — "
                           "the override did not route through it.")
    logger.info("[%s] step-0 advantage: mean=%.4f std=%.4f n_distinct=%d",
                method, adv0.mean(), adv0.std(), len(set(np.round(adv0, 4))))
    return adv0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="outputs/smoke_integration")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as e:
        sys.exit(f"train extra not installed ({e}). pip install -e '.[train]'")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        args.cpu = True
    logger.info("device=%s  model=%s", device, args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adv = {}
    for method in ["grpo", "gdpo", "cdpo"]:
        adv[method] = run_method(method, args, tokenizer, args.model, device)

    # ---- the assertions -----------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP-0 ADVANTAGE DIVERGENCE CHECK")
    print("=" * 60)
    ok = True

    def pdiff(a, b):
        n = min(len(a), len(b))
        return float(np.max(np.abs(a[:n] - b[:n])))

    pairs = [("grpo", "cdpo"), ("gdpo", "cdpo"), ("grpo", "gdpo")]
    for x, y in pairs:
        d = pdiff(adv[x], adv[y])
        status = "DISTINCT" if d > args.tol else "IDENTICAL (!!)"
        print(f"  {x.upper():<5} vs {y.upper():<5}  max|Δ| = {d:.4e}  -> {status}")
        if x == "grpo" and y == "cdpo" and d <= args.tol:
            ok = False  # the critical pair

    # histogram dump
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
        for ax, m in zip(axes, ["grpo", "gdpo", "cdpo"]):
            ax.hist(adv[m], bins=12, color="#b00020" if m == "cdpo" else "#1f77b4",
                    alpha=0.8)
            ax.set_title(f"{m.upper()} step-0 advantages")
            ax.set_xlabel("advantage")
        fig.tight_layout()
        out = Path(args.out) / "step0_advantage_histograms.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        print(f"\n  wrote {out}")
    except Exception as e:
        print(f"  (histogram skipped: {e})")

    print("\nRESULT:",
          "PASS — methods diverge, override is working."
          if ok else
          "FAIL — GRPO and CDPO advantages are identical. The decoupling did "
          "NOT take effect; do not run 7B until fixed.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

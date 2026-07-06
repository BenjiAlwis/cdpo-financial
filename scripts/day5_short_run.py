#!/usr/bin/env python3
"""
scripts/day5_short_run.py
=========================
DAY-5 GATE: run ONE short real 7B training run, measure seconds/step and VRAM,
and project the cost of the full experiment matrix. This is the go/no-go number
before you commit to 27+ runs.

It does NOT replace train.py — it wraps a short, instrumented invocation of the
same decoupled trainer so the timing reflects the real path (generate → bridge →
CDPO advantage → loss), then prints a cost projection table.

Usage
-----
    # after generate_full.py + generate_descriptions.py have produced data/full
    python scripts/day5_short_run.py \
        --model Qwen/Qwen2.5-7B-Instruct --task portfolio \
        --steps 30 --num-generations 8 --use-lora \
        --cost-per-hour 0.69

What it reports
---------------
  - sec/step (median over the timed steps, ignoring step 0 warmup)
  - peak VRAM
  - projected minutes & $ for one 400-step run
  - projected total for the full matrix (methods × tasks × seeds × steps)

Read the projection against your RunPod credit BEFORE launching the matrix. If
it's too expensive, cut seeds or tasks here — not after 10 runs.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("day5")


# task dims (K hard, M soft) — must match finplanenv
TASK_DIMS = {
    "portfolio":  {"K": 4, "M": 3},
    "retirement": {"K": 2, "M": 3},
    "loan":       {"K": 3, "M": 3},
}


class StepTimer:
    """A TrainerCallback that records wall-clock time per step."""
    def __init__(self):
        self.times = []
        self._t = None

    def make_callback(self):
        from transformers import TrainerCallback

        outer = self

        class _CB(TrainerCallback):
            def on_step_begin(self, args, state, control, **kw):
                outer._t = time.time()

            def on_step_end(self, args, state, control, **kw):
                if outer._t is not None:
                    outer.times.append(time.time() - outer._t)

        return _CB()


def resolve_train_path(data_dir: Path, task: str) -> Path:
    wd = data_dir / f"{task}_train_with_desc.jsonl"
    pl = data_dir / f"{task}_train.jsonl"
    if wd.exists():
        return wd
    if pl.exists():
        return pl
    raise SystemExit(
        f"No training data for {task} in {data_dir}. Run generate_full.py and "
        "generate_descriptions.py first."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--task", default="portfolio", choices=list(TASK_DIMS))
    ap.add_argument("--data-dir", default="data/full")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--method", default="cdpo", choices=["grpo", "gdpo", "cdpo"])
    ap.add_argument("--use-lora", action="store_true", default=True)
    ap.add_argument("--no-lora", dest="use_lora", action="store_false")
    ap.add_argument("--cost-per-hour", type=float, default=0.69,
                    help="GPU $/hr for the cost projection (default RTX 4090).")
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false", default=True,
                    help="Disable gradient checkpointing. It is ON by default "
                         "because 7B backprop OOMs without it on <=48GB cards. "
                         "Disable only on a large card (80GB+) for speed.")
    # matrix shape for the projection
    ap.add_argument("--matrix-methods", type=int, default=3)
    ap.add_argument("--matrix-tasks", type=int, default=1)
    ap.add_argument("--matrix-seeds", type=int, default=3)
    ap.add_argument("--matrix-steps", type=int, default=400)
    # vLLM: the throughput fix. On a single-GPU pod, vLLM shares the training GPU
    # (cuda:0). gpu-mem-util reserves a slice of VRAM for vLLM's KV cache; the
    # rest stays for the training model + backward. 0.30 is a safe start on a
    # 44GB card with a 7B; raise it if generation is still slow, lower it if OOM.
    ap.add_argument("--use-vllm", action="store_true",
                    help="Use vLLM for generation (5-20x faster). Requires "
                         "`pip install vllm`. Shares the single GPU by default.")
    ap.add_argument("--vllm-device", default="cuda:0",
                    help="Device for vLLM. On a 1-GPU pod keep cuda:0 (shared "
                         "with training). TRL's 'auto' would pick cuda:1 and fail.")
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.30,
                    help="Fraction of GPU memory vLLM reserves (single-GPU share).")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOConfig
    except Exception as e:
        sys.exit(f"train extra not installed ({e}). pip install -e '.[train]'")

    from finplanenv.cdpo import CDPOConfig
    from finplanenv.baselines import BaselineConfig
    from finplanenv.parser import compute_rewards_from_output
    from training.dataset_builder import build_dataset
    from training.scenario_store import ScenarioStore
    from training.reward_fns import make_advantage_computer, make_zero_reward_fn
    from training.advantage_bridge import AdvantageBridge
    from training.trainers import CDPODecoupledGRPOTrainer
    from finplanenv.dataset import DatasetConfig, DatasetGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("No CUDA — timing on CPU is not representative of a 7B run.")

    K = TASK_DIMS[args.task]["K"]
    M = TASK_DIMS[args.task]["M"]
    G = args.num_generations

    # ---- data ----
    data_dir = Path(args.data_dir)
    train_path = resolve_train_path(data_dir, args.task)
    logger.info("data: %s", train_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(
        metadata_paths={args.task: train_path},
        tokenizer=tokenizer,
    )
    ds_config = DatasetConfig()
    store = ScenarioStore(ds_config)
    store.build(DatasetGenerator(ds_config).load_metadata(train_path))

    # ---- model (NO manual LoRA here) ----
    # IMPORTANT: do NOT pre-apply LoRA with get_peft_model. In TRL 0.14.0,
    # GRPOTrainer only skips building a full deepcopy reference model when it
    # receives `peft_config` directly (then ref_model=None and it disables the
    # adapter to get the reference policy). If we hand it an already-PEFT model,
    # TRL sees peft_config=None and deepcopies the full 7B -> OOM on 24GB.
    logger.info("loading model %s ...", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        logger.info("LoRA will be applied BY TRL (ref_model=None, no deepcopy).")

    # Gradient checkpointing is required for 7B backprop on <=48GB. With LoRA +
    # a frozen base model it needs enable_input_require_grads(), else the loss
    # has no grad_fn ('element 0 ... does not require grad'). use_reentrant=False
    # is the PEFT-compatible checkpointing variant.
    if args.grad_checkpointing:
        model.enable_input_require_grads()
        logger.info("gradient checkpointing ON (+enable_input_require_grads, "
                    "non-reentrant).")

    # ---- advantage path ----
    computer = make_advantage_computer(
        args.method, K=K, M=M, G=G,
        cdpo_config=CDPOConfig(G=G), grpo_config=BaselineConfig(G=G),
    )
    bridge = AdvantageBridge(store, compute_rewards_from_output, K=K, M=M)

    # ---- vLLM availability + single-GPU guard ----
    if args.use_vllm:
        try:
            import vllm  # noqa: F401
        except Exception:
            sys.exit("--use-vllm set but vllm not installed. Run: pip install vllm")
        # On a single-GPU pod, vLLM must share cuda:0 with training. With
        # gradient checkpointing ON, the training model needs most of VRAM, so
        # keep vllm_gpu_memory_utilization modest (~0.30). Generation no longer
        # competes with the backward graph for the same allocation window, which
        # is why vLLM also EASES the memory pressure, not just speeds things up.
        logger.info("vLLM ON: device=%s gpu_mem_util=%.2f (shared single GPU).",
                    args.vllm_device, args.vllm_gpu_mem_util)

    cfg = GRPOConfig(
        num_generations=G,
        max_completion_length=args.max_new_tokens,
        temperature=1.0,
        learning_rate=args.lr,
        per_device_train_batch_size=max(args.batch_size, G),
        gradient_accumulation_steps=1,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.steps,
        seed=42,
        output_dir="outputs/day5_short_run",
        logging_steps=1, save_steps=10_000,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
        use_vllm=args.use_vllm,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
    )

    timer = StepTimer()
    trainer = CDPODecoupledGRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[make_zero_reward_fn(G)],
        args=cfg, train_dataset=train_dataset,
        peft_config=peft_config,
        advantage_computer=computer, advantage_bridge=bridge,
        trajectory_logger=None,
    )
    trainer.add_callback(timer.make_callback())

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logger.info("running %d steps (method=%s, G=%d, max_new_tokens=%d) ...",
                args.steps, args.method, G, args.max_new_tokens)
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    # ---- timing analysis (drop step 0 warmup) ----
    steps = timer.times[1:] if len(timer.times) > 1 else timer.times
    if not steps:
        sys.exit("No steps timed — did training run?")
    sec_per_step = statistics.median(steps)
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0

    # ---- projection ----
    per_run_steps = args.matrix_steps
    per_run_min = sec_per_step * per_run_steps / 60
    per_run_cost = per_run_min / 60 * args.cost_per_hour
    n_runs = args.matrix_methods * args.matrix_tasks * args.matrix_seeds
    total_min = per_run_min * n_runs
    total_cost = per_run_cost * n_runs

    print("\n" + "=" * 60)
    print("  DAY-5 SHORT RUN — TIMING & COST PROJECTION")
    print("=" * 60)
    print(f"  model            : {args.model}")
    print(f"  method/task      : {args.method} / {args.task}  (LoRA={args.use_lora})")
    print(f"  G / max_new_tok  : {G} / {args.max_new_tokens}")
    print(f"  steps timed      : {len(steps)} (warmup dropped)")
    print(f"  median sec/step  : {sec_per_step:.2f}")
    print(f"  wall time        : {wall:.0f}s for {args.steps} steps")
    if device == "cuda":
        print(f"  peak VRAM        : {peak_gb:.1f} GB")
    print("-" * 60)
    print(f"  PER RUN ({per_run_steps} steps):")
    print(f"    time           : {per_run_min:.1f} min")
    print(f"    cost @ ${args.cost_per_hour}/hr : ${per_run_cost:.2f}")
    print("-" * 60)
    print(f"  FULL MATRIX: {args.matrix_methods} methods × "
          f"{args.matrix_tasks} tasks × {args.matrix_seeds} seeds = {n_runs} runs")
    print(f"    total time     : {total_min/60:.1f} hours")
    print(f"    total cost     : ${total_cost:.2f}")
    print("=" * 60)
    print("\n  GO/NO-GO: compare total cost against your RunPod credit. If too")
    print("  high, reduce --matrix-seeds or --matrix-tasks (portfolio-only ×3")
    print("  seeds is the minimum publishable comparison), or shorten")
    print("  --matrix-steps. Re-run this projection with the new shape.")


if __name__ == "__main__":
    main()

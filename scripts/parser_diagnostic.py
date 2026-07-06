#!/usr/bin/env python3
"""
scripts/parser_diagnostic.py
============================
DAY-4 STEP 1: see how the parser behaves on REAL model output.

The Day-3 smoke gate proved the wiring; it used synthetic reward scoring. This
script does the opposite: it runs the actual model on actual FinPlanEnv prompts,
collects the raw completions, runs them through the REAL parser, and reports
exactly how and where parsing succeeds or fails — with representative samples of
each failure mode. You harden the parser against what you SEE here, not what you
guess.

What it does
------------
1. Builds N real prompts per task from ScenarioSampler + build_training_prompt
   (using placeholder client descriptions if real ones aren't generated yet).
2. Generates G completions per prompt with the chosen model.
3. Runs every completion through compute_rewards_from_output and records:
     - parse success / failure,
     - the failure CATEGORY (no tags / fenced JSON / bad JSON / schema / etc.),
     - a saved sample of each category for eyeballing.
4. Prints a breakdown table and writes raw outputs + categorized samples to
   outputs/parser_diagnostic/ so you can read what the model actually produced.

Usage
-----
    python scripts/parser_diagnostic.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --tasks portfolio retirement loan --n-prompts 8 --num-generations 4

    python scripts/parser_diagnostic.py --cpu --n-prompts 2   # tiny local run

Requires the 'train' extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("parser_diagnostic")


# ---------------------------------------------------------------------------
# Failure taxonomy: classify WHY a completion failed to parse, by inspecting
# the raw text + the ParserError message. These categories map directly to the
# hardening you'll do in step 2.
# ---------------------------------------------------------------------------
_PLAN_TAG = re.compile(r"<financial_plan>(.*?)</financial_plan>", re.DOTALL | re.I)
_FENCE = re.compile(r"```(?:json)?", re.I)


def classify_failure(raw: str, err_msg: str) -> str:
    """Bucket a parse failure into an actionable category."""
    has_open = "<financial_plan>" in raw.lower()
    has_close = "</financial_plan>" in raw.lower()

    if not has_open and not has_close:
        # did the model emit JSON at all, just without tags?
        if "{" in raw and "}" in raw:
            return "json_without_tags"
        return "no_tags_no_json"
    if has_open and not has_close:
        return "unterminated_tag"          # ran out of tokens mid-plan

    # tags present — what's inside?
    m = _PLAN_TAG.search(raw)
    inner = m.group(1).strip() if m else ""
    if _FENCE.search(inner):
        return "fenced_json_inside_tags"   # ```json ... ``` wrapping
    if "json parse error" in err_msg.lower():
        # distinguish common JSON syntax problems
        if "'" in inner and '"' not in inner:
            return "single_quoted_json"
        if re.search(r",\s*[}\]]", inner):
            return "trailing_comma"
        return "other_json_syntax"
    # JSON parsed but schema/validation failed
    if "missing" in err_msg.lower():
        return "schema_missing_field"
    if "expected" in err_msg.lower() and "assets" in err_msg.lower():
        return "wrong_asset_count"
    if "outside [0, 1]" in err_msg or "must be" in err_msg.lower():
        return "value_out_of_range"
    return "other_validation"


# ---------------------------------------------------------------------------
def build_real_prompts(task, n_prompts, tokenizer):
    """Build n real prompts for `task` using the dataset sampler.

    Uses a placeholder client description when real GPT-4o/Claude descriptions
    aren't on disk yet (step 3 of Day 4 generates the real ones). The parser
    diagnostic doesn't need polished prose — it needs the real SCHEMA prompt the
    model will actually see.
    """
    from finplanenv.dataset import DatasetGenerator, DatasetConfig

    gen = DatasetGenerator(DatasetConfig())
    rows, meta = [], []
    placeholder = ("A client seeking a sound financial plan. "
                   "Moderate risk tolerance, long-term horizon.")
    for idx in range(n_prompts):
        scenario, constraints, metadata = gen.sampler.sample(task, idx)
        system, user = gen.build_training_prompt(task, metadata, placeholder)
        # embed instance_id so the store could resolve it (not needed here)
        chat = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        if PREFILL[0]:
            # Force the model to start INSIDE the plan tag. This is the single
            # strongest lever for small models that otherwise drift into markdown.
            text = text + "<financial_plan>\n{"
        rows.append(text)
        meta.append((scenario, constraints))
    return rows, meta


# module-level toggle set from CLI (kept simple to avoid threading through calls)
PREFILL = [False]
PREFILL_TEXT = "<financial_plan>\n{"


def generate(model, tokenizer, prompts, G, max_new_tokens, device):
    """Generate G completions per prompt. Returns list[list[str]]."""
    import torch
    outs = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.95,
                num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
            )
        comp = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(comp, skip_special_tokens=True)
        if PREFILL[0]:
            # the model continued from "<financial_plan>\n{"; prepend it back so
            # the parser sees a complete, tagged block.
            decoded = [PREFILL_TEXT + d for d in decoded]
        outs.append(decoded)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tasks", nargs="+", default=["portfolio", "retirement", "loan"])
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--prefill", action="store_true",
                    help="Prefill the assistant turn with '<financial_plan>\\n{' "
                         "to force in-tag JSON generation. NOTE: empirically this "
                         "HURT the 0.5B (trades format errors for schema errors — "
                         "it denies the model reasoning space and it invents fields). "
                         "Left as an opt-in for experimentation; default OFF.")
    ap.add_argument("--out", default="outputs/parser_diagnostic")
    args = ap.parse_args()

    PREFILL[0] = args.prefill

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        sys.exit(f"train extra not installed ({e}). pip install -e '.[train]'")

    from finplanenv.parser import compute_rewards_from_output, ParserError

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("device=%s model=%s", device, args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.cpu else torch.bfloat16,
        device_map=None if args.cpu else "auto",
    )
    if args.cpu:
        model = model.to("cpu")

    grand_total = 0
    grand_success = 0
    all_samples = defaultdict(list)   # category -> [raw text samples]

    for task in args.tasks:
        logger.info("[%s] building %d prompts ...", task, args.n_prompts)
        prompts, meta = build_real_prompts(task, args.n_prompts, tokenizer)
        logger.info("[%s] generating %d completions each ...", task, args.num_generations)
        completions = generate(model, tokenizer, prompts,
                               args.num_generations, args.max_new_tokens, device)

        cats = Counter()
        n_total = n_success = 0
        raw_dump = []

        for pi, comps in enumerate(completions):
            scenario, constraints = meta[pi]
            for ci, comp in enumerate(comps):
                n_total += 1
                # capture the ParserError message (if any) by re-parsing
                err_msg = ""
                try:
                    # parse directly to see if it raises (compute swallows it)
                    from finplanenv import parser as _p
                    if task == "portfolio":
                        _p.parse_portfolio(comp, scenario)
                    elif task == "retirement":
                        _p.parse_retirement(comp, scenario)
                    else:
                        _p.parse_loan(comp, scenario)
                    parsed_ok = True
                except ParserError as e:
                    parsed_ok = False
                    err_msg = str(e)
                except Exception as e:  # unexpected, record it
                    parsed_ok = False
                    err_msg = f"UNEXPECTED {type(e).__name__}: {e}"

                if parsed_ok:
                    n_success += 1
                    cats["success"] += 1
                else:
                    cat = classify_failure(comp, err_msg)
                    cats[cat] += 1
                    if len(all_samples[f"{task}/{cat}"]) < 3:
                        all_samples[f"{task}/{cat}"].append(comp)

                raw_dump.append({
                    "prompt_idx": pi, "gen_idx": ci,
                    "parsed_ok": parsed_ok,
                    "category": "success" if parsed_ok else classify_failure(comp, err_msg),
                    "err": err_msg,
                    "raw": comp,
                })

        # write raw dump for this task
        (out_dir / f"{task}_raw.jsonl").write_text(
            "\n".join(json.dumps(r) for r in raw_dump)
        )

        grand_total += n_total
        grand_success += n_success

        print(f"\n{'='*56}\n  {task.upper()}  —  {n_success}/{n_total} parsed "
              f"({100*n_success/max(n_total,1):.0f}%)\n{'='*56}")
        for cat, n in cats.most_common():
            bar = "#" * int(30 * n / max(n_total, 1))
            print(f"  {cat:<26} {n:>3}  {bar}")

    # save categorized samples
    samples_path = out_dir / "failure_samples.txt"
    with open(samples_path, "w") as f:
        for key, samples in sorted(all_samples.items()):
            f.write(f"\n{'='*70}\n{key}\n{'='*70}\n")
            for i, s in enumerate(samples):
                f.write(f"\n--- sample {i+1} ---\n{s[:1200]}\n")

    print(f"\n{'='*56}")
    print(f"  OVERALL: {grand_success}/{grand_total} parsed "
          f"({100*grand_success/max(grand_total,1):.0f}%)")
    print(f"  raw outputs       -> {out_dir}/<task>_raw.jsonl")
    print(f"  failure samples   -> {samples_path}")
    print(f"{'='*56}")
    print("\nNext: read failure_samples.txt, then harden parser.py against the")
    print("categories that dominate. Re-run this to confirm the rate improves.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/generate_pilot.py
==========================
Generate the pilot dataset (60 instances per task) and report whether
the difficulty calibration matches the 30/40/30 easy/medium/hard target.

This is the Week 1 critical path step — run BEFORE generating the full
500+ instance dataset or starting any RL training.

Usage:
    python scripts/generate_pilot.py [--tasks portfolio retirement loan]
                                     [--seed 42]
                                     [--output data/pilot]

What it does:
    1. Samples 60 scenario instances per task (20 per difficulty tier)
    2. Saves metadata to data/pilot/<task>_pilot.jsonl
    3. Prints calibration report showing tier fractions
    4. Prints generation prompts for the first instance of each task
       (copy-paste into GPT-4o/Claude to generate client descriptions)
    5. Prints a sample training prompt to verify the format

It does NOT run actual LLM inference — that requires your model.
For calibration you need to:
    a) Use the printed prompts to generate LLM responses
    b) Run compute_rewards_from_output() on each response
    c) Feed the CCR values to DatasetGenerator.calibration_report()
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tasks",  nargs="+",
                    default=["portfolio", "retirement", "loan"])
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--output", type=str, default="data/pilot")
parser.add_argument("--n-pilot", type=int, default=60,
                    help="Instances per task (default 60 = 20 per tier)")
args = parser.parse_args()

# ── imports ───────────────────────────────────────────────────────────────────
from finplanenv import DatasetConfig, ScenarioSampler, DatasetGenerator, GENERATION_PROMPTS

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

config  = DatasetConfig(
    n_instances_per_task = args.n_pilot,
    pilot_size           = args.n_pilot,
    seed                 = args.seed,
)
sampler = ScenarioSampler(config)
gen     = DatasetGenerator(config)

print()
print("=" * 65)
print("  finplanenv — Pilot Dataset Generation")
print("=" * 65)

for task in args.tasks:
    print(f"\n{'─'*65}")
    print(f"  Task: {task.upper()}")
    print(f"{'─'*65}")

    # Generate instances
    instances = gen.generate_instances(task, 0, args.n_pilot)

    # Save to JSONL
    out_path = output_dir / f"{task}_pilot.jsonl"
    gen.save_metadata(instances, out_path)
    print(f"  Saved {len(instances)} instances → {out_path}")

    # Tier distribution summary
    tier_counts = {"easy": 0, "medium": 0, "hard": 0}
    for inst in instances:
        tier_counts[inst["difficulty"]] += 1

    print(f"\n  Tier distribution (target: 30/40/30):")
    for tier, count in tier_counts.items():
        frac = count / len(instances)
        bar  = "█" * int(frac * 30)
        ok   = "✓" if abs(frac - {"easy":0.30,"medium":0.40,"hard":0.30}[tier]) < 0.05 else "⚠"
        print(f"    {tier:<8} {count:>3} ({frac:.0%})  {bar}  {ok}")

    # Sample generation prompt (first instance)
    _, _, meta = sampler.sample(task, 0)
    prompt_fn  = GENERATION_PROMPTS[task]
    gen_prompt = prompt_fn(meta, n_profiles=5)

    print(f"\n  Generation prompt for instance 0 (send to GPT-4o/Claude):")
    print(f"  {'─'*55}")
    # Print first 600 chars
    preview = gen_prompt[:600]
    for line in preview.split("\n"):
        print(f"  {line}")
    if len(gen_prompt) > 600:
        print(f"  ... [{len(gen_prompt)-600} more chars]")
    print(f"  {'─'*55}")

    # Sample training prompt
    sample_description = (
        "Jordan Chen, 42, software engineer, San Francisco. "
        "Married with two children. Portfolio of $800,000 built over 15 years. "
        "Moderate risk tolerance, strong ESG preference (especially climate). "
        "Investment horizon: 10 years before retirement."
    )
    system_p, user_p = gen.build_training_prompt(task, instances[0], sample_description)
    print(f"\n  Sample training prompt (system + user) for instance 0:")
    print(f"  System prompt: {len(system_p)} chars")
    preview_user = user_p[:400]
    for line in preview_user.split("\n"):
        print(f"  {line}")
    if len(user_p) > 400:
        print(f"  ... [{len(user_p)-400} more chars]")

print()
print("=" * 65)
print("  Next steps:")
print("  1. Review the generation prompts above")
print("  2. Send each to GPT-4o/Claude to generate client descriptions")
print("  3. Generate LLM plans using the training prompts")
print("  4. Run compute_rewards_from_output() on each plan")
print("  5. Feed CCR values to DatasetGenerator.calibration_report()")
print("  6. If status=OK, run scripts/generate_full.py")
print("=" * 65)
print()

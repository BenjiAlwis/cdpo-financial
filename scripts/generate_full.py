#!/usr/bin/env python3
"""
scripts/generate_full.py
========================
Generate the full 550-instance dataset (500 train + 50 audit) for all tasks.

Run this ONLY after scripts/generate_pilot.py has confirmed calibration is OK.

Usage:
    python scripts/generate_full.py [--tasks portfolio retirement loan]
                                    [--n-per-task 550]
                                    [--seed 42]
                                    [--output data/full]

Output:
    data/full/<task>_train.jsonl    — 500 training instances
    data/full/<task>_audit.jsonl    — 50 manual audit instances
    data/full/summary.json          — dataset statistics
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tasks",       nargs="+",
                    default=["portfolio", "retirement", "loan"])
parser.add_argument("--n-per-task",  type=int, default=550)
parser.add_argument("--n-audit",     type=int, default=50)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--output",      type=str, default="data/full")
parser.add_argument("--conflict-level", type=float, default=0.0,
                    help="Hard/soft conflict injection (0=none/original, "
                         "1.0=full). >0 bans the highest-ESG asset's sector so "
                         "maximizing ESG conflicts with the H3 hard constraint. "
                         "This is the regime where CDPO should beat the baselines.")
parser.add_argument("--no-conflict-by-tier", dest="conflict_by_tier",
                    action="store_false", default=True,
                    help="Apply conflict uniformly instead of scaling easy→hard.")
args = parser.parse_args()

from finplanenv import DatasetConfig, ScenarioSampler, DatasetGenerator

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

config  = DatasetConfig(
    n_instances_per_task = args.n_per_task,
    seed                 = args.seed,
    conflict_level       = args.conflict_level,
    conflict_by_tier     = args.conflict_by_tier,
)
if args.conflict_level > 0:
    print(f"  Conflict calibration: level={args.conflict_level} "
          f"by_tier={args.conflict_by_tier}")
sampler = ScenarioSampler(config)
gen     = DatasetGenerator(config)

summary = {
    "generated_at": datetime.now().isoformat(),
    "seed": args.seed,
    "tasks": {},
}

print()
print("=" * 65)
print("  finplanenv — Full Dataset Generation")
print(f"  {args.n_per_task} instances per task  "
      f"({args.n_per_task - args.n_audit} train + {args.n_audit} audit)")
print("=" * 65)

for task in args.tasks:
    print(f"\n  Generating {task}...", end="", flush=True)

    all_instances = gen.generate_instances(task, 0, args.n_per_task)

    # Split: last n_audit instances → audit set
    train_instances = all_instances[:-args.n_audit]
    audit_instances = all_instances[-args.n_audit:]

    train_path = output_dir / f"{task}_train.jsonl"
    audit_path = output_dir / f"{task}_audit.jsonl"

    gen.save_metadata(train_instances, train_path)
    gen.save_metadata(audit_instances, audit_path)

    # Tier counts
    tier_counts = {"easy": 0, "medium": 0, "hard": 0}
    for inst in all_instances:
        tier_counts[inst["difficulty"]] += 1

    summary["tasks"][task] = {
        "n_train": len(train_instances),
        "n_audit": len(audit_instances),
        "tier_counts": tier_counts,
        "train_path": str(train_path),
        "audit_path": str(audit_path),
    }

    print(f" done.  {len(train_instances)} train, {args.n_audit} audit")
    print(f"    Tiers: easy={tier_counts['easy']} "
          f"medium={tier_counts['medium']} hard={tier_counts['hard']}")

# Save summary
summary_path = output_dir / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 65)
print(f"  Dataset saved to {output_dir}/")
print(f"  Summary: {summary_path}")
print()
print("  Next steps:")
print("  1. Generate LLM client descriptions using GENERATION_PROMPTS")
print("     and merge into the JSONL files")
print("  2. Run the manual audit on <task>_audit.jsonl (50 cases per task)")
print("  3. Confirm constraint checkers produce correct results on audit set")
print("  4. Configure verl/TRL with CDPORewardWrapper")
print("  5. Start Week 2 training runs")
print("=" * 65)
print()

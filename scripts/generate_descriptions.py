#!/usr/bin/env python3
"""
scripts/generate_descriptions.py
================================
Generate natural-language client descriptions for dataset instances, WITHOUT
calling an external LLM. Each description is templated deterministically from the
instance's `narrative_hints`, so it is free, reproducible, and requires no API.

Why templated (not GPT-4o/Claude)?
----------------------------------
`build_training_prompt` slots a `client_description` into the user prompt as the
human-readable client profile. The scenario NUMBERS (constraints, returns) are
rendered separately from metadata, so the model can plan from the numbers alone;
the description is framing. A deterministic template is good enough to train on
and removes an external dependency from the critical path. If you later want more
diversity, regenerate descriptions with an LLM using finplanenv.GENERATION_PROMPTS
and overwrite the description field — the format is identical.

Usage
-----
    # attach descriptions to an existing dataset produced by generate_full.py
    python scripts/generate_descriptions.py --data-dir data/full

    # or for the pilot
    python scripts/generate_descriptions.py --data-dir data/pilot --suffix _pilot

Output
------
For each <task>_train.jsonl / <task>_audit.jsonl (or <task>_pilot.jsonl), writes
a sibling <task>_*_with_desc.jsonl where every instance has a `client_description`
field. The ScenarioStore / training loop reads these.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Templating helpers — deterministic given the hints + a per-instance seed.
# ---------------------------------------------------------------------------
_NAMES = [
    "Jordan Chen", "Priya Nair", "Marcus Webb", "Sofia Russo", "Daniel Okoye",
    "Hannah Kim", "Liam O'Brien", "Aisha Rahman", "Tomas Vega", "Emily Schwartz",
    "Noah Adler", "Mei Lin", "Carlos Mendez", "Fatima Zahra", "Oliver Grant",
    "Yuki Tanaka", "Grace Mbeki", "Ravi Kapoor", "Lena Vogel", "Sam Whitfield",
]


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _describe_portfolio(h: dict, rng: random.Random) -> str:
    name = rng.choice(_NAMES)
    risk = h.get("risk_tolerance", "moderate")
    art = "an" if risk[:1].lower() in "aeiou" else "a"
    esg = h.get("esg_priority", "low")
    horizon = h.get("time_horizon_yr", 10)
    size = h.get("portfolio_size", 500_000)
    esg_clause = {
        "high": "strong ESG/sustainability preferences",
        "medium": "some interest in ESG-aligned holdings",
        "low": "no particular ESG constraints",
    }.get(esg, "no particular ESG constraints")
    return (
        f"{name} is an investor with {art} {risk} risk tolerance and {esg_clause}. "
        f"They hold a portfolio of roughly {_fmt_money(size)} and are planning "
        f"over a {int(horizon)}-year horizon. They want an allocation that "
        f"respects their stated risk limits and diversification preferences."
    )


def _describe_retirement(h: dict, rng: random.Random) -> str:
    name = rng.choice(_NAMES)
    cur = h.get("current_age", 65)
    tgt = h.get("target_age", 90)
    bal = h.get("initial_balance", 1_000_000)
    lifestyle = h.get("lifestyle", "comfortable")
    spouse = h.get("spouse", False)
    pension = h.get("has_pension", False)
    bequest = h.get("bequest_pref", "none")
    spouse_clause = "is married" if spouse else "is single"
    pension_clause = "has a pension" if pension else "has no pension income"
    bequest_clause = {
        "large": "wants to leave a substantial bequest",
        "modest": "would like to leave a modest bequest",
        "none": "has no specific bequest goal",
    }.get(bequest, "has no specific bequest goal")
    first = name.split()[0]
    return (
        f"{name}, age {int(cur)}, is planning for retirement through age {int(tgt)}. "
        f"{first} {spouse_clause}, {pension_clause}, and prefers a {lifestyle} lifestyle. "
        f"Current retirement balance is about {_fmt_money(bal)}. {first} "
        f"{bequest_clause} and needs a withdrawal plan that keeps income above their "
        f"required floor across market scenarios."
    )


def _describe_loan(h: dict, rng: random.Random) -> str:
    name = rng.choice(_NAMES)
    income = h.get("annual_income", 100_000)
    prop = h.get("property_value", 500_000)
    credit = h.get("credit_score", 700)
    employment = h.get("employment", "employed").replace("_", "-")
    purpose = h.get("loan_purpose", "purchase")
    tier = h.get("regulatory_tier", "QM")
    debts = h.get("existing_debts_monthly", 0)
    return (
        f"{name} is a {employment} borrower with an annual income of "
        f"{_fmt_money(income)} and a credit score of {int(credit)}. They are "
        f"seeking a mortgage for a {purpose} on a property valued at "
        f"{_fmt_money(prop)}, and carry about {_fmt_money(debts)} in existing "
        f"monthly debt payments. The loan must comply with {tier} regulatory rules "
        f"and stay within DTI and LTV limits."
    )


_DESCRIBERS = {
    "portfolio": _describe_portfolio,
    "retirement": _describe_retirement,
    "loan": _describe_loan,
}


def task_of(instance_id: str) -> str:
    return instance_id.split("_")[0]


def add_descriptions(in_path: Path, out_path: Path, base_seed: int) -> int:
    n = 0
    with open(in_path) as f, open(out_path, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            task = task_of(inst["instance_id"])
            hints = inst.get("narrative_hints", {})
            # deterministic per-instance rng so names/phrasing are reproducible
            idx = int(inst["instance_id"].split("_")[1])
            rng = random.Random(base_seed * 100_000 + idx)
            describer = _DESCRIBERS.get(task)
            if describer is None:
                raise ValueError(f"No describer for task {task!r}")
            inst["client_description"] = describer(hints, rng)
            g.write(json.dumps(inst) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="directory containing <task>_*.jsonl from generate_*.py")
    ap.add_argument("--tasks", nargs="+",
                    default=["portfolio", "retirement", "loan"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="",
                    help="filename suffix to match (e.g. '_pilot'); default tries "
                         "_train and _audit")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir {data_dir} not found — run generate_full.py first")

    if args.suffix:
        splits = [args.suffix]
    else:
        splits = ["_train", "_audit"]

    total = 0
    for task in args.tasks:
        for split in splits:
            in_path = data_dir / f"{task}{split}.jsonl"
            if not in_path.exists():
                continue
            out_path = data_dir / f"{task}{split}_with_desc.jsonl"
            n = add_descriptions(in_path, out_path, args.seed)
            total += n
            print(f"  {in_path.name:<32} -> {out_path.name}  ({n} instances)")

    if total == 0:
        raise SystemExit(
            "No input files matched. Expected files like "
            f"{data_dir}/portfolio_train.jsonl. Run generate_full.py first, or "
            "pass --suffix _pilot for a pilot dataset."
        )
    print(f"\n  Done. {total} instances now have client_description.")
    print("  Point the training ScenarioStore at the *_with_desc.jsonl files.")


if __name__ == "__main__":
    main()

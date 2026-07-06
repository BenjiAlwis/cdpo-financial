#!/usr/bin/env python3
"""
scripts/measure_conflict.py
===========================
Measure the hard/soft CONFLICT in a generated portfolio dataset — a cheap CPU
sanity check to run BEFORE spending GPU on a training matrix.

Why this exists
---------------
The first training matrix produced a null result because the dataset had no
hard/soft conflict: maximizing the ESG soft-preference cost nothing on the hard
constraints, so CDPO's gate had nothing to arbitrate. This script quantifies
whether a dataset actually contains the conflict CDPO is designed to exploit, so
you never again burn ~$180 / ~7 days discovering the regime was wrong.

What it measures
----------------
For each scenario it draws many random valid portfolios and computes, per tier:
  - H3-pass rate: fraction of portfolios satisfying the banned-sector hard
    constraint (want: satisfiable but non-trivial, roughly 0.1-0.7 on hard tier).
  - conflict_gap = mean ESG(fail H3) - mean ESG(pass H3): how much soft-pref
    (ESG) you must GIVE UP to satisfy the hard constraint. > 0 means real
    conflict; ~0 means none.
  - corr(ESG, hard_pass): global correlation; <0 indicates tension.

Verdict
-------
Prints PASS if the hard tier shows a meaningful conflict gap, FAIL otherwise —
so you know whether the dataset is worth training on before launching the matrix.

Usage
-----
    # measure a dataset dir produced by generate_full.py
    python scripts/measure_conflict.py --data-dir data/full_conflict

    # or measure the sampler directly at a given conflict level (no files)
    python scripts/measure_conflict.py --conflict-level 1.0
    python scripts/measure_conflict.py --conflict-level 0.0   # should FAIL (null)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from finplanenv.rewards import portfolio_rewards, PortfolioPlan
from finplanenv.dataset import ScenarioSampler, DatasetConfig, MARKET_PARAMS


SECTORS = MARKET_PARAMS["portfolio"]["sectors"]
ESG = MARKET_PARAMS["portfolio"]["esg_ratings"]
ER = MARKET_PARAMS["portfolio"]["expected_returns"]
BENCH = MARKET_PARAMS["portfolio"]["benchmark_weights"]


def measure_scenario(scen, cons, rng, n_samples=300):
    """Return arrays (esg, hard_pass_frac, h3_pass) over random valid portfolios."""
    esg, hardfrac, h3 = [], [], []
    cov = scen.cov_matrix
    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(len(SECTORS)))
        plan = PortfolioPlan(
            weights=w, sector_labels=SECTORS, esg_ratings=ESG,
            benchmark_weights=BENCH, return_series=scen.return_series,
            expected_returns=ER, cov_matrix=cov,
        )
        rb = portfolio_rewards(plan, cons)
        esg.append(rb.soft[1])          # S2 = ESG
        hardfrac.append(rb.hard.mean())
        h3.append(rb.hard[2])           # H3 = banned sector (weight-based)
    return np.array(esg), np.array(hardfrac), np.array(h3)


def summarize(tier_name, esg, hardfrac, h3):
    h3rate = float(h3.mean())
    esg_pass = float(esg[h3 == 1].mean()) if (h3 == 1).any() else float("nan")
    esg_fail = float(esg[h3 == 0].mean()) if (h3 == 0).any() else float("nan")
    gap = (esg_fail - esg_pass) if (not np.isnan(esg_pass) and not np.isnan(esg_fail)) else float("nan")
    corr = float(np.corrcoef(esg, hardfrac)[0, 1]) if esg.std() > 0 and hardfrac.std() > 0 else float("nan")
    print(f"  {tier_name:8} H3-pass={h3rate:5.2f}  "
          f"ESG|pass={esg_pass:5.3f}  ESG|fail={esg_fail:5.3f}  "
          f"gap={gap:+.3f}  corr(ESG,hard)={corr:+.3f}")
    return gap, h3rate


def from_sampler(conflict_level, by_tier=True, n_scen=20, seed=1):
    cfg = DatasetConfig(conflict_level=conflict_level, conflict_by_tier=by_tier)
    sampler = ScenarioSampler(cfg)
    rng = np.random.default_rng(seed)
    buckets = {"easy": [[], [], []], "medium": [[], [], []], "hard": [[], [], []]}
    # sample a spread of indices across tiers
    for idx in range(0, cfg.n_instances_per_task, max(1, cfg.n_instances_per_task // (n_scen * 3))):
        scen, cons, meta = sampler.sample_portfolio(idx)
        tier = meta["difficulty"]
        e, hf, h3 = measure_scenario(scen, cons, rng)
        buckets[tier][0].append(e); buckets[tier][1].append(hf); buckets[tier][2].append(h3)
    return buckets


def from_data_dir(data_dir, seed=1):
    """Reconstruct scenarios from a generated dataset's metadata and measure."""
    path = Path(data_dir) / "portfolio_train.jsonl"
    if not path.exists():
        path = Path(data_dir) / "portfolio_train_with_desc.jsonl"
    if not path.exists():
        raise SystemExit(f"No portfolio_train.jsonl in {data_dir}")
    # We need the sampler to rebuild scenarios; the dataset stores instance ids
    # and difficulty. Rebuild via the sampler using the same seed/config is not
    # possible without the exact config, so we read the stored banned_sectors and
    # reconstruct constraints directly from metadata.
    from finplanenv.rewards import PortfolioConstraints
    from finplanenv.dataset import _sample_return_series, _build_portfolio_cov
    rng = np.random.default_rng(seed)
    buckets = {"easy": [[], [], []], "medium": [[], [], []], "hard": [[], [], []]}
    params = MARKET_PARAMS["portfolio"]
    for line in open(path):
        d = json.loads(line)
        tier = d.get("difficulty", "medium")
        banned = set(d.get("banned_sectors", []))
        cons = PortfolioConstraints(
            weight_sum_tol=0.01,
            max_drawdown_limit=d.get("max_drawdown_limit", -0.10),
            banned_sectors=banned,
            hhi_limit=d.get("hhi_limit", 0.25),
            risk_free_rate=0.04,
        )
        # a representative return series + cov (scenario-level randomness ok for
        # a conflict measurement, which is about ESG-vs-banned-sector structure)
        class _S:
            return_series = _sample_return_series(params, rng)
            cov_matrix = _build_portfolio_cov(params)
        e, hf, h3 = measure_scenario(_S(), cons, rng)
        buckets[tier][0].append(e); buckets[tier][1].append(hf); buckets[tier][2].append(h3)
    return buckets


def report(buckets):
    print(f"\n{'='*64}\n  CONFLICT MEASUREMENT (per difficulty tier)\n{'='*64}")
    hard_gap = None
    for tier in ["easy", "medium", "hard"]:
        e, hf, h3 = buckets[tier]
        if not e:
            print(f"  {tier:8} (no scenarios)")
            continue
        e = np.concatenate(e); hf = np.concatenate(hf); h3 = np.concatenate(h3)
        gap, _ = summarize(tier, e, hf, h3)
        if tier == "hard":
            hard_gap = gap
    print("=" * 64)
    if hard_gap is None or np.isnan(hard_gap):
        print("  VERDICT: FAIL — no hard-tier conflict detected. Do NOT train:")
        print("  the dataset has no ESG↔hard tension, so CDPO's gate is inert.")
        print("  Regenerate with:  generate_full.py --conflict-level 1.0")
        return 1
    if hard_gap >= 0.03:
        print(f"  VERDICT: PASS — hard-tier conflict gap = {hard_gap:+.3f}.")
        print("  Satisfying the hard constraint costs ESG => real conflict.")
        print("  This dataset exercises CDPO's mechanism; OK to train.")
        return 0
    print(f"  VERDICT: WEAK — hard-tier gap = {hard_gap:+.3f} (< 0.03).")
    print("  Conflict is marginal; consider --conflict-level 1.0 or stronger.")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Generated dataset dir to measure (portfolio_train.jsonl).")
    ap.add_argument("--conflict-level", type=float, default=None,
                    help="Measure the sampler directly at this level (no files).")
    ap.add_argument("--no-conflict-by-tier", dest="by_tier",
                    action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.data_dir:
        print(f"Measuring dataset: {args.data_dir}")
        buckets = from_data_dir(args.data_dir, seed=args.seed)
    else:
        lvl = args.conflict_level if args.conflict_level is not None else 0.0
        print(f"Measuring sampler directly: conflict_level={lvl}, by_tier={args.by_tier}")
        buckets = from_sampler(lvl, by_tier=args.by_tier, seed=args.seed)

    raise SystemExit(report(buckets))


if __name__ == "__main__":
    main()

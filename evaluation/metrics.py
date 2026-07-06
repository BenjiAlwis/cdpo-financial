"""
evaluation/metrics.py
=====================
Evaluation metrics for CDPO experiments.

Primary metrics (Table 1 in paper):
    CCR  — Constraint Compliance Rate: fraction of plans satisfying ALL hard constraints
    SPS  — Soft Preference Score: mean soft score conditional on compliance
    CQ   — Combined Quality: CCR × SPS

Reported per:
    - Task (portfolio / retirement / loan)
    - Difficulty tier (easy / medium / hard)
    - Method (grpo / gdpo / cdpo)

Also reports:
    Per-constraint pass rates (h1..hK) — for Figure 3
    Per-soft-preference scores (s1..sM)
    Parse failure rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class TaskMetrics:
    """Metrics for one task evaluated on one split."""
    task:       str
    method:     str
    split:      str   # "train" | "eval"
    n_plans:    int   = 0

    # Primary metrics
    ccr:        float = 0.0   # Constraint Compliance Rate
    sps:        float = 0.0   # Soft Preference Score (conditional)
    cq:         float = 0.0   # Combined Quality = CCR × SPS

    # Per-constraint pass rates
    per_hard:   list[float] = field(default_factory=list)

    # Per-soft-preference scores (conditional on compliance)
    per_soft:   list[float] = field(default_factory=list)

    # Parse health
    parse_failure_rate: float = 0.0

    # Per-tier breakdown
    tier_ccr:   dict[str, float] = field(default_factory=dict)
    tier_cq:    dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "task":    self.task,
            "method":  self.method,
            "split":   self.split,
            "n_plans": self.n_plans,
            "ccr":     round(self.ccr, 4),
            "sps":     round(self.sps, 4),
            "cq":      round(self.cq,  4),
            "parse_failure_rate": round(self.parse_failure_rate, 4),
        }
        for i, v in enumerate(self.per_hard):
            d[f"pass_h{i+1}"] = round(v, 4)
        for i, v in enumerate(self.per_soft):
            d[f"score_s{i+1}"] = round(v, 4)
        for tier, ccr in self.tier_ccr.items():
            d[f"ccr_{tier}"] = round(ccr, 4)
        for tier, cq in self.tier_cq.items():
            d[f"cq_{tier}"] = round(cq, 4)
        return d


def compute_metrics(
    results: list[dict],
    task:    str,
    method:  str,
    split:   str = "eval",
) -> TaskMetrics:
    """
    Compute TaskMetrics from a list of evaluation results.

    Each result dict must have:
        hard      : list[int]   — hard signal values {0,1} length K
        soft      : list[float] — soft signal values [0,1] length M
        difficulty: str         — "easy"|"medium"|"hard"
        parsed    : bool        — True if output was successfully parsed

    Parameters
    ----------
    results : list of per-plan result dicts
    task    : task name
    method  : method name
    split   : "train" or "eval"
    """
    if not results:
        return TaskMetrics(task=task, method=method, split=split)

    K = len(results[0]["hard"])
    M = len(results[0]["soft"])

    hard_arr    = np.array([r["hard"] for r in results], dtype=float)  # (N,K)
    soft_arr    = np.array([r["soft"] for r in results], dtype=float)  # (N,M)
    parsed_arr  = np.array([r["parsed"] for r in results], dtype=bool) # (N,)
    tiers       = [r.get("difficulty", "unknown") for r in results]

    all_pass    = hard_arr.min(axis=1) == 1   # (N,)  True iff all K constraints pass
    ccr         = float(all_pass.mean())

    if all_pass.sum() > 0:
        sps = float(soft_arr[all_pass].mean())
    else:
        sps = 0.0

    cq = ccr * sps

    per_hard = [float(hard_arr[:, k].mean()) for k in range(K)]
    per_soft = (
        [float(soft_arr[all_pass, m].mean()) for m in range(M)]
        if all_pass.sum() > 0
        else [0.0] * M
    )

    parse_failure_rate = float((~parsed_arr).mean())

    # Per-tier breakdown
    tier_ccr = {}
    tier_cq  = {}
    for tier in ["easy", "medium", "hard"]:
        mask = np.array([t == tier for t in tiers])
        if mask.sum() == 0:
            continue
        t_ccr = float(all_pass[mask].mean())
        t_sps = (float(soft_arr[all_pass & mask].mean())
                 if (all_pass & mask).sum() > 0 else 0.0)
        tier_ccr[tier] = t_ccr
        tier_cq[tier]  = t_ccr * t_sps

    return TaskMetrics(
        task               = task,
        method             = method,
        split              = split,
        n_plans            = len(results),
        ccr                = ccr,
        sps                = sps,
        cq                 = cq,
        per_hard           = per_hard,
        per_soft           = per_soft,
        parse_failure_rate = parse_failure_rate,
        tier_ccr           = tier_ccr,
        tier_cq            = tier_cq,
    )


def print_comparison_table(
    metrics_by_method: dict[str, list[TaskMetrics]],
) -> None:
    """
    Print a comparison table across methods.
    Format mirrors Table 1 in the paper.
    """
    tasks   = ["portfolio", "retirement", "loan"]
    methods = list(metrics_by_method.keys())

    W = 12

    print(f"\n{'─'*80}")
    print(f"  Evaluation Results — CCR / SPS / CQ per task")
    print(f"{'─'*80}")
    print(f"  {'Task':<14}  {'Metric':<6}  " +
          "  ".join(f"{m.upper():>{W}}" for m in methods))
    print(f"  {'─'*14}  {'─'*6}  " +
          "  ".join("─"*W for _ in methods))

    for task in tasks:
        for metric_key, label in [("ccr","CCR"), ("sps","SPS"), ("cq","CQ")]:
            row = f"  {task if metric_key=='ccr' else '':14}  {label:<6}  "
            for method in methods:
                # Find metrics for this task+method
                ms = [m for m in metrics_by_method[method] if m.task == task]
                val = getattr(ms[0], metric_key) if ms else float("nan")
                row += f"  {val:>{W}.4f}"
            print(row)
        print()

    print(f"{'─'*80}")

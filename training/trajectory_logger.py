"""
training/trajectory_logger.py
=============================
Per-step trajectory logging for the CCR-vs-steps paper figure.

Why this exists
---------------
The CDPO-over-GDPO claim is *dynamic*: it lives in how CCR (constraint compliance
rate) evolves over training steps, not in any single snapshot. To build that
figure you need a clean, per-step record of CCR / SPS / CQ for every run, keyed by
(method, task, seed), that survives even if W&B is unavailable.

This helper:
  • logs every step (not every 10) so the curve is smooth;
  • computes CCR directly from r_hard so it never silently depends on whatever the
    advantage computer happens to put in its metrics dict;
  • mirrors everything to a local CSV so `plot_ccr_curves.py` works offline.

Drop-in usage in training/reward_fns.py
----------------------------------------
    from training.trajectory_logger import TrajectoryLogger

    # build once, alongside the reward_fn (needs the run identity):
    traj = TrajectoryLogger(method=method, task=task, seed=seed,
                            out_dir=out_dir)

    # inside reward_fn(), AFTER r_hard / r_soft are filled and A_hat computed:
    traj.log_step(step=step_counter[0],
                  r_hard=r_hard, r_soft=r_soft, A_hat=A_hat,
                  parse_failures=parse_failures, total=B_G)

That single call replaces the every-10-steps `_log_metrics` block for the figure
metrics (keep `_log_metrics` if you still want the console line).
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np


class TrajectoryLogger:
    """Logs per-step CCR/SPS/CQ to W&B (if live) and to a local CSV always."""

    def __init__(
        self,
        method: str,
        task: str,
        seed: int,
        out_dir: str,
        wandb_prefix: Optional[str] = None,
    ) -> None:
        self.method = method
        self.task = task
        self.seed = seed
        # W&B keys are namespaced so multiple metrics group cleanly in the UI.
        self.prefix = wandb_prefix or method

        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(
            out_dir, f"trajectory_{method}_{task}_seed{seed}.csv"
        )
        self._csv_initialised = False

        # detect a live W&B run once
        self._wandb = None
        try:
            import wandb  # noqa
            if wandb.run is not None:
                self._wandb = wandb
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_ccr(r_hard: np.ndarray) -> float:
        """Fraction of rollouts that satisfy ALL hard constraints.

        r_hard: (B, G, K) array of {0,1}. A rollout is compliant iff every one of
        its K constraints passed, i.e. the min over the K axis is 1.
        """
        # min over constraints == 1  ⇔  all constraints passed
        compliant = r_hard.min(axis=2) == 1          # (B, G) bool
        return float(compliant.mean())

    @staticmethod
    def compute_sps(r_hard: np.ndarray, r_soft: np.ndarray) -> float:
        """Mean soft-preference score, CONDITIONAL on full compliance.

        SPS is only meaningful for plans that are actually usable (all hard
        constraints satisfied). Scoring soft quality on infeasible plans would
        reward good-looking-but-illegal plans. Returns 0.0 if nothing complies.
        """
        compliant = r_hard.min(axis=2) == 1          # (B, G)
        if compliant.sum() == 0:
            return 0.0
        # average soft signals first over M, then over the compliant rollouts
        soft_per_rollout = r_soft.mean(axis=2)        # (B, G)
        return float(soft_per_rollout[compliant].mean())

    @staticmethod
    def count_distinct_advantages(A_hat: np.ndarray, decimals: int = 4) -> int:
        return len(set(np.round(np.asarray(A_hat).flatten(), decimals).tolist()))

    # ------------------------------------------------------------------ #
    def log_step(
        self,
        step: int,
        r_hard: np.ndarray,
        r_soft: np.ndarray,
        A_hat: np.ndarray,
        parse_failures: int = 0,
        total: int = 0,
    ) -> dict:
        """Compute the figure metrics for this step and record them everywhere."""
        ccr = self.compute_ccr(r_hard)
        sps = self.compute_sps(r_hard, r_soft)
        cq = ccr * sps                                   # combined quality
        n_distinct = self.count_distinct_advantages(A_hat)
        parse_rate = (parse_failures / total) if total else 0.0

        row = {
            "step": step,
            "method": self.method,
            "task": self.task,
            "seed": self.seed,
            "ccr": ccr,
            "sps": sps,
            "cq": cq,
            "n_distinct_advantages": n_distinct,
            "parse_failure_rate": parse_rate,
        }

        # --- W&B (per step) ---
        if self._wandb is not None:
            self._wandb.log(
                {
                    f"{self.prefix}/ccr": ccr,
                    f"{self.prefix}/sps": sps,
                    f"{self.prefix}/cq": cq,
                    f"{self.prefix}/n_distinct_advantages": n_distinct,
                    f"{self.prefix}/parse_failure_rate": parse_rate,
                },
                step=step,
            )

        # --- CSV (always; this is what the plotter reads offline) ---
        self._append_csv(row)
        return row

    # ------------------------------------------------------------------ #
    def _append_csv(self, row: dict) -> None:
        write_header = not self._csv_initialised and not os.path.exists(self.csv_path)
        mode = "a"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._csv_initialised = True

#!/usr/bin/env python3
"""
plot_ccr_curves.py
==================
Build the paper's hero dynamic figure: CCR (and SPS, CQ) vs training step,
averaged over seeds, with variance bands, all methods overlaid.

It reads the per-step CSVs written by training/trajectory_logger.py
(`trajectory_<method>_<task>_seed<seed>.csv`). No W&B needed — the CSVs are the
source of truth, so this runs offline / locally after pulling logs off RunPod.

Usage
-----
    python plot_ccr_curves.py --logdir outputs/ --task portfolio
    python plot_ccr_curves.py --logdir outputs/ --task portfolio --metric cq
    python plot_ccr_curves.py --logdir outputs/ --task portfolio --band sem --smooth 5

Options
-------
    --metric {ccr,sps,cq}   which trajectory to plot (default ccr)
    --band   {std,sem,minmax,none}  variance band style (default std)
    --smooth N              rolling-mean window over steps (default 1 = none)
    --out FILE              output image path (default ccr_curve_<task>.png)
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

# headless-safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHOD_ORDER = ["grpo", "gdpo", "cdpo"]
METHOD_COLORS = {           # colour-blind-safe, CDPO highlighted
    "grpo": "#888888",
    "gdpo": "#1f77b4",
    "cdpo": "#b00020",      # maroon — the method of interest
}
METHOD_LABELS = {"grpo": "GRPO", "gdpo": "GDPO", "cdpo": "CDPO"}
METRIC_LABELS = {
    "ccr": "Constraint Compliance Rate (CCR)",
    "sps": "Soft Preference Score (SPS)",
    "cq": "Combined Quality (CQ = CCR × SPS)",
}

FNAME_RE = re.compile(r"trajectory_(?P<method>\w+?)_(?P<task>\w+?)_seed(?P<seed>\d+)\.csv$")


def load_csv(path):
    """Return dict of column -> np.array from a trajectory CSV."""
    import csv
    cols = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols[k].append(v)
    out = {}
    for k, vals in cols.items():
        try:
            out[k] = np.array([float(x) for x in vals])
        except ValueError:
            out[k] = np.array(vals)        # string columns (method/task)
    return out


def rolling_mean(y, w):
    if w <= 1:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def gather(logdir, task, metric):
    """method -> {seed -> (steps, values)} for the requested task & metric."""
    runs = defaultdict(dict)
    pattern = os.path.join(logdir, "**", "trajectory_*.csv")
    for path in glob.glob(pattern, recursive=True):
        m = FNAME_RE.search(os.path.basename(path))
        if not m or m.group("task") != task:
            continue
        method = m.group("method")
        seed = int(m.group("seed"))
        data = load_csv(path)
        if metric not in data or "step" not in data:
            continue
        # sort by step, just in case
        order = np.argsort(data["step"])
        runs[method][seed] = (data["step"][order], data[metric][order])
    return runs


def align_seeds(seed_dict):
    """Stack per-seed curves onto a common step grid (intersection of steps)."""
    if not seed_dict:
        return None, None
    # use the shortest run's step axis as the common grid
    step_axes = [s for s, _ in seed_dict.values()]
    min_len = min(len(s) for s in step_axes)
    steps = next(iter(seed_dict.values()))[0][:min_len]
    stacked = np.vstack([v[:min_len] for _, v in seed_dict.values()])
    return steps, stacked    # (n_seeds, n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--metric", default="ccr", choices=["ccr", "sps", "cq"])
    ap.add_argument("--band", default="std", choices=["std", "sem", "minmax", "none"])
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = gather(args.logdir, args.task, args.metric)
    if not runs:
        raise SystemExit(
            f"No trajectory CSVs found for task='{args.task}' under {args.logdir}. "
            "Expected files like trajectory_cdpo_portfolio_seed42.csv"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_ORDER:
        if method not in runs:
            continue
        steps, stacked = align_seeds(runs[method])
        if steps is None:
            continue
        mean = rolling_mean(stacked.mean(axis=0), args.smooth)
        n = stacked.shape[0]

        color = METHOD_COLORS.get(method, None)
        lw = 2.6 if method == "cdpo" else 1.8
        ax.plot(steps, mean, label=f"{METHOD_LABELS.get(method, method)} (n={n})",
                color=color, linewidth=lw, zorder=3 if method == "cdpo" else 2)

        if args.band != "none" and n > 1:
            if args.band == "std":
                lo, hi = mean - stacked.std(0), mean + stacked.std(0)
            elif args.band == "sem":
                sem = stacked.std(0) / np.sqrt(n)
                lo, hi = mean - sem, mean + sem
            else:  # minmax
                lo, hi = stacked.min(0), stacked.max(0)
            lo = rolling_mean(lo, args.smooth)
            hi = rolling_mean(hi, args.smooth)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15, zorder=1)

    ax.set_xlabel("Training step")
    ax.set_ylabel(METRIC_LABELS[args.metric])
    ax.set_title(f"{METRIC_LABELS[args.metric]} over training — {args.task}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out = args.out or f"ccr_curve_{args.task}_{args.metric}.png"
    fig.savefig(out, dpi=160)
    # also save a vector copy for the paper
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print(f"wrote {out} and {os.path.splitext(out)[0]}.pdf")

    # print the checkpoint table too (handy for the paper text)
    print(f"\n{args.metric.upper()} at checkpoints (mean over seeds):")
    checkpoints = [50, 150, 300]
    header = "  Method   " + "".join(f"Step {c:<6}" for c in checkpoints) + "Final"
    print(header)
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        steps, stacked = align_seeds(runs[method])
        mean = stacked.mean(0)
        cells = []
        for c in checkpoints:
            idx = np.argmin(np.abs(steps - c))
            cells.append(f"{mean[idx]:.3f}    " if steps[idx] <= c + 5 else "  --     ")
        cells.append(f"{mean[-1]:.3f}")
        print(f"  {METHOD_LABELS[method]:<8} " + "".join(cells))


if __name__ == "__main__":
    main()

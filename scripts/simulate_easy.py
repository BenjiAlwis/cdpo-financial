#!/usr/bin/env python3
"""
scripts/simulate_easy.py
========================
Reproduces the EASY-TASK simulation finding from the research session.

Task: K=3 binary hard constraints, M=2 soft preferences, G=8 rollouts.
Objectives are ALIGNED — satisfying constraints also improves soft scores.

Expected finding:
  All three methods (GRPO, GDPO, CDPO) converge to the same final CCR
  (~0.998) and Combined Quality (~0.958). No meaningful difference.
  CDPO is marginally slower at step 50 but matches by step 150.

This is the control condition. It shows CDPO does not hurt on easy tasks.

Usage:
    python scripts/simulate_easy.py [--steps 300] [--seeds 5] [--plot]

Output:
    Prints a summary table.
    If --plot is passed and matplotlib is available, saves easy_task.png.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int,   default=300)
parser.add_argument("--seeds", type=int,   default=5)
parser.add_argument("--lr",    type=float, default=0.05)
parser.add_argument("--G",     type=int,   default=8)
parser.add_argument("--B",     type=int,   default=16)
parser.add_argument("--K",     type=int,   default=3)
parser.add_argument("--M",     type=int,   default=2)
parser.add_argument("--plot",  action="store_true")
args = parser.parse_args()

K, M, G, B = args.K, args.M, args.G, args.B
N_PATTERNS  = 2 ** K


# ── Reward oracle ─────────────────────────────────────────────────────────────

def pattern_to_hard(p):
    return np.array([(p >> k) & 1 for k in range(K)], dtype=float)

def pattern_to_soft(p, rng):
    """Soft score proportional to number of constraints satisfied (ALIGNED)."""
    n_sat  = bin(p).count("1")
    base   = n_sat / K
    return np.clip(base + rng.normal(0, 0.10, M), 0, 1)


# ── Advantage functions ───────────────────────────────────────────────────────

def grpo_adv(r_hard, r_soft):
    """Sum all signals, group-wise z-score. (GDPO paper Eq. 1-2)"""
    R   = r_hard.sum(1) + r_soft.sum(1)
    mu  = R.mean(); sig = R.std() + EPS
    return (R - mu) / sig


def gdpo_adv(r_hard, r_soft):
    """Per-signal z-score, sum, batch-normalise. (GDPO paper Eq. 4-6)"""
    A = np.zeros(G)
    for k in range(K):
        r = r_hard[:, k]; mu, sig = r.mean(), r.std() + EPS
        A += (r - mu) / sig
    for m in range(M):
        s = r_soft[:, m]; mu, sig = s.mean(), s.std() + EPS
        A += (s - mu) / sig
    mu, sig = A.mean(), A.std() + EPS
    return (A - mu) / sig


def cdpo_adv(r_hard, r_soft,
             beta_plus=1.0, beta_minus=2.0,
             alpha_max=0.9, alpha_min=0.3,
             G_min=3, gamma_nc=0.1):
    """CDPO: asymmetric binary hard channel + gated soft channel. (Algorithm 1)"""
    # Hard channel (Step 2)
    A_hard = np.zeros(G)
    for k in range(K):
        r   = r_hard[:, k]
        p_k = r.mean()
        if 0 < p_k < 1:
            A_hard += np.where(r == 1,
                               beta_plus  * (1 - p_k),
                               -beta_minus * p_k)

    # Soft channel (Step 4)
    compliant = r_hard.min(axis=1) == 1
    c = int(compliant.sum())
    A_soft = np.zeros(G)
    if c >= G_min:
        for m in range(M):
            s   = r_soft[:, m]
            mu  = s[compliant].mean()
            sig = s[compliant].std() + EPS
            A_soft += np.where(compliant,
                               (s - mu) / sig,
                               gamma_nc * (s - mu) / sig)

    # Per-group adaptive mixing (Step 5)
    alpha = alpha_max - (alpha_max - alpha_min) * (c / G)
    A_sum = alpha * A_hard + (1 - alpha) * A_soft

    # Batch normalisation (Step 6)
    mu, sig = A_sum.mean(), A_sum.std() + EPS
    return (A_sum - mu) / sig


# ── Training loop ─────────────────────────────────────────────────────────────

def run_experiment(adv_fn, seed=0):
    rng    = np.random.default_rng(seed)
    logits = np.zeros(N_PATTERNS)
    hist   = {"ccr": [], "sps": [], "cq": []}

    for _ in range(args.steps):
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        adv_accum = defaultdict(list)
        step_ccr  = []
        step_sps  = []

        for _ in range(B):
            patterns    = rng.choice(N_PATTERNS, size=G, p=probs)
            r_hard_b    = np.array([pattern_to_hard(p) for p in patterns])
            r_soft_b    = np.array([pattern_to_soft(p, rng) for p in patterns])

            A = adv_fn(r_hard_b, r_soft_b)
            for j, p in enumerate(patterns):
                adv_accum[p].append(A[j])

            all_pass = r_hard_b.min(axis=1) == 1
            step_ccr.append(all_pass.mean())
            if all_pass.sum() > 0:
                step_sps.append(r_soft_b[all_pass].mean())

        for p in range(N_PATTERNS):
            if p in adv_accum:
                logits[p] += args.lr * np.mean(adv_accum[p])

        ccr = np.mean(step_ccr)
        sps = np.mean(step_sps) if step_sps else 0.0
        hist["ccr"].append(ccr)
        hist["sps"].append(sps)
        hist["cq"].append(ccr * sps)

    return hist


# ── Run ───────────────────────────────────────────────────────────────────────

print(f"\nEasy-task simulation  "
      f"(K={K} constraints, M={M} soft, G={G}, {args.steps} steps, "
      f"{args.seeds} seeds, ALIGNED objectives)\n")

METHODS = [
    ("GRPO", grpo_adv),
    ("GDPO", gdpo_adv),
    ("CDPO", cdpo_adv),
]

results = {}
for name, fn in METHODS:
    print(f"  Running {name}...", end="", flush=True)
    runs = [run_experiment(fn, seed=s * 100) for s in range(args.seeds)]
    results[name] = runs
    print(" done")


def tail(histories, key, window=50):
    vals = [np.mean(h[key][-window:]) for h in histories]
    return np.mean(vals), np.std(vals)


def steps_to(histories, key, threshold):
    out = []
    for h in histories:
        arr = np.array(h[key])
        idx = np.where(arr >= threshold)[0]
        out.append(int(idx[0]) if len(idx) > 0 else args.steps)
    return np.mean(out), np.std(out)


W = 14
print(f"\n{'─'*70}")
print(f"{'Metric':<30}  {'GRPO':>{W}}  {'GDPO':>{W}}  {'CDPO':>{W}}")
print(f"{'─'*70}")
for key, label in [
    ("ccr", "Hard Pass Rate (CCR)"),
    ("sps", "Soft Pref Score (SPS)"),
    ("cq",  "Combined Quality (CQ)"),
]:
    row = f"{label:<30}"
    for name in ["GRPO", "GDPO", "CDPO"]:
        mu, sd = tail(results[name], key)
        row += f"  {mu:.3f}±{sd:.3f}"
    print(row)

print(f"\n  Steps to CCR=0.5 (sample efficiency):")
for name in ["GRPO", "GDPO", "CDPO"]:
    mu, sd = steps_to(results[name], "ccr", 0.50)
    print(f"    {name}: {mu:.0f} ± {sd:.0f} steps")

print(f"\n  CCR at steps 50 / 150 / {args.steps}:")
print(f"  {'Method':<6}  {'Step 50':>9}  {'Step 150':>9}  {'Final':>9}")
checkpoints = [49, 149, args.steps - 1]
for name in ["GRPO", "GDPO", "CDPO"]:
    vals = [np.mean([h["ccr"][i] for h in results[name]]) for i in checkpoints]
    print(f"  {name:<6}  {vals[0]:>9.3f}  {vals[1]:>9.3f}  {vals[2]:>9.3f}")

print(f"\n  Expected: all methods converge to CCR~0.998, CQ~0.958")
print(f"  Expected: CDPO slightly slower at step 50, matches by step 150")


# ── Optional plot ─────────────────────────────────────────────────────────────

if args.plot:
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colours = {"GRPO": "#E53935", "GDPO": "#FB8C00", "CDPO": "#1E88E5"}
        linestyles = {"GRPO": "--", "GDPO": "-.", "CDPO": "-"}

        for ax, (key, ylabel) in zip(axes, [
            ("ccr", "Hard Pass Rate (CCR)"),
            ("cq",  "Combined Quality (CQ)"),
        ]):
            for name in ["GRPO", "GDPO", "CDPO"]:
                arr  = np.array([h[key] for h in results[name]])
                mean = arr.mean(0)
                std  = arr.std(0)
                x    = np.arange(len(mean))
                ax.plot(x, mean, label=name, color=colours[name],
                        ls=linestyles[name], lw=2)
                ax.fill_between(x, mean - std, mean + std,
                                color=colours[name], alpha=0.15)
            ax.set_xlabel("Training step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Easy task — {ylabel}")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"Easy task (K={K}, M={M}, ALIGNED objectives)\n"
            "All methods converge equally",
            fontsize=12
        )
        plt.tight_layout()
        plt.savefig("easy_task.png", dpi=150, bbox_inches="tight")
        print("\n  Plot saved → easy_task.png")
    except ImportError:
        print("\n  matplotlib not installed — skipping plot")

print()

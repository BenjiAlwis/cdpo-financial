#!/usr/bin/env python3
"""
scripts/simulate_hard.py
========================
Reproduces the HARD-TASK simulation finding from the research session.

Task: K=4 binary hard constraints, M=3 soft preferences, G=8 rollouts.
Objectives CONFLICT — plans that satisfy soft preferences tend to violate
hard constraints (models financial risk: aggressive portfolio = high Sharpe
but violates drawdown/HHI constraints).

Expected findings:
  1. GRPO and GDPO both collapse to CCR ≈ 0 by step 500. The conflicting
     soft signal dominates and the policy learns to be non-compliant.
  2. GDPO collapses WORSE than GRPO (CCR 0.0002 vs 0.0009) because per-signal
     z-scoring amplifies each conflicting soft signal independently.
  3. CDPO reaches CCR ≈ 0.585 and CQ ≈ 0.293 — the channel separation
     prevents the soft signal from overriding the hard-constraint signal.
  4. CDPO reaches CCR=0.4 in ~51 steps; GRPO/GDPO never reach it.

This is the core result motivating CDPO. It shows the regime where CDPO's
decomposition is essential, not just beneficial.

Usage:
    python scripts/simulate_hard.py [--steps 500] [--seeds 5] [--plot]

Output:
    Prints a summary table.
    If --plot is passed, saves hard_task.png and collapse_detail.png.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int,   default=500)
parser.add_argument("--seeds", type=int,   default=5)
parser.add_argument("--lr",    type=float, default=0.03)
parser.add_argument("--G",     type=int,   default=8)
parser.add_argument("--B",     type=int,   default=16)
parser.add_argument("--K",     type=int,   default=4)
parser.add_argument("--M",     type=int,   default=3)
# Conflict strength: how negatively correlated are soft prefs with hard constraints
parser.add_argument("--conflict", type=float, default=0.4,
                    help="Coefficient of negative correlation (default 0.4)")
parser.add_argument("--plot",  action="store_true")
args = parser.parse_args()

K, M, G, B = args.K, args.M, args.G, args.B
N_PATTERNS  = 2 ** K


# ── Reward oracle (CONFLICTING objectives) ────────────────────────────────────

def sample_rollout(logits, rng):
    """
    Sample one rollout. Hard signals come from the policy's current pattern.
    Soft signals are NEGATIVELY correlated with constraint 0 and 1:
    violating constraints 0 and 1 gives higher soft scores.

    This models: aggressive financial plan (ignores drawdown & HHI limits)
    has higher Sharpe ratio / ESG exposure than a conservative compliant plan.
    """
    probs   = np.exp(logits - logits.max())
    probs  /= probs.sum()
    pattern = rng.choice(N_PATTERNS, p=probs)
    bits    = np.array([(pattern >> k) & 1 for k in range(K)], dtype=float)

    r_hard = bits.copy()

    # Soft score: boosted when constraints 0, 1 are VIOLATED
    aggression = (args.conflict * (1 - bits[0]) +
                  args.conflict * 0.75 * (1 - bits[1]))
    base_soft  = 0.5 + aggression + rng.normal(0, 0.08, M)
    r_soft     = np.clip(base_soft, 0, 1)

    return pattern, r_hard, r_soft


# ── Advantage functions ───────────────────────────────────────────────────────

def grpo_adv(r_hard, r_soft):
    R   = r_hard.sum(1) + r_soft.sum(1)
    mu  = R.mean(); sig = R.std() + EPS
    return (R - mu) / sig


def gdpo_adv(r_hard, r_soft):
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
    A_hard = np.zeros(G)
    for k in range(K):
        r   = r_hard[:, k]
        p_k = r.mean()
        if 0 < p_k < 1:
            A_hard += np.where(r == 1,
                               beta_plus  * (1 - p_k),
                               -beta_minus * p_k)

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

    alpha = alpha_max - (alpha_max - alpha_min) * (c / G)
    A_sum = alpha * A_hard + (1 - alpha) * A_soft
    mu, sig = A_sum.mean(), A_sum.std() + EPS
    return (A_sum - mu) / sig


# ── Training loop ─────────────────────────────────────────────────────────────

def run_experiment(adv_fn, seed=0):
    rng    = np.random.default_rng(seed)
    logits = np.zeros(N_PATTERNS)
    hist   = {"ccr": [], "sps": [], "cq": []}

    for _ in range(args.steps):
        adv_accum = defaultdict(list)
        step_ccr  = []
        step_sps  = []

        for _ in range(B):
            patterns = []
            r_hard_b = np.zeros((G, K))
            r_soft_b = np.zeros((G, M))
            for j in range(G):
                pat, rh, rs = sample_rollout(logits, rng)
                patterns.append(pat)
                r_hard_b[j] = rh
                r_soft_b[j] = rs

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

print(f"\nHard-task simulation  "
      f"(K={K} constraints, M={M} soft, G={G}, {args.steps} steps, "
      f"{args.seeds} seeds, CONFLICTING objectives, "
      f"conflict_strength={args.conflict})\n")

METHODS = [
    ("GRPO", grpo_adv),
    ("GDPO", gdpo_adv),
    ("CDPO", cdpo_adv),
]

results = {}
for name, fn in METHODS:
    print(f"  Running {name}...", end="", flush=True)
    runs = [run_experiment(fn, seed=s * 77) for s in range(args.seeds)]
    results[name] = runs
    print(" done")


def tail(histories, key, window=100):
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
print(f"\n{'─'*72}")
print(f"{'Metric':<32}  {'GRPO':>{W}}  {'GDPO':>{W}}  {'CDPO':>{W}}")
print(f"{'─'*72}")
for key, label in [
    ("ccr", "Hard Pass Rate (CCR)"),
    ("sps", "Soft Pref Score (SPS)"),
    ("cq",  "Combined Quality (CQ)"),
]:
    row = f"{label:<32}"
    best_val = -1
    best_name = ""
    for name in ["GRPO", "GDPO", "CDPO"]:
        mu, _ = tail(results[name], key)
        if mu > best_val:
            best_val = mu
            best_name = name
    for name in ["GRPO", "GDPO", "CDPO"]:
        mu, sd = tail(results[name], key)
        flag = " ←" if name == best_name else ""
        row += f"  {mu:.3f}±{sd:.3f}{flag:<2}"
    print(row)

print(f"\n  Sample efficiency — steps to reach CCR=0.4:")
for name in ["GRPO", "GDPO", "CDPO"]:
    mu, sd = steps_to(results[name], "ccr", 0.40)
    note   = "  ← never reaches it" if mu >= args.steps - 1 else ""
    print(f"    {name}: {mu:.0f} ± {sd:.0f} steps{note}")

print(f"\n  Sample efficiency — steps to reach CQ=0.3:")
for name in ["GRPO", "GDPO", "CDPO"]:
    mu, sd = steps_to(results[name], "cq", 0.30)
    note   = "  ← never reaches it" if mu >= args.steps - 1 else ""
    print(f"    {name}: {mu:.0f} ± {sd:.0f} steps{note}")

checkpoints = [49, 149, 299, args.steps - 1]
checkpoint_labels = ["Step 50", "Step 150", "Step 300", f"Step {args.steps}"]
print(f"\n  CCR trajectory (mean over {args.seeds} seeds):")
print(f"  {'Method':<6}  " +
      "  ".join(f"{l:>9}" for l in checkpoint_labels))
for name in ["GRPO", "GDPO", "CDPO"]:
    vals = [np.mean([h["ccr"][min(i, args.steps-1)] for h in results[name]])
            for i in checkpoints]
    print(f"  {name:<6}  " + "  ".join(f"{v:>9.3f}" for v in vals))

print(f"\n  Combined Quality trajectory:")
print(f"  {'Method':<6}  " +
      "  ".join(f"{l:>9}" for l in checkpoint_labels))
for name in ["GRPO", "GDPO", "CDPO"]:
    vals = [np.mean([h["cq"][min(i, args.steps-1)] for h in results[name]])
            for i in checkpoints]
    print(f"  {name:<6}  " + "  ".join(f"{v:>9.3f}" for v in vals))

print(f"\n  Expected:")
print(f"    GRPO: CCR collapses from ~0.2 at step 50 → ~0.001 at step 500")
print(f"    GDPO: CCR collapses FASTER and LOWER than GRPO (per-signal z-score")
print(f"          amplifies the conflicting soft signal)")
print(f"    CDPO: CCR rises to ~0.58, CQ to ~0.29 — channel separation prevents collapse")


# ── Conflict sensitivity sweep ────────────────────────────────────────────────
print(f"\n  Sensitivity to conflict strength (conflict ∈ [0.1, 0.2, 0.4, 0.6]):")
print(f"  {'Conflict':>10}  {'GRPO CCR':>10}  {'GDPO CCR':>10}  {'CDPO CCR':>10}")

# Store original args.conflict
orig_conflict = args.conflict
for c in [0.1, 0.2, 0.4, 0.6]:
    args.conflict = c
    row_vals = {}
    for name, fn in METHODS:
        runs = [run_experiment(fn, seed=s * 13) for s in range(3)]
        mu, _ = tail(runs, "ccr")
        row_vals[name] = mu
    print(f"  {c:>10.1f}  {row_vals['GRPO']:>10.3f}  "
          f"{row_vals['GDPO']:>10.3f}  {row_vals['CDPO']:>10.3f}")
args.conflict = orig_conflict   # restore


# ── Optional plots ────────────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib.pyplot as plt

        colours    = {"GRPO": "#E53935", "GDPO": "#FB8C00", "CDPO": "#1E88E5"}
        linestyles = {"GRPO": "--", "GDPO": "-.", "CDPO": "-"}

        # Plot 1: main CCR and CQ trajectories
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
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
                        ls=linestyles[name], lw=2.5)
                ax.fill_between(x, mean - std, mean + std,
                                color=colours[name], alpha=0.15)
            ax.set_xlabel("Training step", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"Hard task — {ylabel}", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        fig.suptitle(
            f"Hard task (K={K}, M={M}, CONFLICTING objectives, "
            f"conflict={orig_conflict})\n"
            "GRPO and GDPO collapse; CDPO maintains constraint compliance",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig("hard_task.png", dpi=150, bbox_inches="tight")
        print(f"\n  Main plot saved → hard_task.png")

        # Plot 2: zoom on GRPO/GDPO collapse detail (first 200 steps)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        zoom = 200
        for name in ["GRPO", "GDPO"]:
            arr  = np.array([h["ccr"][:zoom] for h in results[name]])
            mean = arr.mean(0)
            std  = arr.std(0)
            x    = np.arange(zoom)
            ax2.plot(x, mean, label=name, color=colours[name],
                     ls=linestyles[name], lw=2.5)
            ax2.fill_between(x, mean - std, mean + std,
                             color=colours[name], alpha=0.2)
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Hard Pass Rate (CCR)")
        ax2.set_title("GRPO vs GDPO collapse detail (first 200 steps)")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("collapse_detail.png", dpi=150, bbox_inches="tight")
        print(f"  Collapse detail saved → collapse_detail.png")

    except ImportError:
        print("\n  matplotlib not installed — skipping plots")
        print("  Install with: pip install matplotlib")

print()

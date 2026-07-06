"""
training/reward_fns.py
======================
TRL-compatible reward functions for GRPO, GDPO, and CDPO.

TRL's GRPOTrainer calls each reward function as:
    rewards = reward_fn(prompts, completions, **kwargs)

where prompts and completions are lists of strings (length = batch_size × num_generations).

This module provides three reward functions — one per method — that share
the same ScenarioStore and differ only in their advantage computation.

Usage
-----
    from training.reward_fns import make_reward_fns

    grpo_fn, gdpo_fn, cdpo_fn = make_reward_fns(
        store  = scenario_store,
        K      = 4,    # hard constraints (portfolio task)
        M      = 3,    # soft preferences
        G      = 8,    # group size = num_generations in TRL
        method = "cdpo",    # which one to actually use
        config = CDPOConfig(),
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from finplanenv.cdpo      import CDPOConfig, CDPOAdvantage, BatchRewards
from finplanenv.baselines import BaselineConfig, GRPOAdvantage, GDPOAdvantage
from finplanenv.parser    import compute_rewards_from_output

logger = logging.getLogger(__name__)


def make_advantage_computer(
    method: str,
    K: int,
    M: int,
    G: int,
    cdpo_config: CDPOConfig     | None = None,
    grpo_config: BaselineConfig | None = None,
):
    """Return the advantage computer for `method` (cdpo|gdpo|grpo).

    This is the object the *decoupled* trainer (CDPODecoupledGRPOTrainer) calls
    to produce the advantage that goes straight into the policy loss. Unlike the
    legacy make_reward_fn below, the advantage is NOT routed through TRL's reward
    channel, so it is never re-normalized. This is the correct Day-2/3 path.
    """
    cdpo_config = cdpo_config or CDPOConfig(G=G)
    grpo_config = grpo_config or BaselineConfig(G=G)
    if method == "cdpo":
        return CDPOAdvantage(cdpo_config, K=K, M=M)
    if method == "gdpo":
        return GDPOAdvantage(grpo_config, K=K, M=M)
    if method == "grpo":
        return GRPOAdvantage(grpo_config, K=K, M=M)
    raise ValueError(f"Unknown method: {method!r}. Choose 'grpo', 'gdpo', or 'cdpo'.")


def make_zero_reward_fn(G: int):
    """A trivial TRL reward function returning 0.0 for every rollout.

    The decoupled trainer bypasses TRL's reward→advantage step entirely, so the
    reward channel is unused. TRL still requires *a* reward function to be
    present, so we hand it this no-op. Its output never reaches the loss.
    """
    def zero_reward_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    zero_reward_fn.__name__ = "zero_reward"   # TRL uses __name__ for metric keys
    return zero_reward_fn


def make_reward_fn(
    store,          # ScenarioStore instance
    K:      int,    # number of hard constraints
    M:      int,    # number of soft preferences
    G:      int,    # group size (= num_generations in TRL config)
    method: str,    # "grpo" | "gdpo" | "cdpo"
    cdpo_config: CDPOConfig      | None = None,
    grpo_config: BaselineConfig  | None = None,
):
    """
    DEPRECATED — advantage-as-reward path.

    This builds a TRL reward_fn that returns the decomposed advantage Â as if it
    were a reward. With stock GRPOTrainer this is WRONG: TRL re-normalizes the
    returned values with its own group z-score, destroying the channel
    separation (see cdpo_theory.tex §"integration boundary"). It is kept only
    for reference / ablation ("what happens if you DON'T decouple"). For correct
    training use CDPODecoupledGRPOTrainer + make_advantage_computer instead.

    Returns a callable with signature:
        fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]
    """

    cdpo_config = cdpo_config or CDPOConfig(G=G)
    grpo_config = grpo_config or BaselineConfig(G=G)

    # Instantiate the advantage computer
    if method == "cdpo":
        adv_computer = CDPOAdvantage(cdpo_config, K=K, M=M)
        log_prefix   = "cdpo"
    elif method == "gdpo":
        adv_computer = GDPOAdvantage(grpo_config, K=K, M=M)
        log_prefix   = "gdpo"
    elif method == "grpo":
        adv_computer = GRPOAdvantage(grpo_config, K=K, M=M)
        log_prefix   = "grpo"
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'grpo', 'gdpo', or 'cdpo'.")

    step_counter = [0]   # mutable closure

    def reward_fn(
        prompts:     list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        TRL reward_fn. Called once per training step with B*G prompts/completions.
        Returns B*G float advantages.
        """
        B_G = len(completions)
        assert B_G % G == 0, \
            f"Expected multiple of G={G} completions, got {B_G}"
        B = B_G // G

        r_hard = np.zeros((B, G, K), dtype=float)
        r_prox = np.zeros((B, G, K), dtype=float)
        r_soft = np.zeros((B, G, M), dtype=float)

        parse_failures = 0
        lookup_failures = 0

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            i, j = divmod(idx, G)

            # Look up scenario and constraints from the store
            entry = store.lookup(prompt)
            if entry is None:
                lookup_failures += 1
                # Zero rewards for missing scenario (shouldn't happen)
                continue

            # Compute rewards
            bundle = compute_rewards_from_output(
                completion,
                entry.task,
                entry.scenario,
                entry.constraints,
            )
            r_hard[i, j] = bundle.hard
            r_prox[i, j] = bundle.prox
            r_soft[i, j] = bundle.soft

            if bundle.hard.sum() == 0 and bundle.soft.sum() == 0:
                parse_failures += 1

        # Compute advantages
        batch = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)
        A_hat, metrics = adv_computer.compute(batch, step=step_counter[0])

        # Log metrics
        step = step_counter[0]
        step_counter[0] += 1

        if step % 10 == 0:
            _log_metrics(
                log_prefix, step, metrics,
                parse_failures, lookup_failures, B_G,
            )

        return A_hat.reshape(-1).tolist()

    return reward_fn


def _log_metrics(
    prefix:          str,
    step:            int,
    metrics:         dict[str, Any],
    parse_failures:  int,
    lookup_failures: int,
    total:           int,
) -> None:
    """Log metrics to W&B if available, otherwise to logger."""
    parse_rate   = parse_failures  / total
    lookup_rate  = lookup_failures / total

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                **{f"{prefix}/{k}": v for k, v in metrics.items()},
                f"{prefix}/parse_failure_rate":  parse_rate,
                f"{prefix}/lookup_failure_rate": lookup_rate,
            }, step=step)
    except ImportError:
        pass

    logger.info(
        "[%s] step=%d  CCR=%.3f  SPS=%.3f  CQ=%.3f  "
        "distinct_A=%d  parse_fail=%.1f%%",
        prefix.upper(), step,
        metrics.get("ccr",                  0),
        metrics.get("mean_soft_score",      0),
        metrics.get("combined_quality",     0),
        metrics.get("n_distinct_advantages",0),
        100 * parse_rate,
    )

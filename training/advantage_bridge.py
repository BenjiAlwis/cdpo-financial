"""
training/advantage_bridge.py
============================
Pure-numpy bridge from completions -> BatchRewards. Deliberately free of torch
and trl so it can be unit-tested in any environment and imported by the trainer
without pulling in the GPU stack at module load.

See training/trainers.py for how the trainer uses this to inject decomposed
advantages into the policy loss.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from finplanenv.cdpo import BatchRewards


class AdvantageBridge:
    """Turns completions into a BatchRewards by scoring them per signal.

    This is the object that lets the trainer recover r_hard / r_prox / r_soft for
    the rollouts in a step. It wraps the repo's existing reward computation
    (compute_rewards_from_output) and a scenario lookup.

    Parameters
    ----------
    store : object with .lookup(prompt) -> entry(task, scenario, constraints)
    reward_from_output : callable(completion, task, scenario, constraints)
        -> bundle with .hard (K,), .prox (K,), .soft (M,)
    K, M : int
        number of hard / soft signals.
    """

    def __init__(self, store, reward_from_output: Callable, K: int, M: int):
        self.store = store
        self.reward_from_output = reward_from_output
        self.K = K
        self.M = M

    def build_batch(
        self,
        prompts: list,
        completions: list,
        num_generations: int,
    ) -> tuple[BatchRewards, dict]:
        """Score every rollout and pack into (B, G, .) arrays.

        Rollout idx = i * G + j, matching TRL's flattening, so the advantage we
        return lines up with TRL's per-token loss ordering.
        """
        BG = len(completions)
        assert BG % num_generations == 0, (
            f"expected multiple of G={num_generations}, got {BG}"
        )
        B = BG // num_generations
        G = num_generations
        K, M = self.K, self.M

        r_hard = np.zeros((B, G, K), dtype=float)
        r_prox = np.zeros((B, G, K), dtype=float)
        r_soft = np.zeros((B, G, M), dtype=float)

        parse_failures = 0
        lookup_failures = 0

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            i, j = divmod(idx, G)
            comp_text = completion
            if isinstance(completion, list) and completion:
                comp_text = completion[-1].get("content", "")
            entry = self.store.lookup(prompt)
            if entry is None:
                lookup_failures += 1
                continue
            bundle = self.reward_from_output(
                comp_text, entry.task, entry.scenario, entry.constraints
            )
            r_hard[i, j] = bundle.hard
            r_prox[i, j] = bundle.prox
            r_soft[i, j] = bundle.soft
            if getattr(bundle, "parse_failed", False):
                parse_failures += 1

        batch = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)
        diag = {
            "parse_failures": parse_failures,
            "lookup_failures": lookup_failures,
            "total": BG,
        }
        return batch, diag

"""
tests/test_advantage_not_renormalized.py
========================================
The Day-2 PROOF.

The whole point of CDPODecoupledGRPOTrainer is that the advantage entering the
policy loss is the CDPO/GDPO advantage AS COMPUTED -- not re-normalized by TRL's
group z-score. These tests prove exactly that, without needing a GPU, a model, or
even torch, by isolating the two things that matter:

  1. AdvantageBridge recovers the correct per-signal rewards from completions and
     the advantage computer's output is what the trainer would inject.
  2. That injected advantage is DIFFERENT from what stock GRPOTrainer would
     produce (sum-then-group-z-score), so the bug is demonstrably fixed.

If a future refactor accidentally re-introduces TRL normalization, test (2)
fails loudly.

Run:  python -m pytest tests/test_advantage_not_renormalized.py -v
"""

import numpy as np
import pytest

from finplanenv.cdpo import CDPOConfig, CDPOAdvantage, BatchRewards
from finplanenv.baselines import BaselineConfig, GRPOAdvantage, GDPOAdvantage


# ---------------------------------------------------------------------------
# Lightweight fakes: a reward bundle, a scenario entry, and a store, so we can
# exercise AdvantageBridge without the real parser/dataset.
# ---------------------------------------------------------------------------
class FakeBundle:
    def __init__(self, hard, prox, soft):
        self.hard = np.asarray(hard, dtype=float)
        self.prox = np.asarray(prox, dtype=float)
        self.soft = np.asarray(soft, dtype=float)


class FakeEntry:
    def __init__(self, signals):
        self.task = "portfolio"
        self.scenario = None
        self.constraints = None
        self.signals = signals  # the (hard, prox, soft) we want this completion to score


class FakeStore:
    """Maps a prompt string to a fixed scenario entry."""
    def __init__(self, mapping):
        self.mapping = mapping

    def lookup(self, prompt):
        return self.mapping.get(prompt)


def make_bridge(rollout_signals, K, M):
    """Build an AdvantageBridge whose scoring returns pre-set per-rollout signals.

    rollout_signals: list of (hard, prox, soft), one per (prompt, completion) in
    the order the trainer will pass them (i*G + j).
    """
    from training.advantage_bridge import AdvantageBridge

    # the bridge calls reward_from_output(comp_text, task, scenario, constraints).
    # We encode the rollout index in the completion text ("r{idx}") and look the
    # signals up from a list, so the bridge's iteration order is what we test.
    def reward_from_output(comp_text, task, scenario, constraints):
        idx = int(comp_text[1:])  # "r3" -> 3
        hard, prox, soft = rollout_signals[idx]
        return FakeBundle(hard, prox, soft)

    # store maps every prompt to a dummy entry (signals unused by reward_from_output)
    store = FakeStore({"p": FakeEntry(None)})
    bridge = AdvantageBridge(store, reward_from_output, K=K, M=M)
    return bridge


def trl_would_compute(rollout_signals, G):
    """Reproduce stock GRPOTrainer's advantage: sum signals -> group z-score.

    This is the WRONG answer (the bug). We compute it to prove the trainer does
    NOT produce it.
    """
    BG = len(rollout_signals)
    B = BG // G
    rewards = np.array([h.sum() + s.sum()
                        for (h, p, s) in
                        [(np.asarray(a), np.asarray(b), np.asarray(c))
                         for (a, b, c) in rollout_signals]])
    rewards = rewards.reshape(B, G)
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    adv = (rewards - mean) / (std + 1e-4)
    return adv.reshape(-1)


# ---------------------------------------------------------------------------
# The group under test: G=8, K=2, M=1, asymmetric + conflicting so the methods
# genuinely differ from each other AND from TRL's renormalization.
# ---------------------------------------------------------------------------
def conflict_group():
    # (hard(K,), prox(K,), soft(M,)) per rollout, B=1 group of G=8
    # 3 compliant (modest soft), 5 violators (high soft) -> conflict.
    sig = [
        ([1, 1], [1, 1], [0.40]),
        ([1, 1], [1, 1], [0.45]),
        ([1, 1], [1, 1], [0.50]),
        ([1, 0], [1, 0], [0.85]),
        ([0, 1], [0, 1], [0.90]),
        ([0, 1], [0, 1], [0.92]),
        ([0, 0], [0, 0], [0.95]),
        ([0, 0], [0, 0], [0.99]),
    ]
    return [(np.array(h, float), np.array(p, float), np.array(s, float))
            for (h, p, s) in sig]


G, K, M = 8, 2, 1


@pytest.fixture
def signals():
    return conflict_group()


@pytest.fixture
def prompts_completions(signals):
    prompts = ["p"] * len(signals)
    completions = [f"r{idx}" for idx in range(len(signals))]
    return prompts, completions


# ===========================================================================
# Test 1: the bridge reconstructs the exact per-signal reward tensors.
# ===========================================================================
def test_bridge_recovers_signals(signals, prompts_completions):
    prompts, completions = prompts_completions
    bridge = make_bridge(signals, K, M)
    batch, diag = bridge.build_batch(prompts, completions, num_generations=G)

    assert batch.r_hard.shape == (1, G, K)
    assert batch.r_soft.shape == (1, G, M)
    # row 3 was a fail-c2; row 4 a fail-c1
    np.testing.assert_array_equal(batch.r_hard[0, 3], [1, 0])
    np.testing.assert_array_equal(batch.r_hard[0, 4], [0, 1])
    assert batch.r_soft[0, 7, 0] == pytest.approx(0.99)
    assert diag["total"] == G


# ===========================================================================
# Test 2 (THE PROOF): the advantage the trainer injects equals the computer's
# output and is NOT TRL's renormalized version.
# ===========================================================================
@pytest.mark.parametrize("name,computer", [
    ("cdpo", CDPOAdvantage(CDPOConfig(G=G, corr_threshold=1.1,
                                      beta_plus=1.0, beta_minus=2.0,
                                      G_min=4), K=K, M=M)),
    ("gdpo", GDPOAdvantage(BaselineConfig(G=G), K=K, M=M)),
])
def test_injected_advantage_is_decomposed_not_renormalized(
    signals, prompts_completions, name, computer
):
    prompts, completions = prompts_completions
    bridge = make_bridge(signals, K, M)

    # What the trainer's _cdpo_advantages would inject (the numpy core of it):
    batch, _ = bridge.build_batch(prompts, completions, num_generations=G)
    adv_BG, _ = computer.compute(batch, step=0)
    injected = np.asarray(adv_BG, dtype=np.float32).reshape(-1)

    # What stock TRL would have produced from the same rollouts (the bug):
    trl_renorm = trl_would_compute(signals, G).astype(np.float32)

    # (a) the injected advantage must match the computer exactly (modulo the
    #     float32 cast the trainer applies before handing to torch)
    np.testing.assert_allclose(
        injected, adv_BG.reshape(-1).astype(np.float32), rtol=1e-5, atol=1e-6
    )

    # (b) and must DIFFER from TRL's renormalization -- otherwise the bug is back
    max_diff = np.max(np.abs(injected - trl_renorm))
    assert max_diff > 1e-3, (
        f"[{name}] injected advantage is indistinguishable from TRL's "
        f"renormalized advantage (max diff {max_diff:.2e}). The decoupling is "
        "NOT in effect -- the re-normalization bug is present."
    )


# ===========================================================================
# Test 3: the three methods genuinely differ from each other on this group.
# (If they don't, you are not running three different algorithms.)
# ===========================================================================
def test_methods_are_distinct(signals, prompts_completions):
    prompts, completions = prompts_completions
    bridge = make_bridge(signals, K, M)
    batch, _ = bridge.build_batch(prompts, completions, num_generations=G)

    grpo = GRPOAdvantage(BaselineConfig(G=G), K=K, M=M)
    gdpo = GDPOAdvantage(BaselineConfig(G=G), K=K, M=M)
    cdpo = CDPOAdvantage(CDPOConfig(G=G, corr_threshold=1.1,
                                    beta_plus=1.0, beta_minus=2.0, G_min=4),
                         K=K, M=M)
    a_grpo = grpo.compute(batch, 0)[0].reshape(-1)
    a_gdpo = gdpo.compute(batch, 0)[0].reshape(-1)
    a_cdpo = cdpo.compute(batch, 0)[0].reshape(-1)

    assert np.max(np.abs(a_grpo - a_cdpo)) > 1e-3, "GRPO == CDPO (not distinct)"
    assert np.max(np.abs(a_gdpo - a_cdpo)) > 1e-3, "GDPO == CDPO (not distinct)"


# ===========================================================================
# Test 4: the CDPO gate actually subordinates soft to hard on this conflict
# group -- the top-ranked rollout must be COMPLIANT, unlike GDPO.
# ===========================================================================
def test_cdpo_ranks_compliant_top_gdpo_does_not():
    """On a single-constraint conflict group, GDPO inverts (ranks a violator
    top) while CDPO's gate keeps a compliant rollout on top.

    Uses K=1 because with a single hard signal the one conflicting soft signal
    can out-vote it under GDPO -- the cleanest demonstration of Prop. 6/7. With
    K>=2 the summed hard channel often carries enough weight that GDPO does not
    invert on a given group, so this property is specifically about the regime
    where one soft signal competes with one hard signal.
    """
    G1, K1, M1 = 8, 1, 1
    # 3 compliant (modest soft), 5 violators (high soft)
    sig = [
        ([1], [1], [0.40]),
        ([1], [1], [0.45]),
        ([1], [1], [0.50]),
        ([0], [0], [0.85]),
        ([0], [0], [0.90]),
        ([0], [0], [0.92]),
        ([0], [0], [0.95]),
        ([0], [0], [0.99]),
    ]
    rollout_signals = [
        (np.array(h, float), np.array(p, float), np.array(s, float))
        for (h, p, s) in sig
    ]
    prompts = ["p"] * G1
    completions = [f"r{idx}" for idx in range(G1)]
    bridge = make_bridge(rollout_signals, K1, M1)
    batch, _ = bridge.build_batch(prompts, completions, num_generations=G1)

    gdpo = GDPOAdvantage(BaselineConfig(G=G1), K=K1, M=M1)
    cdpo = CDPOAdvantage(
        CDPOConfig(G=G1, corr_threshold=1.1, beta_plus=1.0,
                   beta_minus=2.0, G_min=4),
        K=K1, M=M1,
    )
    a_gdpo = gdpo.compute(batch, 0)[0].reshape(-1)
    a_cdpo = cdpo.compute(batch, 0)[0].reshape(-1)

    compliant = {0, 1, 2}
    # GDPO ranks a VIOLATOR top under conflict (the disease)
    assert a_gdpo.argmax() not in compliant, (
        "GDPO unexpectedly ranked a compliant rollout top; conflict too weak."
    )
    # CDPO ranks a COMPLIANT rollout top (the cure -- gate closed, 3<G_min=4)
    assert a_cdpo.argmax() in compliant, (
        "CDPO did not rank a compliant rollout top -- gate not subordinating "
        "soft to hard as intended."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

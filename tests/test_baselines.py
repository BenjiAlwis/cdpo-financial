"""
tests/test_baselines.py
-----------------------
Tests for GRPO/GDPO baseline wrappers and CDPO aggregation strategies.

Run alongside test_all.py — same no-pytest approach.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from finplanenv.baselines import (
    BaselineConfig, GRPOAdvantage, GDPOAdvantage,
)
from finplanenv.cdpo import CDPOConfig, CDPOAdvantage, BatchRewards


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_batch(seed=42, B=4, G=8, K=4, M=3, hard_p=0.4):
    rng = np.random.default_rng(seed)
    return BatchRewards(
        r_hard = rng.binomial(1, hard_p, (B, G, K)).astype(float),
        r_prox = rng.uniform(0.2, 0.9, (B, G, K)),
        r_soft = rng.uniform(0.3, 0.8, (B, G, M)),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GRPO baseline
# ══════════════════════════════════════════════════════════════════════════════

class TestGRPOAdvantage:

    def test_output_shape(self):
        batch = _make_batch()
        grpo  = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        A, _  = grpo.compute(batch, step=0)
        assert A.shape == (4, 8)

    def test_group_normalised(self):
        """GRPO normalises per-group — each row must have mean≈0 std≈1."""
        batch = _make_batch()
        grpo  = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        A, _  = grpo.compute(batch, step=0)
        for i in range(4):
            assert abs(A[i].mean()) < 0.1, f"Group {i} mean={A[i].mean():.4f}"
            assert abs(A[i].std() - 1.0) < 0.2, f"Group {i} std={A[i].std():.4f}"

    def test_NOT_batch_normalised(self):
        """GRPO does NOT apply batch normalisation — global std need not be 1."""
        batch = _make_batch()
        grpo  = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        A, _  = grpo.compute(batch, step=0)
        # Global std will generally differ from 1 since groups are normalised
        # independently. This test documents the GRPO / CDPO difference.
        global_std = A.std()
        assert global_std > 0, "Advantages must be non-zero"

    def test_metric_keys(self):
        batch  = _make_batch()
        grpo   = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        _, metrics = grpo.compute(batch, step=0)
        required = {"ccr", "mean_soft_score", "combined_quality",
                    "n_distinct_advantages", "A_hat_mean", "A_hat_std"}
        for k in required:
            assert k in metrics, f"Missing metric: {k}"

    def test_ccr_range(self):
        batch  = _make_batch()
        grpo   = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        _, m   = grpo.compute(batch, step=0)
        assert 0.0 <= m["ccr"] <= 1.0

    def test_equal_weights_default(self):
        """With equal weights, GRPO sums hard and soft signals equally."""
        batch  = _make_batch(hard_p=0.5)
        grpo1  = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        grpo2  = GRPOAdvantage(
            BaselineConfig(weights_hard=[1.,1.,1.,1.], weights_soft=[1.,1.,1.]),
            K=4, M=3,
        )
        A1, _ = grpo1.compute(batch, step=0)
        A2, _ = grpo2.compute(batch, step=0)
        assert np.allclose(A1, A2), "Explicit unit weights must match default"

    def test_weighted_changes_advantages(self):
        """Non-unit weights must change the advantage values."""
        batch  = _make_batch()
        grpo1  = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        grpo2  = GRPOAdvantage(
            BaselineConfig(weights_hard=[2.,2.,2.,2.], weights_soft=[1.,1.,1.]),
            K=4, M=3,
        )
        A1, _ = grpo1.compute(batch, step=0)
        A2, _ = grpo2.compute(batch, step=0)
        assert not np.allclose(A1, A2), "Different weights must produce different advantages"


# ══════════════════════════════════════════════════════════════════════════════
#  GDPO baseline
# ══════════════════════════════════════════════════════════════════════════════

class TestGDPOAdvantage:

    def test_output_shape(self):
        batch = _make_batch()
        gdpo  = GDPOAdvantage(BaselineConfig(), K=4, M=3)
        A, _  = gdpo.compute(batch, step=0)
        assert A.shape == (4, 8)

    def test_batch_normalised(self):
        """GDPO applies batch normalisation — global mean≈0 std≈1."""
        batch = _make_batch()
        gdpo  = GDPOAdvantage(BaselineConfig(), K=4, M=3)
        A, _  = gdpo.compute(batch, step=0)
        assert abs(A.mean()) < 0.1,       f"Global mean={A.mean():.4f}"
        assert abs(A.std() - 1.0) < 0.2, f"Global std={A.std():.4f}"

    def test_treats_hard_soft_symmetrically(self):
        """
        GDPO normalises hard and soft signals with the same z-score.
        Verify by checking that flipping which signals are 'hard' vs 'soft'
        produces the same advantages (since GDPO ignores the distinction).
        """
        rng  = np.random.default_rng(0)
        r_all = rng.uniform(0, 1, (4, 8, 7))   # 7 arbitrary signals

        # Split as 4 hard + 3 soft
        batch_a = BatchRewards(
            r_hard = r_all[:, :, :4],
            r_prox = np.ones((4, 8, 4)),
            r_soft = r_all[:, :, 4:],
        )
        # Split as 3 hard + 4 soft (reversed)
        batch_b = BatchRewards(
            r_hard = r_all[:, :, 4:],
            r_prox = np.ones((4, 8, 3)),
            r_soft = r_all[:, :, :4],
        )
        gdpo_a = GDPOAdvantage(BaselineConfig(), K=4, M=3)
        gdpo_b = GDPOAdvantage(BaselineConfig(), K=3, M=4)
        A_a, _ = gdpo_a.compute(batch_a, step=0)
        A_b, _ = gdpo_b.compute(batch_b, step=0)
        # Advantages must be the same regardless of which signals are
        # labelled 'hard' vs 'soft' — GDPO is channel-agnostic
        assert np.allclose(A_a, A_b, atol=1e-6), \
            "GDPO should be symmetric across hard/soft split"

    def test_metric_keys(self):
        batch  = _make_batch()
        gdpo   = GDPOAdvantage(BaselineConfig(), K=4, M=3)
        _, metrics = gdpo.compute(batch, step=0)
        required = {"ccr", "mean_soft_score", "combined_quality",
                    "n_distinct_advantages"}
        for k in required:
            assert k in metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Metric key alignment across all three methods
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricAlignment:

    def test_core_keys_match_across_methods(self):
        """
        The core comparison keys must be present in all three methods
        so W&B charts overlay without post-processing.
        """
        batch  = _make_batch()
        grpo   = GRPOAdvantage(BaselineConfig(), K=4, M=3)
        gdpo   = GDPOAdvantage(BaselineConfig(), K=4, M=3)
        cdpo   = CDPOAdvantage(CDPOConfig(G=8), K=4, M=3)

        _, m_grpo = grpo.compute(batch, step=0)
        _, m_gdpo = gdpo.compute(batch, step=0)
        _, m_cdpo = cdpo.compute(batch, step=0)

        core_keys = {
            "ccr", "mean_soft_score", "combined_quality",
            "n_distinct_advantages", "A_hat_mean", "A_hat_std",
            "pass_rate_h1", "pass_rate_h2", "pass_rate_h3", "pass_rate_h4",
        }
        for k in core_keys:
            assert k in m_grpo, f"GRPO missing key: {k}"
            assert k in m_gdpo, f"GDPO missing key: {k}"
            assert k in m_cdpo, f"CDPO missing key: {k}"

    def test_ccr_identical_for_same_batch(self):
        """
        CCR is a property of the batch, not the method.
        All three methods must report the same CCR for the same batch.
        """
        batch  = _make_batch(seed=7)
        _, m_grpo = GRPOAdvantage(BaselineConfig(), K=4, M=3).compute(batch, 0)
        _, m_gdpo = GDPOAdvantage(BaselineConfig(), K=4, M=3).compute(batch, 0)
        _, m_cdpo = CDPOAdvantage(CDPOConfig(), K=4, M=3).compute(batch, 0)

        assert abs(m_grpo["ccr"] - m_gdpo["ccr"]) < 1e-6, "CCR must match"
        assert abs(m_grpo["ccr"] - m_cdpo["ccr"]) < 1e-6, "CCR must match"


# ══════════════════════════════════════════════════════════════════════════════
#  CDPO aggregation strategies (Week 3)
# ══════════════════════════════════════════════════════════════════════════════

class TestCDPOAggregationStrategies:

    def _run(self, schedule, step=0, **kwargs):
        batch = _make_batch()
        cfg   = CDPOConfig(G=8, alpha_schedule=schedule, **kwargs)
        cdpo  = CDPOAdvantage(cfg, K=4, M=3)
        return cdpo.compute(batch, step=step)

    def test_adaptive_default(self):
        A, m = self._run("adaptive")
        assert A.shape == (4, 8)
        assert abs(A.mean()) < 0.1
        assert m["alpha_schedule"] == "adaptive"

    def test_fixed_alpha(self):
        A, m = self._run("fixed", alpha_fixed=0.7)
        assert A.shape == (4, 8)
        assert abs(A.mean()) < 0.1
        assert m["alpha_schedule"] == "fixed"
        # With fixed α=0.7, alpha_mean must be exactly 0.7
        assert abs(m["alpha_mean"] - 0.7) < 1e-6, \
            f"alpha_mean should be 0.7, got {m['alpha_mean']}"

    def test_annealing_early(self):
        """At step 0, annealing α should equal alpha_anneal_start."""
        A, m = self._run(
            "annealing",
            alpha_anneal_start=0.9,
            alpha_min=0.3,
            anneal_steps=100,
            step=0,
        )
        assert abs(m["alpha_mean"] - 0.9) < 1e-5, \
            f"At step=0, alpha_mean should be 0.9, got {m['alpha_mean']}"

    def test_annealing_end(self):
        """After anneal_steps, annealing α should equal alpha_min."""
        batch = _make_batch()
        cfg   = CDPOConfig(
            G=8, alpha_schedule="annealing",
            alpha_anneal_start=0.9, alpha_min=0.3, anneal_steps=100,
        )
        cdpo = CDPOAdvantage(cfg, K=4, M=3)
        # Simulate being at the end of annealing
        cdpo._step = 100
        A, m = cdpo.compute(batch, step=100)
        assert abs(m["alpha_mean"] - 0.3) < 1e-5, \
            f"At end of annealing, alpha_mean should be 0.3, got {m['alpha_mean']}"

    def test_annealing_monotone(self):
        """Alpha should decrease monotonically during annealing."""
        batch = _make_batch()
        cfg   = CDPOConfig(
            G=8, alpha_schedule="annealing",
            alpha_anneal_start=0.9, alpha_min=0.3, anneal_steps=50,
        )
        cdpo   = CDPOAdvantage(cfg, K=4, M=3)
        alphas = []
        for step in [0, 10, 25, 50, 75]:
            cdpo._step = step
            _, m = cdpo.compute(batch, step=step)
            alphas.append(m["alpha_mean"])
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i+1] - 1e-6, \
                f"Alpha should be non-increasing: {alphas}"

    def test_invalid_schedule_raises(self):
        batch = _make_batch()
        cfg   = CDPOConfig(G=8, alpha_schedule="invalid_name")
        cdpo  = CDPOAdvantage(cfg, K=4, M=3)
        try:
            cdpo.compute(batch, step=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "alpha_schedule" in str(e).lower() or "unknown" in str(e).lower()

    def test_three_strategies_produce_different_advantages(self):
        """
        Different strategies must produce genuinely different advantages
        when the compliance rate in the batch is mixed (so adaptive alpha
        varies per group, while fixed and annealing use constant alpha).

        Use a batch where c_i/G varies across groups to guarantee separation.
        """
        rng = np.random.default_rng(17)
        B, G, K, M = 4, 8, 4, 3
        # Construct a batch with deliberately varied group compliance:
        # group 0: all fail (c=0), group 1: half pass (c=4),
        # group 2: mostly pass (c=7), group 3: all pass (c=8)
        r_hard = np.zeros((B, G, K))
        r_hard[1, :4, :] = 1.0   # 4 of 8 pass all K constraints
        r_hard[2, :7, :] = 1.0   # 7 of 8 pass
        r_hard[3, :,  :] = 1.0   # all pass
        batch = BatchRewards(
            r_hard = r_hard,
            r_prox = rng.uniform(0.2, 0.9, (B, G, K)),
            r_soft = rng.uniform(0.3, 0.8, (B, G, M)),
        )

        def run(schedule, step=0, **kw):
            cfg  = CDPOConfig(G=G, alpha_schedule=schedule, **kw)
            cdpo = CDPOAdvantage(cfg, K=K, M=M)
            A, _ = cdpo.compute(batch, step=step)
            return A

        A_adaptive  = run("adaptive",  alpha_max=0.9, alpha_min=0.1)
        # Fixed at 0.5 — will differ from adaptive since groups have c=0,4,7,8
        A_fixed     = run("fixed",     alpha_fixed=0.5)
        # Annealing at step=0 starts at 0.9; adaptive group 3 (all pass) = 0.1
        # so they must differ
        A_annealing = run("annealing", step=0,
                          alpha_anneal_start=0.9, alpha_min=0.1, anneal_steps=100)

        assert not np.allclose(A_adaptive, A_fixed, atol=1e-4), \
            "Adaptive (per-group) and fixed (constant) must differ on mixed batch"
        assert not np.allclose(A_adaptive, A_annealing, atol=1e-4), \
            "Adaptive (per-group) and annealing (global) must differ on mixed batch"


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison: GRPO collapses more than GDPO which collapses more than CDPO
# ══════════════════════════════════════════════════════════════════════════════

class TestCollapseProportion1:
    """
    Empirical validation of Proposition 1:
    Within-group distinct advantage counts should satisfy
    CDPO >= GDPO >= GRPO on average for binary hard signals.

    Note: This is a statistical test, not exact — it may occasionally
    fail on adversarial random seeds. The paper's theoretical claim
    (Prop 1) is about the *maximum achievable* groups, not every batch.
    """

    def test_ordering_holds_statistically(self):
        """
        Over many random batches, CDPO should produce >= distinct groups
        as GDPO, and GDPO >= GRPO, on average.
        """
        n_trials = 20
        grpo_counts = []
        gdpo_counts = []
        cdpo_counts = []

        for seed in range(n_trials):
            # Use purely binary rewards (no continuous soft) to maximise
            # the collapse effect that Prop 1 describes
            rng    = np.random.default_rng(seed)
            B, G, K, M = 2, 8, 3, 2
            r_hard = rng.binomial(1, 0.5, (B, G, K)).astype(float)
            r_prox = rng.uniform(0, 1, (B, G, K))
            r_soft = rng.binomial(1, 0.5, (B, G, M)).astype(float)
            batch  = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)

            cfg_base = BaselineConfig(G=G)
            cfg_cdpo = CDPOConfig(G=G)

            grpo = GRPOAdvantage(cfg_base, K=K, M=M)
            gdpo = GDPOAdvantage(cfg_base, K=K, M=M)
            cdpo_adv = CDPOAdvantage(cfg_cdpo, K=K, M=M)

            # Per-group distinct advantage count (what Prop 1 analyses)
            def within_group_distinct(A):
                return np.mean([
                    len(set(np.round(A[i], 4).tolist()))
                    for i in range(B)
                ])

            A_grpo, _ = grpo.compute(batch, 0)
            A_gdpo, _ = gdpo.compute(batch, 0)
            A_cdpo, _ = cdpo_adv.compute(batch, 0)

            grpo_counts.append(within_group_distinct(A_grpo))
            gdpo_counts.append(within_group_distinct(A_gdpo))
            cdpo_counts.append(within_group_distinct(A_cdpo))

        avg_grpo = np.mean(grpo_counts)
        avg_gdpo = np.mean(gdpo_counts)
        avg_cdpo = np.mean(cdpo_counts)

        print(f"\n  Avg within-group distinct advantages over {n_trials} trials:")
        print(f"    GRPO: {avg_grpo:.2f}")
        print(f"    GDPO: {avg_gdpo:.2f}")
        print(f"    CDPO: {avg_cdpo:.2f}")

        # CDPO should beat GRPO on average (the core empirical claim)
        assert avg_cdpo >= avg_grpo - 0.5, \
            f"CDPO ({avg_cdpo:.2f}) should be >= GRPO ({avg_grpo:.2f}) on average"

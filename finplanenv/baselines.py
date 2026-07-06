"""
finplanenv/baselines.py
=======================
Full training-loop wrappers for the GRPO and GDPO baselines.

These are drop-in replacements for CDPORewardWrapper / CDPOTRLRewardFn.
They use the same interface, the same logging keys, and the same
parse/reward pipeline — so training curves are directly comparable
across all three methods.

Design
------
GRPOAdvantage   — pure-numpy GRPO advantage computation with metrics
GDPOAdvantage   — pure-numpy GDPO advantage computation with metrics
GRPORewardWrapper   — verl reward_fn interface  (mirrors CDPORewardWrapper)
GDPORewardWrapper   — verl reward_fn interface
GRPOTRLRewardFn     — TRL reward_fn interface   (mirrors CDPOTRLRewardFn)
GDPOTRLRewardFn     — TRL reward_fn interface

Logging keys are identical to CDPOAdvantage.metrics so W&B charts
overlay without any post-processing:
    ccr, mean_soft_score, combined_quality,
    n_distinct_advantages, A_hat_mean, A_hat_std,
    pass_rate_h1 ... pass_rate_hK,
    soft_score_s1 ... soft_score_sM

Paper mapping
-------------
GRPOAdvantage  → Eq. (2) in GDPO paper (Shao et al., DeepSeekMath)
GDPOAdvantage  → Eq. (4)-(6) in GDPO paper (Liu et al., NVIDIA arXiv 2601.05242)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from finplanenv.cdpo import BatchRewards, EPS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Shared config for baselines
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    """
    Hyperparameters shared across GRPO and GDPO baselines.

    Kept deliberately minimal — baselines should not be over-tuned.
    The only meaningful knob is reward weighting for GRPO.
    """
    G:            int   = 8      # group size (must match rollout config)
    log_every:    int   = 10     # log metrics every N steps
    # GRPO-only: optional per-signal weights (None = equal weights)
    weights_hard: list[float] | None = None   # length K, or None
    weights_soft: list[float] | None = None   # length M, or None


# ──────────────────────────────────────────────────────────────────────────────
#  GRPOAdvantage — Steps 2–6 equivalent for GRPO
# ──────────────────────────────────────────────────────────────────────────────

class GRPOAdvantage:
    """
    Standard GRPO multi-reward advantage computation.

    Algorithm:
        1. Sum all K hard + M soft signals into scalar R^(i,j)
           (optionally weighted by w_hard, w_soft)
        2. Group-wise z-score: A^(i,j) = (R^(i,j) - mean_G) / std_G
        3. No batch normalisation (standard GRPO)

    This is the "monolithic normalisation" baseline that Proposition 1
    proves is maximally information-lossy for binary hard signals.

    Maps to:
        R^(i,j) = r^(i,j)_1 + ... + r^(i,j)_n     (Eq. 1, GDPO paper)
        A^(i,j) = (R^(i,j) - mean{R^(i,*)}) /
                  std{R^(i,*)}                       (Eq. 2, GDPO paper)
    """

    def __init__(self, config: BaselineConfig, K: int, M: int):
        self.cfg = config
        self.K   = K
        self.M   = M

    def compute(
        self,
        batch: BatchRewards,
        step:  int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Compute GRPO advantages for one batch.

        Returns
        -------
        A_hat : np.ndarray (B, G)  — group-normalised advantages
        metrics : dict             — logging metrics (same keys as CDPO)
        """
        B, G, K = batch.r_hard.shape
        M = batch.r_soft.shape[2]

        w_h = (np.array(self.cfg.weights_hard)
               if self.cfg.weights_hard is not None
               else np.ones(K))
        w_s = (np.array(self.cfg.weights_soft)
               if self.cfg.weights_soft is not None
               else np.ones(M))

        # Sum all signals into scalar reward (Eq. 1)
        R = (batch.r_hard * w_h).sum(axis=2) + \
            (batch.r_soft * w_s).sum(axis=2)       # (B, G)

        # Group-wise z-score (Eq. 2) — NO batch normalisation
        A = np.zeros_like(R)
        for i in range(B):
            mu  = R[i].mean()
            sig = R[i].std() + EPS
            A[i] = (R[i] - mu) / sig

        metrics = self._compute_metrics(batch, A)
        return A, metrics

    def _compute_metrics(
        self,
        batch: BatchRewards,
        A:     np.ndarray,
    ) -> dict[str, Any]:
        B, G, K = batch.r_hard.shape
        M = batch.r_soft.shape[2]

        all_pass = batch.r_hard.min(axis=2)         # (B, G)
        ccr      = float(all_pass.mean())
        per_k    = batch.r_hard.mean(axis=(0, 1))   # (K,)

        if all_pass.sum() > 0:
            mean_soft = float(
                batch.r_soft[all_pass.astype(bool)].mean()
            )
        else:
            mean_soft = 0.0

        # Distinct advantage groups (paper diagnostic)
        n_distinct = len(set(np.round(A.flatten(), 4).tolist()))

        return {
            "ccr":                  ccr,
            "mean_soft_score":      mean_soft,
            "combined_quality":     ccr * mean_soft,
            "n_distinct_advantages": n_distinct,
            "A_hat_mean":           float(A.mean()),
            "A_hat_std":            float(A.std()),
            **{f"pass_rate_h{k+1}": float(per_k[k]) for k in range(K)},
            **{f"soft_score_s{m+1}": 0.0 for m in range(M)},  # not computed per-signal in GRPO
        }


# ──────────────────────────────────────────────────────────────────────────────
#  GDPOAdvantage — per-signal normalisation (GDPO paper Eqs. 4–6)
# ──────────────────────────────────────────────────────────────────────────────

class GDPOAdvantage:
    """
    GDPO per-signal normalisation.

    Algorithm (GDPO paper Eqs. 4–6):
        1. For each signal k, compute group-wise z-score independently:
               A_k^(i,j) = (r_k^(i,j) - mean_G(r_k)) / std_G(r_k)
        2. Sum normalised advantages across all K+M signals:
               A_sum^(i,j) = sum_k A_k^(i,j) + sum_m A_m^(i,j)
        3. Batch-wise normalisation of A_sum (Eq. 6)

    GDPO treats all signals symmetrically — no hard/soft distinction.
    This is the key difference from CDPO, which normalises the two
    channel types separately and applies asymmetric scaling to binary signals.
    """

    def __init__(self, config: BaselineConfig, K: int, M: int):
        self.cfg = config
        self.K   = K
        self.M   = M

    def compute(
        self,
        batch: BatchRewards,
        step:  int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Compute GDPO advantages for one batch.

        Returns
        -------
        A_hat : np.ndarray (B, G)  — batch-normalised advantages
        metrics : dict             — logging metrics (same keys as CDPO)
        """
        B, G, K = batch.r_hard.shape
        M = batch.r_soft.shape[2]

        A = np.zeros((B, G), dtype=float)

        # Per-signal group-wise z-score (Eq. 4)
        for i in range(B):
            for k in range(K):
                r = batch.r_hard[i, :, k]
                mu, sig = r.mean(), r.std() + EPS
                A[i] += (r - mu) / sig
            for m in range(M):
                s = batch.r_soft[i, :, m]
                mu, sig = s.mean(), s.std() + EPS
                A[i] += (s - mu) / sig

        # Batch-wise normalisation (Eq. 6)
        mu  = A.mean()
        sig = A.std() + EPS
        A_hat = (A - mu) / sig

        metrics = self._compute_metrics(batch, A_hat)
        return A_hat, metrics

    def _compute_metrics(
        self,
        batch: BatchRewards,
        A_hat: np.ndarray,
    ) -> dict[str, Any]:
        B, G, K = batch.r_hard.shape
        M = batch.r_soft.shape[2]

        all_pass = batch.r_hard.min(axis=2)
        ccr      = float(all_pass.mean())
        per_k    = batch.r_hard.mean(axis=(0, 1))

        if all_pass.sum() > 0:
            compliant_soft = batch.r_soft[all_pass.astype(bool)]
            mean_soft_per  = compliant_soft.mean(axis=0)   # (M,)
            mean_soft      = float(mean_soft_per.mean())
        else:
            mean_soft_per = np.zeros(M)
            mean_soft     = 0.0

        n_distinct = len(set(np.round(A_hat.flatten(), 4).tolist()))

        return {
            "ccr":                  ccr,
            "mean_soft_score":      mean_soft,
            "combined_quality":     ccr * mean_soft,
            "n_distinct_advantages": n_distinct,
            "A_hat_mean":           float(A_hat.mean()),
            "A_hat_std":            float(A_hat.std()),
            **{f"pass_rate_h{k+1}": float(per_k[k]) for k in range(K)},
            **{f"soft_score_s{m+1}": float(mean_soft_per[m])
               for m in range(M)},
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Shared verl wrapper base
# ──────────────────────────────────────────────────────────────────────────────

class _BaseVerlWrapper:
    """Internal base class for verl reward_fn wrappers."""

    method_name: str = "unknown"   # override in subclass

    def __init__(
        self,
        advantage_fn,          # GRPOAdvantage or GDPOAdvantage instance
        config: BaselineConfig,
        K: int,
        M: int,
    ):
        self._adv    = advantage_fn
        self.config  = config
        self.K       = K
        self.M       = M
        self._step   = 0

    def __call__(self, data: Any) -> Any:
        import torch
        from finplanenv.parser import compute_rewards_from_output

        responses   = data.non_tensor_batch["response"]
        tasks       = data.non_tensor_batch["task"]
        scenarios   = data.non_tensor_batch["scenario"]
        constraints = data.non_tensor_batch["constraints"]

        B_G = len(responses)
        G   = self.config.G
        assert B_G % G == 0
        B   = B_G // G

        r_hard = np.zeros((B, G, self.K), dtype=float)
        r_prox = np.zeros((B, G, self.K), dtype=float)
        r_soft = np.zeros((B, G, self.M), dtype=float)

        parse_failures = 0
        for idx, (resp, task, scen, cfg) in enumerate(
            zip(responses, tasks, scenarios, constraints)
        ):
            i, j = divmod(idx, G)
            bundle = compute_rewards_from_output(resp, task, scen, cfg)
            r_hard[i, j] = bundle.hard
            r_prox[i, j] = bundle.prox
            r_soft[i, j] = bundle.soft
            if bundle.hard.sum() == 0 and bundle.soft.sum() == 0:
                parse_failures += 1

        batch = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)
        A_hat, metrics = self._adv.compute(batch, step=self._step)

        if self._step % self.config.log_every == 0:
            self._log(metrics, parse_failures, B_G)

        self._step += 1
        return torch.tensor(A_hat.reshape(-1), dtype=torch.float32)

    def _log(
        self,
        metrics:        dict[str, Any],
        parse_failures: int,
        total:          int,
    ) -> None:
        try:
            import wandb
            wandb.log({
                **{f"{self.method_name}/{k}": v for k, v in metrics.items()},
                f"{self.method_name}/parse_failure_rate": parse_failures / total,
            }, step=self._step)
        except ImportError:
            logger.info(
                "[%s] Step %d | CCR=%.3f | SPS=%.3f | Distinct_A=%d | "
                "parse_fail=%.1f%%",
                self.method_name.upper(), self._step,
                metrics.get("ccr", 0),
                metrics.get("mean_soft_score", 0),
                metrics.get("n_distinct_advantages", 0),
                100 * parse_failures / total,
            )


class _BaseTRLRewardFn:
    """Internal base class for TRL reward_fn wrappers."""

    method_name: str = "unknown"

    def __init__(
        self,
        advantage_fn,
        config: BaselineConfig,
        K: int,
        M: int,
        scenario_fn,
    ):
        self._adv        = advantage_fn
        self.config      = config
        self.K           = K
        self.M           = M
        self.scenario_fn = scenario_fn
        self._step       = 0

    def __call__(
        self,
        prompts:     list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        from finplanenv.parser import compute_rewards_from_output

        B_G = len(completions)
        G   = self.config.G
        assert B_G % G == 0
        B   = B_G // G

        r_hard = np.zeros((B, G, self.K), dtype=float)
        r_prox = np.zeros((B, G, self.K), dtype=float)
        r_soft = np.zeros((B, G, self.M), dtype=float)

        parse_failures = 0
        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            i, j = divmod(idx, G)
            scenario, constraints, task = self.scenario_fn(prompt)
            bundle = compute_rewards_from_output(
                completion, task, scenario, constraints
            )
            r_hard[i, j] = bundle.hard
            r_prox[i, j] = bundle.prox
            r_soft[i, j] = bundle.soft
            if bundle.hard.sum() == 0 and bundle.soft.sum() == 0:
                parse_failures += 1

        batch = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)
        A_hat, metrics = self._adv.compute(batch, step=self._step)

        if self._step % self.config.log_every == 0:
            logger.info(
                "[%s] Step %d | CCR=%.3f | SPS=%.3f | Distinct_A=%d | "
                "parse_fail=%.1f%%",
                self.method_name.upper(), self._step,
                metrics.get("ccr", 0),
                metrics.get("mean_soft_score", 0),
                metrics.get("n_distinct_advantages", 0),
                100 * parse_failures / B_G,
            )

        self._step += 1
        return A_hat.reshape(-1).tolist()


# ──────────────────────────────────────────────────────────────────────────────
#  GRPO wrappers
# ──────────────────────────────────────────────────────────────────────────────

class GRPORewardWrapper(_BaseVerlWrapper):
    """
    GRPO verl reward_fn wrapper.

    Drop-in replacement for CDPORewardWrapper.
    Logs under the key prefix "grpo/" in W&B.

    Usage
    -----
    # In verl config:
    reward_model.reward_fn = GRPORewardWrapper(
        config=BaselineConfig(G=8), K=4, M=3
    )
    """

    method_name = "grpo"

    def __init__(self, config: BaselineConfig, K: int, M: int):
        super().__init__(
            advantage_fn = GRPOAdvantage(config, K, M),
            config       = config,
            K            = K,
            M            = M,
        )


class GRPOTRLRewardFn(_BaseTRLRewardFn):
    """
    GRPO TRL GRPOTrainer reward_fn.

    Usage
    -----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=GRPOTRLRewardFn(
            config=BaselineConfig(), K=4, M=3,
            scenario_fn=my_scenario_fn,
        ),
        ...
    )
    """

    method_name = "grpo"

    def __init__(
        self,
        config:      BaselineConfig,
        K:           int,
        M:           int,
        scenario_fn,
    ):
        super().__init__(
            advantage_fn = GRPOAdvantage(config, K, M),
            config       = config,
            K            = K,
            M            = M,
            scenario_fn  = scenario_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  GDPO wrappers
# ──────────────────────────────────────────────────────────────────────────────

class GDPORewardWrapper(_BaseVerlWrapper):
    """
    GDPO verl reward_fn wrapper.

    Drop-in replacement for CDPORewardWrapper.
    Logs under the key prefix "gdpo/" in W&B.

    Usage
    -----
    reward_model.reward_fn = GDPORewardWrapper(
        config=BaselineConfig(G=8), K=4, M=3
    )
    """

    method_name = "gdpo"

    def __init__(self, config: BaselineConfig, K: int, M: int):
        super().__init__(
            advantage_fn = GDPOAdvantage(config, K, M),
            config       = config,
            K            = K,
            M             = M,
        )


class GDPOTRLRewardFn(_BaseTRLRewardFn):
    """
    GDPO TRL GRPOTrainer reward_fn.

    Usage
    -----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=GDPOTRLRewardFn(
            config=BaselineConfig(), K=4, M=3,
            scenario_fn=my_scenario_fn,
        ),
        ...
    )
    """

    method_name = "gdpo"

    def __init__(
        self,
        config:      BaselineConfig,
        K:           int,
        M:           int,
        scenario_fn,
    ):
        super().__init__(
            advantage_fn = GDPOAdvantage(config, K, M),
            config       = config,
            K            = K,
            M             = M,
            scenario_fn  = scenario_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    B, G, K, M = 4, 8, 4, 3
    cfg = BaselineConfig(G=G)

    r_hard = np.random.binomial(1, 0.4, (B, G, K)).astype(float)
    r_prox = np.random.uniform(0.2, 0.9, (B, G, K))
    r_soft = np.random.uniform(0.3, 0.8, (B, G, M))
    batch  = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)

    print("=== GRPO baseline ===")
    grpo = GRPOAdvantage(cfg, K=K, M=M)
    A_grpo, m_grpo = grpo.compute(batch, step=0)
    print(f"  shape={A_grpo.shape}  "
          f"CCR={m_grpo['ccr']:.3f}  "
          f"distinct_A={m_grpo['n_distinct_advantages']}")
    # GRPO: per-group normalised, so each row has mean≈0, std≈1
    for i in range(B):
        assert abs(A_grpo[i].mean()) < 0.1
        assert abs(A_grpo[i].std() - 1.0) < 0.2

    print("=== GDPO baseline ===")
    gdpo = GDPOAdvantage(cfg, K=K, M=M)
    A_gdpo, m_gdpo = gdpo.compute(batch, step=0)
    print(f"  shape={A_gdpo.shape}  "
          f"CCR={m_gdpo['ccr']:.3f}  "
          f"distinct_A={m_gdpo['n_distinct_advantages']}")
    # GDPO: batch-normalised, so global mean≈0, std≈1
    assert abs(A_gdpo.mean()) < 0.1
    assert abs(A_gdpo.std() - 1.0) < 0.2

    print("\n=== Metric key alignment check (GRPO vs GDPO) ===")
    grpo_keys = set(m_grpo.keys())
    gdpo_keys = set(m_gdpo.keys())
    assert grpo_keys == gdpo_keys, \
        f"Key mismatch: GRPO={grpo_keys - gdpo_keys} GDPO={gdpo_keys - grpo_keys}"
    print(f"  All {len(grpo_keys)} metric keys match ✓")

    print("\n=== Metric key alignment check (GDPO vs CDPO) ===")
    import sys; sys.path.insert(0, '..')
    from finplanenv.cdpo import CDPOAdvantage, CDPOConfig
    from finplanenv.cdpo import BatchRewards as BR
    cdpo = CDPOAdvantage(CDPOConfig(G=G), K=K, M=M)
    _, m_cdpo = cdpo.compute(batch, step=0)
    cdpo_keys = set(m_cdpo.keys())
    shared = grpo_keys & cdpo_keys
    cdpo_only = cdpo_keys - grpo_keys
    print(f"  Shared keys:    {len(shared)}")
    print(f"  CDPO-only keys: {sorted(cdpo_only)}")
    # CDPO has extra diagnostic keys (lambda_prox, n_correlated_pairs,
    # ccr_rolling_mean) — these are CDPO-specific and don't need to match
    core_keys = {"ccr", "mean_soft_score", "combined_quality",
                 "n_distinct_advantages", "A_hat_mean", "A_hat_std"}
    assert core_keys.issubset(shared), \
        f"Missing core keys in shared set: {core_keys - shared}"
    print(f"  All {len(core_keys)} core comparison keys present in all methods ✓")

    print("\n✓  All baseline smoke tests passed.")

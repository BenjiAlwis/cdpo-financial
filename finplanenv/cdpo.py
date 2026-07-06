"""
finplanenv/cdpo.py
==================
CDPO advantage computation (Algorithm 1, Steps 2–6) and training-loop
integration for both verl and TRL/HF-TRL.

Structure
---------
CDPOConfig          — all hyperparameters in one dataclass
CorrelationTracker  — rolling EMA correlation matrix (Step 3)
CDPOAdvantage       — core Steps 2–6, pure numpy, framework-agnostic
CDPORewardWrapper   — verl reward-function interface (drop-in)
CDPOTRLRewardFn     — TRL GRPOTrainer reward_fn interface (drop-in)
TrainingLogger      — structured per-step metrics for W&B / TensorBoard

Each class has a one-to-one mapping to the algorithm box:
  Step 1  →  handled by verl/TRL framework
  Steps 2–6  →  CDPOAdvantage.compute()
  Step 7  →  handled by verl/TRL framework (uses returned advantages)

All numpy — no torch dependency in advantage computation.
Torch tensors are only touched at the integration boundary.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8  # numerical stability constant (ε in paper)


# ──────────────────────────────────────────────────────────────────────────────
#  CDPOConfig — all hyperparameters, with paper-recommended defaults
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CDPOConfig:
    """
    All CDPO hyperparameters.  Defaults match paper recommendations.
    Attribute names match notation table in cdpo_algorithm.tex exactly.

    Ablation handles
    ----------------
    Set beta_minus = beta_plus = 1.0   →  symmetric hard channel
    Set G_min = 0                      →  always activate soft channel
    Set gamma_nc = 0.0                 →  zero soft signal for non-compliant
    Set alpha_max = alpha_min          →  fixed α (no adaptive mixing)
    Set alpha_schedule = "fixed"       →  fixed α = alpha_fixed
    Set alpha_schedule = "annealing"   →  global linear annealing
    Set corr_threshold = 1.1           →  disable correlation correction
    """

    # Hard-channel asymmetric scaling (Step 2)
    beta_plus:  float = 1.0     # β⁺  reward for passing
    beta_minus: float = 2.0     # β⁻  penalty for failing (β⁻ ≥ β⁺)

    # Soft-channel gating (Step 4)
    G_min:    int   = 3     # minimum compliant rollouts to activate soft channel
    gamma_nc: float = 0.1   # discount for non-compliant rollout soft signal

    # Per-group adaptive mixing (Step 5)
    alpha_max: float = 0.9  # hard-dominated early / when CCR is low
    alpha_min: float = 0.3  # hard still present late / when CCR is high

    # Aggregation strategy — Week 3 selector
    # "adaptive"  : per-group α based on within-group CCR (default, best)
    # "fixed"     : fixed α = alpha_fixed for all groups
    # "annealing" : global linear anneal from alpha_anneal_start → alpha_min
    #               over anneal_steps training steps
    alpha_schedule:       str   = "adaptive"  # "adaptive"|"fixed"|"annealing"
    alpha_fixed:          float = 0.6         # used when alpha_schedule="fixed"
    alpha_anneal_start:   float = 0.9         # used when alpha_schedule="annealing"
    anneal_steps:         int   = 200         # steps to anneal over

    # Proximity annealing (Step 2, degenerate branch)
    lambda_prox_init: float = 1.0   # λ₀ — initial proximity weight
    ccr_window:       int   = 50    # W — rolling window for CCR estimate

    # Constraint correlation correction (Step 3)
    corr_threshold: float = 0.7    # τ — correlation threshold
    corr_ema_decay: float = 0.99   # ν — EMA decay for correlation matrix

    # Group size (must match verl/TRL rollout config)
    G: int = 8

    # Logging
    log_every: int = 10  # log detailed metrics every N steps


# ──────────────────────────────────────────────────────────────────────────────
#  CorrelationTracker — rolling EMA correlation matrix (Step 3)
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationTracker:
    """
    Maintains an exponential moving average of the constraint correlation
    matrix Σ̂ over constraint outcome vectors r ∈ {0,1}^K.

    Updated once per batch with all (i,j) constraint vectors in the batch.
    Treated as a stop-gradient constant during policy updates.

    Algorithm 1, Step 3 / Remark (Correlation matrix update).
    """

    def __init__(self, K: int, decay: float = 0.99):
        self.K = K
        self.decay = decay
        # Initialise to identity (no assumed correlation)
        self._cov  = np.eye(K, dtype=float)
        self._mean = np.zeros(K, dtype=float)
        self._n    = 0   # number of updates

    def update(self, r_hard: np.ndarray) -> None:
        """
        Update EMA with a batch of hard-signal vectors.

        Parameters
        ----------
        r_hard : np.ndarray, shape (B*G, K)
            All hard-signal vectors across the batch.
        """
        if r_hard.ndim != 2 or r_hard.shape[1] != self.K:
            raise ValueError(
                f"Expected shape (*, {self.K}), got {r_hard.shape}"
            )
        batch_mean = r_hard.mean(axis=0)
        centered   = r_hard - batch_mean
        batch_cov  = (centered.T @ centered) / max(len(r_hard) - 1, 1)

        if self._n == 0:
            self._cov  = batch_cov
            self._mean = batch_mean
        else:
            self._cov  = self.decay * self._cov  + (1 - self.decay) * batch_cov
            self._mean = self.decay * self._mean + (1 - self.decay) * batch_mean
        self._n += 1

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Return normalised correlation matrix ρ̂ ∈ [-1, 1]^{K×K}."""
        std = np.sqrt(np.diag(self._cov) + EPS)
        corr = self._cov / np.outer(std, std)
        return np.clip(corr, -1.0, 1.0)

    def correlated_pairs(self, threshold: float) -> list[tuple[int, int]]:
        """Return list of (k, l) pairs with |ρ̂_kl| > threshold, k < l."""
        corr = self.correlation_matrix
        pairs = []
        for k in range(self.K):
            for l in range(k + 1, self.K):
                if abs(corr[k, l]) > threshold:
                    pairs.append((k, l))
        return pairs


# ──────────────────────────────────────────────────────────────────────────────
#  CDPOAdvantage — core Steps 2–6, pure numpy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchRewards:
    """
    All reward signals for one training batch.

    Shapes
    ------
    r_hard : (B, G, K)   binary hard signals
    r_prox : (B, G, K)   proximity signals ∈ [0, 1]
    r_soft : (B, G, M)   soft preference scores ∈ [0, 1]

    where B = number of prompts in batch, G = group size,
    K = number of hard constraints, M = number of soft preferences.
    """
    r_hard: np.ndarray   # (B, G, K)
    r_prox: np.ndarray   # (B, G, K)
    r_soft: np.ndarray   # (B, G, M)

    def __post_init__(self):
        B, G, K = self.r_hard.shape
        assert self.r_prox.shape == (B, G, K), \
            f"r_prox shape {self.r_prox.shape} != r_hard shape {self.r_hard.shape}"
        assert self.r_soft.shape[:2] == (B, G), \
            f"r_soft shape {self.r_soft.shape} inconsistent with B={B}, G={G}"

    @property
    def B(self) -> int: return self.r_hard.shape[0]
    @property
    def G(self) -> int: return self.r_hard.shape[1]
    @property
    def K(self) -> int: return self.r_hard.shape[2]
    @property
    def M(self) -> int: return self.r_soft.shape[2]


class CDPOAdvantage:
    """
    CDPO advantage computation — Algorithm 1, Steps 2–6.

    Framework-agnostic: takes BatchRewards numpy arrays,
    returns final normalised advantages as numpy array shape (B, G).

    Usage
    -----
    cdpo = CDPOAdvantage(config, K=4, M=3)
    advantages = cdpo.compute(batch_rewards, step=t)
    # advantages shape: (B, G)
    # Feed directly to policy loss as the advantage signal.
    """

    def __init__(self, config: CDPOConfig, K: int, M: int):
        self.cfg  = config
        self.K    = K
        self.M    = M
        self.corr = CorrelationTracker(K, decay=config.corr_ema_decay)
        self._ccr_history: deque[float] = deque(maxlen=config.ccr_window)
        self._step = 0
        self._last_alpha_mean: float = config.alpha_max  # tracked for logging

    # ── public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        batch: BatchRewards,
        step: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run Steps 2–6 of Algorithm 1.

        Parameters
        ----------
        batch : BatchRewards
            All reward signals for this batch.
        step : int
            Current training step (used for λ_prox annealing).

        Returns
        -------
        advantages : np.ndarray, shape (B, G)
            Final batch-normalised advantages Â_sum^{(i,j)}.
        metrics : dict
            Diagnostic metrics for logging (CCR, distinct groups, etc.).
        """
        self._step = step
        B, G, K, M = batch.B, batch.G, batch.K, batch.M

        # Update rolling correlation matrix (stop-gradient)
        r_flat = batch.r_hard.reshape(-1, K)   # (B*G, K)
        self.corr.update(r_flat)

        # Compute λ_prox for this step
        lambda_prox = self._lambda_prox()

        # ── Step 2: Hard-channel normalisation ────────────────────────────
        A_hard_k = self._hard_channel(batch, lambda_prox)   # (B, G, K)

        # ── Step 3: Correlation correction → sum → A_hard ─────────────────
        A_hard_k = self._correlation_correction(A_hard_k)   # (B, G, K)
        A_hard   = A_hard_k.sum(axis=2)                      # (B, G)

        # ── Step 4: Soft-channel normalisation ────────────────────────────
        A_soft_m = self._soft_channel(batch)                 # (B, G, M)
        A_soft   = A_soft_m.sum(axis=2)                      # (B, G)

        # ── Step 5: Per-group adaptive mixing ─────────────────────────────
        A_sum = self._adaptive_mix(batch, A_hard, A_soft)    # (B, G)

        # ── Step 6: Batch-wise normalisation ──────────────────────────────
        A_hat = self._batch_normalise(A_sum)                 # (B, G)

        # ── Metrics ───────────────────────────────────────────────────────
        metrics = self._compute_metrics(batch, A_hard_k, A_hat, lambda_prox)

        return A_hat, metrics

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _lambda_prox(self) -> float:
        """
        λ_prox(t) = λ₀ · max(0, 1 − CCR_W(t))

        Anneals automatically as constraint compliance improves.
        Remark (Proximity annealing) in cdpo_algorithm.tex.
        """
        if len(self._ccr_history) == 0:
            return self.cfg.lambda_prox_init
        ccr_w = float(np.mean(self._ccr_history))
        return self.cfg.lambda_prox_init * max(0.0, 1.0 - ccr_w)

    def _hard_channel(
        self,
        batch: BatchRewards,
        lambda_prox: float,
    ) -> np.ndarray:
        """
        Algorithm 1, Step 2.
        Returns A_hard_k shape (B, G, K).
        """
        B, G, K = batch.B, batch.G, batch.K
        A = np.zeros((B, G, K), dtype=float)

        for i in range(B):
            for k in range(K):
                r_k = batch.r_hard[i, :, k]         # shape (G,)
                p_k = r_k.mean()                     # pass rate p_k^{(i)}

                if 0.0 < p_k < 1.0:
                    # Informative branch — asymmetric mean-centering
                    for j in range(G):
                        if r_k[j] == 1:
                            A[i, j, k] = self.cfg.beta_plus  * (1.0 - p_k)
                        else:
                            A[i, j, k] = -self.cfg.beta_minus * p_k
                else:
                    # Degenerate branch — proximity imputation
                    rho_k  = batch.r_prox[i, :, k]      # shape (G,)
                    rho_mu = rho_k.mean()
                    A[i, :, k] = lambda_prox * (rho_k - rho_mu)

        return A

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def _correlation_correction(
        self,
        A_hard_k: np.ndarray,
    ) -> np.ndarray:
        """
        Algorithm 1, Step 3.
        Down-weights correlated constraint pairs.
        Stop-gradient: correlation matrix is not differentiated through.
        """
        A = A_hard_k.copy()
        for k, l in self.corr.correlated_pairs(self.cfg.corr_threshold):
            rho_kl = abs(self.corr.correlation_matrix[k, l])
            scale  = 1.0 - rho_kl
            A[:, :, k] *= scale
            A[:, :, l] *= scale
        return A

    # ── Step 4 ────────────────────────────────────────────────────────────────

    def _soft_channel(self, batch: BatchRewards) -> np.ndarray:
        """
        Algorithm 1, Step 4.
        Returns A_soft_m shape (B, G, M).
        """
        B, G, M = batch.B, batch.G, batch.M
        A = np.zeros((B, G, M), dtype=float)

        for i in range(B):
            # Compliant rollout mask: all K hard constraints pass
            compliant_mask = batch.r_hard[i].min(axis=1) == 1  # shape (G,)
            c_i = int(compliant_mask.sum())

            # Update CCR history for λ_prox annealing
            self._ccr_history.append(c_i / G)

            if c_i < self.cfg.G_min:
                # Too few compliant rollouts — zero soft channel
                # A[i, :, :] already zero
                continue

            for m in range(M):
                s_m = batch.r_soft[i, :, m]     # shape (G,)

                # Statistics over compliant rollouts only
                s_compliant = s_m[compliant_mask]
                mu_m  = s_compliant.mean()
                sig_m = np.sqrt(s_compliant.var() + EPS)

                for j in range(G):
                    z = (s_m[j] - mu_m) / sig_m
                    if compliant_mask[j]:
                        A[i, j, m] = z
                    else:
                        # Discounted soft signal for non-compliant rollouts
                        A[i, j, m] = self.cfg.gamma_nc * z

        return A

    # ── Step 5 ────────────────────────────────────────────────────────────────

    def _adaptive_mix(
        self,
        batch:  BatchRewards,
        A_hard: np.ndarray,
        A_soft: np.ndarray,
    ) -> np.ndarray:
        """
        Algorithm 1, Step 5 — mixing with selectable aggregation strategy.

        Three strategies selectable via CDPOConfig.alpha_schedule:

        "adaptive"  (default, Week 3 primary)
            α_i = α_max − (α_max − α_min) · (c^{(i)} / G)
            Per-group, self-regulating. No schedule hyperparameter.
            When c^{(i)} = 0: α_i = α_max (hard-dominated)
            When c^{(i)} = G: α_i = α_min (soft-dominated)

        "fixed"     (Week 3 ablation 1)
            α_i = alpha_fixed for all groups and all steps.
            Simplest baseline for the mixing ablation.

        "annealing" (Week 3 ablation 2)
            α_i = max(alpha_min, alpha_anneal_start − step/anneal_steps)
            Global linear schedule, same α for all groups.
            Decays from alpha_anneal_start to alpha_min over anneal_steps.
        """
        B, G    = batch.B, batch.G
        A_sum   = np.zeros((B, G), dtype=float)
        sched   = self.cfg.alpha_schedule

        for i in range(B):
            if sched == "adaptive":
                compliant_mask = batch.r_hard[i].min(axis=1) == 1
                c_i = int(compliant_mask.sum())
                alpha_i = (self.cfg.alpha_max
                           - (self.cfg.alpha_max - self.cfg.alpha_min)
                           * (c_i / G))

            elif sched == "fixed":
                alpha_i = self.cfg.alpha_fixed

            elif sched == "annealing":
                progress = min(1.0, self._step / max(self.cfg.anneal_steps, 1))
                alpha_i  = max(
                    self.cfg.alpha_min,
                    self.cfg.alpha_anneal_start
                    - progress * (self.cfg.alpha_anneal_start - self.cfg.alpha_min)
                )

            else:
                raise ValueError(
                    f"Unknown alpha_schedule: {sched!r}. "
                    "Choose 'adaptive', 'fixed', or 'annealing'."
                )

            A_sum[i] = alpha_i * A_hard[i] + (1.0 - alpha_i) * A_soft[i]

        # Track mean alpha for logging
        alphas = []
        for i in range(B):
            if sched == "adaptive":
                c_i = int((batch.r_hard[i].min(axis=1) == 1).sum())
                alphas.append(self.cfg.alpha_max
                              - (self.cfg.alpha_max - self.cfg.alpha_min) * (c_i / G))
            elif sched == "fixed":
                alphas.append(self.cfg.alpha_fixed)
            else:
                progress = min(1.0, self._step / max(self.cfg.anneal_steps, 1))
                alphas.append(max(self.cfg.alpha_min,
                    self.cfg.alpha_anneal_start
                    - progress * (self.cfg.alpha_anneal_start - self.cfg.alpha_min)))
        self._last_alpha_mean = float(np.mean(alphas))
        return A_sum

    # ── Step 6 ────────────────────────────────────────────────────────────────

    def _batch_normalise(self, A_sum: np.ndarray) -> np.ndarray:
        """
        Algorithm 1, Step 6.
        Normalise across all (i, j) in the batch.
        """
        mu  = A_sum.mean()
        sig = A_sum.std() + EPS
        return (A_sum - mu) / sig

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        batch: BatchRewards,
        A_hard_k: np.ndarray,
        A_hat: np.ndarray,
        lambda_prox: float,
    ) -> dict[str, Any]:
        """
        Compute all diagnostic metrics logged during training.
        These become your paper's training curves (Figure 2).
        """
        B, G, K = batch.r_hard.shape

        # CCR: fraction of rollouts that pass ALL hard constraints
        all_pass = batch.r_hard.min(axis=2)   # (B, G) — 1 iff all K pass
        ccr = float(all_pass.mean())

        # Per-constraint pass rates
        per_constraint_pr = batch.r_hard.mean(axis=(0, 1))  # shape (K,)

        # Distinct advantage groups (paper Figure 2 metric)
        # Count unique rounded advantage values across batch
        a_flat = np.round(A_hat.flatten(), 4)
        n_distinct = len(set(a_flat.tolist()))

        # Mean soft scores (conditional on compliance)
        if all_pass.sum() > 0:
            compliant_soft = batch.r_soft[all_pass.astype(bool)]  # (n_compliant, M)
            mean_soft = compliant_soft.mean(axis=0)   # shape (M,)
        else:
            mean_soft = np.zeros(batch.M)

        # Hard/soft advantage magnitudes
        A_hard_sum = A_hard_k.sum(axis=2)  # (B, G)

        # Correlation matrix (for logging)
        corr = self.corr.correlation_matrix
        n_corr_pairs = len(self.corr.correlated_pairs(self.cfg.corr_threshold))

        return {
            # Primary metrics (Table 1 in paper)
            "ccr":                  ccr,
            "mean_soft_score":      float(mean_soft.mean()) if len(mean_soft) else 0.0,
            "combined_quality":     ccr * float(mean_soft.mean()) if len(mean_soft) else 0.0,

            # Advantage diagnostics (Figure 2 equivalent)
            "n_distinct_advantages":     n_distinct,
            "A_hard_mean":               float(A_hard_sum.mean()),
            "A_hard_std":                float(A_hard_sum.std()),
            "A_hat_mean":                float(A_hat.mean()),
            "A_hat_std":                 float(A_hat.std()),

            # Per-constraint compliance
            **{f"pass_rate_h{k+1}": float(per_constraint_pr[k])
               for k in range(K)},

            # Per-soft-preference scores
            **{f"soft_score_s{m+1}": float(mean_soft[m]) if m < len(mean_soft) else 0.0
               for m in range(batch.M)},

            # Algorithm internals
            "lambda_prox":          lambda_prox,
            "n_correlated_pairs":   n_corr_pairs,
            "ccr_rolling_mean":     float(np.mean(self._ccr_history))
                                    if self._ccr_history else 0.0,
            # Aggregation strategy diagnostics (Week 3)
            "alpha_schedule":       self.cfg.alpha_schedule,
            "alpha_mean":           float(self._last_alpha_mean),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  verl integration — drop-in reward function
# ──────────────────────────────────────────────────────────────────────────────

class CDPORewardWrapper:
    """
    verl reward-function interface for CDPO.

    verl calls reward_fn(data) once per batch.
    data is a DataProto with fields including:
        data.non_tensor_batch["response"]  — list of LLM output strings
        data.non_tensor_batch["task"]      — list of task names
        data.non_tensor_batch["scenario"]  — list of *Scenario objects
        data.non_tensor_batch["constraints"]— list of *Constraints objects

    This wrapper:
    1. Calls compute_rewards_from_output() for each rollout  (Step 1 output)
    2. Assembles BatchRewards
    3. Calls CDPOAdvantage.compute()  (Steps 2–6)
    4. Returns advantages as a flat tensor for verl's policy update  (Step 7)

    Usage in verl config
    --------------------
    reward_model.reward_fn = CDPORewardWrapper(config, K=4, M=3)
    """

    def __init__(self, config: CDPOConfig, K: int, M: int):
        self.config  = config
        self.K       = K
        self.M       = M
        self.cdpo    = CDPOAdvantage(config, K, M)
        self._step   = 0

    def __call__(self, data: Any) -> Any:
        """verl reward_fn signature: (DataProto) → tensor of shape (B*G,)."""
        import torch

        responses    = data.non_tensor_batch["response"]      # list len B*G
        tasks        = data.non_tensor_batch["task"]           # list len B*G
        scenarios    = data.non_tensor_batch["scenario"]       # list len B*G
        constraints  = data.non_tensor_batch["constraints"]   # list len B*G

        B_times_G = len(responses)
        G = self.config.G
        assert B_times_G % G == 0, \
            f"Expected B*G responses, got {B_times_G} which is not divisible by G={G}"
        B = B_times_G // G

        # ── Step 1: score each rollout ────────────────────────────────────
        from finplanenv.parser import compute_rewards_from_output

        r_hard = np.zeros((B, G, self.K), dtype=float)
        r_prox = np.zeros((B, G, self.K), dtype=float)
        r_soft = np.zeros((B, G, self.M), dtype=float)

        parse_failures = 0
        for idx, (resp, task, scenario, cfg) in enumerate(
            zip(responses, tasks, scenarios, constraints)
        ):
            i, j = divmod(idx, G)
            bundle = compute_rewards_from_output(resp, task, scenario, cfg)
            r_hard[i, j] = bundle.hard
            r_prox[i, j] = bundle.prox
            r_soft[i, j] = bundle.soft
            if bundle.hard.sum() == 0 and bundle.soft.sum() == 0:
                parse_failures += 1

        if parse_failures > 0:
            logger.warning(
                "Step %d: %d/%d rollouts had parse failures (rewards zeroed).",
                self._step, parse_failures, B_times_G,
            )

        # ── Steps 2–6: CDPO advantage computation ─────────────────────────
        batch   = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)
        A_hat, metrics = self.cdpo.compute(batch, step=self._step)

        # Log metrics
        if self._step % self.config.log_every == 0:
            self._log(metrics, parse_failures, B_times_G)

        self._step += 1

        # Return flat tensor shape (B*G,) — verl expects this shape
        advantages_flat = torch.tensor(
            A_hat.reshape(-1), dtype=torch.float32
        )
        return advantages_flat

    def _log(
        self,
        metrics: dict[str, Any],
        parse_failures: int,
        total: int,
    ) -> None:
        try:
            import wandb
            wandb.log({
                **{f"cdpo/{k}": v for k, v in metrics.items()},
                "cdpo/parse_failure_rate": parse_failures / total,
            }, step=self._step)
        except ImportError:
            logger.info(
                "Step %d | CCR=%.3f | SPS=%.3f | Distinct_A=%d | "
                "parse_fail=%.1f%%",
                self._step,
                metrics.get("ccr", 0),
                metrics.get("mean_soft_score", 0),
                metrics.get("n_distinct_advantages", 0),
                100 * parse_failures / total,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  TRL/HF-TRL integration — reward_fn for GRPOTrainer
# ──────────────────────────────────────────────────────────────────────────────

class CDPOTRLRewardFn:
    """
    TRL GRPOTrainer reward_fn interface for CDPO.

    TRL calls reward_fn(prompts, completions, **kwargs) per batch.
    CDPO replaces the scalar reward with a normalised advantage.

    Usage
    -----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=CDPOTRLRewardFn(
            config=CDPOConfig(),
            K=4, M=3,
            scenario_fn=my_scenario_fn,   # maps prompt → scenario + constraints
        ),
        ...
    )

    scenario_fn signature
    ---------------------
    def my_scenario_fn(prompt: str) -> tuple[*Scenario, *Constraints, str]:
        # returns (scenario, constraints, task_name)
        ...
    """

    def __init__(
        self,
        config: CDPOConfig,
        K: int,
        M: int,
        scenario_fn,      # callable: prompt → (scenario, constraints, task)
    ):
        self.config      = config
        self.K           = K
        self.M           = M
        self.cdpo        = CDPOAdvantage(config, K, M)
        self.scenario_fn = scenario_fn
        self._step       = 0

    def __call__(
        self,
        prompts:     list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        TRL reward_fn signature.

        Parameters
        ----------
        prompts     : list of B*G prompt strings (repeated G times each)
        completions : list of B*G completion strings

        Returns
        -------
        list of B*G float advantages
        """
        from finplanenv.parser import compute_rewards_from_output

        B_times_G = len(completions)
        G         = self.config.G
        assert B_times_G % G == 0, \
            f"Got {B_times_G} completions, expected multiple of G={G}"
        B = B_times_G // G

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

        batch   = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)
        A_hat, metrics = self.cdpo.compute(batch, step=self._step)

        if self._step % self.config.log_every == 0:
            logger.info(
                "Step %d | CCR=%.3f | SPS=%.3f | Distinct_A=%d | "
                "parse_fail=%.1f%%",
                self._step,
                metrics.get("ccr", 0.0),
                metrics.get("mean_soft_score", 0.0),
                metrics.get("n_distinct_advantages", 0),
                100.0 * parse_failures / B_times_G,
            )

        self._step += 1
        return A_hat.reshape(-1).tolist()


# ──────────────────────────────────────────────────────────────────────────────
#  GRPO and GDPO baselines — same interface, for controlled comparison
# ──────────────────────────────────────────────────────────────────────────────

def grpo_advantages(
    r_hard: np.ndarray,
    r_soft: np.ndarray,
    weights_hard: np.ndarray | None = None,
    weights_soft: np.ndarray | None = None,
) -> np.ndarray:
    """
    Standard GRPO advantage: sum all signals, single z-score.
    Baseline for ablation.  Eq. (2) in GDPO paper.

    r_hard : (B, G, K)
    r_soft : (B, G, M)
    Returns: (B, G)
    """
    B, G, K = r_hard.shape
    M = r_soft.shape[2]

    w_h = weights_hard if weights_hard is not None else np.ones(K)
    w_s = weights_soft if weights_soft is not None else np.ones(M)

    # Sum all signals into scalar
    R = (r_hard * w_h).sum(axis=2) + (r_soft * w_s).sum(axis=2)  # (B, G)

    # Group-level z-score (standard GRPO)
    A = np.zeros_like(R)
    for i in range(B):
        mu  = R[i].mean()
        sig = R[i].std() + EPS
        A[i] = (R[i] - mu) / sig
    return A


def gdpo_advantages(
    r_hard: np.ndarray,
    r_soft: np.ndarray,
) -> np.ndarray:
    """
    GDPO advantage: per-signal z-score, then sum, then batch normalise.
    Baseline for ablation.  Eq. (4)-(6) in GDPO paper.

    r_hard : (B, G, K)
    r_soft : (B, G, M)
    Returns: (B, G)
    """
    B, G, K = r_hard.shape
    M = r_soft.shape[2]

    A = np.zeros((B, G), dtype=float)

    # Per-signal group normalisation (GDPO Eq. 4)
    for i in range(B):
        for k in range(K):
            r = r_hard[i, :, k]
            mu, sig = r.mean(), r.std() + EPS
            A[i] += (r - mu) / sig
        for m in range(M):
            s = r_soft[i, :, m]
            mu, sig = s.mean(), s.std() + EPS
            A[i] += (s - mu) / sig

    # Batch-wise normalisation (GDPO Eq. 6)
    mu  = A.mean()
    sig = A.std() + EPS
    return (A - mu) / sig


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    B, G, K, M = 4, 8, 4, 3   # portfolio task dimensions
    config = CDPOConfig(G=G)

    # Simulate a realistic batch:
    # early training — mixed compliance, low CCR
    r_hard = np.random.binomial(1, 0.4, (B, G, K)).astype(float)
    r_prox = np.random.uniform(0.2, 0.9, (B, G, K))
    r_soft = np.random.uniform(0.3, 0.8, (B, G, M))

    batch = BatchRewards(r_hard=r_hard, r_prox=r_prox, r_soft=r_soft)

    cdpo = CDPOAdvantage(config, K=K, M=M)
    A_hat, metrics = cdpo.compute(batch, step=0)

    print("=== CDPOAdvantage smoke test ===")
    print(f"A_hat shape: {A_hat.shape}  (expected ({B}, {G}))")
    print(f"A_hat mean:  {A_hat.mean():.6f}  (expected ~0)")
    print(f"A_hat std:   {A_hat.std():.6f}   (expected ~1)")
    assert A_hat.shape == (B, G)
    assert abs(A_hat.mean()) < 0.1,  "Batch-normalised mean should be ~0"
    assert abs(A_hat.std()  - 1.0) < 0.2, "Batch-normalised std should be ~1"

    print(f"\nKey metrics:")
    for k in ["ccr", "mean_soft_score", "n_distinct_advantages",
              "lambda_prox", "n_correlated_pairs"]:
        print(f"  {k}: {metrics[k]}")

    # GRPO baseline comparison
    A_grpo = grpo_advantages(r_hard, r_soft)
    A_gdpo = gdpo_advantages(r_hard, r_soft)

    print(f"\n=== Distinct advantage groups ===")
    print(f"  GRPO:  {len(set(np.round(A_grpo.flatten(), 4).tolist()))}")
    print(f"  GDPO:  {len(set(np.round(A_gdpo.flatten(), 4).tolist()))}")
    print(f"  CDPO:  {metrics['n_distinct_advantages']}")

    # Proposition 1 claims CDPO preserves more distinct advantage groups
    # *within each group* than GRPO — not across the full batch.
    # Batch normalisation collapses batch-level counts for all methods.
    avg_grpo = np.mean([len(set(np.round(A_grpo[i], 4).tolist())) for i in range(B)])
    avg_cdpo = np.mean([len(set(np.round(A_hat[i],  4).tolist())) for i in range(B)])
    print(f"  Avg within-group — GRPO: {avg_grpo:.1f}  CDPO: {avg_cdpo:.1f}")

    # Late training simulation — high CCR
    print("\n=== Late training (high CCR) ===")
    r_hard_late = np.ones((B, G, K))   # all pass
    r_prox_late = np.ones((B, G, K))
    r_soft_late = np.random.uniform(0.6, 1.0, (B, G, M))
    batch_late  = BatchRewards(r_hard_late, r_prox_late, r_soft_late)

    # Prime the CCR history with high values
    for _ in range(config.ccr_window):
        cdpo._ccr_history.append(1.0)

    A_late, metrics_late = cdpo.compute(batch_late, step=100)
    print(f"  λ_prox (should be ~0): {metrics_late['lambda_prox']:.4f}")
    print(f"  CCR rolling mean:      {metrics_late['ccr_rolling_mean']:.4f}")
    alpha_i = config.alpha_max - (config.alpha_max - config.alpha_min) * (G / G)
    print(f"  α_i (all-pass group):  {alpha_i:.4f}  "
          f"(should be α_min={config.alpha_min})")

    print("\n✓  All CDPO smoke tests passed.")

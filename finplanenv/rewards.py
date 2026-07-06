"""
finplanenv/rewards.py
=====================
FinPlanEnv reward functions — implementation of the CDPO paper spec.

Every function here corresponds 1-to-1 to a numbered equation in
cdpo_reward_spec.tex.  The docstring of each function cites the
equation number so the paper and code stay in sync.

All hard-constraint functions return int ∈ {0, 1}.
All proximity functions return float ∈ [0, 1].
All soft-preference functions return float ∈ [0, 1].

Usage
-----
Each task returns a RewardBundle dataclass with:
    .hard   : np.ndarray shape (K,)   — binary hard signals
    .prox   : np.ndarray shape (K,)   — proximity signals
    .soft   : np.ndarray shape (M,)   — continuous soft signals

These map directly to the r_hard and r_soft vectors in Algorithm 1.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────────────
#  Shared dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardBundle:
    """Container for one rollout's reward signals across all channels."""
    hard: np.ndarray   # shape (K,)  ∈ {0, 1}
    prox: np.ndarray   # shape (K,)  ∈ [0, 1]
    soft: np.ndarray   # shape (M,)  ∈ [0, 1]
    task: str = ""
    parse_failed: bool = False   # True iff output was unparseable (all-zero by
                                 # policy, but distinct from a valid-but-failing
                                 # plan that also has hard=zeros)

    @property
    def n_hard(self) -> int:
        return len(self.hard)

    @property
    def n_soft(self) -> int:
        return len(self.soft)

    def all_hard_pass(self) -> bool:
        return bool(np.all(self.hard == 1))


# ──────────────────────────────────────────────────────────────────────────────
#  Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _clip01(x: float) -> float:
    """Clip value to [0, 1]."""
    return float(np.clip(x, 0.0, 1.0))


def compute_max_drawdown(
    weights: np.ndarray,
    return_series: np.ndarray,
) -> float:
    """
    Compute maximum drawdown of a portfolio over a historical return series.

    Eq. (MDD) in cdpo_reward_spec.tex.

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
        Portfolio weights (must sum to 1).
    return_series : np.ndarray, shape (T, N)
        Historical asset returns, T timesteps × N assets.

    Returns
    -------
    float
        Maximum drawdown (negative value, e.g. -0.12 = 12% drawdown).
    """
    portfolio_returns = return_series @ weights          # shape (T,)
    cum_value = np.cumprod(1.0 + portfolio_returns)      # shape (T,)
    rolling_max = np.maximum.accumulate(cum_value)
    drawdowns = (cum_value - rolling_max) / rolling_max
    return float(drawdowns.min())


def compute_hhi(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index. Eq. (HHI) in cdpo_reward_spec.tex.

    Returns float ∈ [1/N, 1].  Lower = more diversified.
    """
    return float(np.sum(weights ** 2))


def amortised_monthly_payment(
    principal: float,
    annual_rate: float,
    n_months: int,
) -> float:
    """
    Standard amortisation formula. Eq. (amortisation) in spec.

    m(ρ, L, T) = L · (ρ/12) / (1 − (1 + ρ/12)^{−T})
    """
    if annual_rate == 0.0:
        return principal / n_months
    r = annual_rate / 12.0
    return principal * r / (1.0 - (1.0 + r) ** (-n_months))


def total_interest_cost(
    principal: float,
    annual_rate: float,
    n_months: int,
) -> float:
    """
    TIC(ρ, L, T) = m(ρ, L, T) · T − L.  Eq. (S1 loan) in spec.
    """
    return amortised_monthly_payment(principal, annual_rate, n_months) \
           * n_months - principal


# ──────────────────────────────────────────────────────────────────────────────
#  Task 1 — Portfolio Allocation  (K=4 hard, M=3 soft)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioPlan:
    """Parsed fields from an LLM portfolio-allocation plan."""
    weights: np.ndarray         # shape (N,), must sum ~1
    sector_labels: list[str]    # length N
    expected_returns: np.ndarray  # shape (N,) annual
    cov_matrix: np.ndarray      # shape (N, N) annual
    esg_ratings: np.ndarray     # shape (N,) ∈ [0, 100]
    benchmark_weights: np.ndarray  # shape (N,)
    return_series: np.ndarray   # shape (T, N) historical


@dataclass
class PortfolioConstraints:
    """Per-scenario constraint thresholds."""
    weight_sum_tol: float = 0.01    # δ_w
    max_drawdown_limit: float = -0.10  # D*  (negative)
    banned_sectors: set[str] = field(default_factory=set)
    banned_weight_tol: float = 0.05  # max weight allowed in banned sectors (H3)
    hhi_limit: float = 0.25         # H*
    risk_free_rate: float = 0.04    # r_f


def portfolio_rewards(
    plan: PortfolioPlan,
    cfg: PortfolioConstraints,
) -> RewardBundle:
    """
    Compute all K=4 hard, K=4 proximity, M=3 soft signals
    for a portfolio allocation plan.

    Equations H1–H4, S1–S3 in Section 4.1.1 of spec.
    """
    w = plan.weights

    # ── Hard constraints ──────────────────────────────────────────────────
    # H1: weight-sum  (Eq. pa_h1)
    h1 = int(abs(w.sum() - 1.0) <= cfg.weight_sum_tol)

    # H2: max drawdown  (Eq. pa_h2)
    mdd = compute_max_drawdown(w, plan.return_series)
    h2 = int(mdd >= cfg.max_drawdown_limit)

    # H3: no MEANINGFUL weight in banned sectors  (Eq. pa_h3)
    # Weight-based (not mere presence): the constraint is satisfied when the
    # total weight allocated to banned sectors is below tolerance. This makes H3
    # satisfiable by zeroing out banned-sector holdings — which is what creates a
    # genuine hard/soft conflict when a banned sector also carries high ESG:
    # passing H3 forces you to drop that weight and sacrifice ESG (S2).
    banned_weight = sum(
        float(wi) for wi, s in zip(w, plan.sector_labels)
        if s in cfg.banned_sectors
    )
    h3 = int(banned_weight <= cfg.banned_weight_tol)

    # H4: HHI diversification  (Eq. pa_h4)
    hhi = compute_hhi(w)
    h4 = int(hhi <= cfg.hhi_limit)

    hard = np.array([h1, h2, h3, h4], dtype=float)

    # ── Proximity signals ─────────────────────────────────────────────────
    # ρ1: how close to summing to 1?
    p1 = _clip01(1.0 - abs(w.sum() - 1.0) / cfg.weight_sum_tol)

    # ρ2: how close to drawdown limit? (both are negative)
    p2 = _clip01(1.0 - abs(mdd) / abs(cfg.max_drawdown_limit))

    # ρ3: graded proximity to the banned-weight tolerance (smooth gradient now
    # that H3 is weight-based rather than binary presence)
    p3 = _clip01(1.0 - banned_weight / max(cfg.banned_weight_tol, 1e-6)) \
        if banned_weight > cfg.banned_weight_tol else 1.0

    # ρ4: how close to HHI limit?
    p4 = _clip01(1.0 - hhi / cfg.hhi_limit)

    prox = np.array([p1, p2, p3, p4], dtype=float)

    # ── Soft preferences ──────────────────────────────────────────────────
    # S1: Sharpe ratio (ex-ante)  (Eq. pa_s1)
    C_SR = 3.0
    port_var = float(w @ plan.cov_matrix @ w)
    port_vol = np.sqrt(max(port_var, 1e-10))
    sharpe_raw = (w @ plan.expected_returns - cfg.risk_free_rate) / port_vol
    s1 = _clip01(sharpe_raw / C_SR)

    # S2: ESG score  (Eq. pa_s2)
    s2 = _clip01(float(w @ plan.esg_ratings) / 100.0)

    # S3: tax efficiency via turnover proxy  (Eq. pa_s3)
    turnover = float(np.sum(np.abs(w - plan.benchmark_weights))) / 2.0
    s3 = _clip01(1.0 - turnover)

    soft = np.array([s1, s2, s3], dtype=float)

    return RewardBundle(hard=hard, prox=prox, soft=soft, task="portfolio")


# ──────────────────────────────────────────────────────────────────────────────
#  Task 2 — Retirement Planning  (K=2 hard, M=3 soft)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetirementPlan:
    """Parsed fields from an LLM retirement plan."""
    initial_balance: float          # B_0
    annual_withdrawal: float        # W (or schedule {W_t})
    withdrawal_schedule: np.ndarray # shape (years,) — yearly amounts
    weights: np.ndarray             # asset allocation for MC
    expected_returns: np.ndarray    # shape (N,) annual
    cov_matrix: np.ndarray          # shape (N, N) annual
    current_age: int                # a_0
    target_age: int                 # Y


@dataclass
class RetirementConstraints:
    """Per-scenario constraint thresholds."""
    mc_survival_threshold: float = 0.90   # θ_MC
    income_floor_monthly: float = 3000.0  # F_min
    n_mc: int = 1000
    mc_seed: int = 42


def _run_mc_retirement(
    plan: RetirementPlan,
    cfg: RetirementConstraints,
) -> tuple[float, float]:
    """
    Run Monte Carlo simulation for retirement survival.

    Returns
    -------
    survival_rate : float  ∈ [0, 1]
    median_final_balance : float  ≥ 0
    """
    rng = np.random.default_rng(cfg.mc_seed)
    years = plan.target_age - plan.current_age

    # Annual portfolio return = w'μ + w'Σw noise
    port_mu = float(plan.weights @ plan.expected_returns)
    port_var = float(plan.weights @ plan.cov_matrix @ plan.weights)
    port_std = np.sqrt(max(port_var, 0.0))

    # Annual returns matrix: shape (n_mc, years)
    annual_returns = rng.normal(port_mu, port_std, (cfg.n_mc, years))

    # Simulate balances — Eq. (mc_balance) in spec
    balance = np.full(cfg.n_mc, float(plan.initial_balance))
    depleted = np.zeros(cfg.n_mc, dtype=bool)

    for t in range(years):
        w_t = (plan.withdrawal_schedule[t]
               if t < len(plan.withdrawal_schedule)
               else plan.annual_withdrawal)
        balance = balance * (1.0 + annual_returns[:, t]) - w_t
        depleted |= (balance <= 0.0)
        balance = np.maximum(balance, 0.0)   # floor at zero

    survival_rate = float(1.0 - depleted.mean())
    median_final_balance = float(np.median(balance))
    return survival_rate, median_final_balance


def retirement_rewards(
    plan: RetirementPlan,
    cfg: RetirementConstraints,
) -> RewardBundle:
    """
    Compute all K=2 hard, K=2 proximity, M=3 soft signals
    for a retirement plan.

    Equations H1–H2, S1–S3 in Section 4.1.2 of spec.
    """
    surv_rate, median_balance = _run_mc_retirement(plan, cfg)
    monthly_income = plan.annual_withdrawal / 12.0

    # ── Hard constraints ──────────────────────────────────────────────────
    # H1: MC survival  (Eq. rp_h1)
    h1 = int(surv_rate >= cfg.mc_survival_threshold)

    # H2: income floor  (Eq. rp_h2)
    h2 = int(monthly_income >= cfg.income_floor_monthly)

    hard = np.array([h1, h2], dtype=float)

    # ── Proximity signals ─────────────────────────────────────────────────
    # ρ1: how close to survival threshold?
    p1 = _clip01(surv_rate / cfg.mc_survival_threshold)

    # ρ2: how close to income floor?
    p2 = _clip01(monthly_income / cfg.income_floor_monthly)

    prox = np.array([p1, p2], dtype=float)

    # ── Soft preferences ──────────────────────────────────────────────────
    C_LQ = 1.5

    # S1: lifestyle quality  (Eq. rp_s1)
    s1 = _clip01(monthly_income / (C_LQ * cfg.income_floor_monthly))

    # S2: bequest preference  (Eq. rp_s2)
    s2 = _clip01(median_balance / plan.initial_balance)

    # S3: withdrawal smoothness — CoV of schedule  (Eq. rp_s3)
    ws = plan.withdrawal_schedule
    if len(ws) > 1 and ws.mean() > 0:
        s3 = _clip01(1.0 - ws.std() / ws.mean())
    else:
        s3 = 1.0   # fixed withdrawal = perfectly smooth

    soft = np.array([s1, s2, s3], dtype=float)

    return RewardBundle(hard=hard, prox=prox, soft=soft, task="retirement")


# ──────────────────────────────────────────────────────────────────────────────
#  Task 3 — Loan Structuring  (K=3 hard, M=3 soft)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoanPlan:
    """Parsed fields from an LLM loan structuring plan."""
    loan_amount: float              # L
    property_value: float           # V
    annual_rate: float              # ρ
    term_months: int                # T
    gross_monthly_income: float     # I
    total_monthly_debt: float       # D (all debts incl. proposed loan)
    prepayment_penalty: bool        # p ∈ {0, 1}
    lock_in_years: float            # ℓ
    regulatory_tier: str            # "QM" or "non-QM"


@dataclass
class LoanConstraints:
    """Per-scenario constraint thresholds."""
    dti_limit: float = 0.43         # DTI*
    ltv_limit: float = 0.90         # LTV*
    rate_benchmark: float = 0.12    # ρ_max for TIC normalisation


def _check_qm_compliance(plan: LoanPlan) -> bool:
    """
    Qualified Mortgage regulatory rule engine.

    Checks: DTI ≤ 0.43, no balloon payment (term > 5yr),
    points-and-fees ≤ 3% (approximated as 0 for simplicity),
    no negative amortisation (rate > 0).
    Extend this function with full ATR/QM rules for the final paper.
    """
    dti = plan.total_monthly_debt / plan.gross_monthly_income
    is_qm = (
        dti <= 0.43
        and plan.term_months >= 60        # no balloon (min 5yr)
        and plan.annual_rate > 0.0        # no negative amortisation
        and plan.term_months <= 360       # max 30yr QM
    )
    return is_qm


def _check_non_qm_compliance(plan: LoanPlan) -> bool:
    """
    Simplified non-QM compliance check.
    Requires: rate > 0, term ≥ 12 months, LTV ≤ 0.97.
    """
    ltv = plan.loan_amount / plan.property_value
    return (
        plan.annual_rate > 0.0
        and plan.term_months >= 12
        and ltv <= 0.97
    )


def loan_rewards(
    plan: LoanPlan,
    cfg: LoanConstraints,
) -> RewardBundle:
    """
    Compute all K=3 hard, K=3 proximity, M=3 soft signals
    for a loan structuring plan.

    Equations H1–H3, S1–S3 in Section 4.1.3 of spec.
    """
    dti = plan.total_monthly_debt / plan.gross_monthly_income
    ltv = plan.loan_amount / plan.property_value

    # ── Hard constraints ──────────────────────────────────────────────────
    # H1: DTI ratio  (Eq. ls_h1)
    h1 = int(dti <= cfg.dti_limit)

    # H2: LTV ratio  (Eq. ls_h2)
    h2 = int(ltv <= cfg.ltv_limit)

    # H3: regulatory compliance  (Eq. ls_h3)
    if plan.regulatory_tier.upper() == "QM":
        h3 = int(_check_qm_compliance(plan))
    else:
        h3 = int(_check_non_qm_compliance(plan))

    hard = np.array([h1, h2, h3], dtype=float)

    # ── Proximity signals ─────────────────────────────────────────────────
    p1 = _clip01(1.0 - dti / cfg.dti_limit)
    p2 = _clip01(1.0 - ltv / cfg.ltv_limit)
    p3 = float(h3)   # regulatory: no meaningful gradient proxy

    prox = np.array([p1, p2, p3], dtype=float)

    # ── Soft preferences ──────────────────────────────────────────────────
    # S1: total interest cost  (Eq. ls_s1)
    tic_plan = total_interest_cost(plan.loan_amount, plan.annual_rate,
                                   plan.term_months)
    tic_max  = total_interest_cost(plan.loan_amount, cfg.rate_benchmark,
                                   plan.term_months)
    s1 = _clip01(1.0 - tic_plan / tic_max) if tic_max > 0 else 0.0

    # S2: payment flexibility  (Eq. ls_s2)
    monthly_payment = amortised_monthly_payment(
        plan.loan_amount, plan.annual_rate, plan.term_months
    )
    pti = monthly_payment / plan.gross_monthly_income
    s2 = _clip01(1.0 - 2.0 * pti)

    # S3: prepayment optionality  (Eq. ls_s3)
    p_flag = 0.0 if plan.prepayment_penalty else 1.0
    s3 = _clip01(p_flag * (1.0 - plan.lock_in_years / 10.0))

    soft = np.array([s1, s2, s3], dtype=float)

    return RewardBundle(hard=hard, prox=prox, soft=soft, task="loan")


# ──────────────────────────────────────────────────────────────────────────────
#  Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

TASK_REWARD_FN = {
    "portfolio":  portfolio_rewards,
    "retirement": retirement_rewards,
    "loan":       loan_rewards,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test  (run: python rewards.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    np.random.seed(0)

    # ── Portfolio ──────────────────────────────────────────────────────────
    N, T = 4, 756  # 3 years daily
    plan_p = PortfolioPlan(
        weights             = np.array([0.40, 0.30, 0.20, 0.10]),
        sector_labels       = ["tech", "finance", "energy", "utilities"],
        expected_returns    = np.array([0.12, 0.09, 0.07, 0.04]),
        cov_matrix          = np.diag([0.04, 0.03, 0.02, 0.01]),
        esg_ratings         = np.array([72., 85., 60., 45.]),
        benchmark_weights   = np.array([0.25, 0.25, 0.25, 0.25]),
        return_series       = np.random.normal(0.0004, 0.01, (T, N)),
    )
    cfg_p = PortfolioConstraints(banned_sectors={"tobacco", "weapons"})
    rb_p = portfolio_rewards(plan_p, cfg_p)
    print("Portfolio:")
    print(f"  hard={rb_p.hard}  prox={rb_p.prox.round(3)}  "
          f"soft={rb_p.soft.round(3)}")
    print(f"  all_hard_pass={rb_p.all_hard_pass()}")

    # ── Retirement ─────────────────────────────────────────────────────────
    plan_r = RetirementPlan(
        initial_balance     = 1_000_000,
        annual_withdrawal   = 50_000,
        withdrawal_schedule = np.full(30, 50_000.0),
        weights             = np.array([0.60, 0.40]),
        expected_returns    = np.array([0.08, 0.04]),
        cov_matrix          = np.array([[0.04, 0.005], [0.005, 0.001]]),
        current_age         = 65,
        target_age          = 95,
    )
    cfg_r = RetirementConstraints(mc_survival_threshold=0.90,
                                  income_floor_monthly=3000.0)
    rb_r = retirement_rewards(plan_r, cfg_r)
    print("\nRetirement:")
    print(f"  hard={rb_r.hard}  prox={rb_r.prox.round(3)}  "
          f"soft={rb_r.soft.round(3)}")
    print(f"  all_hard_pass={rb_r.all_hard_pass()}")

    # ── Loan ───────────────────────────────────────────────────────────────
    plan_l = LoanPlan(
        loan_amount          = 320_000,
        property_value       = 400_000,
        annual_rate          = 0.065,
        term_months          = 360,
        gross_monthly_income = 8_000,
        total_monthly_debt   = 2_200,
        prepayment_penalty   = False,
        lock_in_years        = 2.0,
        regulatory_tier      = "QM",
    )
    cfg_l = LoanConstraints(dti_limit=0.43, ltv_limit=0.90)
    rb_l = loan_rewards(plan_l, cfg_l)
    print("\nLoan:")
    print(f"  hard={rb_l.hard}  prox={rb_l.prox.round(3)}  "
          f"soft={rb_l.soft.round(3)}")
    print(f"  all_hard_pass={rb_l.all_hard_pass()}")

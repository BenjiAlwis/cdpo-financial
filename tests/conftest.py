"""
tests/conftest.py
-----------------
Shared fixtures for the finplanenv test suite.
All fixtures use fixed seeds for reproducibility.
"""
import numpy as np
import pytest

from finplanenv.rewards import (
    PortfolioPlan, PortfolioConstraints,
    RetirementPlan, RetirementConstraints,
    LoanPlan, LoanConstraints,
)
from finplanenv.parser import (
    PortfolioScenario, RetirementScenario, LoanScenario,
)
from finplanenv.cdpo import CDPOConfig, BatchRewards
from finplanenv.dataset import DatasetConfig


# ── Scenario fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def portfolio_scenario():
    np.random.seed(0)
    N, T = 4, 756
    return PortfolioScenario(
        n_assets          = N,
        expected_returns  = np.array([0.12, 0.09, 0.07, 0.04]),
        cov_matrix        = np.diag([0.04, 0.03, 0.02, 0.01]),
        esg_ratings       = np.array([72., 85., 60., 45.]),
        benchmark_weights = np.array([0.25, 0.25, 0.25, 0.25]),
        return_series     = np.random.normal(0.0004, 0.01, (T, N)),
        banned_sectors    = {"tobacco", "weapons"},
    )

@pytest.fixture
def portfolio_constraints():
    return PortfolioConstraints(
        weight_sum_tol      = 0.01,
        max_drawdown_limit  = -0.10,
        banned_sectors      = {"tobacco", "weapons"},
        hhi_limit           = 0.25,
        risk_free_rate      = 0.04,
    )

@pytest.fixture
def retirement_scenario():
    return RetirementScenario(
        initial_balance      = 1_000_000,
        current_age          = 65,
        target_age           = 95,
        n_assets             = 2,
        expected_returns     = np.array([0.08, 0.04]),
        cov_matrix           = np.array([[0.04, 0.005], [0.005, 0.001]]),
        income_floor_monthly = 3000.0,
    )

@pytest.fixture
def retirement_constraints():
    return RetirementConstraints(
        mc_survival_threshold = 0.90,
        income_floor_monthly  = 3000.0,
        n_mc                  = 200,   # small for fast tests
        mc_seed               = 42,
    )

@pytest.fixture
def loan_scenario():
    return LoanScenario(
        property_value       = 400_000,
        gross_monthly_income = 8_000,
    )

@pytest.fixture
def loan_constraints():
    return LoanConstraints(
        dti_limit      = 0.43,
        ltv_limit      = 0.90,
        rate_benchmark = 0.12,
    )


# ── LLM output fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def valid_portfolio_output():
    return """
<plan>
<reasoning>Balanced allocation across four sectors.</reasoning>
<financial_plan>
{
    "task": "portfolio_allocation",
    "assets": [
        {"ticker": "AAPL", "sector": "technology",  "weight": 0.35},
        {"ticker": "JPM",  "sector": "financials",  "weight": 0.25},
        {"ticker": "XOM",  "sector": "energy",      "weight": 0.20},
        {"ticker": "NEE",  "sector": "utilities",   "weight": 0.20}
    ],
    "rationale": "Diversified across sectors."
}
</financial_plan>
</plan>
"""

@pytest.fixture
def valid_retirement_output():
    return """
<plan>
<reasoning>60/40 allocation for 30-year horizon.</reasoning>
<financial_plan>
{
    "task": "retirement_planning",
    "annual_withdrawal": 50000,
    "withdrawal_type": "fixed",
    "asset_allocation": [
        {"asset_class": "equities", "weight": 0.60},
        {"asset_class": "bonds",    "weight": 0.40}
    ],
    "rationale": "Conservative 60/40."
}
</financial_plan>
</plan>
"""

@pytest.fixture
def valid_loan_output():
    return """
<plan>
<reasoning>30-year fixed at 6.5% within QM guidelines.</reasoning>
<financial_plan>
{
    "task": "loan_structuring",
    "loan_amount": 320000,
    "annual_rate": 0.065,
    "term_months": 360,
    "total_monthly_debt_payments": 2200,
    "prepayment_penalty": false,
    "lock_in_years": 2,
    "regulatory_tier": "QM",
    "rationale": "Standard 30-year fixed."
}
</financial_plan>
</plan>
"""


# ── CDPO fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def cdpo_config():
    return CDPOConfig(G=8)

@pytest.fixture
def batch_rewards_mixed():
    """Realistic batch: early training, mixed compliance."""
    np.random.seed(42)
    B, G, K, M = 4, 8, 4, 3
    return BatchRewards(
        r_hard = np.random.binomial(1, 0.4, (B, G, K)).astype(float),
        r_prox = np.random.uniform(0.2, 0.9, (B, G, K)),
        r_soft = np.random.uniform(0.3, 0.8, (B, G, M)),
    )

@pytest.fixture
def batch_rewards_all_pass():
    """Late training: all hard constraints satisfied."""
    np.random.seed(1)
    B, G, K, M = 4, 8, 4, 3
    return BatchRewards(
        r_hard = np.ones((B, G, K)),
        r_prox = np.ones((B, G, K)),
        r_soft = np.random.uniform(0.6, 1.0, (B, G, M)),
    )

@pytest.fixture
def batch_rewards_all_fail():
    """Early training: all hard constraints violated."""
    np.random.seed(2)
    B, G, K, M = 4, 8, 4, 3
    return BatchRewards(
        r_hard = np.zeros((B, G, K)),
        r_prox = np.random.uniform(0.0, 0.4, (B, G, K)),
        r_soft = np.random.uniform(0.1, 0.5, (B, G, M)),
    )


# ── Dataset fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def dataset_config():
    return DatasetConfig(
        n_instances_per_task = 30,
        pilot_size           = 15,
        seed                 = 42,
    )

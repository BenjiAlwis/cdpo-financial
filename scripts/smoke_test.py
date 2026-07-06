#!/usr/bin/env python3
"""
scripts/smoke_test.py
=====================
Quick end-to-end smoke test — run this first after installation to confirm
that rewards, parser, CDPO advantage computation, and dataset generation
all work correctly together.

Usage:
    python scripts/smoke_test.py

Expected output: a summary table showing pass/fail for each component.
Takes ~10 seconds.
"""
import sys
import traceback
import numpy as np

PASS = "  ✓"
FAIL = "  ✗"
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
    except Exception as e:
        results.append((FAIL, name))
        results.append(("   ", f"    ERROR: {e}"))
        traceback.print_exc()


# ── 1. Rewards ────────────────────────────────────────────────────────────────
def test_rewards():
    from finplanenv import (
        PortfolioPlan, PortfolioConstraints, portfolio_rewards,
        RetirementPlan, RetirementConstraints, retirement_rewards,
        LoanPlan, LoanConstraints, loan_rewards,
    )
    np.random.seed(0)
    N, T = 4, 756

    # Portfolio
    plan_p = PortfolioPlan(
        weights           = np.array([0.25, 0.25, 0.25, 0.25]),
        sector_labels     = ["technology", "financials", "energy", "utilities"],
        expected_returns  = np.array([0.12, 0.09, 0.07, 0.04]),
        cov_matrix        = np.diag([0.04, 0.03, 0.02, 0.01]),
        esg_ratings       = np.array([72., 85., 60., 45.]),
        benchmark_weights = np.array([0.25, 0.25, 0.25, 0.25]),
        return_series     = np.random.normal(0.0004, 0.01, (T, N)),
    )
    rb_p = portfolio_rewards(plan_p, PortfolioConstraints())
    assert rb_p.hard.shape == (4,)
    assert rb_p.soft.shape == (3,)
    assert np.all(rb_p.soft >= 0) and np.all(rb_p.soft <= 1)

    # Retirement
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
    rb_r = retirement_rewards(plan_r, RetirementConstraints(n_mc=200))
    assert rb_r.hard.shape == (2,)

    # Loan
    plan_l = LoanPlan(
        loan_amount=320_000, property_value=400_000, annual_rate=0.065,
        term_months=360, gross_monthly_income=8_000, total_monthly_debt=2_200,
        prepayment_penalty=False, lock_in_years=2.0, regulatory_tier="QM",
    )
    rb_l = loan_rewards(plan_l, LoanConstraints())
    assert rb_l.all_hard_pass()

check("Reward functions (all 3 tasks)", test_rewards)


# ── 2. Parser ─────────────────────────────────────────────────────────────────
def test_parser():
    from finplanenv import (
        compute_rewards_from_output, ParserError,
        PortfolioScenario, RetirementScenario, LoanScenario,
        PortfolioConstraints, RetirementConstraints, LoanConstraints,
    )
    np.random.seed(1)

    # Valid portfolio
    scenario_p = PortfolioScenario(
        n_assets=4,
        expected_returns=np.array([0.12, 0.09, 0.07, 0.04]),
        cov_matrix=np.diag([0.04, 0.03, 0.02, 0.01]),
        esg_ratings=np.array([72., 85., 60., 45.]),
        benchmark_weights=np.array([0.25, 0.25, 0.25, 0.25]),
        return_series=np.random.normal(0.0004, 0.01, (756, 4)),
        banned_sectors=set(),
    )
    output = """<plan><financial_plan>
{"task":"portfolio_allocation","assets":[
{"ticker":"A","sector":"technology","weight":0.25},
{"ticker":"B","sector":"financials","weight":0.25},
{"ticker":"C","sector":"energy","weight":0.25},
{"ticker":"D","sector":"utilities","weight":0.25}
],"rationale":"equal weight"}</financial_plan></plan>"""
    rb = compute_rewards_from_output(output, "portfolio", scenario_p, PortfolioConstraints())
    assert not (np.all(rb.hard == 0) and np.all(rb.soft == 0)), \
        "Valid output should not produce all-zero rewards"

    # Parse failure → zero rewards
    rb_bad = compute_rewards_from_output("no tags here", "portfolio", scenario_p, PortfolioConstraints())
    assert np.all(rb_bad.hard == 0) and np.all(rb_bad.soft == 0)

check("Parser (valid output + parse failure)", test_parser)


# ── 3. CDPO advantage computation ────────────────────────────────────────────
def test_cdpo():
    from finplanenv import CDPOConfig, CDPOAdvantage, BatchRewards, grpo_advantages, gdpo_advantages
    np.random.seed(42)
    B, G, K, M = 4, 8, 4, 3
    config = CDPOConfig(G=G)
    batch = BatchRewards(
        r_hard = np.random.binomial(1, 0.4, (B, G, K)).astype(float),
        r_prox = np.random.uniform(0.2, 0.9, (B, G, K)),
        r_soft = np.random.uniform(0.3, 0.8, (B, G, M)),
    )

    cdpo = CDPOAdvantage(config, K=K, M=M)
    A_hat, metrics = cdpo.compute(batch, step=0)

    assert A_hat.shape == (B, G)
    assert abs(A_hat.mean()) < 0.1,  f"Mean should be ~0, got {A_hat.mean():.4f}"
    assert abs(A_hat.std() - 1.0) < 0.2, f"Std should be ~1, got {A_hat.std():.4f}"
    assert "ccr" in metrics
    assert 0.0 <= metrics["ccr"] <= 1.0

    # Baselines work too
    A_grpo = grpo_advantages(batch.r_hard, batch.r_soft)
    A_gdpo = gdpo_advantages(batch.r_hard, batch.r_soft)
    assert A_grpo.shape == A_gdpo.shape == A_hat.shape

check("CDPO advantage computation + baselines", test_cdpo)


# ── 4. Correlation tracker ────────────────────────────────────────────────────
def test_correlation():
    from finplanenv import CorrelationTracker
    ct = CorrelationTracker(K=3)
    # Simulate perfectly correlated constraints 0 and 1
    r = np.zeros((100, 3))
    r[:, 0] = np.random.binomial(1, 0.5, 100)
    r[:, 1] = r[:, 0]   # identical
    r[:, 2] = np.random.binomial(1, 0.5, 100)   # independent
    for _ in range(10):
        ct.update(r)
    pairs = ct.correlated_pairs(threshold=0.7)
    assert (0, 1) in pairs, "Should detect correlation between constraints 0 and 1"
    assert (0, 2) not in pairs, "Should NOT flag independent constraints"

check("Correlation tracker", test_correlation)


# ── 5. Dataset generation ─────────────────────────────────────────────────────
def test_dataset():
    from finplanenv import DatasetConfig, ScenarioSampler, DatasetGenerator
    config  = DatasetConfig(n_instances_per_task=30, pilot_size=15, seed=42)
    sampler = ScenarioSampler(config)

    # Check all three tasks
    for task in ["portfolio", "retirement", "loan"]:
        scenario, constraints, meta = sampler.sample(task, 0)
        assert "difficulty" in meta
        assert meta["difficulty"] in {"easy", "medium", "hard"}

    # Tier distribution
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for idx in range(30):
        _, _, meta = sampler.sample("portfolio", idx)
        counts[meta["difficulty"]] += 1
    assert counts["easy"]   == 9
    assert counts["medium"] == 12
    assert counts["hard"]   == 9

    # Calibration report
    gen = DatasetGenerator(config)
    ccrs = np.concatenate([
        np.random.uniform(0.6, 1.0, 9),
        np.random.uniform(0.2, 0.6, 12),
        np.random.uniform(0.0, 0.2, 9),
    ]).tolist()
    report = gen.calibration_report("portfolio", ccrs)
    assert report["status"] in {"OK", "NEEDS_CALIBRATION"}

check("Dataset generation (sampling + calibration)", test_dataset)


# ── 6. Generation prompts ─────────────────────────────────────────────────────
def test_prompts():
    from finplanenv import DatasetConfig, ScenarioSampler, GENERATION_PROMPTS
    config  = DatasetConfig(n_instances_per_task=30, seed=42)
    sampler = ScenarioSampler(config)
    for task in ["portfolio", "retirement", "loan"]:
        _, _, meta = sampler.sample(task, 5)
        prompt_fn = GENERATION_PROMPTS[task]
        prompt = prompt_fn(meta, n_profiles=3)
        assert len(prompt) > 200, f"Prompt too short for task {task}"
        assert "JSON" in prompt

check("Generation prompts (all 3 tasks)", test_prompts)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  finplanenv smoke test results")
print("=" * 55)
for status, name in results:
    print(f"{status}  {name}")
print("=" * 55)

n_fail = sum(1 for s, _ in results if s == FAIL)
n_pass = sum(1 for s, _ in results if s == PASS)
print(f"  {n_pass} passed  |  {n_fail} failed")
print()

if n_fail > 0:
    sys.exit(1)

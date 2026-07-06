"""
tests/test_all.py
-----------------
Full test suite for finplanenv.
Run with:  pytest tests/ -v
"""
import numpy as np
# import pytest  # use standard unittest instead

from finplanenv.rewards import (
    RewardBundle, compute_max_drawdown, compute_hhi,
    amortised_monthly_payment, total_interest_cost,
    portfolio_rewards, retirement_rewards, loan_rewards,
    PortfolioPlan, PortfolioConstraints,
    RetirementPlan, RetirementConstraints,
    LoanPlan, LoanConstraints,
)
from finplanenv.parser import (
    ParserError, ParseWarning,
    extract_json_block, compute_rewards_from_output,
    parse_portfolio, parse_retirement, parse_loan,
)
from finplanenv.cdpo import (
    CDPOConfig, CDPOAdvantage, BatchRewards,
    CorrelationTracker, grpo_advantages, gdpo_advantages,
)
from finplanenv.dataset import (
    DatasetConfig, ScenarioSampler, DatasetGenerator,
    PORTFOLIO_THRESHOLDS, RETIREMENT_THRESHOLDS, LOAN_THRESHOLDS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  REWARDS
# ══════════════════════════════════════════════════════════════════════════════

class TestRewardUtils:

    def test_hhi_equal_weights(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(compute_hhi(w) - 0.25) < 1e-6

    def test_hhi_concentrated(self):
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(compute_hhi(w) - 1.0) < 1e-6

    def test_max_drawdown_negative(self):
        np.random.seed(0)
        w = np.array([0.4, 0.3, 0.2, 0.1])
        ret = np.random.normal(0.0004, 0.01, (756, 4))
        mdd = compute_max_drawdown(w, ret)
        assert mdd < 0, "Max drawdown must be negative"
        assert mdd > -1.0, "Max drawdown must be > -100%"

    def test_amortisation_zero_rate(self):
        mp = amortised_monthly_payment(120_000, 0.0, 120)
        assert abs(mp - 1000.0) < 0.01

    def test_total_interest_positive(self):
        tic = total_interest_cost(300_000, 0.06, 360)
        assert tic > 0


class TestPortfolioRewards:

    def test_passing_plan(self, portfolio_scenario, portfolio_constraints):
        np.random.seed(0)
        plan = PortfolioPlan(
            weights           = np.array([0.25, 0.25, 0.25, 0.25]),
            sector_labels     = ["technology", "financials", "energy", "utilities"],
            expected_returns  = portfolio_scenario.expected_returns,
            cov_matrix        = portfolio_scenario.cov_matrix,
            esg_ratings       = portfolio_scenario.esg_ratings,
            benchmark_weights = portfolio_scenario.benchmark_weights,
            return_series     = portfolio_scenario.return_series,
        )
        rb = portfolio_rewards(plan, portfolio_constraints)
        assert rb.hard.shape == (4,)
        assert rb.prox.shape == (4,)
        assert rb.soft.shape == (3,)
        assert np.all(rb.soft >= 0) and np.all(rb.soft <= 1)
        assert np.all(rb.prox >= 0) and np.all(rb.prox <= 1)
        assert rb.hard[0] == 1   # weight-sum passes (equal weights sum to 1)
        assert rb.hard[2] == 1   # no banned sectors

    def test_banned_sector_fails(self, portfolio_scenario, portfolio_constraints):
        plan = PortfolioPlan(
            weights           = np.array([0.25, 0.25, 0.25, 0.25]),
            sector_labels     = ["tobacco", "financials", "energy", "utilities"],
            expected_returns  = portfolio_scenario.expected_returns,
            cov_matrix        = portfolio_scenario.cov_matrix,
            esg_ratings       = portfolio_scenario.esg_ratings,
            benchmark_weights = portfolio_scenario.benchmark_weights,
            return_series     = portfolio_scenario.return_series,
        )
        rb = portfolio_rewards(plan, portfolio_constraints)
        assert rb.hard[2] == 0   # banned sector → H3 fails

    def test_weight_sum_fails(self, portfolio_scenario, portfolio_constraints):
        plan = PortfolioPlan(
            weights           = np.array([0.30, 0.30, 0.30, 0.30]),  # sums to 1.2
            sector_labels     = ["technology", "financials", "energy", "utilities"],
            expected_returns  = portfolio_scenario.expected_returns,
            cov_matrix        = portfolio_scenario.cov_matrix,
            esg_ratings       = portfolio_scenario.esg_ratings,
            benchmark_weights = portfolio_scenario.benchmark_weights,
            return_series     = portfolio_scenario.return_series,
        )
        rb = portfolio_rewards(plan, portfolio_constraints)
        assert rb.hard[0] == 0   # H1 fails


class TestRetirementRewards:

    def test_bundle_shapes(self, retirement_scenario, retirement_constraints):
        plan = RetirementPlan(
            initial_balance     = 1_000_000,
            annual_withdrawal   = 50_000,
            withdrawal_schedule = np.full(30, 50_000.0),
            weights             = np.array([0.60, 0.40]),
            expected_returns    = retirement_scenario.expected_returns,
            cov_matrix          = retirement_scenario.cov_matrix,
            current_age         = 65,
            target_age          = 95,
        )
        rb = retirement_rewards(plan, retirement_constraints)
        assert rb.hard.shape == (2,)
        assert rb.soft.shape == (3,)
        assert np.all(rb.soft >= 0) and np.all(rb.soft <= 1)

    def test_income_floor_passes(self, retirement_scenario, retirement_constraints):
        plan = RetirementPlan(
            initial_balance     = 1_000_000,
            annual_withdrawal   = 60_000,   # 5000/mo > 3000 floor
            withdrawal_schedule = np.full(30, 60_000.0),
            weights             = np.array([0.60, 0.40]),
            expected_returns    = retirement_scenario.expected_returns,
            cov_matrix          = retirement_scenario.cov_matrix,
            current_age         = 65,
            target_age          = 95,
        )
        rb = retirement_rewards(plan, retirement_constraints)
        assert rb.hard[1] == 1   # H2: income floor

    def test_income_floor_fails(self, retirement_scenario, retirement_constraints):
        plan = RetirementPlan(
            initial_balance     = 1_000_000,
            annual_withdrawal   = 24_000,   # 2000/mo < 3000 floor
            withdrawal_schedule = np.full(30, 24_000.0),
            weights             = np.array([0.60, 0.40]),
            expected_returns    = retirement_scenario.expected_returns,
            cov_matrix          = retirement_scenario.cov_matrix,
            current_age         = 65,
            target_age          = 95,
        )
        rb = retirement_rewards(plan, retirement_constraints)
        assert rb.hard[1] == 0   # H2 fails


class TestLoanRewards:

    def test_passing_loan(self, loan_scenario, loan_constraints):
        plan = LoanPlan(
            loan_amount           = 320_000,
            property_value        = 400_000,
            annual_rate           = 0.065,
            term_months           = 360,
            gross_monthly_income  = 8_000,
            total_monthly_debt    = 2_200,
            prepayment_penalty    = False,
            lock_in_years         = 2.0,
            regulatory_tier       = "QM",
        )
        rb = loan_rewards(plan, loan_constraints)
        assert rb.hard.shape == (3,)
        assert rb.soft.shape == (3,)
        assert rb.all_hard_pass()

    def test_dti_fails(self, loan_scenario, loan_constraints):
        plan = LoanPlan(
            loan_amount           = 500_000,
            property_value        = 400_000,
            annual_rate           = 0.08,
            term_months           = 360,
            gross_monthly_income  = 8_000,
            total_monthly_debt    = 5_000,   # very high DTI
            prepayment_penalty    = False,
            lock_in_years         = 0.0,
            regulatory_tier       = "QM",
        )
        rb = loan_rewards(plan, loan_constraints)
        assert rb.hard[0] == 0   # H1 DTI fails

    def test_ltv_fails(self, loan_scenario, loan_constraints):
        plan = LoanPlan(
            loan_amount           = 390_000,  # LTV = 0.975 > 0.90
            property_value        = 400_000,
            annual_rate           = 0.065,
            term_months           = 360,
            gross_monthly_income  = 8_000,
            total_monthly_debt    = 2_000,
            prepayment_penalty    = False,
            lock_in_years         = 0.0,
            regulatory_tier       = "QM",
        )
        rb = loan_rewards(plan, loan_constraints)
        assert rb.hard[1] == 0   # H2 LTV fails


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractJsonBlock:

    def test_valid_extraction(self):
        output = "<plan><financial_plan>{\"key\": 1}</financial_plan></plan>"
        d = extract_json_block(output)
        assert d == {"key": 1}

    def test_missing_tag_raises(self):
        # "No tags here." has no JSON object at all -> still a parse failure.
        try:
            extract_json_block("No tags here.")
            assert False, "Should have raised ParserError"
        except ParserError as e:
            assert "No parseable" in str(e) or "No <financial_plan>" in str(e)

    def test_invalid_json_raises(self):
        # "{bad json}" is not valid JSON and not repairable -> still fails, even
        # though the hardened parser now brace-matches and attempts recovery.
        try:
            extract_json_block("<financial_plan>{bad json}</financial_plan>")
            assert False, "Should have raised ParserError"
        except ParserError:
            pass

    def test_multiline_json(self):
        output = "<financial_plan>\n{\n  \"a\": 1,\n  \"b\": 2\n}\n</financial_plan>"
        d = extract_json_block(output)
        assert d["a"] == 1 and d["b"] == 2


class TestParserIntegration:

    def test_valid_portfolio(self, valid_portfolio_output,
                              portfolio_scenario, portfolio_constraints):
        rb = compute_rewards_from_output(
            valid_portfolio_output, "portfolio",
            portfolio_scenario, portfolio_constraints,
        )
        assert rb.hard.shape == (4,)
        assert rb.soft.shape == (3,)
        assert not (np.all(rb.hard == 0) and np.all(rb.soft == 0)), \
            "Valid output should not produce all-zero rewards"

    def test_valid_retirement(self, valid_retirement_output,
                               retirement_scenario, retirement_constraints):
        rb = compute_rewards_from_output(
            valid_retirement_output, "retirement",
            retirement_scenario, retirement_constraints,
        )
        assert rb.hard.shape == (2,)
        assert rb.hard[1] == 1   # income floor: 50000/12 > 3000

    def test_valid_loan(self, valid_loan_output,
                         loan_scenario, loan_constraints):
        rb = compute_rewards_from_output(
            valid_loan_output, "loan",
            loan_scenario, loan_constraints,
        )
        assert rb.all_hard_pass()

    def test_no_tag_zeroes_rewards(self, portfolio_scenario, portfolio_constraints):
        rb = compute_rewards_from_output(
            "I recommend a diversified portfolio.",
            "portfolio", portfolio_scenario, portfolio_constraints,
        )
        assert np.all(rb.hard == 0)
        assert np.all(rb.soft == 0)

    def test_rate_as_percentage_zeroes_rewards(self,
                                                loan_scenario, loan_constraints):
        bad_output = """
<plan><financial_plan>
{"task":"loan_structuring","loan_amount":320000,"annual_rate":6.5,
"term_months":360,"total_monthly_debt_payments":2200,
"prepayment_penalty":false,"lock_in_years":2,"regulatory_tier":"QM"}
</financial_plan></plan>"""
        rb = compute_rewards_from_output(
            bad_output, "loan", loan_scenario, loan_constraints,
        )
        assert np.all(rb.hard == 0), "Rate=6.5 should fail validation"

    def test_weight_sum_warning_passes_parser(self, portfolio_scenario,
                                               portfolio_constraints):
        output = """<plan><financial_plan>{"task":"portfolio_allocation","assets":[
{"ticker":"A","sector":"technology","weight":0.30},{"ticker":"B","sector":"financials","weight":0.25},
{"ticker":"C","sector":"energy","weight":0.22},{"ticker":"D","sector":"utilities","weight":0.20}
],"rationale":"test"}</financial_plan></plan>"""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rb = compute_rewards_from_output(output, "portfolio", portfolio_scenario, portfolio_constraints)
        assert rb.hard[0] == 0   # H1 weight-sum fires


# ══════════════════════════════════════════════════════════════════════════════
#  CDPO ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchRewards:

    def test_shape_mismatch_raises(self):
        try:
            BatchRewards(r_hard=np.zeros((4,8,4)), r_prox=np.zeros((4,8,3)), r_soft=np.zeros((4,8,3)))
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass


class TestCorrelationTracker:

    def test_identity_init(self):
        ct = CorrelationTracker(K=4)
        corr = ct.correlation_matrix
        assert corr.shape == (4, 4)
        assert np.allclose(corr, np.eye(4), atol=0.01)

    def test_update_changes_matrix(self):
        ct = CorrelationTracker(K=2)
        # Perfectly correlated: r1 always equals r2
        r = np.column_stack([np.arange(100), np.arange(100)]).astype(float)
        ct.update(r)
        corr = ct.correlation_matrix
        assert corr[0, 1] > 0.9, "Should detect high correlation"

    def test_correlated_pairs(self):
        ct = CorrelationTracker(K=2)
        r = np.column_stack([np.arange(100), np.arange(100)]).astype(float)
        for _ in range(20):
            ct.update(r)
        pairs = ct.correlated_pairs(threshold=0.7)
        assert (0, 1) in pairs


class TestCDPOAdvantage:

    def test_output_shape(self, cdpo_config, batch_rewards_mixed):
        cdpo = CDPOAdvantage(cdpo_config, K=4, M=3)
        A_hat, metrics = cdpo.compute(batch_rewards_mixed, step=0)
        assert A_hat.shape == (4, 8)

    def test_batch_normalisation(self, cdpo_config, batch_rewards_mixed):
        """Batch-normalised advantages must have mean≈0 and std≈1."""
        cdpo = CDPOAdvantage(cdpo_config, K=4, M=3)
        A_hat, _ = cdpo.compute(batch_rewards_mixed, step=0)
        assert abs(A_hat.mean()) < 0.1
        assert abs(A_hat.std() - 1.0) < 0.2

    def test_all_pass_uses_alpha_min(self, cdpo_config, batch_rewards_all_pass):
        """When all rollouts comply, α_i should be α_min for every group."""
        cdpo = CDPOAdvantage(cdpo_config, K=4, M=3)
        # Prime CCR history with high values
        for _ in range(50):
            cdpo._ccr_history.append(1.0)
        A_hat, metrics = cdpo.compute(batch_rewards_all_pass, step=100)
        assert abs(metrics["lambda_prox"]) < 0.01   # proximity annealed to 0

    def test_all_fail_soft_channel_zero(self, cdpo_config, batch_rewards_all_fail):
        """When no rollouts comply, soft channel must be zeroed (G_min not met)."""
        cdpo = CDPOAdvantage(cdpo_config, K=4, M=3)
        # Manually run soft channel and check it zeros out
        A_soft_m = cdpo._soft_channel(batch_rewards_all_fail)
        assert np.all(A_soft_m == 0), "Soft channel should be zero when CCR=0"

    def test_metrics_keys_present(self, cdpo_config, batch_rewards_mixed):
        cdpo = CDPOAdvantage(cdpo_config, K=4, M=3)
        _, metrics = cdpo.compute(batch_rewards_mixed, step=0)
        required_keys = [
            "ccr", "mean_soft_score", "combined_quality",
            "n_distinct_advantages", "lambda_prox", "n_correlated_pairs",
        ]
        for k in required_keys:
            assert k in metrics, f"Missing metric: {k}"

    def test_ccr_metric_range(self, cdpo_config, batch_rewards_mixed):
        cdpo = CDPOAdvantage(cdpo_config, K=4, M=3)
        _, metrics = cdpo.compute(batch_rewards_mixed, step=0)
        assert 0.0 <= metrics["ccr"] <= 1.0


class TestBaselines:

    def test_grpo_shape(self):
        np.random.seed(0)
        r_hard = np.random.binomial(1, 0.5, (4, 8, 4)).astype(float)
        r_soft = np.random.uniform(0, 1, (4, 8, 3))
        A = grpo_advantages(r_hard, r_soft)
        assert A.shape == (4, 8)

    def test_gdpo_shape(self):
        np.random.seed(0)
        r_hard = np.random.binomial(1, 0.5, (4, 8, 4)).astype(float)
        r_soft = np.random.uniform(0, 1, (4, 8, 3))
        A = gdpo_advantages(r_hard, r_soft)
        assert A.shape == (4, 8)

    def test_grpo_within_group_normalised(self):
        """Each group in GRPO should have mean≈0 std≈1."""
        np.random.seed(0)
        r_hard = np.random.binomial(1, 0.5, (4, 8, 4)).astype(float)
        r_soft = np.random.uniform(0, 1, (4, 8, 3))
        A = grpo_advantages(r_hard, r_soft)
        for i in range(4):
            assert abs(A[i].mean()) < 0.1
            assert abs(A[i].std() - 1.0) < 0.2


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioSampler:

    def test_tier_distribution(self, dataset_config):
        """30/40/30 tier distribution must be respected."""
        sampler = ScenarioSampler(dataset_config)
        n = dataset_config.n_instances_per_task
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for idx in range(n):
            _, _, meta = sampler.sample("portfolio", idx)
            counts[meta["difficulty"]] += 1
        assert abs(counts["easy"]   / n - 0.30) < 0.05
        assert abs(counts["medium"] / n - 0.40) < 0.05
        assert abs(counts["hard"]   / n - 0.30) < 0.05

    def test_deterministic(self, dataset_config):
        """Same index must produce identical scenario."""
        sampler = ScenarioSampler(dataset_config)
        _, c1, m1 = sampler.sample("portfolio", 5)
        _, c2, m2 = sampler.sample("portfolio", 5)
        assert c1.max_drawdown_limit == c2.max_drawdown_limit
        assert c1.hhi_limit == c2.hhi_limit
        assert m1["difficulty"] == m2["difficulty"]

    def test_threshold_in_tier_range(self, dataset_config):
        """Constraint thresholds must lie within the tier's configured range."""
        sampler = ScenarioSampler(dataset_config)
        for idx in range(dataset_config.n_instances_per_task):
            _, constraints, meta = sampler.sample("portfolio", idx)
            tier = meta["difficulty"]
            lo, hi = PORTFOLIO_THRESHOLDS[tier]["max_drawdown_limit"]
            assert lo <= constraints.max_drawdown_limit <= hi, \
                f"max_drawdown_limit={constraints.max_drawdown_limit} outside [{lo},{hi}]"

    def test_all_tasks_sample(self, dataset_config):
        sampler = ScenarioSampler(dataset_config)
        for task in ["portfolio", "retirement", "loan"]:
            scenario, constraints, meta = sampler.sample(task, 0)
            assert "instance_id" in meta
            assert "difficulty" in meta
            assert "narrative_hints" in meta

    def test_return_series_shape(self, dataset_config):
        sampler = ScenarioSampler(dataset_config)
        from finplanenv.parser import PortfolioScenario
        scenario, _, _ = sampler.sample("portfolio", 0)
        assert isinstance(scenario, PortfolioScenario)
        assert scenario.return_series.shape == (756, 4)


class TestDatasetGenerator:

    def test_generate_instances(self, dataset_config):
        gen = DatasetGenerator(dataset_config)
        instances = gen.generate_instances("portfolio", 0, 10)
        assert len(instances) == 10
        assert all("instance_id" in d for d in instances)

    def test_save_load_roundtrip(self, dataset_config, tmp_path):
        gen = DatasetGenerator(dataset_config)
        instances = gen.generate_instances("loan", 0, 5)
        path = tmp_path / "test.jsonl"
        gen.save_metadata(instances, path)
        loaded = gen.load_metadata(path)
        assert len(loaded) == 5
        assert loaded[0]["instance_id"] == instances[0]["instance_id"]

    def test_calibration_report_ok(self, dataset_config):
        gen = DatasetGenerator(dataset_config)
        np.random.seed(0)
        # Simulate well-calibrated CCR values
        ccrs = np.concatenate([
            np.random.uniform(0.6, 1.0, 9),
            np.random.uniform(0.2, 0.6, 12),
            np.random.uniform(0.0, 0.2, 9),
        ]).tolist()
        report = gen.calibration_report("portfolio", ccrs)
        assert "status" in report
        assert "tier_fractions" in report
        assert "recommendations" in report

    def test_calibration_report_flags_imbalance(self, dataset_config):
        gen = DatasetGenerator(dataset_config)
        # All easy instances
        ccrs = np.random.uniform(0.7, 1.0, 30).tolist()
        report = gen.calibration_report("portfolio", ccrs)
        assert report["status"] == "NEEDS_CALIBRATION"
        assert len(report["recommendations"]) > 0

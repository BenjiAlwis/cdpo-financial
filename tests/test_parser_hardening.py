"""
tests/test_parser_hardening.py
==============================
Day-4 parser-hardening tests.

Two things must BOTH hold after hardening:
  (A) Real-world-messy-but-VALID output now parses (fences, trailing prose,
      trailing commas, single quotes, unterminated tags).
  (B) Genuinely unparseable / incomplete output STILL fails — the hardening must
      not fabricate fields or rescue plans that don't exist, because a parse
      failure is deliberately treated as a hard-constraint violation (zero
      rewards) and that signal must stay honest.

Run:  python -m pytest tests/test_parser_hardening.py -v
"""

import numpy as np
import pytest

from finplanenv.parser import (
    extract_json_block, ParserError,
    parse_portfolio, PortfolioScenario,
    compute_rewards_from_output,
)
from finplanenv.rewards import PortfolioConstraints


# ---------------------------------------------------------------------------
# extract_json_block: recovery cases (B) and failure cases
# ---------------------------------------------------------------------------
VALID_INNER = '{"task": "portfolio_allocation", "assets": []}'


def test_strict_still_works():
    out = f"<financial_plan>{VALID_INNER}</financial_plan>"
    assert extract_json_block(out)["task"] == "portfolio_allocation"


def test_fenced_json_inside_tags():
    out = f"<financial_plan>\n```json\n{VALID_INNER}\n```\n</financial_plan>"
    assert extract_json_block(out)["task"] == "portfolio_allocation"


def test_trailing_comma_repaired():
    inner = '{"task": "portfolio_allocation", "assets": [],}'
    out = f"<financial_plan>{inner}</financial_plan>"
    assert extract_json_block(out)["assets"] == []


def test_single_quoted_json_repaired():
    inner = "{'task': 'portfolio_allocation', 'assets': []}"
    out = f"<financial_plan>{inner}</financial_plan>"
    assert extract_json_block(out)["task"] == "portfolio_allocation"


def test_unterminated_tag_recovered():
    # model ran out of tokens before closing the tag
    out = f"Here is my plan:\n<financial_plan>\n{VALID_INNER}\n"
    assert extract_json_block(out)["task"] == "portfolio_allocation"


def test_bare_json_no_tags():
    out = f"Sure! Here's the plan:\n{VALID_INNER}\nLet me know if you need changes."
    assert extract_json_block(out)["task"] == "portfolio_allocation"


def test_prose_before_and_after_fenced():
    out = (
        "I recommend the following allocation.\n"
        f"```json\n{VALID_INNER}\n```\n"
        "This balances growth and stability."
    )
    assert extract_json_block(out)["task"] == "portfolio_allocation"


# ---- failure cases: hardening must NOT rescue these -----------------------
def test_no_json_at_all_fails():
    with pytest.raises(ParserError):
        extract_json_block("I recommend investing in diversified assets.")


def test_empty_string_fails():
    with pytest.raises(ParserError):
        extract_json_block("")


def test_non_dict_json_fails():
    # a bare list is not a plan object
    with pytest.raises(ParserError):
        extract_json_block("<financial_plan>[1, 2, 3]</financial_plan>")


def test_garbage_braces_fail():
    with pytest.raises(ParserError):
        extract_json_block("the set {a, b, c} is unordered")


# ---------------------------------------------------------------------------
# End-to-end: incomplete plan still yields ZERO rewards (parse-fail policy)
# ---------------------------------------------------------------------------
def _portfolio_scenario():
    np.random.seed(0)
    N, T = 4, 64
    return PortfolioScenario(
        n_assets=4,
        expected_returns=np.array([0.12, 0.09, 0.07, 0.04]),
        cov_matrix=np.diag([0.04, 0.03, 0.02, 0.01]),
        esg_ratings=np.array([72., 85., 60., 45.]),
        benchmark_weights=np.array([0.25, 0.25, 0.25, 0.25]),
        return_series=np.random.normal(0.0004, 0.01, (T, N)),
        banned_sectors={"tobacco"},
    )


def test_recovered_valid_plan_scores_nonzero():
    """A fenced, trailing-prose plan with the right schema should parse AND
    produce a real (non-degenerate) reward bundle — proving recovery reaches
    the reward path, not just extract_json_block."""
    inner = (
        '{"task": "portfolio_allocation", "assets": ['
        '{"ticker": "A", "sector": "technology", "weight": 0.25},'
        '{"ticker": "B", "sector": "financials", "weight": 0.25},'
        '{"ticker": "C", "sector": "energy", "weight": 0.25},'
        '{"ticker": "D", "sector": "utilities", "weight": 0.25}]}'
    )
    out = f"Here's my plan:\n```json\n{inner}\n```\nHope this helps!"
    rb = compute_rewards_from_output(out, "portfolio", _portfolio_scenario(),
                                     PortfolioConstraints())
    # weights sum to 1 and no banned sector -> at least the weight-sum hard
    # constraint should pass (hard is not all-zero)
    assert rb.hard.sum() > 0, "recovered valid plan should not be all-fail"


def test_incomplete_plan_zeros_rewards():
    """An object that parses as JSON but is missing required fields must still
    zero the rewards — the hardening recovers SYNTAX, never invents SCHEMA."""
    out = '<financial_plan>{"task": "portfolio_allocation"}</financial_plan>'
    rb = compute_rewards_from_output(out, "portfolio", _portfolio_scenario(),
                                     PortfolioConstraints())
    assert np.all(rb.hard == 0) and np.all(rb.soft == 0), (
        "missing 'assets' must remain a parse failure (zero rewards)"
    )


def test_total_garbage_zeros_rewards():
    out = "I cannot help with this request."
    rb = compute_rewards_from_output(out, "portfolio", _portfolio_scenario(),
                                     PortfolioConstraints())
    assert np.all(rb.hard == 0) and np.all(rb.soft == 0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Day-4 round 2: number-comma repair (observed in real 0.5B loan output)
# ---------------------------------------------------------------------------
def test_thousands_separator_repaired():
    inner = ('{"task": "loan_structuring", "loan_amount": 249,000, '
             '"annual_rate": 0.055}')
    out = f"<financial_plan>{inner}</financial_plan>"
    d = extract_json_block(out)
    assert d["loan_amount"] == 249000


def test_multigroup_thousands_separator():
    out = '<financial_plan>{"x": 1,234,567}</financial_plan>'
    assert extract_json_block(out)["x"] == 1234567


def test_list_commas_survive_number_repair():
    # the number-comma repair must NOT collapse legitimate list commas
    inner = '{"task": "x", "weights": [0.25, 0.25, 0.25, 0.25]}'
    out = f"<financial_plan>{inner}</financial_plan>"
    assert extract_json_block(out)["weights"] == [0.25, 0.25, 0.25, 0.25]


def test_arithmetic_expression_still_fails():
    # "8/12" is not valid JSON and we deliberately do NOT eval it
    out = '<financial_plan>{"expected_return": 8/12}</financial_plan>'
    with pytest.raises(ParserError):
        extract_json_block(out)

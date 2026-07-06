"""
finplanenv/parser.py
====================
Output schema, prompt templates, and parser for all three FinPlanEnv tasks.

Design principles
-----------------
1. The LLM is constrained to emit a single <financial_plan>{JSON}</financial_plan>
   block inside a <plan>...</plan> wrapper.  The parser extracts this block
   with a regex, validates JSON, checks required fields and types, and
   merges with scenario data to return a fully-populated *Plan dataclass.

2. Parse failure → all rewards zero.  This is intentional: a plan that
   cannot be parsed is unusable, and the agent must learn to produce
   parseable output as an implicit format constraint.  Log parse failures
   separately so you can monitor the parse-failure rate during training.

3. The schema is minimal: only fields the LLM decides are requested.
   Fields the scenario provides (market data, client demographics) are
   injected by the scorer, never output by the LLM.

4. Validation has two tiers:
   - Hard validation (ParserError): missing fields, wrong types, out-of-
     range values that would crash the reward function.
   - Soft validation (ParseWarning): values that are suspicious but legal
     (e.g. weights summing to 0.97 instead of 1.0).  These are logged
     and passed to the reward function as-is; the reward function handles
     them correctly (the weight-sum hard constraint will fire).

Token budget
------------
JSON block sizes (from audit): portfolio ~115 tokens, retirement ~80,
loan ~70.  With a 1024-token max response and ~200 tokens for reasoning,
all three tasks fit comfortably.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from finplanenv.rewards import (
    LoanConstraints, LoanPlan,
    PortfolioConstraints, PortfolioPlan,
    RetirementConstraints, RetirementPlan,
    RewardBundle,
    loan_rewards, portfolio_rewards, retirement_rewards,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class ParserError(ValueError):
    """Raised when the LLM output cannot be parsed into a valid plan."""


class ParseWarning(UserWarning):
    """Issued for suspicious but legal field values."""


# ──────────────────────────────────────────────────────────────────────────────
#  Sentinel-tag extraction
# ──────────────────────────────────────────────────────────────────────────────

_PLAN_RE = re.compile(
    r"<financial_plan>\s*(.*?)\s*</financial_plan>",
    re.DOTALL | re.IGNORECASE,
)

# Fallbacks for real small-model output that doesn't perfectly follow the schema.
# These are tried ONLY after the strict tag match fails, so well-formed output is
# unaffected. Each fallback is logged so you can monitor how often the model
# needs rescuing (a high rescue rate is itself a signal to fix the prompt).
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_TAG_ONLY_RE = re.compile(r"<financial_plan>\s*(.*)", re.DOTALL | re.IGNORECASE)


def _strip_fences(s: str) -> str:
    """Remove a surrounding ```json ... ``` fence if present."""
    m = _FENCE_RE.search(s)
    return m.group(1).strip() if m else s


def _extract_first_json_object(s: str) -> str | None:
    """Return the first balanced {...} object in s, or None.

    Brace-matching so we don't choke on prose before/after the JSON, which small
    instruct models frequently emit.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def _repair_common_json(raw: str) -> str:
    """Best-effort repair of the most common small-model JSON defects:
    trailing commas before } or ], thousands-separator commas inside numbers
    (249,000 -> 249000), and single-quoted keys/strings. Applied only on the
    fallback path, after strict json.loads has already failed.

    Deliberately does NOT evaluate arithmetic expressions (e.g. "8/12") — those
    remain a parse failure, because guessing the value would corrupt the reward
    signal. Repair recovers syntax, never invents semantics.
    """
    # thousands separators inside numbers:  249,000 -> 249000  (repeat for
    # multi-group numbers like 1,234,567). The lookahead (?=\D|$) ensures we only
    # collapse a comma that sits between a digit and a 3-digit group followed by a
    # non-digit, so list commas like [0.25, 0.25] are untouched.
    prev = None
    repaired = raw
    while prev != repaired:
        prev = repaired
        repaired = re.sub(r"(\d),(\d{3})(?=\D|$)", r"\1\2", repaired)
    # trailing commas:  {"a": 1,}  ->  {"a": 1}
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    # single-quoted -> double-quoted (only if there are no double quotes already,
    # to avoid corrupting valid JSON that legitimately contains apostrophes)
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def extract_json_block(llm_output: str) -> dict[str, Any]:
    """
    Extract and parse the <financial_plan>...</financial_plan> JSON block.

    Strategy (strict first, then bounded fallbacks for real small-model output):
      1. Strict: exact <financial_plan>...</financial_plan> tags + valid JSON.
      2. Fenced: tags present but JSON wrapped in a ```json ... ``` fence.
      3. Unterminated tag: <financial_plan> opened but never closed (ran out of
         tokens) — take everything after the open tag and brace-match.
      4. No tags: model emitted bare JSON — brace-match the first {...} object.
    Each fallback attempts json.loads, then a light repair (trailing commas /
    single quotes) before giving up. Fallback use is logged.

    Raises ParserError if no strategy yields valid JSON.
    """
    def _try_load(candidate: str, how: str) -> dict | None:
        candidate = candidate.strip()
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                obj = json.loads(_repair_common_json(candidate))
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict):
            return None
        if how != "strict":
            logger.debug("extract_json_block: recovered via fallback '%s'", how)
        return obj

    # 1. strict
    m = _PLAN_RE.search(llm_output)
    if m:
        obj = _try_load(m.group(1), "strict")
        if obj is not None:
            return obj
        # tags matched but JSON bad — try stripping a fence inside the tags
        obj = _try_load(_strip_fences(m.group(1)), "fenced_in_tags")
        if obj is not None:
            return obj
        obj = _try_load(_extract_first_json_object(m.group(1)) or "", "braces_in_tags")
        if obj is not None:
            return obj

    # 3. unterminated <financial_plan> (no closing tag)
    if not m:
        mo = _OPEN_TAG_ONLY_RE.search(llm_output)
        if mo:
            tail = _strip_fences(mo.group(1))
            obj = _try_load(_extract_first_json_object(tail) or tail, "unterminated_tag")
            if obj is not None:
                return obj

    # 2/4. no usable tags — try a fenced block, then any bare {...}
    fenced = _strip_fences(llm_output)
    obj = _try_load(_extract_first_json_object(fenced) or "", "bare_json")
    if obj is not None:
        return obj

    raise ParserError(
        "No parseable <financial_plan> JSON found (strict tags, fenced JSON, "
        "unterminated tag, and bare-object fallbacks all failed). "
        f"Output snippet: {llm_output[:200]!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Shared validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require(d: dict, key: str, expected_type: type | tuple) -> Any:
    """Extract a required field, raising ParserError if absent or wrong type."""
    if key not in d:
        raise ParserError(f"Required field '{key}' missing from plan JSON.")
    val = d[key]
    if not isinstance(val, expected_type):
        raise ParserError(
            f"Field '{key}' expected type {expected_type}, "
            f"got {type(val).__name__}: {val!r}"
        )
    return val


def _require_float(d: dict, key: str) -> float:
    """Extract a required numeric field, coercing int → float."""
    if key not in d:
        raise ParserError(f"Required field '{key}' missing from plan JSON.")
    val = d[key]
    if not isinstance(val, (int, float)):
        raise ParserError(
            f"Field '{key}' must be numeric, got {type(val).__name__}: {val!r}"
        )
    return float(val)


def _require_float_list(d: dict, key: str) -> list[float]:
    """Extract a required list of numerics, coercing to float."""
    if key not in d:
        raise ParserError(f"Required field '{key}' missing from plan JSON.")
    val = d[key]
    if not isinstance(val, list):
        raise ParserError(
            f"Field '{key}' must be a list, got {type(val).__name__}: {val!r}"
        )
    try:
        return [float(x) for x in val]
    except (TypeError, ValueError) as e:
        raise ParserError(f"Field '{key}' contains non-numeric values: {e}") from e


def _validate_weights(weights: list[float], key: str = "weights") -> np.ndarray:
    """
    Validate weight list: all in [0,1], warn if sum not close to 1.
    Returns np.ndarray.
    """
    w = np.array(weights, dtype=float)
    if np.any(w < -0.001) or np.any(w > 1.001):
        raise ParserError(
            f"Field '{key}' contains values outside [0, 1]: {weights}"
        )
    w = np.clip(w, 0.0, 1.0)
    weight_sum = w.sum()
    if abs(weight_sum - 1.0) > 0.05:
        warnings.warn(
            f"Field '{key}' sums to {weight_sum:.4f}, expected 1.0. "
            "The weight-sum hard constraint will fire.",
            ParseWarning,
            stacklevel=3,
        )
    return w


# ──────────────────────────────────────────────────────────────────────────────
#  Task 1 — Portfolio Allocation parser
# ──────────────────────────────────────────────────────────────────────────────

# Valid sector labels (extend as needed for your asset universe)
VALID_SECTORS = {
    "technology", "financials", "energy", "utilities", "healthcare",
    "consumer_discretionary", "consumer_staples", "industrials",
    "materials", "real_estate", "communication_services",
}

VALID_SECTORS_ALIASES = {
    "tech": "technology", "finance": "financials",
    "health": "healthcare", "realestate": "real_estate",
    "comms": "communication_services",
}


def _normalise_sector(s: str) -> str:
    s = s.lower().replace(" ", "_").replace("-", "_")
    return VALID_SECTORS_ALIASES.get(s, s)


@dataclass
class PortfolioScenario:
    """
    Fixed scenario data injected by the dataset, not output by the LLM.
    Pass this alongside the raw LLM output to parse_portfolio().
    """
    n_assets: int
    expected_returns: np.ndarray    # shape (N,)
    cov_matrix: np.ndarray          # shape (N, N)
    esg_ratings: np.ndarray         # shape (N,)
    benchmark_weights: np.ndarray   # shape (N,)
    return_series: np.ndarray       # shape (T, N)
    banned_sectors: set[str] = None # from client profile

    def __post_init__(self):
        if self.banned_sectors is None:
            self.banned_sectors = set()


def parse_portfolio(
    llm_output: str,
    scenario: PortfolioScenario,
) -> PortfolioPlan:
    """
    Parse LLM output into a PortfolioPlan.

    Expected JSON schema inside <financial_plan>:
    {
        "task": "portfolio_allocation",
        "assets": [
            {"ticker": "AAPL", "sector": "technology", "weight": 0.35},
            ...
        ],
        "rationale": "..."   // optional, ignored
    }

    Raises ParserError on any hard validation failure.
    """
    d = extract_json_block(llm_output)

    # task type check (soft — don't fail, just warn)
    if d.get("task") != "portfolio_allocation":
        warnings.warn(
            f"Expected task='portfolio_allocation', got {d.get('task')!r}.",
            ParseWarning, stacklevel=2,
        )

    # assets array
    assets = _require(d, "assets", list)
    if len(assets) != scenario.n_assets:
        raise ParserError(
            f"Expected {scenario.n_assets} assets, got {len(assets)}."
        )

    weights = []
    sector_labels = []

    for i, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ParserError(f"Asset {i} is not a dict: {asset!r}")

        # weight
        if "weight" not in asset:
            raise ParserError(f"Asset {i} missing 'weight' field.")
        try:
            w = float(asset["weight"])
        except (TypeError, ValueError):
            raise ParserError(f"Asset {i} 'weight' is not numeric: {asset['weight']!r}")
        weights.append(w)

        # sector
        if "sector" not in asset:
            raise ParserError(f"Asset {i} missing 'sector' field.")
        sector = _normalise_sector(str(asset["sector"]))
        if sector not in VALID_SECTORS:
            warnings.warn(
                f"Asset {i} sector '{sector}' not in VALID_SECTORS. "
                "It will be checked against banned_sectors as-is.",
                ParseWarning, stacklevel=2,
            )
        sector_labels.append(sector)

    w_array = _validate_weights(weights)

    return PortfolioPlan(
        weights          = w_array,
        sector_labels    = sector_labels,
        expected_returns = scenario.expected_returns,
        cov_matrix       = scenario.cov_matrix,
        esg_ratings      = scenario.esg_ratings,
        benchmark_weights= scenario.benchmark_weights,
        return_series    = scenario.return_series,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Task 2 — Retirement Planning parser
# ──────────────────────────────────────────────────────────────────────────────

VALID_ASSET_CLASSES = {"equities", "bonds", "cash", "real_estate", "commodities"}
ASSET_CLASS_ALIASES = {
    "stocks": "equities", "equity": "equities",
    "fixed_income": "bonds", "bond": "bonds",
    "money_market": "cash",
}


@dataclass
class RetirementScenario:
    """Fixed scenario data for retirement task."""
    initial_balance: float
    current_age: int
    target_age: int
    n_assets: int
    expected_returns: np.ndarray    # shape (N,) — one per asset class
    cov_matrix: np.ndarray          # shape (N, N)
    income_floor_monthly: float


def parse_retirement(
    llm_output: str,
    scenario: RetirementScenario,
) -> RetirementPlan:
    """
    Parse LLM output into a RetirementPlan.

    Expected JSON schema inside <financial_plan>:
    {
        "task": "retirement_planning",
        "annual_withdrawal": 52000,
        "withdrawal_type": "fixed",        // "fixed" | "variable"
        "withdrawal_schedule": [52000, ...],  // required if variable;
                                              // omit or length-1 if fixed
        "asset_allocation": [
            {"asset_class": "equities", "weight": 0.60},
            {"asset_class": "bonds",    "weight": 0.40}
        ],
        "rationale": "..."
    }

    Raises ParserError on hard validation failure.
    """
    d = extract_json_block(llm_output)
    years = scenario.target_age - scenario.current_age

    if d.get("task") != "retirement_planning":
        warnings.warn(
            f"Expected task='retirement_planning', got {d.get('task')!r}.",
            ParseWarning, stacklevel=2,
        )

    # annual_withdrawal
    annual_withdrawal = _require_float(d, "annual_withdrawal")
    if annual_withdrawal <= 0:
        raise ParserError(
            f"'annual_withdrawal' must be > 0, got {annual_withdrawal}."
        )

    # withdrawal_schedule
    withdrawal_type = d.get("withdrawal_type", "fixed")
    if withdrawal_type == "variable":
        raw_schedule = _require_float_list(d, "withdrawal_schedule")
        if len(raw_schedule) == 1:
            # treat single value as fixed
            schedule = np.full(years, raw_schedule[0])
        elif len(raw_schedule) != years:
            raise ParserError(
                f"'withdrawal_schedule' length {len(raw_schedule)} "
                f"!= years {years}. Provide one value per year or "
                "set withdrawal_type='fixed'."
            )
        else:
            schedule = np.array(raw_schedule, dtype=float)
        if np.any(schedule <= 0):
            raise ParserError("All values in 'withdrawal_schedule' must be > 0.")
    else:
        # fixed: broadcast annual_withdrawal across all years
        schedule = np.full(years, annual_withdrawal)

    # asset_allocation
    allocations = _require(d, "asset_allocation", list)
    if len(allocations) != scenario.n_assets:
        raise ParserError(
            f"Expected {scenario.n_assets} asset classes in 'asset_allocation', "
            f"got {len(allocations)}."
        )

    weights = []
    for i, alloc in enumerate(allocations):
        if not isinstance(alloc, dict):
            raise ParserError(f"asset_allocation[{i}] is not a dict: {alloc!r}")
        if "weight" not in alloc:
            raise ParserError(f"asset_allocation[{i}] missing 'weight'.")
        try:
            weights.append(float(alloc["weight"]))
        except (TypeError, ValueError):
            raise ParserError(
                f"asset_allocation[{i}] 'weight' not numeric: {alloc['weight']!r}"
            )

    w_array = _validate_weights(weights)

    return RetirementPlan(
        initial_balance    = scenario.initial_balance,
        annual_withdrawal  = annual_withdrawal,
        withdrawal_schedule= schedule,
        weights            = w_array,
        expected_returns   = scenario.expected_returns,
        cov_matrix         = scenario.cov_matrix,
        current_age        = scenario.current_age,
        target_age         = scenario.target_age,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Task 3 — Loan Structuring parser
# ──────────────────────────────────────────────────────────────────────────────

VALID_TERM_MONTHS = {60, 120, 180, 240, 300, 360}
VALID_REG_TIERS   = {"qm", "non-qm", "non_qm"}


@dataclass
class LoanScenario:
    """Fixed scenario data for loan task."""
    property_value: float
    gross_monthly_income: float


def parse_loan(
    llm_output: str,
    scenario: LoanScenario,
) -> LoanPlan:
    """
    Parse LLM output into a LoanPlan.

    Expected JSON schema inside <financial_plan>:
    {
        "task": "loan_structuring",
        "loan_amount": 320000,
        "annual_rate": 0.065,
        "term_months": 360,
        "total_monthly_debt_payments": 2200,
        "prepayment_penalty": false,
        "lock_in_years": 2,
        "regulatory_tier": "QM",
        "rationale": "..."
    }

    Raises ParserError on hard validation failure.
    """
    d = extract_json_block(llm_output)

    if d.get("task") != "loan_structuring":
        warnings.warn(
            f"Expected task='loan_structuring', got {d.get('task')!r}.",
            ParseWarning, stacklevel=2,
        )

    # loan_amount
    loan_amount = _require_float(d, "loan_amount")
    if loan_amount <= 0:
        raise ParserError(f"'loan_amount' must be > 0, got {loan_amount}.")

    # annual_rate
    annual_rate = _require_float(d, "annual_rate")
    if not (0.005 <= annual_rate <= 0.30):
        raise ParserError(
            f"'annual_rate' must be in [0.005, 0.30], got {annual_rate:.4f}. "
            "Rates outside this range indicate a parsing error "
            "(e.g. 6.5 instead of 0.065)."
        )

    # term_months
    raw_term = _require(d, "term_months", (int, float))
    term_months = int(raw_term)
    if term_months not in VALID_TERM_MONTHS:
        # attempt nearest valid term
        nearest = min(VALID_TERM_MONTHS, key=lambda t: abs(t - term_months))
        warnings.warn(
            f"'term_months'={term_months} not in {sorted(VALID_TERM_MONTHS)}. "
            f"Snapping to nearest valid term: {nearest}.",
            ParseWarning, stacklevel=2,
        )
        term_months = nearest

    # total_monthly_debt_payments
    # accept both field names for robustness
    debt_key = ("total_monthly_debt_payments"
                if "total_monthly_debt_payments" in d
                else "total_monthly_debt")
    if debt_key not in d:
        raise ParserError(
            "Required field 'total_monthly_debt_payments' "
            "(or 'total_monthly_debt') missing."
        )
    total_monthly_debt = _require_float(d, debt_key)
    if total_monthly_debt <= 0:
        raise ParserError(
            f"'{debt_key}' must be > 0, got {total_monthly_debt}."
        )

    # prepayment_penalty
    if "prepayment_penalty" not in d:
        raise ParserError("Required field 'prepayment_penalty' missing.")
    pp_raw = d["prepayment_penalty"]
    if isinstance(pp_raw, bool):
        prepayment_penalty = pp_raw
    elif isinstance(pp_raw, str):
        prepayment_penalty = pp_raw.lower() in {"true", "yes", "1"}
    elif isinstance(pp_raw, (int, float)):
        prepayment_penalty = bool(int(pp_raw))
    else:
        raise ParserError(
            f"'prepayment_penalty' must be boolean, got {type(pp_raw).__name__}."
        )

    # lock_in_years
    lock_in_years = _require_float(d, "lock_in_years")
    if not (0.0 <= lock_in_years <= 30.0):
        raise ParserError(
            f"'lock_in_years' must be in [0, 30], got {lock_in_years}."
        )

    # regulatory_tier
    raw_tier = _require(d, "regulatory_tier", str)
    tier_norm = raw_tier.lower().replace(" ", "_")
    if tier_norm not in VALID_REG_TIERS:
        raise ParserError(
            f"'regulatory_tier' must be 'QM' or 'non-QM', got {raw_tier!r}."
        )
    regulatory_tier = "QM" if tier_norm == "qm" else "non-QM"

    return LoanPlan(
        loan_amount           = loan_amount,
        property_value        = scenario.property_value,
        annual_rate           = annual_rate,
        term_months           = term_months,
        gross_monthly_income  = scenario.gross_monthly_income,
        total_monthly_debt    = total_monthly_debt,
        prepayment_penalty    = prepayment_penalty,
        lock_in_years         = lock_in_years,
        regulatory_tier       = regulatory_tier,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Unified entry point — used by the RL training loop
# ──────────────────────────────────────────────────────────────────────────────

def compute_rewards_from_output(
    llm_output: str,
    task: str,
    scenario: PortfolioScenario | RetirementScenario | LoanScenario,
    constraints: PortfolioConstraints | RetirementConstraints | LoanConstraints,
) -> RewardBundle:
    """
    Parse LLM output and compute all rewards in one call.

    On ParserError: returns a zero RewardBundle (hard=0, soft=0, prox=0)
    and logs the error.  The zero bundle is intentional — unparseable
    output is treated as a fully failing plan.

    Parameters
    ----------
    llm_output : str
        Raw text output from the LLM rollout.
    task : str
        One of {"portfolio", "retirement", "loan"}.
    scenario : *Scenario dataclass
        Fixed scenario data for this prompt instance.
    constraints : *Constraints dataclass
        Per-scenario constraint thresholds.

    Returns
    -------
    RewardBundle with .hard, .prox, .soft arrays and .task string.
    """
    K_MAP = {"portfolio": 4, "retirement": 2, "loan": 3}
    M_MAP = {"portfolio": 3, "retirement": 3, "loan": 3}

    try:
        if task == "portfolio":
            plan = parse_portfolio(llm_output, scenario)
            return portfolio_rewards(plan, constraints)
        elif task == "retirement":
            plan = parse_retirement(llm_output, scenario)
            return retirement_rewards(plan, constraints)
        elif task == "loan":
            plan = parse_loan(llm_output, scenario)
            return loan_rewards(plan, constraints)
        else:
            raise ParserError(f"Unknown task: {task!r}")

    except ParserError as e:
        K = K_MAP.get(task, 0)
        M = M_MAP.get(task, 0)
        logger.warning("ParserError for task=%s: %s", task, e)
        return RewardBundle(
            hard = np.zeros(K, dtype=float),
            prox = np.zeros(K, dtype=float),
            soft = np.zeros(M, dtype=float),
            task = task,
            parse_failed = True,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Prompt templates — used by the dataset generator and RL loop
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert financial planning assistant. You output structured plans for automated parsing.

CRITICAL OUTPUT RULES — follow EXACTLY or your answer is discarded:
1. Output ONLY a single <financial_plan>...</financial_plan> block. Nothing before it, nothing after it.
2. Inside the block, output ONE valid JSON object — and nothing else. No markdown, no ```json fences, no headers, no commentary.
3. Do NOT write any reasoning, analysis, or explanation outside the JSON. If you want to justify the plan, use the "rationale" field INSIDE the JSON (one short sentence).
4. JSON number rules: no thousands separators (write 249000, NOT 249,000), no arithmetic (write 0.6667, NOT 8/12), no currency symbols, no units. Decimals only.
5. Follow the exact JSON schema given in the task. Use the exact field names requested — do not invent fields.

Your entire response must look like this and nothing else:
<financial_plan>
{ ...the JSON object matching the task schema... }
</financial_plan>"""


PORTFOLIO_USER_TEMPLATE = """## Client Profile
{client_profile}

## Available Assets ({n_assets} assets)
{asset_descriptions}

## Constraints
- Maximum drawdown limit: {max_drawdown_pct}%
- Banned sectors: {banned_sectors}
- Minimum diversification (HHI ≤ {hhi_limit:.2f})

## Required JSON Schema
Produce a portfolio allocation as:
{{
    "task": "portfolio_allocation",
    "assets": [
        {{"ticker": "<ticker>", "sector": "<sector>", "weight": <float 0-1>}},
        ...  // one entry per asset, weights must sum to 1.0
    ],
    "rationale": "<brief explanation>"
}}"""


RETIREMENT_USER_TEMPLATE = """## Client Profile
{client_profile}

## Financial Situation
- Current portfolio value: ${initial_balance:,.0f}
- Current age: {current_age} | Target retirement end age: {target_age}
- Minimum required monthly income: ${income_floor_monthly:,.0f}

## Available Asset Classes ({n_assets} classes)
{asset_descriptions}

## Survival Requirement
The plan must survive at least {survival_pct}% of Monte Carlo market scenarios.

## Required JSON Schema
{{
    "task": "retirement_planning",
    "annual_withdrawal": <float>,
    "withdrawal_type": "fixed",
    "asset_allocation": [
        {{"asset_class": "<class>", "weight": <float 0-1>}},
        ...  // weights must sum to 1.0
    ],
    "rationale": "<brief explanation>"
}}"""


LOAN_USER_TEMPLATE = """## Client Profile
{client_profile}

## Property Details
- Property value: ${property_value:,.0f}
- Client gross monthly income: ${gross_monthly_income:,.0f}

## Regulatory Context
Applicable tier: {regulatory_tier}

## Required JSON Schema
{{
    "task": "loan_structuring",
    "loan_amount": <float>,
    "annual_rate": <float, e.g. 0.065 for 6.5%>,
    "term_months": <int, one of: 60 120 180 240 300 360>,
    "total_monthly_debt_payments": <float, ALL debt payments incl. this loan>,
    "prepayment_penalty": <true|false>,
    "lock_in_years": <float 0-10>,
    "regulatory_tier": "<QM|non-QM>",
    "rationale": "<brief explanation>"
}}"""


PROMPT_TEMPLATES = {
    "portfolio":  (SYSTEM_PROMPT, PORTFOLIO_USER_TEMPLATE),
    "retirement": (SYSTEM_PROMPT, RETIREMENT_USER_TEMPLATE),
    "loan":       (SYSTEM_PROMPT, LOAN_USER_TEMPLATE),
}


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    # ── Test 1: valid portfolio output ──────────────────────────────────────
    valid_portfolio_output = textwrap.dedent("""
    <plan>
    <reasoning>
    The client needs diversification across four sectors.
    Technology offers high growth, utilities provide stability.
    </reasoning>
    <financial_plan>
    {
        "task": "portfolio_allocation",
        "assets": [
            {"ticker": "AAPL", "sector": "technology",  "weight": 0.35},
            {"ticker": "JPM",  "sector": "financials",  "weight": 0.25},
            {"ticker": "XOM",  "sector": "energy",      "weight": 0.20},
            {"ticker": "NEE",  "sector": "utilities",   "weight": 0.20}
        ],
        "rationale": "Balanced allocation with tech tilt."
    }
    </financial_plan>
    </plan>
    """)

    np.random.seed(0)
    N, T = 4, 756
    scenario_p = PortfolioScenario(
        n_assets          = 4,
        expected_returns  = np.array([0.12, 0.09, 0.07, 0.04]),
        cov_matrix        = np.diag([0.04, 0.03, 0.02, 0.01]),
        esg_ratings       = np.array([72., 85., 60., 45.]),
        benchmark_weights = np.array([0.25, 0.25, 0.25, 0.25]),
        return_series     = np.random.normal(0.0004, 0.01, (T, N)),
        banned_sectors    = {"tobacco", "weapons"},
    )
    cfg_p = PortfolioConstraints()
    rb = compute_rewards_from_output(
        valid_portfolio_output, "portfolio", scenario_p, cfg_p
    )
    print("Test 1 — valid portfolio:")
    print(f"  hard={rb.hard}  prox={rb.prox.round(3)}  soft={rb.soft.round(3)}")

    # ── Test 2: parse failure → zero rewards ────────────────────────────────
    bad_output = "I recommend investing in diversified assets."
    rb_bad = compute_rewards_from_output(bad_output, "portfolio", scenario_p, cfg_p)
    print("\nTest 2 — parse failure (no tag):")
    print(f"  hard={rb_bad.hard}  prox={rb_bad.prox}  soft={rb_bad.soft}")
    assert np.all(rb_bad.hard == 0), "Parse failure must zero hard rewards"
    assert np.all(rb_bad.soft == 0), "Parse failure must zero soft rewards"

    # ── Test 3: valid retirement output ─────────────────────────────────────
    valid_retirement_output = textwrap.dedent("""
    <plan>
    <reasoning>60/40 allocation balances growth and stability.</reasoning>
    <financial_plan>
    {
        "task": "retirement_planning",
        "annual_withdrawal": 52000,
        "withdrawal_type": "fixed",
        "asset_allocation": [
            {"asset_class": "equities", "weight": 0.60},
            {"asset_class": "bonds",    "weight": 0.40}
        ],
        "rationale": "Conservative 60/40 targeting 90% survival."
    }
    </financial_plan>
    </plan>
    """)

    scenario_r = RetirementScenario(
        initial_balance       = 1_000_000,
        current_age           = 65,
        target_age            = 95,
        n_assets              = 2,
        expected_returns      = np.array([0.08, 0.04]),
        cov_matrix            = np.array([[0.04, 0.005], [0.005, 0.001]]),
        income_floor_monthly  = 3000.0,
    )
    cfg_r = RetirementConstraints()
    rb_r = compute_rewards_from_output(
        valid_retirement_output, "retirement", scenario_r, cfg_r
    )
    print("\nTest 3 — valid retirement:")
    print(f"  hard={rb_r.hard}  prox={rb_r.prox.round(3)}  soft={rb_r.soft.round(3)}")

    # ── Test 4: valid loan output ────────────────────────────────────────────
    valid_loan_output = textwrap.dedent("""
    <plan>
    <reasoning>30-year fixed at 6.5% fits within QM guidelines.</reasoning>
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
        "rationale": "30-year fixed, QM compliant, flexible prepayment."
    }
    </financial_plan>
    </plan>
    """)

    scenario_l = LoanScenario(property_value=400_000, gross_monthly_income=8_000)
    cfg_l = LoanConstraints()
    rb_l = compute_rewards_from_output(valid_loan_output, "loan", scenario_l, cfg_l)
    print("\nTest 4 — valid loan:")
    print(f"  hard={rb_l.hard}  prox={rb_l.prox.round(3)}  soft={rb_l.soft.round(3)}")

    # ── Test 5: rate expressed as percentage instead of decimal ─────────────
    rate_bug_output = valid_loan_output.replace('"annual_rate": 0.065',
                                                '"annual_rate": 6.5')
    rb_bug = compute_rewards_from_output(rate_bug_output, "loan", scenario_l, cfg_l)
    print("\nTest 5 — rate as percentage (6.5 instead of 0.065) → parse error:")
    print(f"  hard={rb_bug.hard}  (should be all zeros)")
    assert np.all(rb_bug.hard == 0), "Rate validation must catch percentage bug"

    print("\n✓  All smoke tests passed.")

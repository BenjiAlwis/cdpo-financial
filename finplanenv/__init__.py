"""
finplanenv
==========
Constraint-Decomposed Policy Optimisation (CDPO) for Financial Planning Agents.

Public API
----------
Core algorithm:
    CDPOConfig          — all hyperparameters
    CDPOAdvantage       — Steps 2–6 of Algorithm 1 (pure numpy)
    CDPORewardWrapper   — verl reward_fn interface
    CDPOTRLRewardFn     — TRL GRPOTrainer reward_fn interface
    grpo_advantages     — GRPO baseline (same interface)
    gdpo_advantages     — GDPO baseline (same interface)

Reward functions:
    portfolio_rewards   — Task 1: portfolio allocation
    retirement_rewards  — Task 2: retirement planning
    loan_rewards        — Task 3: loan structuring
    RewardBundle        — container for hard/prox/soft signals

Parser:
    compute_rewards_from_output — unified entry point for RL loop
    extract_json_block          — raw sentinel-tag extraction
    ParserError / ParseWarning

Dataset generation:
    DatasetConfig       — size, seed, task list
    ScenarioSampler     — deterministic scenario instances
    DatasetGenerator    — pilot → calibrate → scale workflow
"""

from finplanenv.rewards import (
    RewardBundle,
    PortfolioPlan, PortfolioConstraints, portfolio_rewards,
    RetirementPlan, RetirementConstraints, retirement_rewards,
    LoanPlan, LoanConstraints, loan_rewards,
)

from finplanenv.parser import (
    ParserError, ParseWarning,
    extract_json_block,
    compute_rewards_from_output,
    PortfolioScenario, RetirementScenario, LoanScenario,
    SYSTEM_PROMPT,
    PORTFOLIO_USER_TEMPLATE,
    RETIREMENT_USER_TEMPLATE,
    LOAN_USER_TEMPLATE,
)

from finplanenv.cdpo import (
    CDPOConfig,
    CDPOAdvantage,
    BatchRewards,
    CorrelationTracker,
    CDPORewardWrapper,
    CDPOTRLRewardFn,
    grpo_advantages,
    gdpo_advantages,
)

from finplanenv.baselines import (
    BaselineConfig,
    GRPOAdvantage,
    GDPOAdvantage,
    GRPORewardWrapper,
    GRPOTRLRewardFn,
    GDPORewardWrapper,
    GDPOTRLRewardFn,
)

from finplanenv.dataset import (
    DatasetConfig,
    ScenarioSampler,
    DatasetGenerator,
    DifficultyTier,
    DIFFICULTY_TIERS,
    MARKET_PARAMS,
    PORTFOLIO_THRESHOLDS,
    RETIREMENT_THRESHOLDS,
    LOAN_THRESHOLDS,
    GENERATION_PROMPTS,
)

__version__ = "0.1.0"
__all__ = [
    # rewards
    "RewardBundle",
    "PortfolioPlan", "PortfolioConstraints", "portfolio_rewards",
    "RetirementPlan", "RetirementConstraints", "retirement_rewards",
    "LoanPlan", "LoanConstraints", "loan_rewards",
    # parser
    "ParserError", "ParseWarning",
    "extract_json_block", "compute_rewards_from_output",
    "PortfolioScenario", "RetirementScenario", "LoanScenario",
    "SYSTEM_PROMPT",
    "PORTFOLIO_USER_TEMPLATE", "RETIREMENT_USER_TEMPLATE", "LOAN_USER_TEMPLATE",
    # cdpo
    "CDPOConfig", "CDPOAdvantage", "BatchRewards", "CorrelationTracker",
    "CDPORewardWrapper", "CDPOTRLRewardFn",
    "grpo_advantages", "gdpo_advantages",
    # baselines
    "BaselineConfig",
    "GRPOAdvantage", "GDPOAdvantage",
    "GRPORewardWrapper", "GRPOTRLRewardFn",
    "GDPORewardWrapper", "GDPOTRLRewardFn",
    # dataset
    "DatasetConfig", "ScenarioSampler", "DatasetGenerator",
    "DifficultyTier", "DIFFICULTY_TIERS", "MARKET_PARAMS",
    "PORTFOLIO_THRESHOLDS", "RETIREMENT_THRESHOLDS", "LOAN_THRESHOLDS",
    "GENERATION_PROMPTS",
]

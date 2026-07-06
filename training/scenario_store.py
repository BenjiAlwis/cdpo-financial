"""
training/scenario_store.py
==========================
ScenarioStore: maps a prompt string back to (scenario, constraints, task_name)
at reward-computation time.

Why this is needed
------------------
TRL's GRPOTrainer calls reward_fn(prompts, completions). The prompts are plain
strings — they carry no Python objects. But compute_rewards_from_output() needs
the scenario and constraints objects to compute rewards. We solve this by:

1. At dataset build time, embedding a unique instance_id into every prompt.
2. Storing the scenario/constraints for each instance_id in ScenarioStore.
3. At reward time, parsing the instance_id out of the prompt and looking up
   the objects in the store.

The instance_id is embedded as a hidden comment in the system prompt:
    <!-- instance_id: portfolio_0042 -->

This is invisible to the LLM (it's in the system message) and reliably
parseable with a regex.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from finplanenv.parser import PortfolioScenario, RetirementScenario, LoanScenario
from finplanenv.rewards import (
    PortfolioConstraints, RetirementConstraints, LoanConstraints,
)
from finplanenv.dataset import DatasetConfig, ScenarioSampler

logger = logging.getLogger(__name__)

# Regex to extract instance_id from a prompt string
_ID_RE = re.compile(r"<!--\s*instance_id:\s*(\S+)\s*-->")

ScenarioObj    = PortfolioScenario | RetirementScenario | LoanScenario
ConstraintsObj = PortfolioConstraints | RetirementConstraints | LoanConstraints


@dataclass
class ScenarioEntry:
    task:        str
    scenario:    ScenarioObj
    constraints: ConstraintsObj
    instance_id: str


class ScenarioStore:
    """
    In-memory store mapping instance_id → ScenarioEntry.

    Built once at training startup from the dataset JSONL metadata.
    Shared across all reward_fn calls during training.

    Thread-safe for read access (all writes happen during __init__).
    """

    def __init__(self, config: DatasetConfig):
        self._store: dict[str, ScenarioEntry] = {}
        self._sampler = ScenarioSampler(config)
        self._config  = config

    def build(self, metadata_list: list[dict]) -> None:
        """
        Populate the store from a list of metadata dicts
        (loaded from <task>_train.jsonl).

        Parameters
        ----------
        metadata_list : list of dicts from DatasetGenerator.load_metadata()
        """
        for meta in metadata_list:
            iid  = meta["instance_id"]
            task = iid.split("_")[0]   # "portfolio_0042" → "portfolio"
            idx  = int(iid.split("_")[1])

            scenario, constraints, _ = self._sampler.sample(task, idx)

            self._store[iid] = ScenarioEntry(
                task        = task,
                scenario    = scenario,
                constraints = constraints,
                instance_id = iid,
            )

        logger.info("ScenarioStore: loaded %d instances", len(self._store))

    def lookup(self, prompt: str) -> ScenarioEntry | None:
        """
        Extract instance_id from prompt and return the stored entry.
        Returns None if the prompt has no embedded instance_id or if the
        id is not in the store (will produce zero rewards — logged as warning).
        """
        match = _ID_RE.search(prompt)
        if not match:
            logger.warning(
                "ScenarioStore.lookup: no instance_id found in prompt. "
                "Prompt snippet: %r", prompt[:120]
            )
            return None
        iid = match.group(1)
        entry = self._store.get(iid)
        if entry is None:
            logger.warning(
                "ScenarioStore.lookup: instance_id %r not in store.", iid
            )
        return entry

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def embed_id(system_prompt: str, instance_id: str) -> str:
        """
        Embed instance_id into a system prompt as a hidden HTML comment.
        Called at dataset build time.
        """
        return system_prompt + f"\n<!-- instance_id: {instance_id} -->"

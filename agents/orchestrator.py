"""
Agent 1 — Orchestrator
Validates the scenario, then delegates to the remaining agents in order.
"""
from __future__ import annotations
import logging

from config.scenarios import SCENARIOS
from config.industry_profiles import get_profile
from core.dynamic_scenarios import resolve_scenario_meta, scenario_exists
from core.llm_client import GeminiClient
from core.state import WorkflowState

logger = logging.getLogger(__name__)

_SYSTEM = """
You are the Orchestrator Agent for a synthetic data generation pipeline covering
any business industry (telecom, banking, retail, healthcare, etc.).
Validate the requested scenario against the target industry's and country's
real-world conventions (product/plan types, regulator, market character) and
return a JSON object with:
  - "valid": bool
  - "reason": str  (empty string when valid)
  - "execution_notes": str
"""


class OrchestratorAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info("[Orchestrator] Scenario=%s count=%d", state.scenario, state.count)

        if not scenario_exists(state.scenario):
            state.errors.append(
                f"Unknown scenario '{state.scenario}'. Valid: {list(SCENARIOS.keys())}"
            )
            return state

        sc = resolve_scenario_meta(state.scenario)
        profile = get_profile(state.industry, state.country)
        prompt = (
            f"Scenario: {sc['label']}\n"
            f"Journey: {sc['journey']}\n"
            f"Description: {sc['description']}\n"
            f"Records requested: {state.count}\n"
            f"Target industry: {profile['industry']}; country: {profile['country_name']} ({state.country}) — "
            f"regulator {profile['regulator']}, market character: {profile['market_character']}\n"
            "Validate and produce execution notes."
        )
        plan = self._llm.generate_json(_SYSTEM, prompt, temperature=0.1)
        if not plan.get("valid", True):
            state.errors.append(f"Orchestrator rejected: {plan.get('reason')}")
            return state

        logger.info("[Orchestrator] Accepted. Notes: %s", plan.get("execution_notes", ""))
        return state
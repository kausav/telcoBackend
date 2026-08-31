"""
Agent 1 — Orchestrator
Validates the scenario, then delegates to the remaining agents in order.
"""
from __future__ import annotations
import logging
import json

from config.scenarios import SCENARIOS
from config.industry_profiles import get_profile
from core.dynamic_scenarios import resolve_scenario_meta, scenario_exists
from core.llm_client import GeminiClient
from core.state import WorkflowState
from core.runtime_cache import get_orchestrator, set_orchestrator

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

        cache_key = (
            state.scenario, state.industry, (state.country or "GLOBAL").upper(), state.type_of_data,
            state.domain or "", state.business_scenario or "", state.business_response or "",
            state.expected_outcome or "", state.scenario_type or "", state.use_case or "",
            state.entity_key or "", str(state.scenario_context.get("events", [])),
        )
        cached = get_orchestrator(cache_key)
        if cached is not None:
            if not cached.get("valid", True):
                state.errors.append(f"Orchestrator rejected: {cached.get('reason')}")
            else:
                logger.info("[Orchestrator] Cache hit; skipping LLM validation.")
            return state

        sc = resolve_scenario_meta(state.scenario)
        profile = get_profile(state.industry, state.country)
        prompt = (
            f"Scenario: {sc['label']}\n"
            f"Journey: {sc['journey']}\n"
            f"Description: {sc['description']}\n"
            f"Domain: {state.domain or sc.get('domain', '')}\n"
            f"Business scenario: {state.business_scenario or sc.get('business_scenario', '')}\n"
            f"Business response: {state.business_response or sc.get('business_response', '')}\n"
            f"Expected outcome: {state.expected_outcome or sc.get('expected_outcome', '')}\n"
            f"Use case: {state.use_case or sc.get('use_case', '')}\n"
            f"Scenario type: {state.scenario_type or sc.get('scenario_type', '')}\n"
            f"Records requested: {state.count}\n"
            f"Complete confirmed scenario context (source of truth): {json.dumps(state.scenario_context, default=str, sort_keys=True)}\n"
            f"Target industry: {profile['industry']}; country: {profile['country_name']} ({state.country}) — "
            f"regulator {profile['regulator']}, market character: {profile['market_character']}\n"
            "Validate and produce execution notes."
        )
        plan = self._llm.generate_json(_SYSTEM, prompt, temperature=0.1)
        set_orchestrator(cache_key, plan)
        if not plan.get("valid", True):
            state.errors.append(f"Orchestrator rejected: {plan.get('reason')}")
            return state

        logger.info("[Orchestrator] Accepted. Notes: %s", plan.get("execution_notes", ""))
        return state
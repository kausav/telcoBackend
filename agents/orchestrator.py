"""
Agent 2 — Orchestrator
Gatekeeper that runs AFTER the Scenario Designer Agent has already produced and
confirmed the scenario's variable data (via /scenario/propose + /scenario/confirm).
Checks that variable data is present and coherent before Schema/EdgeCase/DataGeneration
run. Internally implemented as a small LangGraph StateGraph: variable-data check ->
cache check -> LLM validation, each step short-circuiting to END as soon as a decision
(reject or cache hit) is reached.
"""
from __future__ import annotations
import logging
import json

from langgraph.graph import StateGraph, END

from config.industry_profiles import get_profile
from core.dynamic_scenarios import list_scenarios, resolve_scenario_meta, resolve_variables, scenario_exists
from core.llm_client import GeminiClient
from core.state import WorkflowState
from core.runtime_cache import get_orchestrator, set_orchestrator

logger = logging.getLogger(__name__)

_SYSTEM = """
You are the Orchestrator Agent for a synthetic data generation pipeline covering
any business industry (telecom, banking, retail, healthcare, etc.).
You run AFTER the scenario's variable catalog has already been generated and
confirmed by the Scenario Designer Agent. Gatekeep that variable data against the
target industry's and country's real-world conventions (product/plan types,
regulator, market character) and return a JSON object with:
  - "valid": bool
  - "reason": str  (empty string when valid)
  - "execution_notes": str
"""


class OrchestratorAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm
        # Per-call scratch values, populated by _check_variables/_check_cache and read by
        # _cache_router/_validate. Safe because a fresh OrchestratorAgent is instantiated
        # for each run_pipeline() call.
        self._cache_key_value: tuple | None = None
        self._cache_hit = False
        self._variables: list[dict] = []
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("check_existence", self._check_existence)
        graph.add_node("check_cache", self._check_cache)
        graph.add_node("validate", self._validate)

        graph.set_entry_point("check_existence")
        graph.add_conditional_edges("check_existence", self._exists_router, {"continue": "check_cache", "halt": END})
        graph.add_conditional_edges("check_cache", self._cache_router, {"hit": END, "miss": "validate"})
        graph.add_edge("validate", END)
        return graph.compile()

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info("[Orchestrator] Scenario=%s count=%d", state.scenario, state.count)
        result = self._graph.invoke(state, config={"recursion_limit": 10})
        return result if isinstance(result, WorkflowState) else WorkflowState.model_validate(result)

    @staticmethod
    def _exists_router(state: WorkflowState) -> str:
        return "halt" if state.errors else "continue"

    def _cache_router(self, state: WorkflowState) -> str:
        return "hit" if self._cache_hit else "miss"

    def _check_existence(self, state: WorkflowState) -> WorkflowState:
        """Gatekeeper: confirm the scenario exists AND that the Scenario Designer Agent
        actually produced usable variable data before anything downstream runs."""
        if not scenario_exists(state.scenario):
            valid = [s["id"] for s in list_scenarios()]
            state.errors.append(f"Unknown scenario '{state.scenario}'. Valid: {valid}")
            return state

        dyn = resolve_variables(state.scenario)
        if dyn is None or not dyn[0]:
            state.errors.append(
                f"Scenario '{state.scenario}' has no variable data to gatekeep. The Scenario "
                "Designer Agent's output must be confirmed via /scenario/confirm before generation can run."
            )
            return state
        self._variables = dyn[0]
        return state

    def _compute_cache_key(self, state: WorkflowState) -> tuple:
        return (
            state.scenario, state.industry, (state.country or "GLOBAL").upper(), state.type_of_data,
            state.domain or "", state.business_scenario or "", state.business_response or "",
            state.expected_outcome or "", state.scenario_type or "", state.use_case or "",
            state.entity_key or "", str(state.scenario_context.get("events", [])),
        )

    def _check_cache(self, state: WorkflowState) -> WorkflowState:
        self._cache_key_value = self._compute_cache_key(state)
        cached = get_orchestrator(self._cache_key_value)
        self._cache_hit = cached is not None
        if cached is not None:
            if not cached.get("valid", True):
                state.errors.append(f"Orchestrator rejected: {cached.get('reason')}")
            else:
                logger.info("[Orchestrator] Cache hit; skipping LLM validation.")
        return state

    def _validate(self, state: WorkflowState) -> WorkflowState:
        sc = resolve_scenario_meta(state.scenario)
        profile = get_profile(state.industry, state.country)
        variable_summary = [
            {"name": v.get("name"), "dtype": v.get("dtype"), "gen": v.get("gen")}
            for v in self._variables
        ]
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
            f"Variable data produced by the Scenario Designer Agent ({len(variable_summary)} fields): {variable_summary}\n"
            f"Complete confirmed scenario context (source of truth): {json.dumps(state.scenario_context, default=str, sort_keys=True)}\n"
            f"Target industry: {profile['industry']}; country: {profile['country_name']} ({state.country}) — "
            f"regulator {profile['regulator']}, market character: {profile['market_character']}\n"
            "Validate and produce execution notes."
        )
        plan = self._llm.generate_json(_SYSTEM, prompt, temperature=0.1)
        set_orchestrator(self._cache_key_value, plan)
        if not plan.get("valid", True):
            state.errors.append(f"Orchestrator rejected: {plan.get('reason')}")
        else:
            logger.info("[Orchestrator] Accepted. Notes: %s", plan.get("execution_notes", ""))
        return state
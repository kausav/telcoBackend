"""
Agent 3 — Edge Case Agent
Detects the edge-case definitions carried on the confirmed scenario and
validates them against the schema derived by the Schema Agent: invalid
variable/condition combinations, impossible-to-satisfy scenarios, and
incorrect edge-case allocations (percentage/count). This runs before the Data
Generator Agent so a bad edge-case definition never reaches the hot
generation path.

Internally implemented as a 2-node LangGraph subgraph: detect_edge_cases ->
validate_edge_cases.
"""
from __future__ import annotations
import logging

from langgraph.graph import StateGraph, END

from agents.data_generation_agent import _condition_compatible_with_schema, _edge_case_count, _edge_case_groups
from core.dynamic_scenarios import resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState

logger = logging.getLogger(__name__)


class EdgeCaseAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm  # available for future semantic edge-case reasoning
        # Per-call scratch values, populated by _detect_edge_cases and read by
        # _validate_edge_cases. Safe because a fresh EdgeCaseAgent is instantiated
        # for each run_pipeline() call.
        self._variables: list[dict] = []
        self._edge_groups: dict[str, dict] = {}
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("detect_edge_cases", self._detect_edge_cases)
        graph.add_node("validate_edge_cases", self._validate_edge_cases)
        graph.set_entry_point("detect_edge_cases")
        graph.add_conditional_edges(
            "detect_edge_cases", self._has_edge_cases_router,
            {"validate": "validate_edge_cases", "skip": END},
        )
        graph.add_edge("validate_edge_cases", END)
        return graph.compile()

    def run(self, state: WorkflowState) -> WorkflowState:
        result = self._graph.invoke(state, config={"recursion_limit": 10})
        return result if isinstance(result, WorkflowState) else WorkflowState.model_validate(result)

    def _has_edge_cases_router(self, state: WorkflowState) -> str:
        return "validate" if self._edge_groups else "skip"

    def _detect_edge_cases(self, state: WorkflowState) -> WorkflowState:
        dyn = resolve_variables(state.scenario)
        if dyn is None:
            raise ValueError(f"Unknown scenario '{state.scenario}'")
        self._variables, _ = dyn

        if not state.edge_case_variables or state.edge_case_percentage <= 0:
            logger.info("[EdgeCaseAgent] No edge cases configured for this run; skipping.")
            self._edge_groups = {}
            return state

        self._edge_groups = _edge_case_groups(state.edge_case_variables)
        logger.info("[EdgeCaseAgent] Detected %d edge-case group(s).", len(self._edge_groups))
        return state

    def _validate_edge_cases(self, state: WorkflowState) -> WorkflowState:
        """Edge case validation layer: invalid combinations, impossible scenarios,
        and incorrect allocations, all caught before generation starts."""
        problems: list[str] = []

        if not 0.0 <= state.edge_case_percentage <= 1.0:
            problems.append(f"incorrect allocation: edgeCasePercentage {state.edge_case_percentage} is outside [0, 1]")

        edge_count = _edge_case_count(state.count, state.edge_case_percentage)
        if edge_count <= 0:
            # Floor rounding on a small requested count (e.g. 20% of 3) can legitimately
            # produce zero edge-case units; that is not a misconfiguration by itself.
            logger.info(
                "[EdgeCaseAgent] edgeCasePercentage %.4f produces 0 edge-case units out of %d requested.",
                state.edge_case_percentage, state.count,
            )

        for group_name, group in self._edge_groups.items():
            condition = str(group.get("condition") or "").strip()
            if not condition:
                problems.append(f"invalid combination: edge case '{group_name}' has no condition")
                continue
            override_names = {str(v.get("name")) for v in group.get("variables", []) if v.get("name")}
            if not _condition_compatible_with_schema(condition, self._variables, edge_override_names=override_names):
                problems.append(
                    f"impossible scenario: edge case '{group_name}' condition '{condition}' is unsatisfiable "
                    "under the declared variable schema"
                )

        if problems:
            state.errors.append("Edge case validation failed: " + "; ".join(problems))
        else:
            logger.info("[EdgeCaseAgent] Edge case validation passed for %d group(s).", len(self._edge_groups))
        return state
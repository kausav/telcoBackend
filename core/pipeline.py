"""
Generation pipeline (Gemini-powered), orchestrated as a LangGraph StateGraph.
run_pipeline() is the shared entry point used by server.py's /scenario/generate
endpoint. The Scenario Designer Agent is not part of this pipeline; it runs
earlier, at /scenario/propose time.
"""
from __future__ import annotations
import logging

from langgraph.graph import StateGraph, END

from agents.data_generation_agent import DataGenerationAgent
from agents.edge_case_agent import EdgeCaseAgent
from agents.orchestrator import OrchestratorAgent
from agents.schema_agent import SchemaAgent
from core.dynamic_scenarios import resolve_data_type, resolve_scenario_context
from core.llm_client import GeminiClient, get_default_gemini_client
from core.state import WorkflowState

logger = logging.getLogger(__name__)

_STAGES = [
    ("orchestrator", "1. Orchestrator",       OrchestratorAgent),
    ("schema",       "2. Schema",             SchemaAgent),
    ("edge_case",    "3. Edge Case",          EdgeCaseAgent),
    ("generation",   "4. Data Generation",    DataGenerationAgent),
]


def _make_node(agent, name: str):
    """Wrap an agent's run(state) -> state as a LangGraph node function."""
    def node(state: WorkflowState) -> WorkflowState:
        logger.info("━━━ Running Agent %s ━━━", name)
        return agent.run(state)
    return node


def _halt_if_errors(state: WorkflowState) -> str:
    """Conditional edge: any agent appending to state.errors stops the pipeline immediately,
    matching the previous sequential-loop behavior exactly."""
    if state.errors:
        logger.error("Pipeline halted: %s", state.errors)
        return "halt"
    return "continue"


def _build_graph(llm: GeminiClient):
    """Build a fresh graph per run so each stage's agent is bound to this call's llm client."""
    graph = StateGraph(WorkflowState)
    for node_id, name, agent_cls in _STAGES:
        graph.add_node(node_id, _make_node(agent_cls(llm), name))

    graph.set_entry_point(_STAGES[0][0])
    for (node_id, _, _), next_stage in zip(_STAGES, _STAGES[1:] + [None]):
        if next_stage is None:
            graph.add_edge(node_id, END)
        else:
            graph.add_conditional_edges(node_id, _halt_if_errors, {"continue": next_stage[0], "halt": END})

    return graph.compile()


def run_pipeline(scenario: str, count: int, industry: str = "generic", country: str | None = None, api_key: str = "", type_of_data: str | None = None, scenario_context: dict | None = None) -> WorkflowState:
    llm = get_default_gemini_client() if not api_key else GeminiClient(api_key=api_key)
    resolved_type = type_of_data or resolve_data_type(scenario)
    context = scenario_context or resolve_scenario_context(scenario)
    # The resolver may only have partial metadata; never let that override explicit function arguments.
    context["industry"] = industry
    context["country"] = country
    context["type_of_data"] = resolved_type
    state = WorkflowState(
        scenario=scenario, count=count, industry=industry, country=country,
        type_of_data=resolved_type, domain=context.get("domain", ""),
        business_scenario=context.get("business_scenario", ""),
        business_response=context.get("business_response"),
        expected_outcome=context.get("expected_outcome"),
        scenario_type=context.get("scenario_type"), use_case=context.get("use_case"),
        entity_key=context.get("entity_key"), scenario_context=context,
        edge_case_variables=context.get("edge_case_variables", []),
        edge_case_percentage=float(context.get("edge_case_percentage", 0.0) or 0.0),
    )

    graph = _build_graph(llm)
    result = graph.invoke(state, config={"recursion_limit": len(_STAGES) + 5})
    state = result if isinstance(result, WorkflowState) else WorkflowState.model_validate(result)

    if not state.final_records:
        state.final_records = state.raw_records

    return state
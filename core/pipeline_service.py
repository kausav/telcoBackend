"""Application pipeline service for scenario generation.

This module contains deterministic workflow orchestration. It is intentionally a
normal service, not an LLM agent: coordinating agents is application plumbing,
not an additional agent capability.
"""
from __future__ import annotations

import json
import logging

from agents.generator_agent import DataGeneratorAgent
from agents.qa_agent import QAAgent
from agents.rules_agent import RulesAgent
from config.industry_profiles import get_profile
from config.scenarios import SCENARIOS
from core.csv_scenario import parse_variables_csv
from core.dynamic_scenarios import (
    confirm_scenario,
    resolve_data_type,
    resolve_scenario_context,
    resolve_scenario_meta,
    scenario_exists,
)
from core.llm_client import GeminiClient
from core.runtime_cache import get_pipeline_validation, set_pipeline_validation
from core.state import WorkflowState

logger = logging.getLogger(__name__)

_VALIDATION_SYSTEM_PROMPT = """
You validate a synthetic-data scenario before rules, generation, and QA run.
The system supports any business industry and country. Validate the requested
scenario against the target industry's and country's real-world conventions
(product/plan types, regulator, market character) and return JSON with:
  - valid: boolean
  - reason: string (empty when valid)
  - execution_notes: string
Do not invent missing scenario requirements; use the confirmed scenario context
as the source of truth.
"""


def build_pipeline_validation_key(state: WorkflowState) -> tuple:
    """Build the cache key for scenario-level execution validation."""
    return (
        state.scenario,
        state.industry,
        (state.country or "GLOBAL").upper(),
        state.type_of_data,
        state.domain or "",
        state.business_scenario or "",
        state.business_response or "",
        state.expected_outcome or "",
        state.scenario_type or "",
        state.use_case or "",
        state.entity_key or "",
        str(state.scenario_context.get("events", [])),
    )


def validate_scenario_for_execution(state: WorkflowState, llm: GeminiClient) -> WorkflowState:
    """Validate the confirmed scenario before executing the generation pipeline.

    This preserves the previous pre-generation validation behavior without using
    an additional agent slot. The result is cached because it is immutable for
    a given confirmed scenario context.
    """
    logger.info("Validating scenario for execution: %s", state.scenario)

    if not scenario_exists(state.scenario):
        state.errors.append(
            f"Unknown scenario '{state.scenario}'. Valid: {list(SCENARIOS.keys())}"
        )
        return state

    cache_key = build_pipeline_validation_key(state)
    cached = get_pipeline_validation(cache_key)
    if cached is not None:
        if not cached.get("valid", True):
            state.errors.append(
                f"Scenario validation rejected: {cached.get('reason', '')}"
            )
        else:
            logger.info("Scenario validation cache hit; skipping LLM validation.")
        return state

    scenario_meta = resolve_scenario_meta(state.scenario)
    profile = get_profile(state.industry, state.country)
    prompt = (
        f"Scenario: {scenario_meta['label']}\n"
        f"Journey: {scenario_meta['journey']}\n"
        f"Description: {scenario_meta['description']}\n"
        f"Domain: {state.domain or scenario_meta.get('domain', '')}\n"
        f"Business scenario: {state.business_scenario or scenario_meta.get('business_scenario', '')}\n"
        f"Business response: {state.business_response or scenario_meta.get('business_response', '')}\n"
        f"Expected outcome: {state.expected_outcome or scenario_meta.get('expected_outcome', '')}\n"
        f"Use case: {state.use_case or scenario_meta.get('use_case', '')}\n"
        f"Scenario type: {state.scenario_type or scenario_meta.get('scenario_type', '')}\n"
        f"Records requested: {state.count}\n"
        f"Complete confirmed scenario context (source of truth): "
        f"{json.dumps(state.scenario_context, default=str, sort_keys=True)}\n"
        f"Target industry: {profile['industry']}; country: {profile['country_name']} ({state.country}) — "
        f"regulator {profile['regulator']}, market character: {profile['market_character']}\n"
        "Validate and produce execution notes."
    )

    validation = llm.generate_json(
        _VALIDATION_SYSTEM_PROMPT,
        prompt,
        temperature=0.1,
    )
    set_pipeline_validation(cache_key, validation)

    if not validation.get("valid", True):
        state.errors.append(
            f"Scenario validation rejected: {validation.get('reason', '')}"
        )
        return state

    logger.info(
        "Scenario accepted. Execution notes: %s",
        validation.get("execution_notes", ""),
    )
    return state


def register_scenario_from_csv(scenario: str, csv_path: str, label: str = "") -> None:
    """Register an industry-supplied variable-definition CSV as a scenario."""
    with open(csv_path, "r", encoding="utf-8-sig") as file:
        csv_text = file.read()
    variables, field_order = parse_variables_csv(csv_text)
    metadata = {
        "label": label or scenario,
        "journey": scenario,
        "description": f"Imported from CSV: {csv_path}",
    }
    confirm_scenario(scenario, metadata, variables, field_order)
    logger.info(
        "Registered scenario '%s' from CSV '%s' (%d variables)",
        scenario,
        csv_path,
        len(variables),
    )


def run_generation_pipeline(
    scenario: str,
    count: int,
    industry: str = "generic",
    country: str | None = None,
    api_key: str = "",
    type_of_data: str | None = None,
    scenario_context: dict | None = None,
) -> WorkflowState:
    """Run validation, rules, generation, and QA for a confirmed scenario."""
    llm = GeminiClient(api_key=api_key or None)
    resolved_type = type_of_data or resolve_data_type(scenario)
    context = dict(scenario_context or resolve_scenario_context(scenario))
    context.update({
        "industry": industry,
        "country": country,
        "type_of_data": resolved_type,
    })

    state = WorkflowState(
        scenario=scenario,
        count=count,
        industry=industry,
        country=country,
        type_of_data=resolved_type,
        domain=context.get("domain", ""),
        business_scenario=context.get("business_scenario", ""),
        business_response=context.get("business_response"),
        expected_outcome=context.get("expected_outcome"),
        scenario_type=context.get("scenario_type"),
        use_case=context.get("use_case"),
        entity_key=context.get("entity_key"),
        scenario_context=context,
        edge_case_variables=context.get("edge_case_variables", []),
        edge_case_percentage=float(
            context.get("edge_case_percentage", 0.0) or 0.0
        ),
    )

    validate_scenario_for_execution(state, llm)
    if state.errors:
        return state

    pipeline = (
        ("Rules", RulesAgent(llm)),
        ("Data Generator", DataGeneratorAgent(llm)),
        ("QA Validation", QAAgent(llm)),
    )

    for stage_name, agent in pipeline:
        if state.errors:
            logger.error(
                "Pipeline halted before %s: %s",
                stage_name,
                state.errors,
            )
            break
        logger.info("Running pipeline stage: %s", stage_name)
        state = agent.run(state)

    if not state.final_records:
        state.final_records = state.raw_records

    return state


# Backward-compatible function name for any external caller that used the
# previous service entry point. It is intentionally not used by the API.
def run_pipeline(*args, **kwargs) -> WorkflowState:
    """Backward-compatible alias for :func:`run_generation_pipeline`."""
    return run_generation_pipeline(*args, **kwargs)

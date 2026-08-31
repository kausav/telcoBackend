"""
Agent 2 — Rules Agent
Uses Gemini to articulate business rules and validation invariants for the
selected scenario. Downstream QA agent uses these rules to validate records.
"""
from __future__ import annotations
import logging

from config.industry_profiles import get_profile
from config.variables import VARIABLES
from core.dynamic_scenarios import resolve_scenario_meta, resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState
from core.runtime_cache import get_rules, set_rules

logger = logging.getLogger(__name__)

_SYSTEM = """
You are the Rules Agent for a synthetic data generation pipeline covering any
business industry (telecom, banking, retail, healthcare, etc.).
Given a scenario, its variable definitions, and the target industry's and
country's real-world conventions (currency, product/plan types, regulator,
market character), produce a precise JSON rules document consistent with that
industry and country's actual standards.

Return a JSON object with:
  - "scenario_summary": str
  - "business_rules": [str]   — logical invariants that MUST hold across all records
  - "field_constraints": {field_name: {description: str, valid_values: str, nullable: bool}}
  - "cross_field_rules": [str] — mathematical / temporal dependencies between fields
  - "generation_constraints": {field_name: {valid_values: [str], min: number, max: number,
      preferred_values: [str]}} — machine-readable constraints the generator should use
  - "formula_rules": [{"field": str, "expression": str}] — authoritative formulas to
      calculate and validate generated fields
"""


class RulesAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info("[RulesAgent] Deriving rules for scenario=%s", state.scenario)

        cache_key = (
            state.scenario, state.industry, (state.country or "GLOBAL").upper(), state.type_of_data,
            state.domain or "", state.business_scenario or "", state.business_response or "",
            state.expected_outcome or "", state.scenario_type or "", state.use_case or "",
            state.entity_key or "", str(state.scenario_context.get("events", [])),
        )
        cached = get_rules(cache_key)
        if cached is not None:
            state.rules = cached
            logger.info("[RulesAgent] Cache hit; skipping LLM rules generation.")
            return state

        sc = resolve_scenario_meta(state.scenario)
        dyn = resolve_variables(state.scenario)
        if dyn is not None:
            VARS, _ = dyn
        else:
            VARS = VARIABLES

        field_summary = [
            {"name": v["name"], "dtype": v["dtype"], "gen": v["gen"],
             "depends_on": v.get("depends_on", [])}
            for v in VARS
        ]

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
            f"Target industry: {profile['industry']}; country: {profile['country_name']} ({state.country})\n"
            f"Currency: {profile['currency']}; Regulator: {profile['regulator']}\n"
            f"Country-appropriate payment methods: {profile.get('payment_methods', [])}\n"
            f"Market character: {profile['market_character']}\n"
            f"Output data type: {state.type_of_data}\n"
            f"Transactional events: {sc.get('events', [])}\n"
            f"Typical product/plan types: {profile['product_types']}\n"
            f"Variables: {field_summary}\n\n"
            "IMPORTANT: generation_constraints and formula_rules are machine-readable and must be "
            "derived from the supplied scenario/use case, not generic filler. Do not invent a constraint "
            "unless it is supported by the scenario, industry, country, or explicit variable definition.\n"
            f"Produce a complete rules document covering all {len(VARS)} variables, "
            f"consistent with this industry and country's real-world standards. "
            f"For transactional output, also validate event_sequence is increasing within each journey, "
            f"transaction_id is unique, and event_timestamp is non-decreasing within each journey."
        )

        rules = self._llm.generate_json(_SYSTEM, prompt, temperature=0.1)
        if not isinstance(rules, dict):
            rules = {}

        # Variable definitions are authoritative. Always preserve their formulas
        # as machine-readable rules so validation/generation cannot lose a formula
        # because the LLM omitted it from its response.
        formula_rules = list(rules.get("formula_rules", []) or [])
        existing_formula_fields = {
            str(item.get("field")) for item in formula_rules
            if isinstance(item, dict) and item.get("field")
        }
        for var in VARS:
            field = str(var.get("name", ""))
            expression = var.get("formula")
            if field and expression and field not in existing_formula_fields:
                formula_rules.append({"field": field, "expression": str(expression)})
        rules["formula_rules"] = formula_rules

        # Ensure the generation constraints contain the explicit scenario choices
        # and bounds as a safe baseline; LLM/use-case constraints may add to these.
        generation_constraints = rules.get("generation_constraints", {})
        if not isinstance(generation_constraints, dict):
            generation_constraints = {}
        for var in VARS:
            field = str(var.get("name", ""))
            if not field:
                continue
            gc = generation_constraints.get(field)
            if not isinstance(gc, dict):
                gc = {}
            params = var.get("params") if isinstance(var.get("params"), dict) else {}
            if "choices" in params and "valid_values" not in gc:
                gc["valid_values"] = list(params.get("choices") or [])
            if "min" in params and "min" not in gc:
                gc["min"] = params.get("min")
            if "max" in params and "max" not in gc:
                gc["max"] = params.get("max")
            generation_constraints[field] = gc
        rules["generation_constraints"] = generation_constraints

        set_rules(cache_key, rules)
        state.rules = rules
        logger.info("[RulesAgent] Rules generated. Business rules: %d cross-field: %d",
                    len(rules.get("business_rules", [])),
                    len(rules.get("cross_field_rules", [])))
        return state

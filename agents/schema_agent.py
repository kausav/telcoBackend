"""
Agent 2 — Schema Agent
Uses Gemini to articulate business rules, field constraints and formulas for the
selected scenario, then deterministically validates the derived schema for
missing variables, missing rules, and incomplete logic before edge-case
handling or generation ever run.

Internally implemented as a 2-node LangGraph subgraph: derive_schema ->
validate_schema. A failed validation appends to state.errors, which the outer
pipeline graph (core/pipeline.py) uses to halt before the Edge Case Agent runs.
"""
from __future__ import annotations
import ast
import logging
import json

from langgraph.graph import StateGraph, END

from config.industry_profiles import get_profile
from core.dynamic_scenarios import resolve_scenario_meta, resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState
from core.runtime_cache import get_schema, set_schema

logger = logging.getLogger(__name__)

_SYSTEM = """
You are the Schema Agent for a synthetic data generation pipeline covering any
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
  - "field_relationships": [{"controller_field": str, "dependent_field": str, "mapping": {controller_value: [allowed_dependent_values]}}] — executable categorical dependencies
  - "generation_constraints": {field_name: {valid_values: [str], min: number, max: number,
      preferred_values: [str]}} — machine-readable constraints the generator should use
  - "formula_rules": [{"field": str, "expression": str}] — authoritative formulas to
      calculate and validate generated fields

SEMANTIC ACCURACY IS MANDATORY:
- Every field and every categorical value must be relevant to the target industry,
  country, domain, and scenario.
- Reject placeholder values such as Provider_A, Company_A, Product_A, Gateway_A,
  Synthetic_Provider, or similar when the field represents a real-world entity.
- Use real-world industry/country vocabulary supplied in the profile whenever available.
- Do not treat syntactic validity as semantic validity. A value that fits a datatype is
  still invalid if it does not make business sense for the target industry.
- Preserve the scenario variable definitions as the source of truth, and make
  generation_constraints reflect their actual business vocabulary.
- Check cross-field semantics, not only individual fields. If two fields describe
  related business states, explicitly encode the dependency in cross_field_rules.
- Perform a VALUE-BY-VALUE INDUSTRY AUDIT before returning rules: every categorical
  value must belong to the target industry's real vocabulary. Do not accept a value
  merely because it is syntactically valid or common in another industry.
- REAL ENTITY FIELDS require real target-industry/country entities. For example, a
  telecom service_provider must be an actual telecom operator; a banking provider
  must be a bank/payment institution when such a field is appropriate; a retail
  provider/merchant must be retail-appropriate. Never use Provider_A, Company_A,
  telecom brands in banking/retail/etc., or any other cross-industry placeholder.
- If the supplied profile has no authoritative entity list for a field, do NOT borrow
  an entity list from another industry. Use scenario-supported domain vocabulary or
  leave the field unconstrained rather than introducing unrelated entities.
- Treat semantic relevance as a HARD validation rule. If a variable/value fails the
  industry + country + scenario audit, reject or replace it before generation.
- Encode important state dependencies explicitly. Example: if a scenario states that
  transaction failure causes recharge failure, encode that as a machine-checkable
  conditional rule; do not leave the relationship only in prose.
- FIELD RELATIONSHIP REQUIREMENT: whenever one categorical field determines or constrains
  another categorical field (provider -> plan, product -> product-specific attribute,
  account type -> account behavior, order state -> fulfillment state, etc.), emit a
  field_relationships entry with an explicit mapping. The generator will execute this
  mapping, so do not leave the dependency only in cross_field_rules prose.
- Do not invent impossible combinations merely because each value is individually valid.
  Validate complete RECORDS and ENTITY PROFILES, not isolated columns.
"""


class SchemaAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm
        # Per-call scratch value, populated by _derive_schema and read by _validate_schema.
        # Safe because a fresh SchemaAgent is instantiated for each run_pipeline() call.
        self._variables: list[dict] = []
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("derive_schema", self._derive_schema)
        graph.add_node("validate_schema", self._validate_schema)
        graph.set_entry_point("derive_schema")
        graph.add_edge("derive_schema", "validate_schema")
        graph.add_edge("validate_schema", END)
        return graph.compile()

    def run(self, state: WorkflowState) -> WorkflowState:
        result = self._graph.invoke(state, config={"recursion_limit": 10})
        return result if isinstance(result, WorkflowState) else WorkflowState.model_validate(result)

    def _cache_key(self, state: WorkflowState) -> tuple:
        return (
            state.scenario, state.industry, (state.country or "GLOBAL").upper(), state.type_of_data,
            state.domain or "", state.business_scenario or "", state.business_response or "",
            state.expected_outcome or "", state.scenario_type or "", state.use_case or "",
            state.entity_key or "", str(state.scenario_context.get("events", [])),
        )

    def _derive_schema(self, state: WorkflowState) -> WorkflowState:
        logger.info("[SchemaAgent] Deriving schema for scenario=%s", state.scenario)

        dyn = resolve_variables(state.scenario)
        if dyn is None:
            raise ValueError(f"Unknown scenario '{state.scenario}'")
        VARS, _ = dyn
        self._variables = VARS

        cache_key = self._cache_key(state)
        cached = get_schema(cache_key)
        if cached is not None:
            state.rules = cached
            logger.info("[SchemaAgent] Cache hit; skipping LLM schema derivation.")
            return state

        sc = resolve_scenario_meta(state.scenario)
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
            f"Industry-appropriate service providers/operators (when applicable): {profile.get('service_providers', [])}\n"
            f"Market character: {profile['market_character']}\n"
            f"Output data type: {state.type_of_data}\n"
            f"Transactional events: {sc.get('events', [])}\n"
            f"Typical product/plan types: {profile['product_types']}\n"
            f"Variables: {field_summary}\n"
            f"Complete confirmed scenario context (source of truth): {json.dumps(state.scenario_context, default=str, sort_keys=True)}\n\n"
            "IMPORTANT: generation_constraints, cross_field_rules, and formula_rules are machine-readable and must be "
            "derived from the supplied scenario/use case, not generic filler. Do not invent a constraint "
            "unless it is supported by the scenario, industry, country, or explicit variable definition. "
            "Before returning, audit every field AND every categorical value for industry/country relevance. "
            "A syntactically valid value from another industry is still invalid and must be replaced. "
            "Do not use telecom entities as generic providers for non-telecom industries. "
            "For every meaningful relationship between business-state fields, add an explicit conditional rule "
            "that the data generator can enforce.\n"
            f"Produce a complete rules document covering all {len(VARS)} variables, "
            f"consistent with this industry and country's real-world standards. "
            f"For transactional output, also validate event_sequence is increasing within each journey, "
            f"transaction_id is unique, and event_timestamp is non-decreasing within each journey."
        )

        rules = self._llm.generate_json(_SYSTEM, prompt, temperature=0.1)
        if not isinstance(rules, dict):
            rules = {}
        if not isinstance(rules.get("field_relationships"), list):
            rules["field_relationships"] = []

        # Variable definitions are authoritative. Always preserve their formulas
        # as machine-readable rules so validation/generation cannot lose a formula
        # because the LLM omitted it from its response.
        formula_rules = list(rules.get("formula_rules", []) or [])
        # Variable definitions are authoritative. If Gemini supplied a formula_rule
        # for the same field, replace it with the explicit variable formula rather
        # than allowing a malformed/alternate LLM expression to win.
        authoritative_formulas = {
            str(var.get("name")): str(var.get("formula"))
            for var in VARS
            if var.get("name") and var.get("formula")
        }
        formula_rules = [
            item for item in formula_rules
            if not (isinstance(item, dict) and str(item.get("field", "")) in authoritative_formulas)
        ]
        formula_rules.extend(
            {"field": field, "expression": expression}
            for field, expression in authoritative_formulas.items()
        )
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

        set_schema(cache_key, rules)
        state.rules = rules
        logger.info("[SchemaAgent] Schema generated. Business rules: %d cross-field: %d",
                    len(rules.get("business_rules", [])),
                    len(rules.get("cross_field_rules", [])))
        return state

    def _validate_schema(self, state: WorkflowState) -> WorkflowState:
        """Schema validation layer: catch missing variables, missing rules, and
        incomplete logic (formulas referencing undefined fields) before any
        downstream agent trusts state.rules."""
        variables = self._variables
        var_names = {str(v.get("name")) for v in variables if v.get("name")}
        rules = state.rules or {}
        problems: list[str] = []

        # Missing variable: a declared dependency that doesn't exist in the schema.
        for v in variables:
            for dep in v.get("depends_on", []) or []:
                if str(dep) not in var_names:
                    problems.append(f"variable '{v.get('name')}' depends_on missing variable '{dep}'")

        # Missing rule: a variable declares a formula but has no formula_rules entry.
        formula_fields = {
            str(item.get("field")) for item in rules.get("formula_rules", []) or []
            if isinstance(item, dict) and item.get("field")
        }
        for v in variables:
            if v.get("formula") and str(v.get("name")) not in formula_fields:
                problems.append(f"variable '{v.get('name')}' has a formula but no formula_rules entry")

        # Incomplete logic: a formula expression that doesn't parse, or that
        # references a variable outside the declared schema. Only variable-declared
        # formulas are authoritative and strictly checked here; formula_rules entries
        # the LLM added on its own (with no matching variable.formula) are supplementary
        # and already degrade gracefully downstream (QA skips an unevaluable formula
        # instead of failing the record), so they are not a hard gate.
        authoritative_fields = {str(v.get("name")) for v in variables if v.get("formula")}
        for item in rules.get("formula_rules", []) or []:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field", ""))
            if field not in authoritative_fields:
                continue
            expression = str(item.get("expression", ""))
            try:
                tree = ast.parse(expression, mode="eval")
                refs = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            except SyntaxError as exc:
                problems.append(f"formula for '{field}' is not a valid expression: {exc}")
                continue
            unknown = sorted(
                name for name in refs
                if name not in var_names and name not in {"round", "min", "max", "abs", "sum"}
            )
            if unknown:
                problems.append(f"formula for '{field}' references undefined variable(s): {unknown}")

        if problems:
            state.errors.append("Schema validation failed: " + "; ".join(problems))
        else:
            logger.info("[SchemaAgent] Schema validation passed for %d variables.", len(variables))
        return state
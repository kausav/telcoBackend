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
"""


class RulesAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info("[RulesAgent] Deriving rules for scenario=%s", state.scenario)

        cache_key = (state.scenario, state.industry, (state.country or "GLOBAL").upper(), state.type_of_data)
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
            f"Target industry: {profile['industry']}; country: {profile['country_name']} ({state.country})\n"
            f"Currency: {profile['currency']}; Regulator: {profile['regulator']}\n"
            f"Country-appropriate payment methods: {profile.get('payment_methods', [])}\n"
            f"Market character: {profile['market_character']}\n"
            f"Output data type: {state.type_of_data}\n"
            f"Transactional events: {sc.get('events', [])}\n"
            f"Typical product/plan types: {profile['product_types']}\n"
            f"Variables: {field_summary}\n\n"
            f"Produce a complete rules document covering all {len(VARS)} variables, "
            f"consistent with this industry and country's real-world standards. "
            f"For transactional output, also validate event_sequence is increasing within each journey, "
            f"transaction_id is unique, and event_timestamp is non-decreasing within each journey."
        )

        rules = self._llm.generate_json(_SYSTEM, prompt, temperature=0.1)
        set_rules(cache_key, rules)
        state.rules = rules
        logger.info("[RulesAgent] Rules generated. Business rules: %d cross-field: %d",
                    len(rules.get("business_rules", [])),
                    len(rules.get("cross_field_rules", [])))
        return state

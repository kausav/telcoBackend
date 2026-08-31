"""
Telco Agentic SDG — Scenario-based 4-Agent Pipeline (Gemini-powered)

Usage:
  python main.py --scenario LB-01 --count 100

Set GEMINI_API_KEY in .env before running.
"""
from __future__ import annotations
import argparse
import json
import logging

from agents.generator_agent import DataGeneratorAgent
from agents.orchestrator import OrchestratorAgent
from agents.qa_agent import QAAgent
from agents.rules_agent import RulesAgent
from config.scenarios import SCENARIO_LABELS
from core.csv_scenario import parse_variables_csv
from core.dynamic_scenarios import confirm_scenario, resolve_data_type, resolve_scenario_context
from core.llm_client import GeminiClient
from core.state import WorkflowState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telco Agentic SDG")
    p.add_argument("--scenario", required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--industry", default="generic", help="Industry name driving industry conventions (see config/industry_profiles.py)")
    p.add_argument("--country", default=None, help="ISO 3166-1 alpha-2 country code; omit for GLOBAL/non-country-specific data")
    p.add_argument("--variables-csv", default="", help="Path to an industry-supplied variable-definition CSV (see core/csv_scenario.py); "
                                                        "registers --scenario from this CSV instead of using the LLM-invented catalog")
    p.add_argument("--api-key", default="", dest="api_key")
    return p.parse_args()


def register_scenario_from_csv(scenario: str, csv_path: str, label: str = "") -> None:
    """Parse an industry-supplied CSV and register it in the dynamic scenario store
    under `scenario`, so the rest of the pipeline (orchestrator/rules/generator/qa)
    can resolve it exactly like an LLM-proposed-and-confirmed scenario."""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        csv_text = f.read()
    variables, field_order = parse_variables_csv(csv_text)
    meta = {"label": label or scenario, "journey": scenario, "description": f"Imported from CSV: {csv_path}"}
    confirm_scenario(scenario, meta, variables, field_order)
    logger.info("Registered scenario '%s' from CSV '%s' (%d variables)", scenario, csv_path, len(variables))


def run_pipeline(scenario: str, count: int, industry: str = "generic", country: str | None = None, api_key: str = "", type_of_data: str | None = None, scenario_context: dict | None = None) -> WorkflowState:
    llm = GeminiClient(api_key=api_key or None)
    resolved_type = type_of_data or resolve_data_type(scenario)
    context = scenario_context or resolve_scenario_context(scenario)
    # For static/CLI scenarios the resolver may only have partial metadata;
    # never let that override explicit function arguments.
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
    )

    pipeline = [
        ("1. Orchestrator",   OrchestratorAgent(llm)),
        ("2. Rules",          RulesAgent(llm)),
        ("3. Data Generator", DataGeneratorAgent(llm)),
        ("4. QA Validation",  QAAgent(llm)),
    ]

    for name, agent in pipeline:
        if state.errors:
            logger.error("Pipeline halted before %s: %s", name, state.errors)
            break
        logger.info("━━━ Running Agent %s ━━━", name)
        state = agent.run(state)

    if not state.final_records:
        state.final_records = state.raw_records

    return state


def main() -> None:
    args = _parse_args()
    logger.info("═" * 60)
    logger.info("  Scenario : %s", SCENARIO_LABELS.get(args.scenario, args.scenario))
    logger.info("  Count    : %d", args.count)
    logger.info("═" * 60)

    if args.variables_csv:
        register_scenario_from_csv(args.scenario, args.variables_csv)

    state = run_pipeline(scenario=args.scenario, count=args.count, industry=args.industry, country=args.country, api_key=args.api_key)

    for e in state.errors:
        logger.warning("  • %s", e)

    logger.info("═" * 60)
    logger.info("  Final records    : %d", len(state.final_records))
    logger.info("  Validation report: %s", json.dumps(state.validation_report))
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
"""MongoDB-backed store for drafts, confirmed scenarios and feedback."""
from __future__ import annotations
from typing import Any
from models.scenario import ScenarioModel
from models.scenario_draft import ScenarioDraftModel
from models.scenario_feedback import ScenarioFeedbackModel
from core.scenario_variable_store import get_recommended
from core.industry_source_store import has_sources


def new_draft_id() -> str:
    return ScenarioDraftModel.new_id()

def save_draft(draft_id: str, data: dict[str, Any]) -> None:
    ScenarioDraftModel.save(draft_id, data)

def get_draft(draft_id: str) -> dict[str, Any] | None:
    return ScenarioDraftModel.get(draft_id)

def pop_draft(draft_id: str) -> dict[str, Any] | None:
    return ScenarioDraftModel.pop(draft_id)

def confirm_scenario(requested_scenario_id: str | None, meta: dict[str, Any], variables: list[dict[str, Any]], field_order: list[str], draft_id: str | None = None) -> tuple[str, bool]:
    requested_id = str(requested_scenario_id or "").strip() or None
    return ScenarioModel.create_with_allocation(requested_id, draft_id, meta, variables, field_order)

def resolve_requested_scenario_id_from_draft(draft_id: str) -> str | None:
    return ScenarioModel.by_draft_id(draft_id)

def get_confirmed(requested_scenario_id: str) -> dict[str, Any] | None:
    return ScenarioModel.get(requested_scenario_id)

def scenario_exists(requested_scenario_id: str) -> bool:
    return ScenarioModel.exists(requested_scenario_id)

def resolve_scenario_meta(requested_scenario_id: str) -> dict[str, Any] | None:
    dyn = get_confirmed(requested_scenario_id)
    return dyn["meta"] if dyn else None

def resolve_variables(requested_scenario_id: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return the confirmed executable variables exactly as persisted."""
    dyn = get_confirmed(requested_scenario_id)
    if not dyn:
        return None

    raw_variables = [dict(v) for v in (dyn.get("variables") or []) if isinstance(v, dict)]
    meta = dyn.get("meta") or {}
    domain = meta.get("domain") or meta.get("journey") or ""
    industry = meta.get("industry") or "generic"
    source_available = bool(domain and has_sources(industry, domain))
    scenario_version = int(meta.get("scenario_version", 1) or 1)
    source_policy = str(meta.get("source_policy") or "").strip()

    # A confirmed agentic JSON-grounded scenario already contains the immutable executable
    # source-backed contract selected during proposal/confirmation. Do not re-query the live
    # source registry on every generation call; a source document may legitimately be
    # replaced/deactivated after confirmation while the approved scenario must remain
    # reproducible. Legacy scenarios without this policy continue to use the live check.
    if bool(meta.get("agentic")) and source_policy == "mongodb_industry_source_documents":
        source_available = True

    # When an exact industry/domain JSON source is not registered, scenario_variables is the
    # only supported alternate variable source. A confirmed scenario may already contain the
    # same definitions (the normal path), but if it does not, use the enabled recommendations
    # persisted for this scenario/version rather than falling back to static files or legacy
    # industry vocabulary. The generation service performs the fail-closed source gate before
    # execution, so reaching this branch with no DB variables is treated as empty.
    recommended_fallback = []
    if not source_available:
        recommended_fallback = [dict(v) for v in get_recommended(requested_scenario_id, scenario_version) if isinstance(v, dict)]
        if recommended_fallback:
            raw_variables = recommended_fallback
            source_policy = "scenario_variables"

    # Confirmed variables are the executable contract: they were compiled from the registered source
    # documents or persisted scenario variables, or explicitly defined, and are never rewritten here.
    variables = raw_variables
    allowed = {str(v.get("name")) for v in variables if isinstance(v, dict) and v.get("name")}
    candidate_order = (
        [str(v.get("name")) for v in recommended_fallback if str(v.get("name") or "").strip()]
        if recommended_fallback
        else list(dyn.get("field_order") or [])
    )
    field_order = [name for name in candidate_order if str(name) in allowed]
    return variables, field_order


def resolve_scenario_context(requested_scenario_id: str) -> dict[str, Any]:
    """Return the complete confirmed scenario context used by generation agents."""
    meta = resolve_scenario_meta(requested_scenario_id) or {}
    return {
        "scenario_id": requested_scenario_id,
        "requested_scenario_id": meta.get("requested_scenario_id", requested_scenario_id),
        "label": meta.get("label", requested_scenario_id),
        "journey": meta.get("journey", ""),
        "description": meta.get("description", ""),
        "domain": meta.get("domain", ""),
        "business_scenario": meta.get("business_scenario", ""),
        "business_response": meta.get("business_response"),
        "expected_outcome": meta.get("expected_outcome"),
        "scenario_type": meta.get("scenario_type"),
        "use_case": meta.get("use_case"),
        "industry": meta.get("industry", "generic"),
        "country": meta.get("country"),
        "type_of_data": meta.get("type_of_data", "aggregational"),
        "entity_key": meta.get("entity_key"),
        "variables": [dict(v) for v in (resolve_variables(requested_scenario_id) or ([], []))[0] if isinstance(v, dict)],
        "records_per_user": int(meta.get("records_per_user", 10) or 10),
        "agentic": bool(meta.get("agentic", False)),
        "scenario_version": int(meta.get("scenario_version", 1) or 1),
        "variable_sources": dict(meta.get("variable_sources") or {}),
        "source_policy": str(meta.get("source_policy") or "").strip(),
        "variable_source_ids": {
            str(name).strip().casefold(): str(source_id).strip()
            for name, source_id in (meta.get("variable_source_ids") or {}).items()
            if str(name).strip() and str(source_id).strip()
        },
        "db_variable_names": sorted(set(meta.get("db_variable_names") or []) | set((meta.get("db_variable_definitions") or {}).keys())),
        "db_variable_definitions": dict(meta.get("db_variable_definitions") or {}),
        "behavioral_rules": list(meta.get("behavioral_rules") or []),
        "generation_spec_key": meta.get("generation_spec_key"),
    }


def resolve_data_type(requested_scenario_id: str) -> str:
    """Return the persisted data type for a scenario.

    Old scenarios created before typeOfData was introduced are treated as
    aggregational so existing scenarios continue to work unchanged.
    """
    meta = resolve_scenario_meta(requested_scenario_id) or {}
    value = str(meta.get("type_of_data", "aggregational")).strip().lower()
    return value if value in {"transactional", "aggregational"} else "aggregational"


def resolve_entity_key(requested_scenario_id: str) -> str | None:
    meta = resolve_scenario_meta(requested_scenario_id) or {}
    value = meta.get("entity_key")
    return str(value) if value else None


def list_scenarios() -> list[dict[str, Any]]:
    return ScenarioModel.list()


def add_feedback(requested_scenario_id: str, domain: str, business_scenario: str, feedback: str) -> None:
    if not feedback:
        return
    requested = str(requested_scenario_id or "").strip()
    if not requested:
        raise ValueError("requested_scenario_id is required")
    ScenarioFeedbackModel.add(requested, domain, business_scenario, feedback)


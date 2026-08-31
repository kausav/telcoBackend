from __future__ import annotations
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from main import run_pipeline
from agents.scenario_designer_agent import ScenarioDesignerAgent
from core.csv_scenario import parse_definition_csv, parse_variables_csv
from core.dynamic_scenarios import (
    add_feedback,
    confirm_scenario,
    get_draft,
    new_draft_id,
    next_scenario_id,
    pop_draft,
    resolve_scenario_id_from_draft,
    resolve_scenario_meta,
    resolve_data_type,
    resolve_scenario_context,
    resolve_variables,
    save_draft,
    scenario_exists,
)
from core.llm_client import GeminiClient
from core.compiled_schema import invalidate_scenario
from core.runtime_cache import clear_scenario


def _get_field_order(scenario: str) -> list[str]:
    """Return the field order for any scenario without importing at module level."""
    dyn = resolve_variables(scenario)
    if dyn is not None:
        return dyn[1]
    from config.variables import FIELD_ORDER
    return FIELD_ORDER


def _is_placeholder(value) -> bool:
    """Treat Swagger's default "string" placeholder, blanks, and null as "no value"."""
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "string"))


def _clean_dict(d: dict) -> dict:
    """Drop keys whose value is a placeholder, recursing into nested dicts."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _clean_dict(v)
            if not v:
                continue
        elif _is_placeholder(v):
            continue
        cleaned[k] = v
    return cleaned

logger = logging.getLogger(__name__)
app = FastAPI(title="Telco Agentic SDG", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    scenario: str | None = Field(None, examples=["LB-01"])
    draftId: str | None = Field(None, description="Confirmed draft id; disambiguates when scenario ids collide across users")
    count: int = Field(20, ge=1, le=5000)


class GenerateResponse(BaseModel):
    scenario_id: str
    typeOfData: Literal["transactional", "aggregational"]
    events: list[dict] = Field(default_factory=list)
    requested_scenario_id: str | None = Field(None, description="The scenarioId originally requested at /scenario/propose time, persisted from confirm — may differ from scenario_id if it was reassigned due to a collision")
    draft_id: str | None = None
    scenario_label: str
    fields: list[str]
    total_records: int
    validation_report: dict
    records: list[dict]
    entityKey: str | None = Field(None, description="Field used to group transactional records by business entity")
    totalCount: int = Field(0, description="Total number of unique entities in the response dataset")
    eventData: list[dict] = Field(default_factory=list, description="Deprecated compatibility field; transactional data is grouped under records by entityKey")
    errors: list[str]
    edgeCasePercentage: float = Field(0.0, ge=0.0, le=1.0)


class ProposeRequest(BaseModel):
    scenarioId: str = Field(..., examples=["LB-04"])
    scenarioType: str = Field(..., examples=["Normal", "No Response", "Customer Declines"])
    industryType: str = Field(..., examples=["Telecom"])
    domain: str = Field(..., examples=["Low Balance & Top-up"])
    businessScenario: str = Field(..., examples=["Balance reaches the defined low-balance threshold"])
    businessResponse: str | None = Field(None, examples=["Recognise the situation"])
    expectedOutcome: str | None = Field(None, examples=["Potential recharge opportunity identified"])
    country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code; if omitted, generic/global (non-country-specific) conventions are used instead of assuming a country", examples=["US", "IN", "GB", "AE"])
    useCase: str | None = Field(None, description="Specific use case within the domain, gives the designer sharper context than domain alone", examples=["Proactive low-balance recharge nudge"])
    typeOfData: Literal["transactional", "aggregational"] = Field(..., description="Output data grain: one journey-level record or multiple event/transaction records per entity")


class ProposeResponse(BaseModel):
    draft_id: str
    scenario_id: str
    scenario_id_available: bool = Field(True, description="False if this scenarioId is already confirmed under a different draft — /scenario/confirm will mint a new id in that case")
    label: str
    journey: str
    description: str
    variables: list[dict]
    field_order: list[str]
    typeOfData: Literal["transactional", "aggregational"]
    events: list[dict] = Field(default_factory=list)
    edgeCaseVariables: list[dict] = Field(default_factory=list)


class VariableEdit(BaseModel):
    name: str
    changes: dict


class EventEdit(BaseModel):
    event_type: str
    changes: dict


class ConfirmRequest(BaseModel):
    draft_id: str

    # Variable-level changes (supported for all scenario types).
    add: list[dict] = Field(default_factory=list, description="New variable definitions to add")
    edit: list[VariableEdit] = Field(default_factory=list, description="Existing variables to edit by name")
    delete: list[str] = Field(default_factory=list, description="Variable names to delete")

    # Transactional event-level changes. Event changes are intentionally separate
    # from variable changes so the API contract is explicit about the two grains.
    eventAdd: list[dict] = Field(default_factory=list, description="Transactional events to add")
    eventEdit: list[EventEdit] = Field(default_factory=list, description="Transactional events to edit by event_type")
    eventDelete: list[str] = Field(default_factory=list, description="Transactional event_type values to delete")

    # Edge-case variable changes are supported for both data types.
    edgeCaseAdd: list[dict] = Field(default_factory=list, description="Edge-case variable definitions to add")
    edgeCaseEdit: list[VariableEdit] = Field(default_factory=list, description="Edge-case variables to edit by name")
    edgeCaseDelete: list[str] = Field(default_factory=list, description="Edge-case variable names to delete")
    edgeCasePercentage: float | None = Field(None, ge=0.0, le=1.0, description="Fraction of generated records/entities that must be edge-case data (0.02 = 2%)")

    feedback: str | None = None


class ConfirmResponse(BaseModel):
    scenario_id: str
    requested_scenario_id: str | None = Field(None, description="The scenarioId originally requested at propose time, for comparison")
    scenario_id_reassigned: bool = Field(False, description="True if scenario_id differs from requested_scenario_id because the requested id was already confirmed under a different draft")
    draft_id: str
    label: str
    journey: str
    description: str
    variables: list[dict]
    field_order: list[str]
    typeOfData: Literal["transactional", "aggregational"]
    events: list[dict] = Field(default_factory=list)
    edgeCaseVariables: list[dict] = Field(default_factory=list)
    edgeCasePercentage: float = Field(0.0, ge=0.0, le=1.0)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scenario/propose", response_model=ProposeResponse)
def propose_scenario(req: ProposeRequest):
    """API 1 — ask the designer agent to invent a new scenario + variable catalog."""
    # scenarioId is just the user's preferred label at this stage; it's not reserved here
    # because two different users can propose the same one concurrently. draftId (minted
    # below) is the only unique handle — /scenario/confirm resolves any scenarioId clash.
    try:
        llm = GeminiClient()
        agent = ScenarioDesignerAgent(llm)
        draft = agent.propose(
            industry_type=req.industryType,
            domain=req.domain,
            business_scenario=req.businessScenario,
            business_response=req.businessResponse,
            expected_outcome=req.expectedOutcome,
            scenario_id=req.scenarioId,
            scenario_type=req.scenarioType,
            country=req.country,
            use_case=req.useCase,
            type_of_data=req.typeOfData,
        )
    except EnvironmentError as exc:
        raise HTTPException(500, detail={"error": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(502, detail={"error": f"Scenario proposal failed: {exc}"}) from exc

    draft_id = new_draft_id()
    draft["domain"] = req.domain
    draft["business_scenario"] = req.businessScenario
    draft["scenario_id"] = req.scenarioId
    draft["scenario_type"] = req.scenarioType
    draft["industry_type"] = req.industryType
    draft["country"] = req.country
    draft["use_case"] = req.useCase
    draft["business_response"] = req.businessResponse
    draft["expected_outcome"] = req.expectedOutcome
    draft["type_of_data"] = req.typeOfData
    draft.setdefault("edge_case_variables", [])
    draft.setdefault("edge_case_percentage", 0.0)
    save_draft(draft_id, draft)
    scenario_id_available = not scenario_exists(req.scenarioId)
    if not scenario_id_available:
        logger.warning(
            "[propose] scenarioId '%s' is already confirmed under a different draft; "
            "/scenario/confirm for draft_id=%s will mint a new id unless that scenario_id is freed up.",
            req.scenarioId, draft_id,
        )
    return ProposeResponse(
        draft_id=draft_id,
        scenario_id=req.scenarioId,
        scenario_id_available=scenario_id_available,
        label=draft.get("label", ""),
        journey=draft.get("journey", req.domain),
        description=draft.get("description", ""),
        variables=draft.get("variables", []),
        field_order=draft.get("field_order", []),
        typeOfData=draft.get("type_of_data", "aggregational"),
        events=draft.get("events", []),
        edgeCaseVariables=draft.get("edge_case_variables", []),
    )


@app.post("/scenario/import-csv", response_model=ProposeResponse)
def import_scenario_csv(
    file: UploadFile = File(..., description="CSV scenario definition: variables and, for transactional data, events"),
    scenarioId: str = Form(...),
    domain: str = Form(...),
    typeOfData: Literal["transactional", "aggregational"] = Form(...),
    industryType: str = Form("generic"),
    country: str | None = Form(None),
    businessScenario: str = Form(""),
    businessResponse: str | None = Form(None),
    expectedOutcome: str | None = Form(None),
    scenarioType: str = Form(""),
    useCase: str | None = Form(None),
    label: str = Form(""),
    entityKey: str | None = Form(None),
    edgeCasePercentage: float | None = Form(None),
):
    """CSV alternative to /scenario/propose.

    The uploaded CSV is a *scenario definition*, not sample/output data. It tells
    the backend which variables the user wants and, for transactional scenarios,
    which events and event fields the user wants. The resulting draft is stored in
    exactly the same draft store consumed by /scenario/confirm and /scenario/generate.
    """
    raw = file.file.read()
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, detail={"error": f"CSV must be UTF-8 encoded: {exc}"}) from exc

    try:
        variables, field_order, events, edge_case_variables, csv_metadata = parse_definition_csv(
            csv_text, type_of_data=typeOfData
        )
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc

    if typeOfData == "transactional":
        if not entityKey:
            raise HTTPException(400, detail={
                "error": "entityKey is required for transactional CSV imports",
                "example": "subscriber_id",
            })
        if entityKey not in {v["name"] for v in variables}:
            raise HTTPException(400, detail={
                "error": f"entityKey '{entityKey}' must be one of the CSV variable names"
            })
    else:
        entityKey = None

    # CSV can define edgeCasePercentage; an explicit form value overrides it.
    import math
    edge_case_percentage = float(csv_metadata.get("edge_case_percentage", 0.0) or 0.0)
    if edgeCasePercentage is not None:
        edge_case_percentage = edgeCasePercentage
    if not math.isfinite(edge_case_percentage) or not 0.0 <= edge_case_percentage <= 1.0:
        raise HTTPException(400, detail={
            "error": "edgeCasePercentage must be between 0 and 1",
            "value": edge_case_percentage,
        })
    if not edge_case_variables:
        edge_case_percentage = 0.0
    else:
        valid_names = {v["name"] for v in variables}
        for item in edge_case_variables:
            if item.get("name") not in valid_names:
                raise HTTPException(400, detail={"error": f"Edge-case variable '{item.get('name')}' is not a normal CSV variable"})
            if not str(item.get("condition", "")).strip():
                raise HTTPException(400, detail={"error": f"Edge-case variable '{item.get('name')}' requires condition"})

    draft_id = new_draft_id()
    draft = {
        "label": label or scenarioId,
        "journey": domain,
        "description": businessScenario or f"Scenario imported from CSV for {domain}",
        "variables": variables,
        "field_order": field_order,
        "domain": domain,
        "business_scenario": businessScenario,
        "business_response": businessResponse,
        "expected_outcome": expectedOutcome,
        "use_case": useCase,
        "scenario_id": scenarioId,
        "scenario_type": scenarioType,
        "industry_type": industryType,
        "country": country,
        "type_of_data": typeOfData,
        "entity_key": entityKey,
        "events": events,
        "edge_case_variables": edge_case_variables,
        "edge_case_percentage": edge_case_percentage,
    }
    save_draft(draft_id, draft)
    return ProposeResponse(
        draft_id=draft_id,
        scenario_id=scenarioId,
        scenario_id_available=not scenario_exists(scenarioId),
        label=draft["label"],
        journey=draft["journey"],
        description=draft["description"],
        variables=variables,
        field_order=field_order,
        typeOfData=typeOfData,
        events=events,
        edgeCaseVariables=edge_case_variables,
    )


@app.post("/scenario/confirm", response_model=ConfirmResponse)
def confirm_scenario_route(req: ConfirmRequest):
    """API 2 — apply user add/edit/delete + feedback, then finalize the scenario."""
    draft = get_draft(req.draft_id)
    if draft is None:
        raise HTTPException(404, detail={"error": f"Unknown or expired draft_id '{req.draft_id}'"})

    variables: list[dict] = [dict(v) for v in draft.get("variables", [])]
    by_name = {v["name"]: v for v in variables}

    for name in req.delete:
        if _is_placeholder(name):
            continue
        by_name.pop(name, None)
    variables = list(by_name.values())
    by_name = {v["name"]: v for v in variables}

    for e in req.edit:
        if _is_placeholder(e.name):
            continue
        changes = _clean_dict(e.changes or {})
        if e.name in by_name and changes:
            by_name[e.name].update(changes)

    for new_var in req.add:
        cleaned_var = _clean_dict(new_var)
        if not _is_placeholder(cleaned_var.get("name")):
            by_name[cleaned_var["name"]] = cleaned_var

    variables = list(by_name.values())
    field_order = [v["name"] for v in variables]

    # Placeholder/blank event entries are no-ops, same as variable add/edit/delete above.
    clean_event_delete = [v for v in req.eventDelete if not _is_placeholder(v)]
    clean_event_edits: list[tuple[str, dict]] = []
    for ee in req.eventEdit:
        if _is_placeholder(ee.event_type):
            continue
        changes = _clean_dict(ee.changes or {})
        if changes:
            clean_event_edits.append((ee.event_type, changes))
    clean_event_adds = [_clean_dict(ea) for ea in req.eventAdd if isinstance(ea, dict)]
    clean_event_adds = [ea for ea in clean_event_adds if ea]

    # Event-level changes are available ONLY for transactional scenarios.
    # Aggregational scenarios have a variables-only confirm contract.
    type_of_data = draft.get("type_of_data", "aggregational")
    has_event_changes = bool(clean_event_adds or clean_event_edits or clean_event_delete)
    if type_of_data != "transactional" and has_event_changes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Event add/edit/delete is only supported for transactional scenarios",
                "typeOfData": type_of_data,
                "allowedChanges": ["add", "edit", "delete"],
            },
        )

    events: list[dict] = [dict(e) for e in draft.get("events", []) if isinstance(e, dict)]
    if type_of_data == "transactional":
        delete_event_types = {str(v).strip().upper() for v in clean_event_delete}
        events = [
            e for e in events
            if str(e.get("event_type", "")).strip().upper() not in delete_event_types
        ]

        event_by_type = {str(e.get("event_type", "")).strip().upper(): e for e in events}

        # Apply edits against the pre-edit event_type. If an edit renames the event,
        # rebuild the lookup map so subsequent edits/adds operate on the new name.
        for event_type, changes in clean_event_edits:
            old_key = event_type.strip().upper().replace(" ", "_")
            if old_key not in event_by_type:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Event not found", "event_type": old_key},
                )
            if "event_type" in changes:
                new_key = str(changes["event_type"]).strip().upper().replace(" ", "_")
                if not new_key:
                    raise HTTPException(status_code=400, detail={"error": "event_type cannot be empty"})
                if new_key != old_key and new_key in event_by_type:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "Event type already exists", "event_type": new_key},
                    )
                changes["event_type"] = new_key
            event = event_by_type.pop(old_key)
            event.update(changes)
            new_key = str(event.get("event_type", old_key)).strip().upper().replace(" ", "_")
            event["event_type"] = new_key
            event_by_type[new_key] = event

        for new_event in clean_event_adds:
            event_type = str(new_event.get("event_type", "")).strip().upper().replace(" ", "_")
            if not event_type:
                raise HTTPException(status_code=400, detail={"error": "eventAdd requires event_type"})
            if event_type in event_by_type:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Event type already exists", "event_type": event_type},
                )
            event = dict(new_event)
            event["event_type"] = event_type
            event_by_type[event_type] = event

        events = list(event_by_type.values())
        for index, event in enumerate(events, start=1):
            event["sequence"] = index
            fields = event.get("fields", [])
            event["fields"] = fields if isinstance(fields, list) else []
            # Each transactional event can repeat for the same entity. The API
            # returns at most 10 records per event; these defaults make newly
            # proposed transactional events capable of producing multiple rows.
            min_occ = max(1, int(event.get("min_occurrences", 1)))
            max_occ = max(min_occ, min(1000, int(event.get("max_occurrences", 10))))
            event["min_occurrences"] = min_occ
            event["max_occurrences"] = max_occ

    # Edge-case variables: add/edit/delete independently from normal variables.
    edge_case_variables: list[dict] = [dict(v) for v in draft.get("edge_case_variables", []) if isinstance(v, dict)]
    edge_by_name = {str(v.get("name", "")): v for v in edge_case_variables if v.get("name")}
    for name in req.edgeCaseDelete:
        if not _is_placeholder(name):
            edge_by_name.pop(str(name), None)
    for e in req.edgeCaseEdit:
        if _is_placeholder(e.name):
            continue
        changes = _clean_dict(e.changes or {})
        if e.name in edge_by_name and changes:
            edge_by_name[e.name].update(changes)
    for new_var in req.edgeCaseAdd:
        cleaned = _clean_dict(new_var)
        if not _is_placeholder(cleaned.get("name")):
            edge_by_name[str(cleaned["name"])] = cleaned
    edge_case_variables = list(edge_by_name.values())

    edge_case_percentage = draft.get("edge_case_percentage", 0.0)
    if req.edgeCasePercentage is not None:
        edge_case_percentage = req.edgeCasePercentage
    try:
        edge_case_percentage = float(edge_case_percentage or 0.0)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, detail={"error": "edgeCasePercentage must be a finite float between 0 and 1"}) from exc
    import math
    if not math.isfinite(edge_case_percentage) or not 0.0 <= edge_case_percentage <= 1.0:
        raise HTTPException(400, detail={"error": "edgeCasePercentage must be between 0 and 1", "value": edge_case_percentage})

    # Edge cases are optional. If the confirmed scenario has no edge-case
    # definitions, percentage is normalized to 0 rather than producing an
    # impossible request.
    if not edge_case_variables:
        edge_case_percentage = 0.0

    scenario_id = draft.get("scenario_id") or next_scenario_id()
    requested_scenario_id = draft.get("scenario_id")
    # scenarioId is user-chosen at propose time, so two different users' drafts can pick the
    # same one; if it's already confirmed under a different draft, don't clobber it — mint a
    # fresh id and keep the original draft_id as the unambiguous handle for /generate.
    if scenario_exists(scenario_id) and resolve_scenario_id_from_draft(req.draft_id) != scenario_id:
        reassigned_from = scenario_id
        scenario_id = next_scenario_id()
        logger.warning(
            "[confirm] requested scenarioId '%s' (draft_id=%s) is already confirmed under a "
            "different draft; reassigning this draft to '%s' instead.",
            reassigned_from, req.draft_id, scenario_id,
        )
    meta = {
        "label": draft.get("label", scenario_id),
        "journey": draft.get("journey", draft.get("domain", "")),
        "description": draft.get("description", ""),
        "domain": draft.get("domain", ""),
        "business_scenario": draft.get("business_scenario", ""),
        "business_response": draft.get("business_response"),
        "expected_outcome": draft.get("expected_outcome"),
        "scenario_type": draft.get("scenario_type"),
        "use_case": draft.get("use_case"),
        "industry": draft.get("industry_type", "generic"),
        "country": draft.get("country"),
        "requested_scenario_id": requested_scenario_id,
        "type_of_data": type_of_data,
        "events": events,
        "entity_key": draft.get("entity_key"),
        "edge_case_variables": edge_case_variables,
        "edge_case_percentage": edge_case_percentage,
    }
    confirm_scenario(scenario_id, meta, variables, field_order, draft_id=req.draft_id)
    # Confirmed scenario changed: invalidate all runtime artifacts compiled from the old version.
    invalidate_scenario(scenario_id)
    clear_scenario(scenario_id)

    if not _is_placeholder(req.feedback):
        add_feedback(draft.get("domain", ""), draft.get("business_scenario", ""), req.feedback)

    pop_draft(req.draft_id)

    return ConfirmResponse(
        scenario_id=scenario_id,
        requested_scenario_id=requested_scenario_id,
        scenario_id_reassigned=requested_scenario_id is not None and scenario_id != requested_scenario_id,
        draft_id=req.draft_id,
        label=meta["label"],
        journey=meta["journey"],
        description=meta["description"],
        variables=variables,
        field_order=field_order,
        typeOfData=meta["type_of_data"],
        events=meta.get("events", []),
        edgeCaseVariables=edge_case_variables,
        edgeCasePercentage=edge_case_percentage,
    )


@app.post("/scenario/generate", response_model=GenerateResponse)
def generate_dynamic(req: GenerateRequest):
    """API 4 — confirm a draft/scenario into records via the 4-agent pipeline."""
    scenario_id = req.scenario
    if req.draftId:
        # draftId is the unambiguous handle when two users' scenarioId choices collided.
        resolved = resolve_scenario_id_from_draft(req.draftId)
        if resolved is None:
            raise HTTPException(404, detail={"error": f"Unknown or unconfirmed draftId '{req.draftId}'"})
        if req.scenario and req.scenario != resolved:
            raise HTTPException(400, detail={
                "error": f"draftId '{req.draftId}' does not match scenario '{req.scenario}'",
                "draft_scenario_id": resolved,
            })
        scenario_id = resolved
    if not scenario_id:
        raise HTTPException(400, detail={"error": "Either 'scenario' or 'draftId' is required"})
    if not scenario_exists(scenario_id):
        raise HTTPException(400, detail={"error": f"Unknown scenario '{scenario_id}'"})
    scenario_context = resolve_scenario_context(scenario_id)
    state = run_pipeline(
        scenario=scenario_id,
        count=req.count,
        industry=scenario_context.get("industry", "generic"),
        country=scenario_context.get("country"),
        type_of_data=scenario_context.get("type_of_data", resolve_data_type(scenario_id)),
        scenario_context=scenario_context,
    )
    if state.errors and not state.final_records:
        raise HTTPException(500, detail={"errors": state.errors})
    meta = resolve_scenario_meta(scenario_id) or {}
    final_records = state.final_records

    # Transactional responses are grouped by the scenario-defined entity key.
    # Only the latest 10 entities are returned. Within each entity, each event
    # contains its own totalCount and latest 10 records.
    event_data: list[dict] = []
    response_records: list[dict] = final_records
    entity_key = meta.get("entity_key")
    total_count = len(final_records)

    if state.type_of_data == "transactional":
        if not entity_key:
            raise HTTPException(500, detail={"error": "Transactional scenario is missing entity_key"})

        grouped_entities: dict[str, list[dict]] = {}
        for record in final_records:
            if entity_key not in record or record.get(entity_key) in (None, ""):
                continue
            key = str(record[entity_key])
            grouped_entities.setdefault(key, []).append(record)

        # The requested count represents the full number of entities in the conceptual
        # dataset. Only the latest 10 entities are materialized for the response.
        total_count = state.count
        # Generated records are ordered by generation time, so insertion order
        # preserves the most recently generated entities at the end.
        latest_entity_items = list(grouped_entities.items())[-10:]

        entity_records: list[dict] = []
        for entity_value, entity_rows in latest_entity_items:
            # Transactional edgeCasePercentage is defined at the entity/journey
            # grain, but the public flag belongs to the individual event record.
            # Do not expose isEdgeCaseData at the entity level. The generator has
            # already assigned the journey-consistent flag to each event row.
            entity_output = {entity_key: entity_value}

            # Include useful identity fields alongside the grouping key.
            first = entity_rows[-1]
            for common_name in ("subscriber_msisdn", "phone_number", "account_id", "customer_id"):
                if common_name in first and common_name != entity_key:
                    entity_output[common_name] = first[common_name]

            grouped_events: dict[str, list[dict]] = {}
            for row in entity_rows:
                event_type = str(row.get("event_type") or "BUSINESS_EVENT")
                grouped_events.setdefault(event_type, []).append(row)

            ordered_event_types: list[str] = []
            for event in meta.get("events", []):
                event_type = str(event.get("event_type", "BUSINESS_EVENT"))
                if event_type not in ordered_event_types:
                    ordered_event_types.append(event_type)
            for event_type in grouped_events:
                if event_type not in ordered_event_types:
                    ordered_event_types.append(event_type)

            events_for_entity: list[dict] = []
            for event_type in ordered_event_types:
                rows = grouped_events.get(event_type, [])
                if not rows:
                    continue
                actual_count = state.transactional_event_counts.get(str(entity_value), {}).get(event_type, len(rows))
                # isEdgeCaseData intentionally lives inside each event record.
                # Preserve it here; do not move it to the entity wrapper.
                clean_rows = [dict(event_row) for event_row in rows[-10:]]
                events_for_entity.append({
                    "event_type": event_type,
                    "totalCount": actual_count,
                    "records": clean_rows,
                })

            entity_output["events"] = events_for_entity
            entity_records.append(entity_output)

        response_records = entity_records
        # Keep eventData populated for backward compatibility, but it is no
        # longer the primary transactional representation.
        event_data = []

    return GenerateResponse(
        scenario_id=scenario_id,
        typeOfData=state.type_of_data,
        entityKey=entity_key,
        totalCount=total_count,
        events=meta.get("events", []),
        requested_scenario_id=meta.get("requested_scenario_id"),
        draft_id=req.draftId,
        scenario_label=meta["label"],
        fields=state.field_order or _get_field_order(scenario_id),
        total_records=len(final_records),
        validation_report=state.validation_report,
        records=response_records,
        eventData=event_data,
        errors=state.errors,
        edgeCasePercentage=float(meta.get("edge_case_percentage", 0.0) or 0.0),
    )
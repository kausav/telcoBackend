from __future__ import annotations
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from main import run_pipeline
from agents.scenario_designer_agent import ScenarioDesignerAgent
from core.csv_scenario import parse_variables_csv
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
    resolve_variables,
    save_draft,
    scenario_exists,
)
from core.llm_client import GeminiClient


def _get_field_order(scenario: str) -> list[str]:
    """Return the field order for any scenario without importing at module level."""
    dyn = resolve_variables(scenario)
    if dyn is not None:
        return dyn[1]
    from config.variables import FIELD_ORDER
    return FIELD_ORDER

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


class VariableEdit(BaseModel):
    name: str
    changes: dict


class ConfirmRequest(BaseModel):
    draft_id: str
    add: list[dict] = Field(default_factory=list)
    edit: list[VariableEdit] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)
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
    draft["type_of_data"] = req.typeOfData
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
    )


@app.post("/scenario/import-csv", response_model=ProposeResponse)
def import_scenario_csv(
    file: UploadFile = File(..., description="CSV with columns: name, dtype, gen, params, description, depends_on, nullable, formula"),
    scenarioId: str = Form(...),
    domain: str = Form(...),
    industryType: str = Form("generic"),
    country: str | None = Form(None),
    businessScenario: str = Form(""),
    scenarioType: str = Form(""),
    label: str = Form(""),
):
    """Alternative to API 1 — instead of asking the LLM to invent variables,
    load an industry-supplied CSV variable catalog directly. Produces a draft
    in the same shape /scenario/propose does, so it goes through the same
    /scenario/confirm review step before becoming a usable scenario."""
    raw = file.file.read()
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, detail={"error": f"CSV must be UTF-8 encoded: {exc}"}) from exc

    try:
        variables, field_order = parse_variables_csv(csv_text)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc

    draft_id = new_draft_id()
    draft = {
        "label": label or scenarioId,
        "journey": domain,
        "description": businessScenario or f"Scenario imported from CSV for {domain}",
        "variables": variables,
        "field_order": field_order,
        "domain": domain,
        "business_scenario": businessScenario,
        "scenario_id": scenarioId,
        "scenario_type": scenarioType,
        "industry_type": industryType,
        "country": country,
        "type_of_data": "aggregational",
        "events": [],
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
        typeOfData=draft.get("type_of_data", "aggregational"),
        events=draft.get("events", []),
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
        by_name.pop(name, None)
    variables = list(by_name.values())
    by_name = {v["name"]: v for v in variables}

    for e in req.edit:
        if e.name in by_name:
            by_name[e.name].update(e.changes)

    for new_var in req.add:
        if "name" in new_var:
            by_name[new_var["name"]] = new_var

    variables = list(by_name.values())
    field_order = [v["name"] for v in variables]

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
        "industry": draft.get("industry_type", "generic"),
        "country": draft.get("country"),
        "requested_scenario_id": requested_scenario_id,
        "type_of_data": draft.get("type_of_data", "aggregational"),
        "events": draft.get("events", []),
        "entity_key": draft.get("entity_key"),
    }
    confirm_scenario(scenario_id, meta, variables, field_order, draft_id=req.draft_id)

    if req.feedback:
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
    state = run_pipeline(
        scenario=scenario_id,
        count=req.count,
        industry=resolve_scenario_meta(scenario_id).get("industry", "generic"),
        country=resolve_scenario_meta(scenario_id).get("country"),
        type_of_data=resolve_data_type(scenario_id),
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

        total_count = len(grouped_entities)
        # Generated records are ordered by generation time, so insertion order
        # preserves the most recently generated entities at the end.
        latest_entity_items = list(grouped_entities.items())[-10:]

        entity_records: list[dict] = []
        for entity_value, entity_rows in latest_entity_items:
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
                events_for_entity.append({
                    "event_type": event_type,
                    "totalCount": len(rows),
                    "records": rows[-10:],
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
    )
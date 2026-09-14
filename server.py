from __future__ import annotations
import logging
import re
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Literal

from core.pipeline import run_pipeline
from core.csv_scenario import infer_type_of_data, parse_definition_csv
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
from core.compiled_schema import invalidate_scenario
from core.runtime_cache import clear_scenario
from config.industry_profiles import COUNTRY_BASE




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

from core.error_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from core.errors import ErrorResponse

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Telco Agentic SDG",
    version="2.0.0",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign one correlation id to every request and return it to the client."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


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
    success: bool = Field(True, description="True when the API request completed successfully")
    scenario_id: str
    typeOfData: Literal["transactional", "aggregational"]
    events: list[dict] = Field(default_factory=list)
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
    record_errors: list[dict] = Field(default_factory=list, description="Errors for individual records that could not be generated or validated; successful records are still returned")




class ScenarioImportResponse(BaseModel):
    success: bool = Field(True, description="True when the API request completed successfully")
    draft_id: str
    scenario_id: str
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


    feedback: str | None = None


class ConfirmResponse(BaseModel):
    success: bool = Field(True, description="True when the API request completed successfully")
    scenario_id: str
    requested_scenario_id: str | None = Field(None, description="The scenarioId supplied during CSV import, for comparison")
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
    """Liveness check."""
    return {"success": True, "status": "ok"}


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


def _country_from_csv_params(variables: list[dict]) -> str | None:
    """Infer country code/name from CSV variable params when present.

    This lets CSV definitions be self-contained. If country is declared in params,
    the importer prefers that over payload country.
    """
    currency_to_country: dict[str, str] = {}
    phone_to_country: dict[str, str] = {}
    country_name_to_code: dict[str, str] = {}
    for code, data in COUNTRY_BASE.items():
        if code == "GLOBAL":
            continue
        currency = str(data.get("currency", "")).strip().upper()
        if currency and currency not in currency_to_country:
            currency_to_country[currency] = code
        phone_cc = str(data.get("phone_country_code", "")).strip()
        if phone_cc:
            phone_to_country[phone_cc] = code
        country_name = str(data.get("country_name", "")).strip().lower()
        if country_name:
            country_name_to_code[country_name] = code

    known_country_codes = {code for code in COUNTRY_BASE if code != "GLOBAL"}
    known_currency_codes = set(currency_to_country.keys())

    def _normalize_country_hint(raw: object) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        upper = text.upper()
        if upper in known_country_codes:
            return upper
        if upper in currency_to_country:
            return currency_to_country[upper]
        if text in phone_to_country:
            return phone_to_country[text]
        lower = text.lower()
        if lower in country_name_to_code:
            return country_name_to_code[lower]
        return None

    def _scan_text_hints(text: str) -> list[str]:
        """Extract country signals from free text (description/examples/params strings)."""
        hits: list[str] = []
        if not text:
            return hits
        normalized_text = str(text)
        # Tokenize while keeping + for phone country codes.
        tokens = [t for t in re.split(r"[^A-Za-z0-9+]+", normalized_text) if t]
        for token in tokens:
            mapped = _normalize_country_hint(token)
            if mapped:
                hits.append(mapped)
        # Country names can include spaces, so also scan full lowercase text.
        lower_text = normalized_text.lower()
        for name, code in country_name_to_code.items():
            if name in lower_text:
                hits.append(code)
        # Prefer currency signals in textual phrases like "amount in INR".
        for currency in known_currency_codes:
            if re.search(rf"\b{re.escape(currency)}\b", normalized_text, flags=re.IGNORECASE):
                hits.append(currency_to_country[currency])
        return hits

    candidates: list[str] = []
    for var in variables:
        if not isinstance(var, dict):
            continue
        # Free-text hints outside params.
        for text_key in ("description", "name"):
            candidates.extend(_scan_text_hints(str(var.get(text_key, ""))))

        params = var.get("params")
        if not isinstance(params, dict):
            continue

        # Direct explicit keys.
        for key in ("country", "country_code", "countryCode"):
            value = params.get(key)
            if value is None:
                continue
            normalized = _normalize_country_hint(value)
            if normalized:
                candidates.append(normalized)
        currency_value = params.get("currency")
        if currency_value is not None:
            normalized = _normalize_country_hint(currency_value)
            if normalized:
                candidates.append(normalized)

        # List-style country hints.
        country_codes = params.get("country_codes")
        if isinstance(country_codes, list):
            for entry in country_codes:
                normalized = _normalize_country_hint(entry)
                if normalized:
                    candidates.append(normalized)

        # Generic scan across all string/list params for hints embedded in arbitrary keys.
        for value in params.values():
            if isinstance(value, str):
                candidates.extend(_scan_text_hints(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        candidates.extend(_scan_text_hints(item))

    if not candidates:
        return None
    # Most frequent normalized token wins for deterministic behavior.
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


@app.post("/scenario/import-csv", response_model=ScenarioImportResponse)
def import_scenario_csv(
    file: UploadFile = File(..., description="CSV scenario definition: variables and, for transactional data, events"),
    scenarioId: str = Form(...),
    domain: str = Form(...),
    typeOfData: Literal["transactional", "aggregational"] | None = Form(None),
    industryType: str = Form("generic"),
    country: str | None = Form(None),
    businessScenario: str = Form(""),
    businessResponse: str | None = Form(None),
    expectedOutcome: str | None = Form(None),
    scenarioType: str = Form(""),
    useCase: str | None = Form(None),
    label: str = Form(""),
    entityKey: str | None = Form(None),
):
    """Import a CSV scenario definition and create a draft for confirmation/generation."""
    raw = file.file.read()
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, detail={"error": f"CSV must be UTF-8 encoded: {exc}"}) from exc

    try:
        detected_type = infer_type_of_data(csv_text)
        variables, field_order, events = parse_definition_csv(
            csv_text, type_of_data=detected_type
        )
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc

    typeOfData = detected_type
    csv_country = _country_from_csv_params(variables)
    effective_country = csv_country or country
    variable_names = {v["name"] for v in variables}
    if typeOfData == "transactional":
        # entityKey is optional for the new CSV contract. Prefer an explicit key;
        # otherwise infer a stable entity identifier from the non-event fields.
        if entityKey:
            token = str(entityKey).strip()
            if token in variable_names:
                entityKey = token
            else:
                lowered = {name.lower(): name for name in variable_names}
                normalized = lowered.get(token.lower()) if token else None
                # Common UI/default alias drift: map between customer/subscriber keys
                # when one side is present in the CSV schema.
                if not normalized:
                    alias_map = {
                        "customer_id": "subscriber_id",
                        "subscriber_id": "customer_id",
                    }
                    alias = alias_map.get(token.lower()) if token else None
                    if alias and alias in variable_names:
                        normalized = alias
                # Do not hard-fail on stale/invalid entityKey from client payload.
                # Fall back to schema inference below.
                entityKey = normalized
        if not entityKey:
            preferred = (
                "subscriber_id", "customer_id", "account_id", "user_id",
                "entity_id", "customer_key", "entity_key", "id",
            )
            event_fields = {f for e in events for f in e.get("fields", [])}
            entity_names = [v["name"] for v in variables if v["name"] not in event_fields]
            # Some client event containers enumerate every field, including the
            # entity identifier. In that shape, derive from the full schema.
            if not entity_names:
                entity_names = [v["name"] for v in variables]
            entityKey = next((name for name in preferred if name in entity_names), None)
            entityKey = entityKey or (entity_names[0] if entity_names else None)
        if not entityKey:
            raise HTTPException(400, detail={
                "error": "Transactional CSV must contain at least one entity field so a journey entity can be identified"
            })
    else:
        entityKey = None

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
        "country": effective_country,
        "type_of_data": typeOfData,
        "entity_key": entityKey,
        "events": events,
    }
    save_draft(draft_id, draft)
    return ScenarioImportResponse(
        success=True,
        draft_id=draft_id,
        scenario_id=scenarioId,
        label=draft["label"],
        journey=draft["journey"],
        description=draft["description"],
        variables=variables,
        field_order=field_order,
        typeOfData=typeOfData,
        events=events,
    )


@app.post("/scenario/confirm", response_model=ConfirmResponse)
def confirm_scenario_route(req: ConfirmRequest):
    """Finalize an imported CSV draft, applying optional variable/event edits."""
    draft = get_draft(req.draft_id)
    if draft is None:
        raise HTTPException(404, detail={"error": f"Unknown or expired draft_id '{req.draft_id}'"})

    variables = [dict(v) for v in draft.get("variables", []) if isinstance(v, dict)]
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}

    for name in req.delete:
        if not _is_placeholder(name):
            by_name.pop(name, None)
    for edit in req.edit:
        if _is_placeholder(edit.name):
            continue
        changes = _clean_dict(edit.changes or {})
        if edit.name in by_name and changes:
            by_name[edit.name].update(changes)
    for new_var in req.add:
        cleaned = _clean_dict(new_var)
        name = cleaned.get("name")
        if not _is_placeholder(name):
            by_name[str(name)] = cleaned

    variables = list(by_name.values())
    field_order = [str(v["name"]) for v in variables if v.get("name")]

    type_of_data = draft.get("type_of_data", "aggregational")
    clean_event_delete = [v for v in req.eventDelete if not _is_placeholder(v)]
    clean_event_edits = []
    for edit in req.eventEdit:
        if not _is_placeholder(edit.event_type):
            changes = _clean_dict(edit.changes or {})
            if changes:
                clean_event_edits.append((edit.event_type, changes))
    clean_event_adds = [_clean_dict(v) for v in req.eventAdd if isinstance(v, dict)]
    clean_event_adds = [v for v in clean_event_adds if v]

    if type_of_data != "transactional" and (clean_event_adds or clean_event_edits or clean_event_delete):
        raise HTTPException(400, detail={
            "error": "Event add/edit/delete is only supported for transactional scenarios",
            "typeOfData": type_of_data,
            "allowedChanges": ["add", "edit", "delete"],
        })

    events = [dict(e) for e in draft.get("events", []) if isinstance(e, dict)]
    if type_of_data == "transactional":
        delete_types = {str(v).strip().upper().replace(" ", "_") for v in clean_event_delete}
        events = [e for e in events if str(e.get("event_type", "")).strip().upper().replace(" ", "_") not in delete_types]
        event_by_type = {str(e.get("event_type", "")).strip().upper().replace(" ", "_"): e for e in events}

        for event_type, changes in clean_event_edits:
            old_key = str(event_type).strip().upper().replace(" ", "_")
            if old_key not in event_by_type:
                raise HTTPException(400, detail={"error": "Event not found", "event_type": old_key})
            if "event_type" in changes:
                new_key = str(changes["event_type"]).strip().upper().replace(" ", "_")
                if not new_key:
                    raise HTTPException(400, detail={"error": "event_type cannot be empty"})
                if new_key != old_key and new_key in event_by_type:
                    raise HTTPException(400, detail={"error": "Event type already exists", "event_type": new_key})
                changes["event_type"] = new_key
            event = event_by_type.pop(old_key)
            event.update(changes)
            new_key = str(event.get("event_type", old_key)).strip().upper().replace(" ", "_")
            event["event_type"] = new_key
            event_by_type[new_key] = event

        for new_event in clean_event_adds:
            event_type = str(new_event.get("event_type", "")).strip().upper().replace(" ", "_")
            if not event_type:
                raise HTTPException(400, detail={"error": "eventAdd requires event_type"})
            if event_type in event_by_type:
                raise HTTPException(400, detail={"error": "Event type already exists", "event_type": event_type})
            event = dict(new_event)
            event["event_type"] = event_type
            event_by_type[event_type] = event

        events = list(event_by_type.values())
        for index, event in enumerate(events, start=1):
            event["sequence"] = index
            fields = event.get("fields", [])
            event["fields"] = fields if isinstance(fields, list) else []
            min_occ = max(1, int(event.get("min_occurrences", 1)))
            max_occ = max(min_occ, min(1000, int(event.get("max_occurrences", 10))))
            event["min_occurrences"] = min_occ
            event["max_occurrences"] = max_occ

    scenario_id = draft.get("scenario_id") or next_scenario_id()
    requested_scenario_id = draft.get("scenario_id")
    if scenario_exists(scenario_id) and resolve_scenario_id_from_draft(req.draft_id) != scenario_id:
        reassigned_from = scenario_id
        scenario_id = next_scenario_id()
        logger.warning("[confirm] scenarioId '%s' already exists; reassigned to '%s'", reassigned_from, scenario_id)

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
    }
    confirm_scenario(scenario_id, meta, variables, field_order, draft_id=req.draft_id)
    invalidate_scenario(scenario_id)
    clear_scenario(scenario_id)

    if not _is_placeholder(req.feedback):
        add_feedback(draft.get("domain", ""), draft.get("business_scenario", ""), req.feedback)
    pop_draft(req.draft_id)

    return ConfirmResponse(
        success=True,
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
def generate_scenario(req: GenerateRequest):
    """Generate records from a confirmed CSV-defined scenario."""
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
    try:
        state = run_pipeline(
            scenario=scenario_id,
            count=req.count,
            industry=scenario_context.get("industry", "generic"),
            country=scenario_context.get("country"),
            type_of_data=scenario_context.get("type_of_data", resolve_data_type(scenario_id)),
            scenario_context=scenario_context,
        )
    except ValueError as exc:
        # Deterministic scenario-definition/data-contract failures are client/config
        # errors, not server crashes. Surface the exact reason as HTTP 400.
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    # Scenario/pipeline errors still fail when no record-level work succeeded.
    # Individual record failures are returned as record_errors and must never turn
    # the whole /scenario/generate request into an HTTP 500, even when every
    # requested record failed.
    if state.errors and not state.final_records and not state.record_errors:
        raise HTTPException(500, detail={"errors": state.errors})
    meta = resolve_scenario_meta(scenario_id) or {}
    final_records = state.final_records

    # Transactional responses are grouped by the scenario-defined entity key.
    # Return the full generated entity set for the requested count.
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

        # The requested count represents the full number of entities in the dataset.
        total_count = state.count
        entity_items = list(grouped_entities.items())

        entity_records: list[dict] = []
        for entity_value, entity_rows in entity_items:
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
                clean_rows = [dict(event_row) for event_row in rows]
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
        success=True,
        scenario_id=scenario_id,
        typeOfData=state.type_of_data,
        entityKey=entity_key,
        totalCount=total_count,
        events=meta.get("events", []),
        draft_id=req.draftId,
        scenario_label=meta["label"],
        fields=state.field_order or ((resolve_variables(scenario_id) or ([], []))[1]),
        total_records=len(final_records),
        validation_report=state.validation_report,
        records=response_records,
        eventData=event_data,
        errors=state.errors,
        record_errors=state.record_errors,
    )
	
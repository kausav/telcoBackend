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
    count: int = Field(20, ge=1, le=5000, description="Number of users/entities to generate for a transactional scenario")
    recordsPerUser: int = Field(10, ge=1, le=10, description="Number of most-recent historical records returned per user for a transactional scenario")


class GenerateResponse(BaseModel):
    success: bool = True
    scenario_id: str
    typeOfData: Literal["transactional", "aggregational"]
    draft_id: str | None = None
    scenario_label: str
    fields: list[str]
    total_records: int
    validation_report: dict
    records: list[dict]
    entityKey: str | None = Field(None, description="Field used as the user/entity key for transactional history")
    totalCount: int = Field(0, description="Number of users/entities in the response dataset")
    recordsPerUser: int = Field(10, description="Number of recent records returned for each transactional user")
    errors: list[str]
    record_errors: list[dict] = Field(default_factory=list)


class ScenarioImportResponse(BaseModel):
    success: bool = True
    draft_id: str
    scenario_id: str
    label: str
    journey: str
    description: str
    variables: list[dict]
    field_order: list[str]
    typeOfData: Literal["transactional", "aggregational"]
    entityKey: str | None = None


class VariableEdit(BaseModel):
    name: str
    changes: dict


class ConfirmRequest(BaseModel):
    draft_id: str
    add: list[dict] = Field(default_factory=list, description="New variable definitions to add")
    edit: list[VariableEdit] = Field(default_factory=list, description="Existing variables to edit by name")
    delete: list[str] = Field(default_factory=list, description="Variable names to delete")
    feedback: str | None = None


class ConfirmResponse(BaseModel):
    success: bool = True
    scenario_id: str
    requested_scenario_id: str | None = None
    scenario_id_reassigned: bool = False
    draft_id: str
    label: str
    journey: str
    description: str
    variables: list[dict]
    field_order: list[str]
    typeOfData: Literal["transactional", "aggregational"]
    entityKey: str | None = None


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
    file: UploadFile = File(..., description="CSV scenario definition containing variables; transactional schemas can mark scope=user/record"),
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
    """Import a CSV variable definition and create a draft for confirmation/generation."""
    raw=file.file.read()
    try:
        csv_text=raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400,detail={"error":f"CSV must be UTF-8 encoded: {exc}"}) from exc
    try:
        requested_type=(str(typeOfData).strip().lower() if typeOfData else None)
        detected_type=requested_type or infer_type_of_data(csv_text)
        variables,field_order=parse_definition_csv(csv_text,type_of_data=detected_type)
    except ValueError as exc:
        raise HTTPException(400,detail={"error":str(exc)}) from exc
    typeOfData=detected_type
    csv_country=_country_from_csv_params(variables)
    effective_country=csv_country or country
    variable_names={v["name"] for v in variables}
    if typeOfData=="transactional":
        if entityKey:
            token=str(entityKey).strip(); entityKey=next((n for n in variable_names if n.lower()==token.lower()),None)
        if not entityKey:
            preferred=("subscriber_id","customer_id","account_id","user_id","entity_id","customer_key","entity_key","id")
            entityKey=next((n for n in preferred if n in variable_names),None) or next(iter(variable_names),None)
        if not entityKey:
            raise HTTPException(400,detail={"error":"Transactional CSV must contain at least one user/entity identifier field"})
    else:
        entityKey=None

    draft_id=new_draft_id()
    draft={
        "label":label or scenarioId,
        "journey":domain,
        "description":businessScenario or f"Scenario imported from CSV for {domain}",
        "variables":variables,
        "field_order":field_order,
        "domain":domain,
        "business_scenario":businessScenario,
        "business_response":businessResponse,
        "expected_outcome":expectedOutcome,
        "use_case":useCase,
        "scenario_id":scenarioId,
        "scenario_type":scenarioType,
        "industry_type":industryType,
        "country":effective_country,
        "type_of_data":typeOfData,
        "entity_key":entityKey,
        "records_per_user":10,
    }
    save_draft(draft_id,draft)
    return ScenarioImportResponse(success=True,draft_id=draft_id,scenario_id=scenarioId,label=draft["label"],journey=draft["journey"],description=draft["description"],variables=variables,field_order=field_order,typeOfData=typeOfData,entityKey=entityKey)


@app.post("/scenario/confirm", response_model=ConfirmResponse)
def confirm_scenario_route(req: ConfirmRequest):
    """Finalize an imported CSV draft, applying optional variable edits."""
    draft=get_draft(req.draft_id)
    if draft is None:
        raise HTTPException(404,detail={"error":f"Unknown or expired draft_id '{req.draft_id}'"})
    variables=[dict(v) for v in draft.get("variables",[]) if isinstance(v,dict)]
    by_name={str(v.get("name")):v for v in variables if v.get("name")}
    for name in req.delete:
        if not _is_placeholder(name): by_name.pop(name,None)
    for edit in req.edit:
        if _is_placeholder(edit.name): continue
        changes=_clean_dict(edit.changes or {})
        if edit.name in by_name and changes: by_name[edit.name].update(changes)
    for new_var in req.add:
        cleaned=_clean_dict(new_var); name=cleaned.get("name")
        if not _is_placeholder(name): by_name[str(name)]=cleaned
    variables=list(by_name.values()); field_order=[str(v["name"]) for v in variables if v.get("name")]
    type_of_data=draft.get("type_of_data","aggregational")

    entity_key=draft.get("entity_key") if type_of_data=="transactional" else None
    if type_of_data=="transactional" and entity_key not in {v.get("name") for v in variables}:
        preferred=("subscriber_id","customer_id","account_id","user_id","entity_id","customer_key","entity_key","id")
        names={str(v.get("name")) for v in variables if v.get("name")}
        entity_key=next((n for n in preferred if n in names),None) or (next(iter(names),None) if names else None)
    scenario_id=draft.get("scenario_id") or next_scenario_id()
    requested_scenario_id=draft.get("scenario_id")
    if scenario_exists(scenario_id) and resolve_scenario_id_from_draft(req.draft_id)!=scenario_id:
        reassigned_from=scenario_id; scenario_id=next_scenario_id()
        logger.warning("[confirm] scenarioId '%s' already exists; reassigned to '%s'",reassigned_from,scenario_id)
    meta={
        "label":draft.get("label",scenario_id),"journey":draft.get("journey",draft.get("domain","")),"description":draft.get("description",""),
        "domain":draft.get("domain",""),"business_scenario":draft.get("business_scenario",""),"business_response":draft.get("business_response"),
        "expected_outcome":draft.get("expected_outcome"),"scenario_type":draft.get("scenario_type"),"use_case":draft.get("use_case"),
        "industry":draft.get("industry_type","generic"),"country":draft.get("country"),"requested_scenario_id":requested_scenario_id,
        "type_of_data":type_of_data,"entity_key":entity_key,"records_per_user":10,
    }
    confirm_scenario(scenario_id,meta,variables,field_order,draft_id=req.draft_id)
    invalidate_scenario(scenario_id); clear_scenario(scenario_id)
    if not _is_placeholder(req.feedback): add_feedback(draft.get("domain",""),draft.get("business_scenario",""),req.feedback)
    pop_draft(req.draft_id)
    return ConfirmResponse(success=True,scenario_id=scenario_id,requested_scenario_id=requested_scenario_id,scenario_id_reassigned=requested_scenario_id is not None and scenario_id!=requested_scenario_id,draft_id=req.draft_id,label=meta["label"],journey=meta["journey"],description=meta["description"],variables=variables,field_order=field_order,typeOfData=type_of_data,entityKey=entity_key)


@app.post("/scenario/generate", response_model=GenerateResponse)
def generate_scenario(req: GenerateRequest):
    """Generate flat records, grouped directly by user/entity for transactional histories."""
    scenario_id=req.scenario
    if req.draftId:
        resolved=resolve_scenario_id_from_draft(req.draftId)
        if resolved is None: raise HTTPException(404,detail={"error":f"Unknown or unconfirmed draftId '{req.draftId}'"})
        if req.scenario and req.scenario!=resolved:
            raise HTTPException(400,detail={"error":f"draftId '{req.draftId}' does not match scenario '{req.scenario}'","draft_scenario_id":resolved})
        scenario_id=resolved
    if not scenario_id: raise HTTPException(400,detail={"error":"Either 'scenario' or 'draftId' is required"})
    if not scenario_exists(scenario_id): raise HTTPException(400,detail={"error":f"Unknown scenario '{scenario_id}'"})
    scenario_context=resolve_scenario_context(scenario_id)
    try:
        state=run_pipeline(
            scenario=scenario_id,count=req.count,industry=scenario_context.get("industry","generic"),country=scenario_context.get("country"),
            type_of_data=scenario_context.get("type_of_data",resolve_data_type(scenario_id)),scenario_context=scenario_context,
            records_per_user=req.recordsPerUser,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400,detail={"error":str(exc)}) from exc
    if state.errors and not state.final_records and not state.record_errors:
        raise HTTPException(500,detail={"errors":state.errors})
    meta=resolve_scenario_meta(scenario_id) or {}
    final_records=state.final_records
    entity_key=meta.get("entity_key")
    response_records=final_records
    total_count=len(final_records); total_records=len(final_records)
    records_per_user=req.recordsPerUser

    if state.type_of_data=="transactional":
        if not entity_key: raise HTTPException(500,detail={"error":"Transactional scenario is missing entity_key"})
        grouped={}
        for row in final_records:
            value=row.get(entity_key)
            if value in (None,""): continue
            grouped.setdefault(str(value),[]).append(row)
        entity_records=[]
        resolved_vars,_ = resolve_variables(scenario_id) or ([],[])
        user_field_names={str(v.get("name")) for v in resolved_vars if str(v.get("scope","record")).lower()=="user" and v.get("name")}
        user_field_names.add(entity_key)
        for entity_value,rows in grouped.items():
            # Newest first, with the user-level context outside the history rows.
            timestamp_field=next((f for f in ("record_timestamp","transaction_timestamp","timestamp","created_at","updated_at") if f in rows[0]),None)
            if timestamp_field:
                rows=sorted(rows,key=lambda r: str(r.get(timestamp_field,"")), reverse=True)
            latest=rows[0] if rows else {}
            user_output={name: latest.get(name) for name in user_field_names if name in latest}
            user_output[entity_key]=entity_value
            clean_history=[]
            for row in rows[:records_per_user]:
                clean_history.append({k:v for k,v in dict(row).items() if k not in user_field_names})
            user_output["records"]=clean_history
            entity_records.append(user_output)
        response_records=entity_records
        total_count=len(entity_records)
        total_records=sum(len(x.get("records",[])) for x in entity_records)

    return GenerateResponse(
        success=True,scenario_id=scenario_id,typeOfData=state.type_of_data,entityKey=entity_key,totalCount=total_count,recordsPerUser=records_per_user,
        draft_id=req.draftId,scenario_label=meta.get("label",scenario_id),fields=state.field_order or ((resolve_variables(scenario_id) or ([],[]))[1]),
        total_records=total_records,validation_report=state.validation_report,records=response_records,errors=state.errors,record_errors=state.record_errors,
    )


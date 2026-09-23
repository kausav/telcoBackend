from __future__ import annotations
import logging
import re
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

from core.dynamic_scenarios import (
    add_feedback,
    confirm_scenario,
    get_draft,
    new_draft_id,
    save_draft,
    pop_draft,
    resolve_scenario_id_from_draft,
    resolve_scenario_meta,
    scenario_exists,
    resolve_data_type,
    resolve_scenario_context,
    resolve_variables,
)
from core.compiled_schema import invalidate_scenario, infer_history_field_sets
from core.runtime_cache import clear_scenario
from core.agentic_models import ScenarioProposeRequest, ScenarioImportResponse
from core.csv_scenario import infer_type_of_data, parse_definition_csv
from config.industry_profiles import COUNTRY_BASE
from config.runtime import CORS_ALLOW_ORIGINS, MAX_CSV_BYTES
from core.agentic_workflow import AgenticSchemaWorkflow, get_agentic_workflow
from core.telecom_registry import RegistryError, get_registry
from core.scenario_variable_store import upsert_recommended, get_recommended, set_user_variables, get_user_variables, get_user_variable_records, delete_user_variable
from models.database import ping as ping_mongodb
from models.model_registry import ensure_indexes as ensure_model_indexes
from models.generation_job import GenerationJobModel
from core.generation_jobs import submit_generation_job




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
    llm_upstream_exception_handler,
)
from core.errors import ErrorResponse, LLMUpstreamError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate immutable model inputs and initialize mutable runtime state once per process."""
    try:
        ping_mongodb()
        ensure_model_indexes()
        registry = get_registry()
        health = registry.health()
        if not health["healthy"]:
            raise RegistryError("Telecom standards registry is empty")
        logger.info(
            "Standards registry ready: standards=%s entities=%s attributes=%s relationships=%s",
            health["standards"], health["entities"], health["attributes"], health["relationships"],
        )
        GenerationJobModel.ensure_indexes()
    except Exception:
        logger.exception("Application startup validation failed")
        raise
    yield


app = FastAPI(
    title="Telco Agentic SDG",
    version="2.6.0",
    lifespan=lifespan,
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
app.add_exception_handler(LLMUpstreamError, llm_upstream_exception_handler)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign one bounded correlation id to every request and return it to the client."""
    incoming = str(request.headers.get("X-Request-ID") or "").strip()
    request_id = incoming if 1 <= len(incoming) <= 128 and re.fullmatch(r"[A-Za-z0-9._:-]+", incoming) else str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


class GenerateRequest(BaseModel):
    scenario: str | None = Field(None, examples=["LB-01"])
    draftId: str | None = Field(None, description="Confirmed draft id; disambiguates when scenario ids collide across users")
    count: int = Field(35, ge=1, le=5000, description="Number of users/entities to generate for a transactional scenario")
    recordsPerUser: int = Field(10, ge=1, le=10, description="Number of most-recent historical records returned per user for a transactional scenario")


class GenerateResponse(BaseModel):
    success: bool = True
    scenario_id: str
    requested_scenario_id: str | None = None
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


class GenerateAcceptedResponse(BaseModel):
    success: bool = True
    status: Literal["queued", "running"] = "queued"
    jobId: str
    scenario_id: str
    draft_id: str | None = None
    totalCount: int
    recordsPerUser: int


class GenerateJobResponse(BaseModel):
    success: bool = True
    status: Literal["queued", "running", "completed", "failed"]
    jobId: str
    scenario_id: str | None = None
    draft_id: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    error: str | None = None
    result: GenerateResponse | None = None


class VariableEdit(BaseModel):
    name: str
    changes: dict


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "draft_id": "draft-169b76e5e0854c868c4110ada99afc7a",
            "add": [],
            "edit": [],
            "delete": [],
            "feedback": None,
        }
    })

    draft_id: str
    add: list[dict] = Field(default_factory=list, description="Optional new variable definitions for legacy/imported drafts; leave empty for agentic proposals")
    edit: list[VariableEdit] = Field(default_factory=list, description="Optional HITL edits to existing variables")
    delete: list[str] = Field(default_factory=list, description="Optional variable names to delete")
    feedback: str | None = None


class ScenarioVariableRecommendationRequest(BaseModel):
    scenarioId: str = Field(min_length=1, max_length=200)
    scenarioVersion: int = Field(1, ge=1)
    variables: list[dict] = Field(default_factory=list)
    userId: str | None = Field(None, max_length=200)

class UserScenarioVariablesRequest(BaseModel):
    userId: str = Field(min_length=1, max_length=200)
    scenarioId: str = Field(min_length=1, max_length=200)
    scenarioVersion: int = Field(1, ge=1)
    variables: list[dict] = Field(default_factory=list)


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
    """Liveness check; does not depend on external providers."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness check for deployment probes."""
    registry_health = get_registry().health()
    if not registry_health.get("healthy"):
        raise HTTPException(status_code=503, detail={"error": "Telecom standards registry is not ready"})
    return {"status": "ready", "registry": {
        "standards": registry_health["standards"],
        "entities": registry_health["entities"],
    }}


def _read_csv_upload(file: UploadFile) -> str:
    """Read a UTF-8 CSV with an explicit size ceiling."""
    filename = (file.filename or "").strip().lower()
    if filename and not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail={"error": "Only .csv files are accepted"})

    raw = file.file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": f"CSV exceeds the configured size limit of {MAX_CSV_BYTES} bytes"},
        )
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail={"error": f"CSV must be UTF-8 encoded: {exc}"}) from exc


def _country_from_csv_params(variables: list[dict]) -> str | None:
    """Infer country code/name from CSV variable params when present."""
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
        hits: list[str] = []
        if not text:
            return hits
        tokens = [t for t in re.split(r"[^A-Za-z0-9+]+", str(text)) if t]
        for token in tokens:
            mapped = _normalize_country_hint(token)
            if mapped:
                hits.append(mapped)
        lower_text = str(text).lower()
        for name, code in country_name_to_code.items():
            if name in lower_text:
                hits.append(code)
        for currency in known_currency_codes:
            if re.search(rf"\b{re.escape(currency)}\b", str(text), flags=re.IGNORECASE):
                hits.append(currency_to_country[currency])
        return hits

    candidates: list[str] = []
    for var in variables:
        if not isinstance(var, dict):
            continue
        for text_key in ("description", "name"):
            candidates.extend(_scan_text_hints(str(var.get(text_key, ""))))
        params = var.get("params")
        if not isinstance(params, dict):
            continue
        for key in ("country", "country_code", "countryCode"):
            value = params.get(key)
            normalized = _normalize_country_hint(value)
            if normalized:
                candidates.append(normalized)
        for value in params.values():
            if isinstance(value, str):
                candidates.extend(_scan_text_hints(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        candidates.extend(_scan_text_hints(item))
    if not candidates:
        return None
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate] = counts.get(candidate, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


@app.post("/scenario/variables/recommend")
def recommend_scenario_variables(req: ScenarioVariableRecommendationRequest):
    """Persist a selected subset of a proposal as scenario-level DB recommendations."""
    try:
        count = upsert_recommended(req.scenarioId.strip(), req.scenarioVersion, req.variables, req.userId)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, "scenarioId": req.scenarioId, "scenarioVersion": req.scenarioVersion, "saved": count}

@app.get("/scenario/{scenario_id}/variables/recommended")
def recommended_scenario_variables(scenario_id: str, scenarioVersion: int = 1):
    return {"success": True, "scenarioId": scenario_id, "scenarioVersion": scenarioVersion, "variables": get_recommended(scenario_id, scenarioVersion)}

@app.post("/scenario/variables/user")
def save_user_scenario_variables(req: UserScenarioVariablesRequest):
    try:
        count = set_user_variables(req.userId.strip(), req.scenarioId.strip(), req.scenarioVersion, req.variables)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, "userId": req.userId, "scenarioId": req.scenarioId, "scenarioVersion": req.scenarioVersion, "saved": count}

@app.get("/scenario/{scenario_id}/variables/user/{user_id}")
def get_saved_user_scenario_variables(scenario_id: str, user_id: str, scenarioVersion: int = 1):
    return {"success": True, "userId": user_id, "scenarioId": scenario_id, "scenarioVersion": scenarioVersion, "variables": get_user_variables(user_id, scenario_id, scenarioVersion), "records": get_user_variable_records(user_id, scenario_id, scenarioVersion)}

@app.patch("/scenario/{scenario_id}/variables/user/{user_id}/{variable_key}")
def edit_user_scenario_variable(scenario_id: str, user_id: str, variable_key: str, changes: dict, scenarioVersion: int = 1):
    current = get_user_variable_records(user_id, scenario_id, scenarioVersion)
    match = next((item for item in current if item.get("variable_key") == variable_key), None)
    if not match:
        raise HTTPException(404, detail={"error": "User scenario variable not found"})
    definition = dict(match.get("definition") or {})
    definition.update(changes or {})
    try:
        set_user_variables(user_id, scenario_id, scenarioVersion, [definition], state="OVERRIDDEN")
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, "userId": user_id, "scenarioId": scenario_id, "variableKey": variable_key, "updated": True}

@app.delete("/scenario/{scenario_id}/variables/user/{user_id}/{variable_key}")
def remove_user_scenario_variable(scenario_id: str, user_id: str, variable_key: str, scenarioVersion: int = 1):
    deleted = delete_user_variable(user_id, scenario_id, scenarioVersion, variable_key)
    if not deleted:
        raise HTTPException(404, detail={"error": "User scenario variable not found"})
    return {"success": True, "userId": user_id, "scenarioId": scenario_id, "variableKey": variable_key, "deleted": True}

@app.post("/scenario/import-csv", response_model=ScenarioImportResponse)
def import_scenario_csv(
    file: UploadFile = File(..., description="CSV scenario definition containing variables"),
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
    """Import a CSV scenario definition and create an explicit CSV/HITL draft.

    This path remains intentionally separate from /scenario/propose so clients can
    choose between an explicit CSV schema definition and agentic schema proposal.
    """
    csv_text = _read_csv_upload(file)

    try:
        requested_type = str(typeOfData).strip().lower() if typeOfData else None
        detected_type = requested_type or infer_type_of_data(csv_text)
        variables, field_order = parse_definition_csv(csv_text, type_of_data=detected_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    csv_country = _country_from_csv_params(variables)
    effective_country = csv_country or country
    variable_names = {str(v.get("name")) for v in variables if isinstance(v, dict) and v.get("name")}

    if detected_type == "transactional":
        if entityKey:
            token = str(entityKey).strip()
            entityKey = next((name for name in variable_names if name.lower() == token.lower()), None)
        if not entityKey:
            preferred = (
                "subscriber_id", "customer_id", "account_id", "user_id",
                "entity_id", "customer_key", "entity_key", "id",
            )
            entityKey = next((name for name in preferred if name in variable_names), None)
            entityKey = entityKey or next(iter(variable_names), None)
        if not entityKey:
            raise HTTPException(
                status_code=400,
                detail={"error": "Transactional CSV must contain at least one user/entity identifier field"},
            )
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
        "type_of_data": detected_type,
        "entity_key": entityKey,
        "records_per_user": 10,
        "agentic": False,
        "source": "csv_import",
    }
    save_draft(draft_id, draft)

    return ScenarioImportResponse(
        success=True,
        draft_id=draft_id,
        scenario_id=scenarioId,
        requested_scenario_id=scenarioId,
        journey=draft["journey"],
        description=draft["description"],
        variables=variables,
        field_order=field_order,
        typeOfData=detected_type,
        entityKey=entityKey,
    )



@app.post("/scenario/confirm", response_model=ConfirmResponse)
def confirm_scenario_route(req: ConfirmRequest):
    """HITL approval boundary: approve/edit the draft and persist it for /scenario/generate."""
    draft=get_draft(req.draft_id)
    if draft is None:
        raise HTTPException(404,detail={"error":f"Unknown or expired draft_id '{req.draft_id}'"})

    if draft.get("agentic"):
        try:
            variables, field_order = AgenticSchemaWorkflow.validate_hitl_changes(
                draft, req.add, req.edit, req.delete
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    else:
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
    requested_scenario_id=draft.get("scenario_id")
    scenario_id=requested_scenario_id
    meta={
        "label":draft.get("label",scenario_id),"journey":draft.get("journey",draft.get("domain","")),"description":draft.get("description",""),
        "domain":draft.get("domain",""),"business_scenario":draft.get("business_scenario",""),"business_response":draft.get("business_response"),
        "expected_outcome":draft.get("expected_outcome"),"scenario_type":draft.get("scenario_type"),"use_case":draft.get("use_case"),
        "industry":draft.get("industry_type","generic"),"country":draft.get("country"),"requested_scenario_id":requested_scenario_id,
        "type_of_data":type_of_data,"entity_key":entity_key,"records_per_user":10,
        "agentic": bool(draft.get("agentic", False)),
    }
    scenario_id, scenario_id_reassigned = confirm_scenario(
        scenario_id, meta, variables, field_order, draft_id=req.draft_id
    )
    invalidate_scenario(scenario_id)
    clear_scenario(scenario_id)
    if not _is_placeholder(req.feedback):
        add_feedback(draft.get("domain", ""), draft.get("business_scenario", ""), req.feedback)
    pop_draft(req.draft_id)
    return ConfirmResponse(
        success=True,
        scenario_id=scenario_id,
        requested_scenario_id=requested_scenario_id,
        scenario_id_reassigned=scenario_id_reassigned,
        draft_id=req.draft_id,
        label=meta["label"],
        journey=meta["journey"],
        description=meta["description"],
        variables=variables,
        field_order=field_order,
        typeOfData=type_of_data,
        entityKey=entity_key,
    )


@app.post("/scenario/propose", response_model=ScenarioImportResponse)
def propose_scenario(req: ScenarioProposeRequest):
    """Create a standards-backed dynamic schema draft from the JSON request body.

    The response intentionally preserves the former scenario-import response contract so the existing frontend
    can reuse the same HITL review screen and call /scenario/confirm unchanged.
    """
    try:
        return get_agentic_workflow(registry=get_registry()).propose(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    except (RuntimeError, EnvironmentError) as exc:
        # Configuration/dependency errors are service-unavailable errors. Unexpected
        # runtime bugs must continue to reach the global 500 handler instead of being
        # mislabeled as a provider outage.
        raise HTTPException(status_code=503, detail={"error": str(exc)}) from exc


def _timestamp_sort_key(value):
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = None
            for fmt in (
                "%d/%m/%Y %I:%M %p",
                "%d/%m/%Y %I:%M:%S %p",
                "%d/%m/%Y %H:%M",
                "%d/%m/%Y %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@app.post("/scenario/generate", response_model=GenerateAcceptedResponse, status_code=202)
def generate_scenario(req: GenerateRequest):
    """Queue scenario generation and return immediately; full deterministic QA runs in a worker."""
    scenario_id = req.scenario
    if req.draftId:
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

    payload = {
        "scenario": scenario_id,
        "draftId": req.draftId,
        "count": req.count,
        "recordsPerUser": req.recordsPerUser,
    }
    job = submit_generation_job(payload)
    return GenerateAcceptedResponse(
        success=True,
        status="queued",
        jobId=job["job_id"],
        scenario_id=scenario_id,
        draft_id=req.draftId,
        totalCount=req.count,
        recordsPerUser=req.recordsPerUser,
    )


@app.get("/scenario/generate/{job_id}", response_model=GenerateJobResponse)
def get_generation_job(job_id: str):
    """Return generation-job status and the fully validated result when complete."""
    job = GenerationJobModel.get(job_id)
    if job is None:
        raise HTTPException(404, detail={"error": f"Unknown generation job '{job_id}'"})

    result = None
    if job.get("status") == "completed":
        payload = GenerationJobModel.result(job_id)
        if payload is not None:
            result = GenerateResponse.model_validate(payload)

    return GenerateJobResponse(
        success=job.get("status") != "failed",
        status=job.get("status", "failed"),
        jobId=job_id,
        scenario_id=job.get("scenario_id"),
        draft_id=job.get("draft_id"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        failed_at=job.get("failed_at"),
        error=job.get("error"),
        result=result,
    )


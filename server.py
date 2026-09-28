from __future__ import annotations
import logging
import re
import secrets
import uuid
from time import perf_counter
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from starlette.concurrency import run_in_threadpool
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
    resolve_requested_scenario_id_from_draft,
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
from config.country_metadata import COUNTRY_BASE
from config.runtime import CORS_ALLOW_ORIGINS, MAX_CSV_BYTES, INDUSTRY_SOURCE_MAX_JSON_BYTES, INDUSTRY_SOURCE_ADMIN_TOKEN
from core.agentic_workflow import AgenticSchemaWorkflow, get_agentic_workflow
from core.json_domain_policy import is_json_grounded_domain, is_low_balance_domain
from core.low_balance_variable_policy import validate_low_balance_variable_sources
from core.scenario_variable_store import upsert_recommended, get_recommended, set_user_variables, get_user_variables, get_user_variable_records, delete_user_variable
from models.database import ping as ping_mongodb
from models.model_registry import ensure_indexes as ensure_model_indexes
from core.generation_service import build_generation_response
from core.industry_source_store import (
    delete_source_document,
    get_source_document,
    list_source_documents,
    save_source_document,
    set_source_active,
    generate_internal_source_id,
    invalidate_catalog_cache,
)




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
        logger.info("Industry source registry ready: source_of_truth=mongodb")
    except Exception:
        logger.exception("Application startup validation failed")
        raise
    yield


app = FastAPI(
    title="Telco Agentic SDG",
    version="2.9.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
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


# Swagger UI compatibility: FastAPI/Pydantic versions that emit OpenAPI 3.1
# may describe UploadFile[] with `contentMediaType`, which Swagger UI renders
# as array<string> text fields in /docs. Keep the actual FastAPI endpoint as
# list[UploadFile] and normalize only the documented request-body schema to the
# widely-supported `format: binary` representation for an array of files.
_default_openapi = app.openapi


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = _default_openapi()
    upload_path = schema.get("paths", {}).get("/industry-sources/upload")
    if upload_path:
        operation = upload_path.get("post", {})
        request_body = operation.get("requestBody", {})
        multipart = request_body.get("content", {}).get("multipart/form-data")
        if multipart is not None:
            multipart["schema"] = {
                "type": "object",
                "required": ["industryType", "domain", "file"],
                "properties": {
                    "industryType": {
                        "type": "string",
                        "title": "Industry Type",
                        "description": "Industry type for all uploaded JSON sources",
                    },
                    "domain": {
                        "type": "string",
                        "title": "Domain",
                        "description": "Domain for all uploaded JSON sources",
                    },
                    "file": {
                        "type": "array",
                        "title": "JSON Files",
                        "description": "Select one or more JSON source files for this industry/domain",
                        "minItems": 1,
                        "maxItems": 25,
                        "items": {
                            "type": "string",
                            "format": "binary",
                        },
                    },
                },
            }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


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
    allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-Industry-Source-Token"],
)

# Compress large JSON generation/proposal responses to reduce transfer time without changing payload semantics.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    requested_scenario_id: str | None = Field(None, alias="requestedScenarioId", examples=["LB-01"], description="Canonical requested scenario identifier")
    draftId: str | None = Field(None, description="Confirmed draft id associated with the active confirmed scenario definition")
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
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    requested_scenario_id: str = Field(alias="requestedScenarioId", min_length=1, max_length=200)
    scenarioVersion: int = Field(1, ge=1)
    variables: list[dict] = Field(default_factory=list)
    userId: str | None = Field(None, max_length=200)


class UserScenarioVariablesRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    userId: str = Field(min_length=1, max_length=200)
    requested_scenario_id: str = Field(alias="requestedScenarioId", min_length=1, max_length=200)
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
    variableSources: dict[str, str] = Field(default_factory=dict, description="Internal provenance for source-backed scenario variables")


class IndustrySourceStatusRequest(BaseModel):
    active: bool


class IndustrySourceResponse(BaseModel):
    success: bool = True
    source: dict | None = None
    sources: list[dict] = Field(default_factory=list)
    uploaded: int = 0
    errors: list[dict] = Field(default_factory=list)


def _require_industry_source_admin_token(request: Request) -> None:
    """Protect destructive source-registry mutations with a deployment-provided admin token."""
    expected = INDUSTRY_SOURCE_ADMIN_TOKEN
    if not expected:
        raise HTTPException(503, detail={"error": "Industry source administration is disabled; set INDUSTRY_SOURCE_ADMIN_TOKEN"})
    supplied = str(request.headers.get("X-Industry-Source-Token") or "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, detail={"error": "Invalid industry source administration token"})


def _normalize_upload_metadata(values: list[str] | None, file_count: int, field_name: str) -> list[str | None]:
    """Normalize repeated multipart metadata so per-file source IDs/names are unambiguous."""
    if not values:
        return [None] * file_count
    cleaned = [str(value).strip() for value in values]
    if len(cleaned) == 1 and file_count == 1:
        return cleaned
    if len(cleaned) != file_count:
        raise HTTPException(400, detail={
            "error": f"{field_name} must be omitted or supplied once per uploaded file",
            "expected": file_count,
            "received": len(cleaned),
        })
    return cleaned


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
    try:
        ping_mongodb()
        source_count = len(list_source_documents(active_only=True))
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": f"MongoDB/source registry is not ready: {exc}"}) from exc
    return {"status": "ready", "sourceOfTruth": "mongodb", "industrySources": {"active": source_count}}


def _read_json_source_upload(file: UploadFile) -> tuple[bytes, dict]:
    """Read and parse one standards JSON with a bounded upload size."""
    filename = (file.filename or "").strip().lower()
    if filename and not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail={"error": "Only .json files are accepted"})
    raw = file.file.read(INDUSTRY_SOURCE_MAX_JSON_BYTES + 1)
    if len(raw) > INDUSTRY_SOURCE_MAX_JSON_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": f"JSON source exceeds the configured size limit of {INDUSTRY_SOURCE_MAX_JSON_BYTES} bytes"},
        )
    try:
        import json
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": f"Source file must contain valid UTF-8 JSON: {exc}"}) from exc
    if not isinstance(document, dict):
        raise HTTPException(status_code=400, detail={"error": "Source JSON root must be an object"})
    return raw, document


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
        count = upsert_recommended(req.requested_scenario_id.strip(), req.scenarioVersion, req.variables, req.userId)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, "requestedScenarioId": req.requested_scenario_id, "scenarioId": req.requested_scenario_id, "scenarioVersion": req.scenarioVersion, "saved": count}

@app.get("/scenario/{requested_scenario_id}/variables/recommended")
def recommended_scenario_variables(requested_scenario_id: str, scenarioVersion: int = 1):
    return {"success": True, "requestedScenarioId": requested_scenario_id, "scenarioId": requested_scenario_id, "scenarioVersion": scenarioVersion, "variables": get_recommended(requested_scenario_id, scenarioVersion)}

@app.post("/scenario/variables/user")
def save_user_scenario_variables(req: UserScenarioVariablesRequest):
    try:
        count = set_user_variables(req.userId.strip(), req.requested_scenario_id.strip(), req.scenarioVersion, req.variables)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, "userId": req.userId, "requestedScenarioId": req.requested_scenario_id, "scenarioId": req.requested_scenario_id, "scenarioVersion": req.scenarioVersion, "saved": count}

@app.get("/scenario/{requested_scenario_id}/variables/user/{user_id}")
def get_saved_user_scenario_variables(requested_scenario_id: str, user_id: str, scenarioVersion: int = 1):
    return {"success": True, "userId": user_id, "requestedScenarioId": requested_scenario_id, "scenarioId": requested_scenario_id, "scenarioVersion": scenarioVersion, "variables": get_user_variables(user_id, requested_scenario_id, scenarioVersion), "records": get_user_variable_records(user_id, requested_scenario_id, scenarioVersion)}

@app.patch("/scenario/{requested_scenario_id}/variables/user/{user_id}/{variable_key}")
def edit_user_scenario_variable(requested_scenario_id: str, user_id: str, variable_key: str, changes: dict, scenarioVersion: int = 1):
    current = get_user_variable_records(user_id, requested_scenario_id, scenarioVersion)
    match = next((item for item in current if item.get("variable_key") == variable_key), None)
    if not match:
        raise HTTPException(404, detail={"error": "User scenario variable not found"})
    definition = dict(match.get("definition") or {})
    definition.update(changes or {})
    try:
        set_user_variables(user_id, requested_scenario_id, scenarioVersion, [definition], state="OVERRIDDEN")
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, "userId": user_id, "requestedScenarioId": requested_scenario_id, "scenarioId": requested_scenario_id, "variableKey": variable_key, "updated": True}

@app.delete("/scenario/{requested_scenario_id}/variables/user/{user_id}/{variable_key}")
def remove_user_scenario_variable(requested_scenario_id: str, user_id: str, variable_key: str, scenarioVersion: int = 1):
    deleted = delete_user_variable(user_id, requested_scenario_id, scenarioVersion, variable_key)
    if not deleted:
        raise HTTPException(404, detail={"error": "User scenario variable not found"})
    return {"success": True, "userId": user_id, "requestedScenarioId": requested_scenario_id, "scenarioId": requested_scenario_id, "variableKey": variable_key, "deleted": True}

@app.get("/industry-sources")
def get_industry_sources(
    industryType: str | None = None,
    domain: str | None = None,
    activeOnly: bool = True,
):
    """List source documents registered for an optional industry/domain filter."""
    return {
        "success": True,
        "industryType": industryType,
        "domain": domain,
        "sources": list_source_documents(
            industry_type=industryType,
            domain=domain,
            active_only=activeOnly,
        ),
    }


@app.get("/industry-sources/{source_id}")
def get_industry_source(source_id: str, includeDocument: bool = False):
    source = get_source_document(source_id, include_document=includeDocument)
    if source is None:
        raise HTTPException(404, detail={"error": f"Unknown industry source '{source_id}'"})
    return {"success": True, "source": source}


@app.post("/industry-sources/upload", response_model=IndustrySourceResponse)
def upload_industry_sources(
    request: Request,
    industryType: str = Form(..., description="Industry type for all uploaded JSON sources"),
    domain: str = Form(..., description="Domain for all uploaded JSON sources"),
    file: list[UploadFile] = File(..., description="One or more Swagger/OpenAPI/JSON Schema JSON files for this industryType/domain"),
):
    _require_industry_source_admin_token(request)
    """Upload multiple standards JSON files for one exact industryType/domain pair.

    The multipart request intentionally exposes only three inputs:
      * industryType
      * domain
      * file (one or more JSON files)

    Per-file source identifiers and metadata are generated by the server from the uploaded
    filename and parsed document. This keeps the API simple and ensures every file in a batch
    belongs to the same industry/domain pair.
    """
    upload_files = list(file or [])
    if not upload_files:
        raise HTTPException(status_code=400, detail={"error": "At least one JSON file is required"})
    if len(upload_files) > 25:
        raise HTTPException(status_code=400, detail={"error": "A maximum of 25 files can be uploaded in one request"})

    prepared: list[dict] = []
    for upload in upload_files:
        raw, document = _read_json_source_upload(upload)
        prepared.append({
            "file": upload,
            "raw": raw,
            "document": document,
        })

    uploaded_sources: list[dict] = []
    errors: list[dict] = []
    for index, item in enumerate(prepared):
        upload = item["file"]
        file_name = upload.filename or f"source_{index + 1}.json"
        info = item["document"].get("info") if isinstance(item["document"].get("info"), dict) else {}
        try:
            internal_source_id = generate_internal_source_id(
                industry_type=industryType,
                domain=domain,
                file_name=file_name,
                document=item["document"],
                raw_bytes=item["raw"],
            )
            source = save_source_document(
                industry_type=industryType,
                domain=domain,
                document=item["document"],
                file_name=file_name,
                source_name=str(info.get("title") or file_name).strip(),
                standard=None,
                version=str(info.get("version") or "").strip() or None,
                description=str(info.get("description") or "").strip() or None,
                active=True,
                source_id=internal_source_id,
                raw_bytes=item["raw"],
            )
            uploaded_sources.append(source)
        except ValueError as exc:
            errors.append({
                "file": file_name,
                "index": index,
                "error": str(exc),
            })

    if uploaded_sources:
        invalidate_catalog_cache(industryType, domain)
    if not uploaded_sources and errors:
        raise HTTPException(status_code=400, detail={
            "error": "No source files were uploaded successfully",
            "files": errors,
        })

    return {
        "success": not errors,
        "source": uploaded_sources[0] if len(uploaded_sources) == 1 else None,
        "sources": uploaded_sources,
        "uploaded": len(uploaded_sources),
        "errors": errors,
    }


@app.patch("/industry-sources/{source_id}", response_model=IndustrySourceResponse)
def update_industry_source(source_id: str, req: IndustrySourceStatusRequest, request: Request):
    _require_industry_source_admin_token(request)
    if not set_source_active(source_id, req.active):
        raise HTTPException(404, detail={"error": f"Unknown industry source '{source_id}'"})
    invalidate_catalog_cache()
    source = get_source_document(source_id)
    return {"success": True, "source": source}


@app.delete("/industry-sources/{source_id}")
def remove_industry_source(source_id: str, request: Request):
    _require_industry_source_admin_token(request)
    if not delete_source_document(source_id):
        raise HTTPException(404, detail={"error": f"Unknown industry source '{source_id}'"})
    invalidate_catalog_cache()
    return {"success": True, "sourceId": source_id, "deleted": True}


@app.post("/scenario/import-csv", response_model=ScenarioImportResponse)
def import_scenario_csv(
    file: UploadFile = File(..., description="CSV scenario definition containing variables"),
    requestedScenarioId: str = Form(...),
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
    Low Balance & Top-up is intentionally source-locked to TMF654/TMF629 or MongoDB
    variables, so CSV cannot become a third variable source for that domain.
    """
    if is_low_balance_domain(domain, industryType) or is_json_grounded_domain(domain, industryType):
        message = (
            "Low Balance & Top-up variables cannot be defined through CSV. Use the active MongoDB industry source documents or scenario variables."
            if is_low_balance_domain(domain, industryType)
            else "This JSON-source-backed industry/domain cannot be defined through CSV. Use the active MongoDB industry source documents."
        )
        raise HTTPException(status_code=400, detail={"error": message})
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
        "label": label or requestedScenarioId,
        "journey": domain,
        "description": businessScenario or f"Scenario imported from CSV for {domain}",
        "variables": variables,
        "field_order": field_order,
        "domain": domain,
        "business_scenario": businessScenario,
        "business_response": businessResponse,
        "expected_outcome": expectedOutcome,
        "use_case": useCase,
        "scenario_id": requestedScenarioId,
        "requested_scenario_id": requestedScenarioId,
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
        scenario_id=requestedScenarioId,
        requested_scenario_id=requestedScenarioId,
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

    variable_sources = {}
    db_variable_names = set()
    if draft.get("agentic"):
        try:
            variables, field_order, variable_sources, db_variable_names = AgenticSchemaWorkflow.validate_hitl_changes(
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
    if is_low_balance_domain(draft.get("domain"), draft.get("industry_type")) and not draft.get("agentic"):
        try:
            validate_low_balance_variable_sources(
                variables,
                variable_sources,
                db_variable_names=db_variable_names,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    type_of_data=draft.get("type_of_data","aggregational")

    entity_key=draft.get("entity_key") if type_of_data=="transactional" else None
    if type_of_data=="transactional" and entity_key not in {v.get("name") for v in variables}:
        preferred=("subscriber_id","customer_id","account_id","user_id","entity_id","customer_key","entity_key","id")
        names={str(v.get("name")) for v in variables if v.get("name")}
        entity_key=next((n for n in preferred if n in names),None) or (next(iter(names),None) if names else None)
    requested_scenario_id=str(draft.get("requested_scenario_id") or draft.get("scenario_id") or "").strip() or None
    scenario_id=requested_scenario_id
    meta={
        "label":draft.get("label",scenario_id),"journey":draft.get("journey",draft.get("domain","")),"description":draft.get("description",""),
        "domain":draft.get("domain",""),"business_scenario":draft.get("business_scenario",""),"business_response":draft.get("business_response"),
        "expected_outcome":draft.get("expected_outcome"),"scenario_type":draft.get("scenario_type"),"use_case":draft.get("use_case"),
        "industry":draft.get("industry_type","generic"),"country":draft.get("country"),"requested_scenario_id":requested_scenario_id,
        "type_of_data":type_of_data,"entity_key":entity_key,"records_per_user":10,
        "agentic": bool(draft.get("agentic", False)),
        # Preserve the source boundary through confirmation so generation can enforce it
        # without modifying DB-owned variable definitions.
        "variable_sources": {
            str(name).strip().casefold(): str(source).strip().upper()
            for name, source in (draft.get("variable_sources") or {}).items()
            if str(name).strip() and str(source).strip()
        },
        "db_variable_names": sorted(str(name).strip().casefold() for name in (draft.get("db_variable_names") or []) if str(name).strip()),
        "source_policy": draft.get("source_policy", "scenario_variables"),
        "source_documents": list(draft.get("source_documents") or []),
    }
    try:
        scenario_id, scenario_id_reassigned = confirm_scenario(
            requested_scenario_id, meta, variables, field_order, draft_id=req.draft_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc
    invalidate_scenario(scenario_id)
    clear_scenario(scenario_id)
    if not _is_placeholder(req.feedback):
        add_feedback(requested_scenario_id, draft.get("domain", ""), draft.get("business_scenario", ""), req.feedback)
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
        variableSources=meta.get("variable_sources") or {},
    )


@app.post("/scenario/propose", response_model=ScenarioImportResponse)
async def propose_scenario(req: ScenarioProposeRequest):
    """Create a standards-backed dynamic schema draft from the JSON request body.

    The response intentionally preserves the former scenario-import response contract so the existing frontend
    can reuse the same HITL review screen and call /scenario/confirm unchanged.
    """
    started = perf_counter()
    try:
        result = await run_in_threadpool(get_agentic_workflow().propose, req)
        logger.info("[Latency] /scenario/propose scenario=%s elapsed_ms=%.1f", req.requested_scenario_id, (perf_counter() - started) * 1000.0)
        return result
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


@app.post("/scenario/generate", response_model=GenerateResponse)
async def generate_scenario(req: GenerateRequest):
    """Generate and return the complete validated dataset synchronously.

    The request remains open until deterministic generation and final QA are complete. This
    preserves the exact response contract while the generation engine itself is optimized
    to avoid repeated dependency planning, formula parsing, and duplicate validation passes.
    """
    requested_scenario_id = req.requested_scenario_id
    if req.draftId:
        resolved = resolve_requested_scenario_id_from_draft(req.draftId)
        if resolved is None:
            raise HTTPException(404, detail={"error": f"Unknown or unconfirmed draftId '{req.draftId}'"})
        if requested_scenario_id and requested_scenario_id != resolved:
            raise HTTPException(400, detail={
                "error": f"draftId '{req.draftId}' does not match requested scenario '{requested_scenario_id}'",
                "draft_scenario_id": resolved,
            })
        requested_scenario_id = resolved
    if not requested_scenario_id:
        raise HTTPException(400, detail={"error": "Either 'requestedScenarioId' or 'draftId' is required"})
    if not scenario_exists(requested_scenario_id):
        raise HTTPException(400, detail={"error": f"Unknown requested scenario '{requested_scenario_id}'"})

    started = perf_counter()
    try:
        payload = await run_in_threadpool(build_generation_response, {
            "requested_scenario_id": requested_scenario_id,
            "draftId": req.draftId,
            "count": req.count,
            "recordsPerUser": req.recordsPerUser,
        })
        # build_generation_response performs the full contract/type/semantic validation. Avoid a
        # second recursive Pydantic traversal of potentially millions of generated values. The
        # response_model remains the OpenAPI contract; returning a Response bypasses duplicate work.
        logger.info(
            "[Latency] /scenario/generate scenario=%s records=%d fields=%d elapsed_ms=%.1f",
            requested_scenario_id,
            int(payload.get("total_records", 0) or 0),
            len(payload.get("fields") or []),
            (perf_counter() - started) * 1000.0,
        )
        return ORJSONResponse(content=payload)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc



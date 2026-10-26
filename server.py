from __future__ import annotations
import logging
import re
import uuid
from time import perf_counter
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

from core import catalog_store
from core.catalog_models import (
    CatalogImportRequest,
    CatalogCountry,
    CatalogImportResponse,
    CatalogItem,
    CatalogJourney,
    CatalogScenarioDetail,
    CatalogScenarioList,
    CatalogTree,
)
from core.seed import ensure_seeded
from core.dynamic_scenarios import (
    confirm_scenario,
    get_draft,
    new_draft_id,
    save_draft,
    pop_draft,
    resolve_requested_scenario_id_from_draft,
    scenario_exists,
)
from core.compiled_schema import invalidate_scenario
from core.runtime_cache import clear_scenario
from core.agentic_models import CsvImportResponse, ScenarioImportResponse, ScenarioProposeRequest
from core.csv_scenario import infer_type_of_data, parse_definition_csv
from config.country_metadata import COUNTRY_BASE
from config.runtime import CORS_ALLOW_ORIGINS, MAX_CSV_BYTES, INDUSTRY_SOURCE_MAX_JSON_BYTES
from core.agentic_workflow import AgenticSchemaWorkflow, get_agentic_workflow
from core.json_domain_policy import is_json_grounded_domain
from core.scenario_variable_store import upsert_recommended, get_recommended, set_user_variables, get_user_variables, get_user_variable_records, delete_user_variable
from models.database import ping as ping_mongodb
from models.model_registry import ensure_indexes as ensure_model_indexes
from core.generation_service import build_generation_response, generation_status
from synth.service import SpecNotReady
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

from core.api_auth import api_token_middleware, log_startup as log_auth_startup, openapi_security
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
    log_auth_startup()
    try:
        ping_mongodb()
        ensure_model_indexes()
        ensure_seeded()
        logger.info("Industry source registry ready: source_of_truth=mongodb")
    except Exception:
        logger.exception("Application startup validation failed")
        raise
    yield


app = FastAPI(
    title="Telco Agentic SDG",
    version="2.14.2+er-poc",
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

    app.openapi_schema = openapi_security(schema)
    return app.openapi_schema


app.openapi = _custom_openapi


# Registered before the request-id middleware so it runs inside it (the 401 carries the request id) and inside CORS (the 401 carries
# the CORS headers, so a browser shows the real error and not a blocked request).
app.middleware("http")(api_token_middleware)


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
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    # So a browser can read the download's file name (the ER zip export) and quote the request id when reporting a problem.
    expose_headers=["Content-Disposition", "X-Request-ID"],
)

# Compress large JSON generation/proposal responses to reduce transfer time without changing payload semantics.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

# ER-model proof of concept (branch feature/er-model-poc): GET /er/samples, POST /er/validate, /er/generate, /er/export.
# Standard library only, no database and no model call; the API-token middleware above protects these routes like any other.
from er.api import router as er_router  # noqa: E402
app.include_router(er_router)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    requested_scenario_id: str | None = Field(None, alias="requestedScenarioId", examples=["LB-01"], description="Canonical requested scenario identifier")
    draftId: str | None = Field(None, description="Confirmed draft id associated with the active confirmed scenario definition")
    count: int = Field(35, ge=1, le=5000, description="Number of users/entities to generate for a transactional scenario")
    recordsPerUser: int = Field(10, ge=1, le=10, description="Number of most-recent historical records returned per user for a transactional scenario")
    seed: int | None = Field(None, ge=0, le=2_147_483_647, description="Optional deterministic generation seed for reproducible synthetic-data runs")
    as_of: str | None = Field(None, alias="asOf", max_length=40, description="Optional ISO-8601 reference time (\"now\") for generation; with a seed it makes a run exactly reproducible")


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
    add: list[dict] = Field(default_factory=list, description="Optional new variables: name, dtype, gen (required) and params, formula, depends_on, nullable, description, scope. Where the column sits is set by the system (story order) unless a sequence is given")
    edit: list[VariableEdit] = Field(default_factory=list, description="Optional HITL edits to existing variables")
    delete: list[str] = Field(default_factory=list, description="Optional variable names to delete")
    feedback: str | None = Field(default=None, description="Accepted for compatibility; feedback is not stored")


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
    generationSpecKey: str | None = Field(default=None, description="Key of the verified generation spec pinned to this scenario, if one could be designed")
    specReport: dict | None = Field(default=None, description="How the generation spec was obtained, what it assumes, and any problem")


class IndustrySourceStatusRequest(BaseModel):
    active: bool


class IndustrySourceResponse(BaseModel):
    success: bool = True
    source: dict | None = None
    sources: list[dict] = Field(default_factory=list)
    uploaded: int = 0
    errors: list[dict] = Field(default_factory=list)


@app.get("/")
def root():
    """Liveness check; ``version`` says which build is running."""
    return {"success": True, "status": "ok", "version": app.version}


@app.get("/health")
def health():
    """Liveness check; does not depend on external providers. ``version`` says which build is running."""
    return {"status": "ok", "version": app.version}


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
    industryType: str = Form(..., description="Industry type for all uploaded JSON sources"),
    domain: str = Form(..., description="Domain for all uploaded JSON sources"),
    file: list[UploadFile] = File(..., description="One or more Swagger/OpenAPI/JSON Schema JSON files for this industryType/domain"),
):
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
def update_industry_source(source_id: str, req: IndustrySourceStatusRequest):
    if not set_source_active(source_id, req.active):
        raise HTTPException(404, detail={"error": f"Unknown industry source '{source_id}'"})
    invalidate_catalog_cache()
    source = get_source_document(source_id)
    return {"success": True, "source": source}


@app.delete("/industry-sources/{source_id}")
def remove_industry_source(source_id: str):
    if not delete_source_document(source_id):
        raise HTTPException(404, detail={"error": f"Unknown industry source '{source_id}'"})
    invalidate_catalog_cache()
    return {"success": True, "sourceId": source_id, "deleted": True}


class CountryRail(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = Field(None, description="Short id of the rail, for example upi")
    label: str | None = Field(None, description="Display name of the rail")
    aliases: list[str] = Field(..., min_length=1, description="Words by which a column value names the rail (whole words, case-insensitive)")
    kind: str | None = None
    min_amount: float | None = Field(None, description="Floor of one transaction on this rail (a high-value rail), in the country's currency")
    max_amount: float | None = Field(None, description="Cap of one transaction on this rail, in the country's currency")
    fee: float | None = Field(None, description="Fixed fee on this rail; 0 means the rail is free")
    note: str | None = None


class CountryLimitsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    currency: str | None = Field(None, description="ISO 4217 code the amounts are in")
    cash_denomination: float | None = Field(None, description="Smallest note an ATM dispenses; cash withdrawals are multiples of it")
    note: str | None = None
    rails: list[CountryRail] = Field(default_factory=list)


@app.get("/country-limits")
def list_country_limits():
    """The countries whose payment-rail limits (floors, caps, fees, cash denomination) are stored."""
    from models.country_limit import CountryLimitModel

    return {"success": True, "countries": CountryLimitModel.list_all()}


@app.get("/country-limits/{country}")
def get_country_limits(country: str):
    """The payment-rail limits used when a scenario is proposed for ``country`` (a code such as IN, or its name)."""
    from core import country_limits

    found = country_limits.for_country(country)
    if not found:
        raise HTTPException(404, detail={"error": f"No payment-rail limits are stored for '{country}'"})
    return {"success": True, "limits": found}


@app.put("/country-limits/{country}")
def put_country_limits(country: str, req: CountryLimitsRequest):
    """Store (replace) the payment-rail limits of a country. Specs designed for that country afterwards use them."""
    from core import country_limits
    from models.country_limit import CountryLimitModel

    code = country_limits.country_code(country)
    if not code:
        raise HTTPException(400, detail={"error": f"'{country}' is not a country code or a known country name"})
    document = req.model_dump(exclude_none=True)
    if not country_limits.clean({**document, "country": code}):
        raise HTTPException(400, detail={"error": "The document holds no usable rail (every rail needs aliases, and min_amount may not exceed max_amount)"})
    CountryLimitModel.put(code, document)
    country_limits.clear_cache()
    return {"success": True, "country": code, "limits": country_limits.for_country(code)}


@app.delete("/country-limits/{country}")
def delete_country_limits(country: str):
    from core import country_limits
    from models.country_limit import CountryLimitModel

    code = country_limits.country_code(country)
    if not code or not CountryLimitModel.delete(code):
        raise HTTPException(404, detail={"error": f"No payment-rail limits are stored for '{country}'"})
    country_limits.clear_cache()
    return {"success": True, "country": code, "deleted": True}


def _catalog_lookup(read, *args):
    """Run a catalog read; an id that is not in the catalog is a 404 and not an empty dropdown."""
    try:
        return read(*args)
    except catalog_store.CatalogNotFound as exc:
        raise HTTPException(404, detail={"error": str(exc)}) from exc


@app.get("/catalog/countries", response_model=list[CatalogCountry])
def catalog_countries():
    """Countries a scenario can be proposed for (the country field of /scenario/propose)."""
    return catalog_store.countries()


@app.get("/catalog/industries", response_model=list[CatalogItem])
def catalog_industries():
    """Industry domains for the first dropdown."""
    return catalog_store.industries()


@app.get("/catalog/sub-domains", response_model=list[CatalogItem])
def catalog_sub_domains(industryId: str):
    """Sub-domains of one industry."""
    return _catalog_lookup(catalog_store.sub_domains, industryId)


@app.get("/catalog/use-cases", response_model=list[CatalogItem])
def catalog_use_cases(subDomainId: str):
    """Use cases of one sub-domain."""
    return _catalog_lookup(catalog_store.use_cases, subDomainId)


@app.get("/catalog/journeys", response_model=list[CatalogJourney])
def catalog_journeys(useCaseId: str):
    """Customer journeys of one use case, with the data types each supports."""
    return _catalog_lookup(catalog_store.journeys, useCaseId)


@app.get("/catalog/scenarios", response_model=CatalogScenarioList)
def catalog_scenarios(journeyId: str):
    """Scenarios of one journey, with the 'all scenarios' label."""
    return _catalog_lookup(catalog_store.scenarios, journeyId)


@app.get("/catalog/scenarios/{scenario_id}", response_model=CatalogScenarioDetail)
def catalog_scenario(scenario_id: str):
    """One scenario with its path through the catalog and the /scenario/propose fields the catalog knows."""
    return _catalog_lookup(catalog_store.scenario_detail, scenario_id)


@app.get("/catalog/tree", response_model=CatalogTree)
def catalog_tree():
    """The whole active catalog in one call."""
    return catalog_store.tree()


@app.post("/catalog/import", response_model=CatalogImportResponse)
def catalog_import(req: CatalogImportRequest):
    """Add catalog entries and update the ones that exist; switch an entry off with ``active: false``. Nothing is deleted."""
    try:
        counts = catalog_store.import_catalog(req)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc
    return {"success": True, **counts}


@app.post("/scenario/import-csv", response_model=CsvImportResponse)
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
    A domain backed by registered source documents is source-locked, so CSV cannot become a
    second variable source for it.
    """
    if is_json_grounded_domain(domain, industryType):
        raise HTTPException(status_code=400, detail={
            "error": "This JSON-source-backed industry/domain cannot be defined through CSV. "
                     "Use the active MongoDB industry source documents."})
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
                "customer_id", "account_id", "user_id",
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

    from synth.sequence import assign as assign_sequence

    variables = assign_sequence(variables, entity_key=entityKey)           # each variable carries its place in the data's story
    field_order = [str(v.get("name")) for v in variables if v.get("name")]

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
    # A CSV definition is as much a source of truth as an industry document: the model designs how its columns behave
    # together for this scenario (verified against the CSV's own allowed values, ranges and patterns), once, here.
    spec_report = _prepare_generation_spec(
        variables, {**draft, "industry": industryType, "scenario_type": scenarioType})
    draft["generation_spec_key"] = spec_report.get("spec_key")
    draft["spec_report"] = spec_report
    save_draft(draft_id, draft)

    return CsvImportResponse(
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
        generationSpecKey=draft["generation_spec_key"],
        specReport=spec_report,
    )


@app.post("/scenario/confirm", response_model=ConfirmResponse)
def confirm_scenario_route(req: ConfirmRequest):
    """HITL approval boundary: approve/edit the draft and persist it for /scenario/generate."""
    draft=get_draft(req.draft_id)
    if draft is None:
        raise HTTPException(404,detail={"error":f"Unknown or expired draft_id '{req.draft_id}'"})

    variable_sources = {}
    added_sources: dict[str, str] = {}
    db_variable_names = set()
    if draft.get("agentic"):
        try:
            variables, field_order, variable_sources, db_variable_names = AgenticSchemaWorkflow.validate_hitl_changes(
                draft, req.add, req.edit, req.delete
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        # Confirmed definitions keep the source label the proposal gave them, so the scenario is described to the
        # generation design exactly as it was proposed.
        for variable in variables:
            label = variable_sources.get(str(variable.get("name") or "").strip().casefold())
            if label and not variable.get("source"):
                variable["source"] = label
        added_sources = {str(n).strip().casefold(): "USER_ADDED" for n, lab in variable_sources.items() if lab == "USER_ADDED"}
        # A variable the reviewer added is placed by the intent agent, like every proposed one (unless the reviewer gave a number).
        given = {str(item.get("name") or "").strip().casefold() for item in req.add if isinstance(item, dict) and item.get("sequence") is not None}
        to_place = [str(v.get("name")) for v in variables if str(v.get("name") or "").strip().casefold() in added_sources
                    and str(v.get("name") or "").strip().casefold() not in given]
        if to_place:
            settled = [str(v.get("name")) for v in variables if str(v.get("name")) not in to_place]
            variables = [{k: x for k, x in v.items() if k != "sequence"} if str(v.get("name")) in to_place else v for v in variables]
            variables = get_agentic_workflow().order_columns(
                variables,
                context={"industry": draft.get("industry_type"), "domain": draft.get("domain"), "use_case": draft.get("use_case"),
                         "country": draft.get("country"), "scenario_type": draft.get("scenario_type")},
                scenario_id=str(draft.get("requested_scenario_id") or draft.get("scenario_id") or ""),
                entity_key=draft.get("entity_key"),
                existing=settled,
            )
            field_order = [str(v.get("name")) for v in variables if v.get("name")]
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
        preferred=("customer_id","account_id","user_id","entity_id","customer_key","entity_key","id")
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
            for name, source in {**(draft.get("variable_sources") or {}), **added_sources}.items()
            if str(name).strip() and str(source).strip()
        },
        "db_variable_names": sorted(str(name).strip().casefold() for name in (draft.get("db_variable_names") or []) if str(name).strip()),
        "source_policy": draft.get("source_policy", "scenario_variables"),
        "source_documents": list(draft.get("source_documents") or []),
        "behavioral_rules": [dict(rule) for rule in (draft.get("behavioral_rules") or []) if isinstance(rule, dict)],
        # Preserve the immutable JSON source provenance through HITL confirmation.
        # Generation must rely on the confirmed contract rather than querying live source
        # document IDs on every request; source IDs remain useful for audit/debugging.
        "variable_source_ids": {
            str(name).strip().casefold(): str(source_id).strip()
            for name, source_id in (draft.get("variable_source_ids") or {}).items()
            if str(name).strip() and str(source_id).strip()
        },
    }
    # Pin the verified generation spec for exactly these variables: the proposal's own spec when nothing changed, that
    # spec narrowed when columns were removed, a fresh design when columns were added. If none can be designed the
    # scenario is still confirmed and generation says so in its report.
    spec_report = _prepare_generation_spec(variables, meta, hint_key=draft.get("generation_spec_key"))
    meta["generation_spec_key"] = spec_report.get("spec_key")
    try:
        scenario_id, scenario_id_reassigned = confirm_scenario(
            requested_scenario_id, meta, variables, field_order, draft_id=req.draft_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc
    invalidate_scenario(scenario_id)
    clear_scenario(scenario_id)
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
        generationSpecKey=meta["generation_spec_key"],
        specReport=spec_report,
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


def _prepare_generation_spec(variables: list[dict], meta: dict, *, hint_key: str | None = None) -> dict:
    """Name the generation spec for ``variables`` and make sure its design is under way (never waits, never raises).

    The report's ``spec_key`` is what the scenario pins; ``status`` is ``ready`` when the verified spec already exists and
    ``designing`` when it is being designed in the background (generation waits for it if it is still running).
    """
    try:
        from synth.service import make_brief, pin_spec

        return dict(pin_spec(variables, make_brief(meta), hint_key=hint_key))
    except Exception as exc:
        logger.exception("generation spec could not be prepared")
        return {"status": "unavailable", "spec_key": None, "problems": [f"{type(exc).__name__}: {exc}"[:300]]}


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


@app.get("/scenario/{requested_scenario_id}/generation-status")
async def scenario_generation_status(requested_scenario_id: str):
    """Where the scenario's data design stands: ``ready`` | ``designing`` | ``refining`` | ``unavailable``. Never waits.

    Poll this (every ``retryAfterSeconds``) after confirm; when it says ``ready``, one call to /scenario/generate returns the data.
    Calling it also starts the design if nobody is running it.
    """
    try:
        return await run_in_threadpool(generation_status, requested_scenario_id)
    except ValueError as exc:
        raise HTTPException(404, detail={"error": str(exc)}) from exc


@app.post(
    "/scenario/generate",
    response_model=GenerateResponse,
    responses={202: {"description": "The data design for this scenario is still being built (status designing) or repaired (status refining). "
                                    "Not an error: ask again after retryAfterSeconds, or poll GET /scenario/{id}/generation-status until ready.",
                     "content": {"application/json": {"example": {"success": True, "status": "designing", "requestedScenarioId": "CBG-01",
                                                                  "message": "The generation spec for this scenario is still being designed.",
                                                                  "retryAfterSeconds": 30}}}}},
)
async def generate_scenario(req: GenerateRequest) -> GenerateResponse:
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
            "seed": req.seed,
            "asOf": req.as_of,
        })
        # build_generation_response performs the domain-level generation and QA checks.
        # Returning the plain payload lets FastAPI apply the declared GenerateResponse contract
        # for response validation, filtering, and JSON serialization. This is the supported
        # FastAPI path for response_model endpoints and avoids a deprecated custom JSON response class.
        logger.info(
            "[Latency] /scenario/generate scenario=%s records=%d fields=%d elapsed_ms=%.1f",
            requested_scenario_id,
            int(payload.get("total_records", 0) or 0),
            len(payload.get("fields") or []),
            (perf_counter() - started) * 1000.0,
        )
        return payload
    except SpecNotReady as exc:
        # The scenario's data design is still running in the background. That is not a failure: answer 202 Accepted at once
        # (with Retry-After) instead of holding the request open or reporting a server error. The caller asks again, or polls
        # GET /scenario/{id}/generation-status until it says "ready".
        return JSONResponse(
            status_code=202,
            content={"success": True, "status": exc.phase, "requestedScenarioId": requested_scenario_id,
                     "message": str(exc), "retryAfterSeconds": exc.retry_after},
            headers={"Retry-After": str(exc.retry_after)},
        )
    except ValueError as exc:
        raise HTTPException(400, detail={"error": str(exc)}) from exc



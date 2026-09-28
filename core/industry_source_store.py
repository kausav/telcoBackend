"""Data-driven industry/domain standard JSON source registry.

Operational generation must never depend on a filesystem path for industry/domain source JSONs.
This module stores uploaded machine-readable source documents in MongoDB and exposes a
normalized scalar catalog used by the intent agent and schema compiler. MongoDB is the sole
source of truth for executable industry/domain JSON documents.

Supported source shapes:
- OpenAPI/Swagger 2: ``definitions``
- OpenAPI 3.x: ``components.schemas``
- JSON Schema: root ``properties`` / ``$defs`` / ``definitions``

Arrays/objects themselves are not emitted as flat executable variables; scalar leaves inside
referenced objects are emitted with their full model/property path.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_BYTES = 8 * 1024 * 1024
SUPPORTED_SOURCE_FORMATS = {"openapi", "swagger", "json_schema"}


def _collection():
    """Resolve the Mongo collection lazily so pure catalog parsing remains testable offline."""
    from models.industry_source import IndustrySourceModel

    return IndustrySourceModel.collection


def normalize_lookup_key(value: str | None) -> str:
    """Normalize human labels to deterministic Mongo lookup keys."""
    text = str(value or "").strip().casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", text)


def normalize_industry_key(value: str | None) -> str:
    """Normalize industry labels without consulting any static industry/standards registry."""
    raw = str(value or "").strip()
    if not raw:
        return "generic"
    compact = normalize_lookup_key(raw)
    if compact in {"telecom", "telecommunication", "telecommunications"}:
        return "telecom"
    return compact or "generic"


def normalize_domain_key(value: str | None) -> str:
    return normalize_lookup_key(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _env_max_bytes() -> int:
    import os

    raw = os.getenv("INDUSTRY_SOURCE_MAX_JSON_BYTES", str(DEFAULT_MAX_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("INDUSTRY_SOURCE_MAX_JSON_BYTES must be an integer") from exc
    return max(1024, value)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_file_name(value: str | None) -> str:
    raw = Path(str(value or "source.json")).name
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return raw or "source.json"


def generate_internal_source_id(
    *,
    industry_type: str,
    domain: str,
    file_name: str | None,
    document: dict[str, Any],
    raw_bytes: bytes,
) -> str:
    """Create a stable server-owned source ID without exposing per-file ID fields in the API.

    Known Low Balance TM Forum artifacts retain their historical IDs so the existing telecom
    policy can continue to recognize TMF654/TMF629. Other uploads get a deterministic ID derived
    from the exact industry/domain, filename, and content fingerprint.
    """
    safe_name = _safe_file_name(file_name)
    compact_name = normalize_lookup_key(Path(safe_name).stem)
    lower_name = safe_name.casefold()
    if "tmf654" in lower_name and "prepay" in lower_name and "balance" in lower_name:
        return "tmf654_v4"
    if "tmf629" in lower_name and "customer" in lower_name and "management" in lower_name:
        return "tmf629_v4"

    industry_key = normalize_industry_key(industry_type)
    domain_key = normalize_domain_key(domain)
    info = document.get("info") if isinstance(document.get("info"), dict) else {}
    source_hint = compact_name or normalize_lookup_key(str(info.get("title") or "source"))
    source_hint = re.sub(r"_+", "_", source_hint).strip("_") or "source"
    digest = _sha256_bytes(raw_bytes)[:12]
    prefix = f"src_{industry_key}_{domain_key}_"
    max_hint = max(8, 128 - len(prefix) - len(digest) - 1)
    source_hint = source_hint[:max_hint].rstrip("._:-") or "source"
    return f"{prefix}{source_hint}_{digest}"


def _deepcopy_mapping(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _source_schema_map(document: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return ``(format, schemas)`` where schemas are named model definitions."""
    definitions = document.get("definitions")
    if isinstance(definitions, dict) and definitions:
        return "swagger", definitions

    components = document.get("components")
    if isinstance(components, dict) and isinstance(components.get("schemas"), dict) and components["schemas"]:
        return "openapi", components["schemas"]

    properties = document.get("properties")
    if isinstance(properties, dict) and properties:
        return "json_schema", {str(document.get("title") or "Root"): document}

    defs = document.get("$defs")
    if isinstance(defs, dict) and defs:
        return "json_schema", defs

    raise ValueError(
        "JSON source is not a supported standard schema. Expected Swagger/OpenAPI definitions, "
        "OpenAPI components.schemas, or JSON Schema properties/$defs."
    )


def _ref_name(ref: str) -> str | None:
    token = str(ref or "").strip()
    if not token.startswith("#/"):
        return None
    return token.split("/")[-1] or None


def _schema_type(schema: dict[str, Any]) -> str | None:
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        values = [str(item).strip().lower() for item in raw_type if str(item).strip().lower() != "null"]
        return values[0] if values else None
    if raw_type is None:
        return None
    return str(raw_type).strip().lower() or None


def _merge_composed_schema(
    schema: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    *,
    seen: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve local refs and simple allOf/oneOf/anyOf compositions without mutating source JSON."""
    current = deepcopy(schema)
    ref = _ref_name(str(current.get("$ref") or ""))
    if ref and ref in schemas and ref not in seen:
        base = _merge_composed_schema(schemas[ref], schemas, seen=seen + (ref,))
        base.update({key: value for key, value in current.items() if key != "$ref"})
        current = base

    merged = {}
    for key in ("allOf", "oneOf", "anyOf"):
        pieces = current.get(key)
        if not isinstance(pieces, list):
            continue
        for piece in pieces:
            if not isinstance(piece, dict):
                continue
            child = _merge_composed_schema(piece, schemas, seen=seen)
            for prop_key, prop_value in child.items():
                if prop_key == "properties" and isinstance(prop_value, dict):
                    merged.setdefault("properties", {}).update(deepcopy(prop_value))
                elif prop_key == "required" and isinstance(prop_value, list):
                    merged.setdefault("required", []).extend(str(item) for item in prop_value)
                elif prop_key not in merged:
                    merged[prop_key] = deepcopy(prop_value)
    for key, value in current.items():
        if key not in {"allOf", "oneOf", "anyOf"}:
            if key == "properties" and isinstance(value, dict):
                merged.setdefault("properties", {}).update(deepcopy(value))
            elif key == "required" and isinstance(value, list):
                merged.setdefault("required", []).extend(str(item) for item in value)
            else:
                merged[key] = deepcopy(value)
    if isinstance(merged.get("required"), list):
        merged["required"] = sorted(set(str(item) for item in merged["required"]))
    return merged


def _property_constraints(schema: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "pattern", "minItems", "maxItems", "format",
        "default", "example", "examples",
    )
    return {key: deepcopy(schema[key]) for key in keys if key in schema}


def _scalar_spec(
    *,
    source_id: str,
    source_name: str,
    standard: str | None,
    version: str | None,
    model: str,
    field_name: str,
    path: str,
    schema: dict[str, Any],
    required: bool,
    depth: int,
) -> dict[str, Any] | None:
    schema_type = _schema_type(schema)
    enum_values = list(schema.get("enum") or [])
    if enum_values and any(isinstance(value, (dict, list)) for value in enum_values):
        return None
    if schema_type in {"object", "array"} and not enum_values:
        return None
    if schema_type is None and not enum_values:
        return None

    return {
        "source_id": source_id,
        "source_name": source_name,
        "standard": standard,
        "version": version,
        "model": model,
        "field": field_name,
        "path": path,
        "name": _snake_case(path),
        "dtype": schema_type or "string",
        "description": str(schema.get("description") or ""),
        "enum_values": enum_values,
        "format": schema.get("format"),
        "required": bool(required),
        "depth": depth,
        **_property_constraints(schema),
    }


def _snake_case(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)


def _expand_model(
    *,
    source_id: str,
    source_name: str,
    standard: str | None,
    version: str | None,
    model_name: str,
    model_schema: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    max_depth: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(def_name: str | None, current_schema: dict[str, Any], prefix: str, parent_required: bool, stack: tuple[str, ...], depth: int) -> None:
        if depth > max_depth:
            return
        resolved = _merge_composed_schema(current_schema, schemas, seen=stack)
        required_names = set(str(item) for item in resolved.get("required") or [])
        properties = resolved.get("properties")
        if not isinstance(properties, dict):
            return

        for field_name, raw_prop in properties.items():
            if str(field_name).startswith("@") or not isinstance(raw_prop, dict):
                continue
            path = f"{prefix}.{field_name}" if prefix else str(field_name)
            is_required = parent_required and str(field_name) in required_names
            prop = _merge_composed_schema(raw_prop, schemas, seen=stack + ((def_name,) if def_name else ()))
            ref = _ref_name(str(raw_prop.get("$ref") or ""))
            prop_type = _schema_type(prop)
            if ref and ref in schemas and isinstance(prop.get("properties"), dict):
                visit(ref, prop, path, is_required, stack + (ref,), depth + 1)
                continue
            if isinstance(prop.get("properties"), dict):
                visit(None, prop, path, is_required, stack, depth + 1)
                continue
            if prop_type == "array":
                continue
            spec = _scalar_spec(
                source_id=source_id,
                source_name=source_name,
                standard=standard,
                version=version,
                model=model_name,
                field_name=str(field_name),
                path=path,
                schema=prop,
                required=is_required,
                depth=depth,
            )
            if spec is not None:
                rows.append(spec)

    visit(model_name, model_schema, model_name, True, (model_name,), 0)
    return rows


def extract_scalar_catalog(
    document: dict[str, Any],
    *,
    source_id: str,
    source_name: str,
    standard: str | None = None,
    version: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten a standard JSON document into deterministic scalar variable definitions."""
    if not isinstance(document, dict):
        raise ValueError("Industry source JSON root must be an object")
    _format, schemas = _source_schema_map(document)
    rows: list[dict[str, Any]] = []
    for model_name, schema in schemas.items():
        if not isinstance(schema, dict):
            continue
        rows.extend(
            _expand_model(
                source_id=source_id,
                source_name=source_name,
                standard=standard,
                version=version,
                model_name=str(model_name),
                model_schema=schema,
                schemas=schemas,
            )
        )

    # Exact normalized variable names are the executable boundary. Preserve the first
    # definition deterministically and retain all contributing source IDs for provenance.
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = normalize_lookup_key(row.get("name"))
        if not name:
            continue
        row = dict(row)
        row["name"] = name
        existing = by_name.get(name)
        if existing is None:
            row["source_ids"] = [str(row.get("source_id") or "")]
            by_name[name] = row
        else:
            source_id_value = str(row.get("source_id") or "")
            if source_id_value and source_id_value not in existing["source_ids"]:
                existing["source_ids"].append(source_id_value)
    return sorted(by_name.values(), key=lambda item: str(item.get("name") or ""))


def _validate_source_document(document: dict[str, Any]) -> str:
    try:
        detected_format, _ = _source_schema_map(document)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return detected_format


def save_source_document(
    *,
    industry_type: str,
    domain: str,
    document: dict[str, Any],
    file_name: str,
    source_name: str | None = None,
    standard: str | None = None,
    version: str | None = None,
    description: str | None = None,
    active: bool = True,
    source_id: str | None = None,
    raw_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not str(industry_type or "").strip():
        raise ValueError("industryType is required")
    if not str(domain or "").strip():
        raise ValueError("domain is required")
    if not isinstance(document, dict):
        raise ValueError("Industry source JSON root must be an object")

    detected_format = _validate_source_document(document)
    payload_bytes = raw_bytes if raw_bytes is not None else json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    max_bytes = _env_max_bytes()
    if len(payload_bytes) > max_bytes:
        raise ValueError(f"JSON source exceeds the configured size limit of {max_bytes} bytes")

    industry_key = normalize_industry_key(industry_type)
    domain_key = normalize_domain_key(domain)
    digest = _sha256_bytes(payload_bytes)
    now = _utc_now()
    resolved_id = str(source_id or "").strip() or uuid.uuid4().hex
    if len(resolved_id) > 128 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", resolved_id):
        raise ValueError("sourceId must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    safe_name = _safe_file_name(file_name)
    doc = {
        "source_id": resolved_id,
        "industry_type": str(industry_type).strip(),
        "industry_key": industry_key,
        "domain": str(domain).strip(),
        "domain_key": domain_key,
        "source_name": str(source_name or safe_name).strip()[:300] or safe_name,
        "standard": str(standard or "").strip()[:300] or None,
        "version": str(version or "").strip()[:100] or None,
        "description": str(description or "").strip()[:2000] or None,
        "file_name": safe_name,
        "format": detected_format,
        "mime_type": "application/json",
        "sha256": digest,
        "size_bytes": len(payload_bytes),
        "active": bool(active),
        "document": deepcopy(document),
        "updated_at": now,
    }

    collection = _collection()
    from pymongo.errors import DuplicateKeyError
    existing_id = collection.find_one({"source_id": resolved_id}, {"_id": 1, "industry_key": 1, "domain_key": 1, "created_at": 1})
    if existing_id:
        if existing_id.get("industry_key") != industry_key or existing_id.get("domain_key") != domain_key:
            raise ValueError(
                f"sourceId '{resolved_id}' already belongs to a different industryType/domain and cannot be reassigned"
            )
        conflicting_hash = collection.find_one(
            {"industry_key": industry_key, "domain_key": domain_key, "sha256": digest},
            {"_id": 1, "source_id": 1},
        )
        if conflicting_hash and str(conflicting_hash.get("source_id") or "") != resolved_id:
            raise ValueError(
                f"The uploaded JSON content is already registered for this industryType/domain under sourceId "
                f"'{conflicting_hash.get('source_id')}'"
            )
        doc["created_at"] = existing_id.get("created_at") or now
        try:
            collection.update_one({"_id": existing_id["_id"]}, {"$set": doc})
        except DuplicateKeyError as exc:
            raise ValueError("The uploaded JSON conflicts with another source document in this industry/domain pair") from exc
        return serialize_source_document(collection.find_one({"_id": existing_id["_id"]}))

    existing_hash = collection.find_one(
        {"industry_key": industry_key, "domain_key": domain_key, "sha256": digest},
        {"_id": 1, "source_id": 1, "created_at": 1},
    )
    if existing_hash:
        doc["source_id"] = str(existing_hash.get("source_id") or resolved_id)
        doc["created_at"] = existing_hash.get("created_at") or now
        collection.update_one({"_id": existing_hash["_id"]}, {"$set": doc})
        return serialize_source_document(collection.find_one({"_id": existing_hash["_id"]}))

    doc["created_at"] = now
    try:
        collection.insert_one(doc)
    except DuplicateKeyError:
        # Race-safe retry for either the unique sourceId or content fingerprint.
        existing_id = collection.find_one({"source_id": resolved_id})
        if existing_id:
            if existing_id.get("industry_key") != industry_key or existing_id.get("domain_key") != domain_key:
                raise ValueError(
                    f"sourceId '{resolved_id}' already belongs to a different industryType/domain and cannot be reassigned"
                )
            collection.update_one({"_id": existing_id["_id"]}, {"$set": {**doc, "created_at": existing_id.get("created_at") or now}})
            return serialize_source_document(collection.find_one({"_id": existing_id["_id"]}))
        existing_hash = collection.find_one({"industry_key": industry_key, "domain_key": domain_key, "sha256": digest})
        if not existing_hash:
            raise
        collection.update_one({"_id": existing_hash["_id"]}, {"$set": {**doc, "source_id": existing_hash.get("source_id") or resolved_id, "created_at": existing_hash.get("created_at") or now}})
        return serialize_source_document(collection.find_one({"_id": existing_hash["_id"]}))
    return serialize_source_document(doc)


def serialize_source_document(doc: dict[str, Any] | None, *, include_document: bool = False) -> dict[str, Any] | None:
    if not doc:
        return None
    result = {
        "source_id": str(doc.get("source_id") or ""),
        "industryType": str(doc.get("industry_type") or ""),
        "industryKey": str(doc.get("industry_key") or ""),
        "domain": str(doc.get("domain") or ""),
        "domainKey": str(doc.get("domain_key") or ""),
        "sourceName": str(doc.get("source_name") or ""),
        "standard": doc.get("standard"),
        "version": doc.get("version"),
        "description": doc.get("description"),
        "fileName": str(doc.get("file_name") or ""),
        "format": str(doc.get("format") or ""),
        "mimeType": str(doc.get("mime_type") or "application/json"),
        "sha256": str(doc.get("sha256") or ""),
        "sizeBytes": int(doc.get("size_bytes") or 0),
        "active": bool(doc.get("active", True)),
        "createdAt": doc.get("created_at"),
        "updatedAt": doc.get("updated_at"),
    }
    if include_document:
        result["document"] = deepcopy(doc.get("document") or {})
    return result


def list_source_documents(
    *,
    industry_type: str | None = None,
    domain: str | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if industry_type:
        query["industry_key"] = normalize_industry_key(industry_type)
    if domain:
        query["domain_key"] = normalize_domain_key(domain)
    if active_only:
        query["active"] = True
    rows = _collection().find(
        query,
        {"document": 0},
    ).sort([("source_name", 1), ("version", 1), ("source_id", 1)])
    return [serialize_source_document(row) for row in rows]


def get_source_document(source_id: str, *, include_document: bool = False) -> dict[str, Any] | None:
    row = _collection().find_one({"source_id": str(source_id).strip()})
    return serialize_source_document(row, include_document=include_document)


def load_source_payloads(industry_type: str, domain: str) -> list[dict[str, Any]]:
    query = {
        "industry_key": normalize_industry_key(industry_type),
        "domain_key": normalize_domain_key(domain),
        "active": True,
    }
    rows = _collection().find(query).sort(
        [("source_name", 1), ("version", 1), ("source_id", 1)]
    )
    return [dict(row) for row in rows]


def source_manifest(industry_type: str, domain: str) -> list[dict[str, Any]]:
    rows = list_source_documents(industry_type=industry_type, domain=domain, active_only=True)
    return [
        {
            "source_id": row["source_id"],
            "name": row["sourceName"],
            "filename": row["fileName"],
            "standard": row.get("standard"),
            "version": row.get("version"),
            "sha256": row["sha256"],
            "size_bytes": row["sizeBytes"],
        }
        for row in rows
    ]


def has_sources(industry_type: str, domain: str) -> bool:
    return _collection().count_documents(
        {
            "industry_key": normalize_industry_key(industry_type),
            "domain_key": normalize_domain_key(domain),
            "active": True,
        },
        limit=1,
    ) > 0


def catalog_for_request(industry_type: str, domain: str) -> dict[str, Any]:
    payloads = load_source_payloads(industry_type, domain)
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(
            extract_scalar_catalog(
                payload["document"],
                source_id=str(payload.get("source_id") or ""),
                source_name=str(payload.get("source_name") or payload.get("file_name") or ""),
                standard=payload.get("standard"),
                version=payload.get("version"),
            )
        )

    # Exact variable-name collision across multiple source documents is treated as the same
    # executable business concept. The first source in deterministic order defines the contract;
    # provenance retains all source IDs supporting that name.
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = normalize_lookup_key(row.get("name"))
        if not name:
            continue
        current = by_name.get(name)
        if current is None:
            by_name[name] = dict(row)
            continue
        for key in ("source_ids",):
            merged = list(current.get(key) or [])
            for value in row.get(key) or []:
                if value not in merged:
                    merged.append(value)
            current[key] = merged

    models = [by_name[key] for key in sorted(by_name)]
    return {
        "source_policy": "mongodb_industry_source_documents",
        "industryType": str(industry_type).strip(),
        "industryKey": normalize_industry_key(industry_type),
        "domain": str(domain).strip(),
        "domainKey": normalize_domain_key(domain),
        "sources": source_manifest(industry_type, domain),
        "models": models,
        "notes": [
            "Variables are restricted to scalar leaves extracted from the active MongoDB source documents for this exact industry/domain pair.",
            "Swagger/OpenAPI arrays and objects are not flattened into fake scalar variables; scalar leaves inside referenced/composed objects are retained.",
            "Exact duplicate variable names across multiple source documents are treated as one business concept and retain all supporting source IDs in provenance.",
            "The LLM may select variables from this catalog but cannot invent, rename, alias, or derive executable source variables.",
        ],
    }


def delete_source_document(source_id: str) -> bool:
    result = _collection().delete_one({"source_id": str(source_id).strip()})
    return result.deleted_count > 0


def set_source_active(source_id: str, active: bool) -> bool:
    result = _collection().update_one(
        {"source_id": str(source_id).strip()},
        {"$set": {"active": bool(active), "updated_at": _utc_now()}},
    )
    return result.matched_count > 0


def validate_catalog_selection(variables: Iterable[dict[str, Any]], catalog: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only exact normalized source-backed variable names returned by the LLM."""
    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for raw in variables or []:
        if not isinstance(raw, dict):
            continue
        name = normalize_lookup_key(raw.get("name"))
        if not name:
            continue
        if name not in catalog:
            rejected.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        spec = dict(catalog[name])
        canonical = dict(raw)
        canonical["name"] = name
        canonical["description"] = str(spec.get("description") or canonical.get("description") or "")[:500]
        canonical["dtype"] = _canonical_catalog_dtype(spec.get("dtype"), bool(spec.get("enum_values")))
        canonical["grain"] = _catalog_grain(spec)
        role = _catalog_role(spec)
        canonical["role"] = "other" if role == "categorical" else role
        selected.append(canonical)
    return selected, sorted(set(rejected))


def dedupe_catalog_against_db(variables: list[dict[str, Any]], db_variables: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Conservative exact-name duplicate suppression for non-Low-Balance JSON sources."""
    db_names = {normalize_lookup_key(item.get("name")) for item in db_variables or [] if isinstance(item, dict)}
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    seen: set[str] = set()
    for item in variables or []:
        name = normalize_lookup_key(item.get("name")) if isinstance(item, dict) else ""
        if not name or name in seen:
            continue
        seen.add(name)
        if name in db_names:
            removed.append(name)
            continue
        kept.append(dict(item))
    return kept, sorted(set(removed))


def _canonical_catalog_dtype(raw_dtype: Any, has_enum: bool) -> str:
    dtype = str(raw_dtype or "string").strip().lower()
    if dtype in {"boolean", "bool"}:
        return "boolean"
    if dtype in {"integer", "int", "bigint", "smallint"}:
        return "integer"
    if dtype in {"number", "float", "double", "decimal", "numeric"}:
        return "float"
    if dtype in {"date-time", "datetime", "timestamp"}:
        return "datetime"
    if dtype == "date":
        return "date"
    if has_enum:
        return "categorical"
    return "string"


def _catalog_grain(spec: dict[str, Any]) -> str:
    text = f"{spec.get('model', '')} {spec.get('path', '')}".lower()
    if any(token in text for token in ("customer", "account", "party", "subscriber", "member", "patient", "policyholder", "user")) and not any(token in text for token in ("transaction", "event", "order", "payment", "claim", "encounter", "visit")):
        return "entity"
    return "transaction"


def _catalog_role(spec: dict[str, Any]) -> str:
    text = f"{spec.get('name', '')} {spec.get('path', '')} {spec.get('description', '')}".lower()
    dtype = _canonical_catalog_dtype(spec.get("dtype"), bool(spec.get("enum_values")))
    if dtype in {"datetime", "date"} or any(token in text for token in ("timestamp", "date", "time", "effective", "expiry", "expiration")):
        return "timing"
    if dtype in {"integer", "float"} or any(token in text for token in ("amount", "balance", "value", "quantity", "count", "rate", "score", "limit", "price")):
        return "measurement"
    if bool(spec.get("enum_values")) or any(token in text for token in ("status", "state", "reason", "type", "category", "class", "code", "method", "channel")):
        return "status" if any(token in text for token in ("status", "state", "reason")) else "categorical"
    if str(spec.get("name") or "").lower().endswith(("_id", "_key")) or str(spec.get("path") or "").lower().endswith(".id"):
        return "identity"
    return "other"


def select_json_source_catalog(
    catalog_rows: list[dict[str, Any]],
    *,
    business_context: str = "",
    preferred_names: set[str] | None = None,
    excluded_names: set[str] | None = None,
    max_fields: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically rank source-backed variables using request semantics.

    This is deliberately domain-neutral: the source JSON defines the vocabulary, while the
    request context determines relevance. No new field names are ever created here.
    """
    excluded = {normalize_lookup_key(value) for value in (excluded_names or set()) if normalize_lookup_key(value)}
    preferred = {normalize_lookup_key(value) for value in (preferred_names or set()) if normalize_lookup_key(value)}
    context_tokens = set(re.findall(r"[a-z0-9]+", str(business_context or "").casefold()))

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, raw in enumerate(catalog_rows or []):
        row = dict(raw)
        name = normalize_lookup_key(row.get("name"))
        if not name or name in excluded:
            continue
        text_tokens = set(re.findall(r"[a-z0-9]+", " ".join(str(row.get(key) or "") for key in ("name", "path", "description", "model")).casefold()))
        overlap = len(context_tokens & text_tokens)
        score = float(overlap * 5)
        if name in preferred:
            score += 7.0
        if row.get("required"):
            score += 4.0
        role = _catalog_role(row)
        if role == "identity":
            score += 3.0
        elif role == "timing":
            score += 2.5
        elif role == "measurement":
            score += 2.0
        elif role in {"status", "categorical"}:
            score += 2.0
        if row.get("enum_values"):
            score += 1.0
        ranked.append((score, index, row))

    ranked.sort(key=lambda item: (-item[0], item[1], normalize_lookup_key(item[2].get("name"))))
    limit = max(1, int(max_fields))
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()

    # Explicit model-selected exact names are the strongest semantic signal. Required and
    # identity fields follow, then the remaining relevance-ranked source variables. This
    # guarantees a tight schema budget cannot silently discard a requested source field.
    preferred_items = [item for item in ranked if normalize_lookup_key(item[2].get("name")) in preferred]
    priority = [
        item for item in ranked
        if item in preferred_items or item[2].get("required") or _catalog_role(item[2]) == "identity"
    ]
    priority.sort(key=lambda item: (0 if normalize_lookup_key(item[2].get("name")) in preferred else 1, -item[0], item[1]))
    remainder = [item for item in ranked if item not in priority]
    for item in priority + remainder:
        name = normalize_lookup_key(item[2].get("name"))
        if name in selected_names or len(selected) >= limit:
            continue
        selected.append(dict(item[2]))
        selected_names.add(name)

    return selected, {
        "candidate_count": len(ranked),
        "selected_count": len(selected),
        "max_fields": limit,
        "preferred_names_used": sorted(preferred & selected_names),
        "selected_names": [str(item.get("name") or "") for item in selected],
    }

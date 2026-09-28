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
from threading import RLock
from time import monotonic


DEFAULT_MAX_BYTES = 8 * 1024 * 1024
SUPPORTED_SOURCE_FORMATS = {"openapi", "swagger", "json_schema"}
_CATALOG_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_CATALOG_CACHE_LOCK = RLock()


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
    """Create a deterministic server-owned source ID scoped to an industry/domain pair.

    The same JSON document may be uploaded for multiple domains or industries. Its internal ID
    therefore includes the normalized industry/domain pair plus filename/content fingerprint,
    so a source can be reused without global-ID reassignment conflicts.
    """
    safe_name = _safe_file_name(file_name)
    compact_name = normalize_lookup_key(Path(safe_name).stem)
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


_SCHEMA_WRAPPER_SUFFIXES = (
    "_attribute_value_change_event_payload",
    "_state_change_event_payload",
    "_cancel_event_payload",
    "_failure_event_payload",
    "_create_event_payload",
    "_update_event_payload",
    "_delete_event_payload",
    "_attribute_value_change_event",
    "_state_change_event",
    "_cancel_event",
    "_failure_event",
    "_create_event",
    "_update_event",
    "_delete_event",
    "_event_payload",
    "_event",
    "_create",
    "_update",
    "_delete",
    "_payload",
    "_input",
)


_AUXILIARY_MODEL_MARKERS = {
    "error", "event_subscription", "event_subscription_input", "addressable", "any",
}


def _normalize_model_name(value: Any) -> str:
    return _snake_case(str(value or ""))


def _strip_schema_wrappers(value: Any) -> str:
    """Return the business root represented by a CRUD/event wrapper schema name."""
    name = _normalize_model_name(value)
    changed = True
    while name and changed:
        changed = False
        for suffix in _SCHEMA_WRAPPER_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)].rstrip("_")
                changed = True
                break
    return name or _normalize_model_name(value)


def _business_model_from_reference(value: Any) -> str:
    name = _strip_schema_wrappers(value)
    if name.endswith("_ref"):
        name = name[:-4].rstrip("_")
    return name


def _is_event_model_name(value: Any) -> bool:
    name = _normalize_model_name(value)
    return "_event" in name or name.endswith("_event") or name.endswith("_event_payload")


def _is_crud_wrapper_model_name(value: Any) -> bool:
    name = _normalize_model_name(value)
    return any(name.endswith(suffix) for suffix in ("_create", "_update", "_delete", "_input", "_payload"))


def _is_reference_model_name(value: Any, schema: dict[str, Any] | None = None) -> bool:
    name = _normalize_model_name(value)
    if name.endswith("_ref") or name in {"related_party", "related_topup_balance"}:
        return True
    description = str((schema or {}).get("description") or "").casefold().strip()
    # Only descriptions that explicitly define the whole schema as a reference/support object are
    # treated as reference models. A business resource may legitimately mention the word "reference"
    # in its normal prose and must not be demoted because of that.
    return bool(re.match(r"^(reference|link)\b|^related (entity|resource) reference\b", description))


def _schema_refs_from_paths(document: dict[str, Any]) -> set[str]:
    """Collect top-level definition refs used directly by API operations.

    This keeps standalone helper/reference schemas out of the breadth selector while still allowing
    their scalar leaves to be traversed when they occur under an actual business resource.
    """
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return set()
    refs: set[str] = set()

    def walk(value: Any, *, under_schema: bool = False) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str):
                ref_name = _ref_name(ref)
                if ref_name:
                    refs.add(ref_name)
            for child in value.values():
                walk(child, under_schema=under_schema)
        elif isinstance(value, list):
            for child in value:
                walk(child, under_schema=under_schema)

    for path_item in paths.values():
        walk(path_item)
    return refs


def _model_kind(model_name: str, model_schema: dict[str, Any], api_root_models: set[str]) -> str:
    normalized = _normalize_model_name(model_name)
    description = str(model_schema.get("description") or "").casefold()
    if normalized in _AUXILIARY_MODEL_MARKERS or any(marker in normalized for marker in ("error", "event_subscription")):
        return "auxiliary"
    if normalized.endswith("_ref") or _is_reference_model_name(model_name, model_schema):
        return "support_reference"
    if _is_event_model_name(model_name):
        return "event_wrapper"
    if _is_crud_wrapper_model_name(model_name):
        return "crud_wrapper"
    if "abstract resource" in description or normalized == "addressable":
        return "abstract"
    normalized_roots = {_normalize_model_name(item) for item in api_root_models}
    # Pure JSON Schema documents often have no API paths. In that shape every concrete top-level
    # definition is a candidate business resource unless it is explicitly a reference/support model.
    if normalized in normalized_roots or not api_root_models:
        return "resource"
    return "support"


def _canonical_event_root(model_name: str) -> str:
    base = _strip_schema_wrappers(model_name)
    return f"event_{base}" if base else "event"


def _canonical_semantic_path(
    *,
    model_name: str,
    raw_path: str,
    api_root_models: set[str],
) -> str:
    """Normalize wrapper paths without collapsing distinct business resources.

    CRUD and event schemas often repeat exactly the same underlying resource fields. Those wrapper
    layers are removed only when the path contains the underlying business resource. Standalone
    referenced/value objects keep their owner context so `customer.valid_for.start_date_time` and
    `topup_balance.valid_for.start_date_time` remain distinct concepts.
    """
    root_model = _normalize_model_name(model_name)
    base_model = _strip_schema_wrappers(root_model)
    segments = [_normalize_model_name(part) for part in str(raw_path or "").split(".") if _normalize_model_name(part)]
    if not segments:
        return ""

    if _is_event_model_name(model_name):
        # Prefer the embedded business resource after event/payload wrapper segments.
        candidate_roots = {
            _strip_schema_wrappers(ref) for ref in api_root_models
            if not _is_event_model_name(ref)
        }
        for idx in range(1, len(segments)):
            segment = segments[idx]
            if segment in candidate_roots and segment != root_model:
                return ".".join([segment, *segments[idx + 1:]])
        return ".".join([_canonical_event_root(model_name), *segments[1:]])

    # CRUD wrappers have no event envelope; simply replace the wrapper root with its business root.
    if _is_crud_wrapper_model_name(model_name):
        return ".".join([base_model, *segments[1:]])

    # Normal business resources keep their root model, including nested value objects.
    return ".".join([base_model, *segments[1:]])


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
    semantic_path: str | None = None,
    model_kind: str = "support",
    source_ref: str | None = None,
    model_description: str | None = None,
) -> dict[str, Any] | None:
    schema_type = _schema_type(schema)
    enum_values = list(schema.get("enum") or [])
    if enum_values and any(isinstance(value, (dict, list)) for value in enum_values):
        return None
    if schema_type in {"object", "array"} and not enum_values:
        return None
    if schema_type is None and not enum_values:
        return None

    canonical_path = semantic_path or _snake_case(path)
    normalized_business_model = _strip_schema_wrappers(model)
    if model_kind == "event_wrapper" and _normalize_model_name(canonical_path).startswith("event_"):
        # Event envelope fields belong to one canonical event resource model (for example
        # event_customer or event_topup_balance). Keep the field name in semantic_key, but never
        # use the full field path as the business model or relevance unit.
        normalized_business_model = str(canonical_path).split(".", 1)[0].strip().casefold()
    return {
        "source_id": source_id,
        "source_name": source_name,
        "standard": standard,
        "version": version,
        "model": model,
        "business_model": normalized_business_model,
        "model_kind": model_kind,
        "field": field_name,
        "path": path,
        "name": _snake_case(canonical_path.replace(".", "_")),
        "semantic_key": canonical_path,
        "dtype": schema_type or "string",
        "description": str(schema.get("description") or ""),
        "model_description": str(model_description or ""),
        "enum_values": enum_values,
        "format": schema.get("format"),
        "required": bool(required),
        "depth": depth,
        "source_ref": source_ref,
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
    api_root_models: set[str] | None = None,
    max_depth: int = 6,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    api_root_models = api_root_models or set()
    normalized_schema_models = {
        _normalize_model_name(name): name for name in schemas
    }
    root_kind = _model_kind(model_name, model_schema, api_root_models)

    def visit(
        def_name: str | None,
        current_schema: dict[str, Any],
        prefix: str,
        parent_required: bool,
        stack: tuple[str, ...],
        depth: int,
        linked_business_models: tuple[str, ...] = (),
    ) -> None:
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

            # Follow referenced objects, including arrays of referenced objects. The old extractor
            # skipped every array and therefore lost a large portion of legitimate business fields.
            if prop_type == "array":
                items = raw_prop.get("items") if isinstance(raw_prop.get("items"), dict) else prop.get("items")
                if isinstance(items, dict):
                    item_ref = _ref_name(str(items.get("$ref") or ""))
                    item_schema = _merge_composed_schema(items, schemas, seen=stack + ((def_name,) if def_name else ()))
                    if item_ref and item_ref in schemas and isinstance(item_schema.get("properties"), dict):
                        visit(
                            item_ref, item_schema, path + "[]", is_required, stack + (item_ref,), depth + 1,
                            linked_business_models,
                        )
                        continue
                    if isinstance(item_schema.get("properties"), dict):
                        visit(None, item_schema, path + "[]", is_required, stack, depth + 1, linked_business_models)
                        continue
                # Scalar arrays cannot be represented as one scalar executable variable without inventing
                # a serialization contract, so they remain intentionally excluded.
                continue

            if ref and ref in schemas and isinstance(prop.get("properties"), dict):
                target_model = _business_model_from_reference(ref)
                next_links = linked_business_models
                target_schema_name = normalized_schema_models.get(target_model)
                if target_schema_name and target_model not in {_normalize_model_name(model_name), *linked_business_models}:
                    target_schema = schemas.get(target_schema_name) or {}
                    target_kind = _model_kind(target_schema_name, target_schema if isinstance(target_schema, dict) else {}, api_root_models)
                    if target_kind in {"resource", "crud_wrapper", "event_wrapper"}:
                        next_links = tuple(dict.fromkeys((*linked_business_models, target_model)))
                visit(ref, prop, path, is_required, stack + (ref,), depth + 1, next_links)
                continue
            if isinstance(prop.get("properties"), dict):
                visit(None, prop, path, is_required, stack, depth + 1, linked_business_models)
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
                semantic_path=_canonical_semantic_path(
                    model_name=model_name,
                    raw_path=path,
                    api_root_models=api_root_models,
                ),
                model_kind=root_kind,
                source_ref=ref,
                model_description=str(model_schema.get("description") or ""),
            )
            if spec is not None and linked_business_models:
                spec["linked_business_models"] = list(linked_business_models)
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
    """Flatten a standard JSON document into source-backed scalar leaves.

    Extraction deliberately preserves *all* scalar leaves, including scalar fields nested inside
    arrays of referenced objects. Semantic deduplication happens later at the source-catalog boundary,
    where multiple CRUD/event representations can be reconciled without losing provenance.
    """
    if not isinstance(document, dict):
        raise ValueError("Industry source JSON root must be an object")
    _format, schemas = _source_schema_map(document)
    path_refs = _schema_refs_from_paths(document)
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
                api_root_models=path_refs,
            )
        )
    # Derive source-defined business-resource relationships from top-level reference properties.
    # Example: TopupBalance.bucket -> BucketRef points to the business Bucket resource. These links
    # are used only for one-hop breadth expansion; they never invent a relationship absent from the source.
    normalized_schema_models = {_normalize_model_name(name): name for name in schemas}
    for row in rows:
        model_name = str(row.get("model") or "")
        schema = schemas.get(model_name)
        if not isinstance(schema, dict):
            continue
        path_segments = [segment for segment in str(row.get("path") or "").split(".") if segment]
        if len(path_segments) < 2:
            continue
        root_property = path_segments[1].replace("[]", "")
        properties = schema.get("properties")
        raw_prop = properties.get(root_property) if isinstance(properties, dict) else None
        if not isinstance(raw_prop, dict):
            continue
        ref = _ref_name(str(raw_prop.get("$ref") or ""))
        if not ref and _schema_type(raw_prop) == "array" and isinstance(raw_prop.get("items"), dict):
            ref = _ref_name(str(raw_prop["items"].get("$ref") or ""))
        target = _business_model_from_reference(ref) if ref else ""
        target_name = normalized_schema_models.get(target)
        if target_name and target != _strip_schema_wrappers(model_name):
            target_schema = schemas.get(target_name) or {}
            if _model_kind(target_name, target_schema, path_refs) in {"resource", "crud_wrapper", "event_wrapper"}:
                row["linked_business_models"] = [target]

    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda item: (
                normalize_lookup_key(item.get("semantic_key") or item.get("name")),
                normalize_lookup_key(item.get("name")),
                str(item.get("model") or ""),
                str(item.get("path") or ""),
            ),
        )
        if normalize_lookup_key(row.get("name"))
    ]


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


def _merge_semantic_catalog_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse source-wrapper duplicates into one canonical business concept.

    TMF/OpenAPI documents commonly repeat the same field in base, create/update/delete and event
    schemas. The executable catalog must treat those wrappers as one concept while retaining every
    supporting source location in provenance.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = normalize_lookup_key(row.get("semantic_key") or row.get("name"))
        if key:
            groups.setdefault(key, []).append(dict(row))

    kind_rank = {
        "resource": 0,
        "crud_wrapper": 1,
        "event_wrapper": 2,
        "abstract": 3,
        "support": 4,
        "support_reference": 5,
        "auxiliary": 6,
    }

    merged_rows: list[dict[str, Any]] = []
    for semantic_key in sorted(groups):
        candidates = groups[semantic_key]
        candidates.sort(
            key=lambda row: (
                kind_rank.get(str(row.get("model_kind") or "support"), 99),
                0 if row.get("required") else 1,
                0 if row.get("enum_values") else 1,
                0 if row.get("format") else 1,
                int(row.get("depth", 0) or 0),
                normalize_lookup_key(row.get("model")),
                str(row.get("source_id") or ""),
                str(row.get("path") or ""),
            )
        )
        base = dict(candidates[0])

        source_ids: list[str] = []
        source_paths: list[str] = []
        source_models: list[str] = []
        source_names: list[str] = []
        linked_models: list[str] = []
        conflicts: dict[str, list[Any]] = {}
        for candidate in candidates:
            for field, target in (
                ("source_ids", source_ids),
                ("source_paths", source_paths),
                ("source_models", source_models),
                ("source_names", source_names),
                ("linked_business_models", linked_models),
            ):
                values = candidate.get(field)
                if values is None:
                    scalar = candidate.get(field[:-1]) if field.endswith("s") else None
                    values = [scalar] if scalar else []
                elif not isinstance(values, list):
                    values = [values]
                for value in values:
                    text = str(value or "").strip()
                    if text and text not in target:
                        target.append(text)

            for field in ("dtype", "enum_values", "format", "minimum", "maximum", "pattern"):
                if field not in candidate or candidate.get(field) in (None, [], ""):
                    continue
                value = candidate.get(field)
                existing = base.get(field)
                if existing not in (None, [], "") and existing != value:
                    values = conflicts.setdefault(field, [])
                    for option in (existing, value):
                        if option not in values:
                            values.append(deepcopy(option))

        if source_ids:
            base["source_ids"] = source_ids
        if source_paths:
            base["source_paths"] = source_paths
        if source_models:
            base["source_models"] = source_models
        if source_names:
            base["source_names"] = source_names
        if linked_models:
            base["linked_business_models"] = linked_models
        base["semantic_key"] = semantic_key
        base["name"] = _snake_case(semantic_key.replace(".", "_"))
        if conflicts:
            base["contract_conflicts"] = conflicts
        merged_rows.append(base)

    return sorted(
        merged_rows,
        key=lambda row: (
            normalize_lookup_key(row.get("semantic_key") or row.get("name")),
            normalize_lookup_key(row.get("name")),
        ),
    )


def canonical_variable_semantic_key(variable: dict[str, Any] | Any) -> str:
    """Return a conservative, domain-neutral semantic key for DB or schema variables.

    This is intentionally weaker than source extraction: arbitrary application variable names are
    never guessed into source fields. It only normalizes well-known structural aliases so an
    existing persisted variable can suppress an equivalent source field.
    """
    data = variable if isinstance(variable, dict) else {}
    explicit = data.get("semantic_key") or data.get("_json_source_semantic_key")
    if explicit:
        return normalize_lookup_key(str(explicit).replace("[]", ""))
    raw = normalize_lookup_key(data.get("name") or variable)
    if not raw:
        return ""
    replacements = (
        ("recharge", "topup"),
        ("top_up", "topup"),
        ("automatic", "auto"),
        ("lifecycle_state", "status"),
        ("lifecycle_status", "status"),
        ("requested_date_time", "requested_timestamp"),
        ("requested_datetime", "requested_timestamp"),
        ("confirmation_date_time", "confirmation_timestamp"),
        ("confirmation_datetime", "confirmation_timestamp"),
        ("valid_for_start_date_time", "valid_from"),
        ("valid_for_end_date_time", "valid_to"),
        ("identifier", "id"),
    )
    for old, new in replacements:
        if old in raw:
            raw = raw.replace(old, new)
    raw = re.sub(r"_amount_amount$", "_amount", raw)
    return raw


def _semantic_aliases_for_row(row: dict[str, Any]) -> set[str]:
    """Return conservative semantic aliases for a canonical source field.

    Do not emit generic leaf aliases such as ``status`` or ``amount``: those concepts legitimately
    occur on many unrelated resources. Aliases are limited to structural equivalents that preserve
    the owning business operation/model context.
    """
    return semantic_exclusion_aliases(dict(row))


def invalidate_catalog_cache(industry_type: str | None = None, domain: str | None = None) -> None:
    """Invalidate process-local source catalog cache after an admin source mutation."""
    industry_key = normalize_industry_key(industry_type) if industry_type else None
    domain_key = normalize_domain_key(domain) if domain else None
    with _CATALOG_CACHE_LOCK:
        if industry_key is None and domain_key is None:
            _CATALOG_CACHE.clear()
            return
        for key in list(_CATALOG_CACHE):
            if industry_key is not None and key[0] != industry_key:
                continue
            if domain_key is not None and key[1] != domain_key:
                continue
            _CATALOG_CACHE.pop(key, None)


def catalog_for_request(industry_type: str, domain: str) -> dict[str, Any]:
    """Return the complete source-backed catalog with a short-lived process-local cache."""
    from config.runtime import SOURCE_CATALOG_CACHE_TTL_SECONDS

    cache_key = (normalize_industry_key(industry_type), normalize_domain_key(domain))
    now = monotonic()
    with _CATALOG_CACHE_LOCK:
        cached = _CATALOG_CACHE.get(cache_key)
        if cached and now - cached[0] < SOURCE_CATALOG_CACHE_TTL_SECONDS:
            return cached[1]
        if cached:
            _CATALOG_CACHE.pop(cache_key, None)

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

    models = _merge_semantic_catalog_rows(rows)
    selectable_models = [
        row for row in models
        if str(row.get("model_kind") or "resource") in {"resource", "crud_wrapper", "event_wrapper"}
        and str(row.get("semantic_key") or "")
    ]

    # We already have all source metadata from the payload query; building the manifest again would
    # perform another MongoDB query and materially slow /scenario/propose. Keep it local to this result.
    sources = []
    for payload in payloads:
        sources.append({
            "source_id": str(payload.get("source_id") or ""),
            "name": str(payload.get("source_name") or payload.get("file_name") or ""),
            "filename": str(payload.get("file_name") or ""),
            "standard": payload.get("standard"),
            "version": payload.get("version"),
            "sha256": str(payload.get("sha256") or ""),
            "size_bytes": int(payload.get("size_bytes") or 0),
        })
    result = {
        "source_policy": "mongodb_industry_source_documents",
        "industryType": str(industry_type).strip(),
        "industryKey": cache_key[0],
        "domain": str(domain).strip(),
        "domainKey": cache_key[1],
        "sources": sources,
        "models": selectable_models,
        "notes": [
            "Variables are restricted to scalar leaves extracted from active MongoDB source documents for this exact industry/domain pair.",
            "Referenced objects and arrays are traversed to their scalar leaves; scalar arrays themselves are not invented as serialized variables.",
            "Base, create/update/delete, and event representations of the same business field are collapsed by semantic path and retain all supporting source provenance.",
            "Reference/support schemas remain available through their owning business resource paths but are not selected as standalone breadth models.",
            "The LLM may provide relevance hints, but deterministic selection is recall-first and can expand relevant source-backed models without inventing vocabulary.",
        ],
    }
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE[cache_key] = (monotonic(), result)
    return result


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
    """Validate LLM-selected names against a canonical source catalog."""
    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen_semantics: set[str] = set()
    for raw in variables or []:
        if not isinstance(raw, dict):
            continue
        name = normalize_lookup_key(raw.get("name"))
        if not name:
            continue
        spec = catalog.get(name)
        if spec is None:
            semantic = canonical_variable_semantic_key(raw)
            spec = next((row for row in catalog.values() if canonical_variable_semantic_key(row) == semantic), None)
        if spec is None:
            rejected.append(name)
            continue
        semantic = canonical_variable_semantic_key(spec)
        if semantic and semantic in seen_semantics:
            continue
        if semantic:
            seen_semantics.add(semantic)
        canonical = dict(raw)
        canonical["name"] = str(spec.get("name") or name)
        canonical["description"] = str(spec.get("description") or canonical.get("description") or "")[:500]
        canonical["dtype"] = _canonical_catalog_dtype(spec.get("dtype"), bool(spec.get("enum_values")))
        canonical["grain"] = _catalog_grain(spec)
        role = _catalog_role(spec)
        canonical["role"] = "other" if role == "categorical" else role
        canonical["_json_source_spec"] = dict(spec)
        selected.append(canonical)
    return selected, sorted(set(rejected))


def dedupe_catalog_against_db(variables: list[dict[str, Any]], db_variables: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove source fields that are semantically represented by authoritative DB variables.

    Duplicate suppression is intentionally conservative. Generic leaves (``status``, ``amount``,
    ``id``, etc.) never suppress one another; only exact names or structural aliases that retain
    their business-operation context are considered equivalent.
    """
    db_aliases: set[str] = set()
    db_names = {normalize_lookup_key(item.get("name")) for item in db_variables or [] if isinstance(item, dict)}
    for item in db_variables or []:
        if isinstance(item, dict):
            db_aliases.update(semantic_exclusion_aliases(item))

    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    seen_aliases: set[str] = set()
    for item in variables or []:
        if not isinstance(item, dict):
            continue
        name = normalize_lookup_key(item.get("name"))
        aliases = semantic_exclusion_aliases(item)
        if not name:
            continue
        if name in db_names or aliases & db_aliases:
            removed.append(name)
            continue
        if aliases & seen_aliases:
            continue
        seen_aliases.update(aliases or {name})
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


def _catalog_text_tokens(value: Any) -> set[str]:
    """Tokenize source/request text, including CamelCase schema names.

    TMF/OpenAPI definitions frequently describe business resources as names such as
    ``TopupBalance`` or ``BalanceActionHistory``. Splitting those names before scoring
    keeps model relevance semantic instead of depending on exact storage spelling.
    """
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) > 1}


_CATALOG_GENERIC_OVERLAP_TOKENS = {
    "id", "key", "name", "type", "status", "state", "date", "time",
    "timestamp", "value", "code", "role", "description", "lifecycle",
}


def _catalog_business_metadata_penalty(row: dict[str, Any]) -> float:
    """Return a domain-neutral penalty for source fields that are primarily transport/display metadata."""
    name = normalize_lookup_key(row.get("name"))
    description = str(row.get("description") or "").casefold()
    tokens = _catalog_text_tokens(f"{name} {description}")
    penalty = 0.0
    if name.endswith(("_href", "_uri", "_url", "_referred_type", "_schema_location", "_base_type")):
        penalty += 30.0
    if tokens & {"href", "uri", "url", "transport", "schema", "referred", "discriminator", "disambiguation"}:
        penalty += 18.0
    if name.endswith(("_description", "_display_name", "_display_label", "_formatted", "_friendly_name")):
        penalty += 16.0
    return penalty


def _catalog_display_only(row: dict[str, Any], *, context_tokens: set[str], preferred: set[str]) -> bool:
    """Identify source leaves that usually add presentation noise instead of business signal."""
    name = normalize_lookup_key(row.get("name"))
    leaf = name.rsplit("_", 1)[-1] if name else ""
    if name in preferred:
        return False
    if leaf in {"href", "url", "uri", "schema_location", "base_type", "referred_type", "display_label", "display_name", "formatted"}:
        return True
    if leaf == "description":
        return "description" not in context_tokens
    if leaf == "name":
        # Names are retained when the request explicitly discusses named products/plans/channels/offers/resources.
        return not bool(context_tokens & {"product", "plan", "offer", "channel", "resource_name", "named", "name"})
    if leaf == "remaining_value_name":
        return True
    return False


def _catalog_privacy_sensitive(row: dict[str, Any], *, context_tokens: set[str]) -> bool:
    """Detect direct/sensitive personal-contact fields when the requested scenario says not to expose PII."""
    privacy_requested = bool(context_tokens & {
        "privacy", "pii", "personal", "personally", "sensitive", "confidential", "anonymized", "anonymous", "deidentified",
    })
    if not privacy_requested:
        return False
    text = " ".join(str(row.get(key) or "") for key in ("semantic_key", "name", "path", "description")).casefold()
    normalized_text = re.sub(r"[^a-z0-9]+", "_", text)
    pii_patterns = (
        r"\bemail(?:_address)?\b",
        r"\b(?:phone|mobile|telephone)(?:_number)?\b",
        r"\bfax(?:_number)?\b",
        r"\bstreet(?:_1|_2)?\b",
        r"\b(?:post_code|postcode|postal_code)\b",
        r"\baddress\b",
        r"\bsocial(?:_network)?(?:_id)?\b",
        r"\b(?:first|last|full)_?name\b",
        r"\b(?:birth|date_of_birth|dob)\b",
        r"\b(?:ssn|passport|national_id|government_id|tax_id|identity_number)\b",
    )
    if any(re.search(pattern, normalized_text) for pattern in pii_patterns):
        return True

    model_context = _catalog_text_tokens(f"{row.get('business_model','')} {row.get('model','')} {row.get('path','')}")
    personal_model = bool(model_context & {"customer", "person", "individual", "party", "member", "patient", "subscriber", "user", "policyholder"})
    if personal_model and re.search(r"\b(?:name|username|nickname)\b", normalized_text):
        return True
    # When privacy is explicitly requested, the entire contact-medium branch is PII-adjacent even
    # when an individual leaf is only geographic or validity metadata.
    if personal_model and "contact_medium" in normalized_text:
        return True
    return False


def _catalog_relevance_score(
    row: dict[str, Any],
    *,
    context_tokens: set[str],
    preferred: set[str],
    privacy_context_tokens: set[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score a canonical source concept from source metadata and request context."""
    name = normalize_lookup_key(row.get("name"))
    name_tokens = _catalog_text_tokens(name)
    semantic_tokens = _catalog_text_tokens(row.get("semantic_key"))
    path_tokens = _catalog_text_tokens(row.get("path"))
    business_model = normalize_lookup_key(row.get("business_model") or row.get("model"))
    model_tokens = _catalog_text_tokens(business_model)
    model_description_tokens = _catalog_text_tokens(row.get("model_description"))
    description_tokens = _catalog_text_tokens(row.get("description"))
    name_overlap = name_tokens & context_tokens
    semantic_overlap = semantic_tokens & context_tokens
    path_overlap = path_tokens & context_tokens
    model_overlap = model_tokens & context_tokens
    model_description_overlap = model_description_tokens & context_tokens
    description_overlap = description_tokens & context_tokens
    informative_overlap = (
        name_overlap | semantic_overlap | path_overlap | description_overlap | model_overlap | model_description_overlap
    ) - _CATALOG_GENERIC_OVERLAP_TOKENS

    score = 0.0
    reasons: list[str] = []
    if name in preferred:
        score += 120.0
        reasons.append("llm_preferred")
    if row.get("required"):
        score += 22.0
        reasons.append("source_required")

    role = _catalog_role(row)
    role_bonus = {
        "identity": 16.0,
        "timing": 12.0,
        "measurement": 12.0,
        "status": 12.0,
        "categorical": 9.0,
        "transaction": 10.0,
        "other": 2.0,
    }.get(role, 2.0)
    score += role_bonus
    reasons.append(f"role_{role}")

    if informative_overlap:
        score += min(42.0, 10.0 * len(informative_overlap))
        reasons.append("informative_context_overlap")
    if model_overlap:
        score += min(18.0, 9.0 * len(model_overlap))
        reasons.append("business_model_context_overlap")
    if model_description_overlap:
        score += min(24.0, 6.0 * len(model_description_overlap))
        reasons.append("business_model_description_context_overlap")
    if semantic_overlap:
        score += min(18.0, 6.0 * len(semantic_overlap))
        reasons.append("semantic_context_overlap")
    if description_overlap - _CATALOG_GENERIC_OVERLAP_TOKENS:
        score += min(12.0, 3.0 * len(description_overlap - _CATALOG_GENERIC_OVERLAP_TOKENS))
        reasons.append("description_context_overlap")

    kind = str(row.get("model_kind") or "resource")
    if kind == "event_wrapper":
        if context_tokens & {"event", "history", "audit", "notification", "lifecycle", "timeline"}:
            score += 4.0
        else:
            score -= 8.0
            reasons.append("event_wrapper_penalty")
    elif kind == "crud_wrapper":
        score -= 2.0
        reasons.append("crud_wrapper_penalty")
    elif kind in {"support", "support_reference", "abstract", "auxiliary"}:
        score -= 30.0
        reasons.append("non_business_model_penalty")

    if row.get("enum_values"):
        score += 2.0
        reasons.append("declared_vocabulary")
    depth = int(row.get("depth", 0) or 0)
    if depth >= 3:
        score -= min(6.0, 1.5 * (depth - 2))
        reasons.append("deep_source_path")

    metadata_penalty = _catalog_business_metadata_penalty(row)
    if metadata_penalty:
        score -= metadata_penalty
        reasons.append("technical_or_display_metadata")

    privacy_sensitive = _catalog_privacy_sensitive(row, context_tokens=(privacy_context_tokens or context_tokens))
    if privacy_sensitive:
        score -= 70.0
        reasons.append("privacy_sensitive_field")

    return score, {
        "name_overlap": sorted(name_overlap),
        "semantic_overlap": sorted(semantic_overlap),
        "path_overlap": sorted(path_overlap),
        "model_overlap": sorted(model_overlap),
        "model_description_overlap": sorted(model_description_overlap),
        "description_overlap": sorted(description_overlap),
        "informative_overlap": sorted(informative_overlap),
        "role": role,
        "metadata_penalty": metadata_penalty,
        "privacy_sensitive": privacy_sensitive,
        "model_kind": kind,
        "business_model": business_model,
        "reasons": reasons,
    }


_CATALOG_GENERIC_CONTEXT_TOKENS = {
    "generate", "dataset", "synthetic", "data", "type", "transactional", "aggregational",
    "industry", "telecommunications", "telecommunication", "telecom", "normal", "scenario",
    "country", "india", "in", "use", "case", "high", "fidelity", "sensitive", "without",
    "exposing", "customer", "customers", "pii", "personal", "privacy", "the", "and", "or",
    "for", "to", "of", "in", "on", "with", "an", "a", "is", "are", "be",
}

# Terms such as `balance`, `status`, `amount`, and `id` occur in almost every telecom schema.
# They should never establish model relevance by themselves. More domain-specific words can.
_CATALOG_WEAK_CONTEXT_TOKENS = {
    "balance", "status", "state", "amount", "value", "id", "key", "operation", "resource",
    "date", "time", "timestamp", "type", "name", "description", "reason", "code", "unit",
    "units", "reference", "customer",
}

# Generic model terms are useful for explaining a schema, but they do not identify a business
# resource by themselves. This prevents descriptions such as "original BalanceTopup" from making
# AdjustBalance/ReserveBalance/TransferBalance sibling resources relevant to a top-up-only scenario.
_CATALOG_MODEL_GENERIC_TOKENS = {
    "data", "entity", "resource", "object", "model", "record", "records", "value", "values",
    "reference", "ref", "management", "service", "operation", "action", "event", "events",
    "history", "audit", "timeline", "log", "ledger", "summary", "snapshot", "aggregate",
    "aggregated", "rollup", "balance", "customer", "account", "party", "product", "profile",
    "detail", "details",
}

_CATALOG_RELATIONSHIP_MODEL_TOKENS = {
    "history", "audit", "timeline", "log", "ledger", "summary", "snapshot", "aggregate",
    "aggregated", "rollup", "accumulated",
}


_CATALOG_CONTEXT_ALIASES = {
    "top_up": "topup",
    "recharge": "topup",
    "subscriber": "customer",
    "client": "customer",
    "member": "customer",
    "mobile": "subscriber",
    "retention": "retention",
}


_CATALOG_CONTEXT_STOPWORDS = {
    "to", "in", "of", "the", "and", "or", "for", "with", "without", "on", "an", "a",
    "is", "are", "be", "this", "that", "from", "by", "as", "it", "its", "can", "will",
    "used", "use", "using", "more", "high", "fidelity", "synthetic", "generate", "dataset",
    "scenario", "industry", "country", "type", "case", "normal", "transactional", "data",
    "sensitive", "exposing", "customer", "customers", "pii", "personal", "privacy",
}


def _canonical_context_tokens(tokens: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for token in tokens or []:
        value = str(token or "").casefold()
        if not value or value in _CATALOG_CONTEXT_STOPWORDS:
            continue
        result.add(_CATALOG_CONTEXT_ALIASES.get(value, value))
    return result


def semantic_exclusion_aliases(variable: dict[str, Any]) -> set[str]:
    """Return non-global aliases suitable for DB-vs-source duplicate suppression."""
    raw = canonical_variable_semantic_key(variable)
    if not raw:
        return set()
    parts = [part for part in raw.split("_") if part]
    aliases = {raw}
    # Normalize repeated leaf segments generated by flattening Quantity/Money-like value objects.
    collapsed = re.sub(r"(_[a-z0-9]+)\1$", r"\1", raw)
    aliases.add(collapsed)
    if len(parts) >= 2:
        aliases.add("_".join(parts[-2:]))
    if len(parts) >= 3:
        root = "_".join(parts[:2])
        leaf = "_".join(parts[-2:])
        aliases.add(f"{root}_{leaf}")
        if root.endswith("_balance"):
            short_root = root[:-7].rstrip("_")
            aliases.add(f"{short_root}_{leaf}")
            if leaf.endswith("_amount"):
                aliases.add(f"{short_root}_amount")
            if leaf.endswith("_status"):
                aliases.add(f"{short_root}_status")
            if leaf.endswith("_is_auto_topup"):
                aliases.add(f"{short_root}_is_auto_topup")
    # Known, domain-neutral lexical aliases for a few ubiquitous business operations.
    aliases.update(alias.replace("recharge", "topup") for alias in list(aliases))
    aliases.update(alias.replace("top_up", "topup") for alias in list(aliases))
    return {normalize_lookup_key(alias) for alias in aliases if normalize_lookup_key(alias)}


def select_json_source_catalog(
    catalog_rows: list[dict[str, Any]],
    *,
    business_context: str = "",
    preferred_names: set[str] | None = None,
    excluded_names: set[str] | None = None,
    excluded_semantic_keys: set[str] | None = None,
    max_fields: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the maximum number of distinct, source-backed concepts that fit the scenario.

    The algorithm is deliberately domain-neutral. It uses the request only to identify relevant
    business models, then expands those models across all distinct scalar concepts available in the
    authoritative source. Source wrappers (create/update/delete/events) are already canonicalized
    upstream, so they cannot multiply the executable variable count.
    """
    limit = max(0, int(max_fields))
    if limit == 0:
        return [], {
            "candidate_count": len(catalog_rows or []),
            "eligible_count_before_cap": 0,
            "selected_count": 0,
            "max_fields": 0,
            "capped": False,
            "truncated_count": 0,
            "selection_mode": "recall_first_domain_neutral_source_expansion",
            "relevant_models": [],
            "selected_names": [],
        }

    excluded = {
        normalize_lookup_key(value) for value in (excluded_names or set()) if normalize_lookup_key(value)
    }
    excluded_semantics = {
        normalize_lookup_key(value) for value in (excluded_semantic_keys or set()) if normalize_lookup_key(value)
    }
    preferred = {
        normalize_lookup_key(value) for value in (preferred_names or set()) if normalize_lookup_key(value)
    }
    raw_context_tokens = _catalog_text_tokens(business_context)
    context_tokens = _canonical_context_tokens(raw_context_tokens)

    rows: list[dict[str, Any]] = []
    seen_semantics: set[str] = set()
    for raw in catalog_rows or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        name = normalize_lookup_key(row.get("name"))
        semantic = canonical_variable_semantic_key(row)
        if not name or name in excluded or (semantic and (semantic in seen_semantics or semantic in excluded_semantics)):
            continue
        if str(row.get("model_kind") or "resource") in {"support", "support_reference", "abstract", "auxiliary"}:
            continue
        if semantic:
            seen_semantics.add(semantic)
        rows.append(row)

    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        score, meta = _catalog_relevance_score(row, context_tokens=context_tokens, preferred=preferred, privacy_context_tokens=raw_context_tokens)
        scored.append({"row": row, "index": index, "score": score, "meta": meta})

    # Identify business models from meaningful context signals. Only model-name terms that
    # survive canonicalization can open a new model. Field descriptions are used as supporting
    # evidence, never as a standalone reason to pull an unrelated resource.
    model_scores: dict[str, float] = {}
    model_evidence: dict[str, set[str]] = {}
    relationship_models: set[str] = set()
    preferred_model_hits: dict[str, int] = {}
    for item in scored:
        row = item["row"]
        model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        if not model:
            continue
        meta = item["meta"]
        if normalize_lookup_key(row.get("name")) in preferred:
            preferred_model_hits[model] = preferred_model_hits.get(model, 0) + 1
        model_tokens = _canonical_context_tokens(_catalog_text_tokens(model))
        model_signal = model_tokens - _CATALOG_WEAK_CONTEXT_TOKENS
        model_specific_tokens = model_tokens - _CATALOG_MODEL_GENERIC_TOKENS - _CATALOG_WEAK_CONTEXT_TOKENS
        model_specific_overlap = model_specific_tokens & context_tokens
        model_description_signal = (
            _canonical_context_tokens(_catalog_text_tokens(row.get("model_description")))
        ) & context_tokens
        model_description_signal -= _CATALOG_WEAK_CONTEXT_TOKENS
        relationship_model = bool(model_tokens & _CATALOG_RELATIONSHIP_MODEL_TOKENS)
        if str(row.get("model_kind") or "") == "event_wrapper":
            # Event is a scope qualifier, not a business resource selector. Only open an event
            # model when the request explicitly asks for event/history/notification data AND the
            # event's underlying business resource is named in the request. This prevents one word
            # such as "event" from opening every event schema in a TMF document.
            event_tokens = {"event", "history", "audit", "notification", "timeline", "listener"}
            event_requested = bool(context_tokens & event_tokens)
            base_model = model.removeprefix("event_")
            base_model_terms = {
                _CATALOG_CONTEXT_ALIASES.get(token, token)
                for token in _catalog_text_tokens(base_model)
            }
            strong_base_overlap = (base_model_terms & context_tokens) - _CATALOG_WEAK_CONTEXT_TOKENS
            # Exact generic model names (for example ``customer``) are still meaningful for an
            # explicitly requested event resource even though they are weak as standalone model
            # selectors. A generic word such as ``balance`` never qualifies by itself.
            exact_model_token = base_model if base_model in context_tokens else ""
            base_context_overlap = strong_base_overlap or ({exact_model_token} if exact_model_token else set())
            if not event_requested or not base_context_overlap:
                model_signal = set()
            else:
                model_signal = base_context_overlap
            model_specific_overlap = base_context_overlap
            model_description_signal = set()
        # A source model is a primary relevance candidate when its own distinctive name token is
        # requested. Generic model/category terms require either multiple independent field signals
        # or a relationship-model description explicitly referring to an already relevant resource.
        field_signal = (
            _canonical_context_tokens(meta.get("name_overlap") or [])
            | _canonical_context_tokens(meta.get("semantic_overlap") or [])
            | _canonical_context_tokens(meta.get("path_overlap") or [])
            | _canonical_context_tokens(meta.get("description_overlap") or [])
        ) - _CATALOG_WEAK_CONTEXT_TOKENS - _CATALOG_MODEL_GENERIC_TOKENS
        preferred_hit = normalize_lookup_key(row.get("name")) in preferred
        # One LLM-selected field is only a ranking preference. Multiple independent preferred fields
        # from the same model are stronger evidence that the model itself is relevant. This lets the
        # LLM surface domain semantics that are not present in the model name while still preventing
        # one generic/accidental field from opening an unrelated resource.
        # A single LLM-selected field is only a ranking preference. It must NOT open an otherwise
        # unrelated source model, because repeated operation resources often expose identically
        # shaped ``status``, ``amount`` and timestamp fields. A model becomes relevant only from
        # explicit model/context evidence, the requested entity anchor, or multiple independent
        # field-level signals.
        is_event_model = str(row.get("model_kind") or "") == "event_wrapper"
        relationship_support = (
            relationship_model
            and bool(model_description_signal)
            and len(model_description_signal) >= 1
        )
        preferred_model_support = preferred_model_hits.get(model, 0) >= 2
        qualifies = (
            bool(model_specific_overlap)
            or (len(field_signal) >= 2)
            or preferred_model_support
            or relationship_support
        )
        if is_event_model:
            qualifies = bool(model_signal) and bool(model_specific_overlap)
        if qualifies:
            signal = float(item["score"]) + (30.0 if model_specific_overlap else 0.0) + (10.0 if relationship_support else 0.0)
            model_scores[model] = max(model_scores.get(model, float("-inf")), signal)
            if relationship_model:
                relationship_models.add(model)
            reasons = model_evidence.setdefault(model, set())
            if preferred_hit:
                reasons.add("preferred_field_in_relevant_model")
            if preferred_model_support:
                reasons.add("multiple_preferred_fields_in_model")
            if model_specific_overlap:
                reasons.add("business_model_specific_context_overlap")
            if field_signal and not is_event_model:
                reasons.add("field_context_overlap")
            if relationship_support:
                reasons.add("source_model_relationship_context")

    # Requested entity keys are an explicit business anchor, but do not by themselves open every
    # model that happens to contain an ID. Add only the model owning the exact requested key.
    # Only an explicit entity/anchor name may seed a model before contextual relevance is established.
    # Other LLM-preferred fields remain preferences inside already relevant models.
    entity_anchor_names = {
        normalize_lookup_key(value)
        for value in preferred
        if normalize_lookup_key(value).endswith(("_id", "_key"))
    }
    preferred_models: set[str] = set()
    for row in rows:
        name = normalize_lookup_key(row.get("name"))
        if name in entity_anchor_names:
            model = normalize_lookup_key(row.get("business_model") or row.get("model"))
            if model:
                preferred_models.add(model)
    relevant_models = set(model_scores) | preferred_models

    # Relationship expansion comes from actual source references. This is how a primary model can
    # reach a source-defined related business model (e.g. TopupBalance -> Bucket) without guessing
    # industry concepts. Use one hop to avoid graph explosion.
    model_graph: dict[str, set[str]] = {}
    all_business_models = {
        normalize_lookup_key(row.get("business_model") or row.get("model"))
        for row in rows
        if normalize_lookup_key(row.get("business_model") or row.get("model"))
    }
    for row in rows:
        source_model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        for linked in row.get("linked_business_models") or []:
            target = normalize_lookup_key(linked)
            if target and target != source_model and target in all_business_models:
                model_graph.setdefault(source_model, set()).add(target)

    for source_model in sorted(list(relevant_models)):
        for linked in sorted(model_graph.get(source_model, set())):
            relevant_models.add(linked)
            model_scores[linked] = max(model_scores.get(linked, 0.0), model_scores.get(source_model, 0.0) - 8.0)
            model_evidence.setdefault(linked, set()).add("source_relationship_expansion")

    selected: dict[str, dict[str, Any]] = {}
    selected_reasons: dict[str, set[str]] = {}

    def add(item: dict[str, Any], reason: str) -> None:
        semantic = canonical_variable_semantic_key(item["row"]) or normalize_lookup_key(item["row"].get("name"))
        if not semantic or semantic in selected:
            if semantic in selected:
                selected_reasons.setdefault(semantic, set()).add(reason)
            return
        selected[semantic] = item
        selected_reasons[semantic] = {reason}

    # Preserve explicit preferred/entity-key/required fields first, unless privacy rules mark them
    # as sensitive. The requested entity key is validated separately against the source boundary.
    seed_items = sorted(
        scored,
        key=lambda item: (
            0 if normalize_lookup_key(item["row"].get("name")) in preferred else 1,
            0 if item["row"].get("required") else 1,
            -float(item["score"]),
            item["index"],
        ),
    )
    for item in seed_items:
        row = item["row"]
        meta = item["meta"]
        name = normalize_lookup_key(row.get("name"))
        model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        if meta.get("privacy_sensitive") or meta.get("metadata_penalty", 0.0) >= 30.0:
            continue
        if _catalog_display_only(row, context_tokens=context_tokens, preferred=preferred):
            continue
        # Supporting history/audit/summary models are evaluated separately after primary-model
        # expansion. Do not let their required fields or a single LLM preference bypass the
        # cross-model duplicate/relevance guard.
        if model in relationship_models:
            continue
        if name in preferred and (model in relevant_models or name in entity_anchor_names):
            add(item, "llm_preferred")
        elif row.get("required") and model in relevant_models:
            add(item, "source_required")
        elif model in relevant_models and (meta["informative_overlap"] or meta.get("semantic_overlap") or meta.get("model_overlap")):
            add(item, "direct_context_match")

    # Expand every meaningful field from primary relevant business models. Technical/display metadata
    # and privacy-sensitive fields are excluded regardless of score so the cap is not spent on noise.
    # Supporting history/audit/summary models are deliberately handled separately below: these models
    # often repeat the same status/amount/date/reference fields as the primary transaction model.
    primary_models = relevant_models - relationship_models
    for item in sorted(
        scored,
        key=lambda item: (
            0 if normalize_lookup_key(item["row"].get("business_model") or item["row"].get("model")) in primary_models else 1,
            -float(item["score"]),
            int(item["row"].get("depth", 0) or 0),
            item["index"],
        ),
    ):
        row = item["row"]
        meta = item["meta"]
        model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        if model not in primary_models:
            continue
        if meta["privacy_sensitive"] or meta["metadata_penalty"] >= 30.0:
            continue
        if _catalog_display_only(row, context_tokens=context_tokens, preferred=preferred):
            continue
        add(item, "relevant_model_expansion" if model in model_scores and "source_relationship_expansion" not in model_evidence.get(model, set()) else "related_model_expansion")

    def _model_leaf_signature(row: dict[str, Any]) -> str:
        semantic = canonical_variable_semantic_key(row)
        model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        prefix = f"{model}_" if model else ""
        if prefix and semantic.startswith(prefix):
            return semantic[len(prefix):]
        parts = semantic.split(".", 1)
        return parts[1] if len(parts) == 2 else semantic

    primary_leaf_signatures: set[str] = set()
    for selected_item in selected.values():
        selected_row = selected_item["row"]
        selected_model = normalize_lookup_key(selected_row.get("business_model") or selected_row.get("model"))
        if selected_model in primary_models:
            leaf_signature = _model_leaf_signature(selected_row)
            if leaf_signature:
                primary_leaf_signatures.add(leaf_signature)

    # Supporting relationship models contribute only fields that are independently evidenced by the
    # request and are not the same leaf concept already represented by a primary resource. This is the
    # root guard against turning a generic history/audit resource into hundreds of repeated status,
    # amount, identifier, and timestamp columns.
    for item in sorted(
        scored,
        key=lambda item: (-float(item["score"]), item["index"], normalize_lookup_key(item["row"].get("name"))),
    ):
        row = item["row"]
        meta = item["meta"]
        model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        if model not in relationship_models:
            continue
        if meta["privacy_sensitive"] or meta["metadata_penalty"] >= 30.0:
            continue
        if _catalog_display_only(row, context_tokens=context_tokens, preferred=preferred):
            continue
        semantic = canonical_variable_semantic_key(row)
        leaf_signature = _model_leaf_signature(row)
        if leaf_signature in primary_leaf_signatures:
            continue
        field_signal = (
            _canonical_context_tokens(meta.get("name_overlap") or [])
            | _canonical_context_tokens(meta.get("semantic_overlap") or [])
            | _canonical_context_tokens(meta.get("path_overlap") or [])
            | _canonical_context_tokens(meta.get("description_overlap") or [])
        ) - _CATALOG_WEAK_CONTEXT_TOKENS - _CATALOG_MODEL_GENERIC_TOKENS
        if not field_signal:
            continue
        add(item, "relevant_supporting_model_field")

    # Relevant fields from models not selected above are admitted only if their own evidence is
    # strong. This prevents the generic word 'customer' from accidentally opening the entire source.
    for item in sorted(scored, key=lambda x: (-float(x["score"]), x["index"], normalize_lookup_key(x["row"].get("name")))):
        semantic = canonical_variable_semantic_key(item["row"]) or normalize_lookup_key(item["row"].get("name"))
        if semantic in selected:
            continue
        meta = item["meta"]
        row = item["row"]
        model = normalize_lookup_key(row.get("business_model") or row.get("model"))
        # Models already classified as relevant were handled by the primary/supporting expansion
        # above. This fallback is only for a genuinely independent model with strong field-level
        # evidence; it must not re-add supporting history/audit models wholesale.
        if model in relevant_models:
            continue
        if meta["privacy_sensitive"] or meta["metadata_penalty"] >= 30.0:
            continue
        if _catalog_display_only(row, context_tokens=context_tokens, preferred=preferred):
            continue
        if str(row.get("model_kind") or "") == "event_wrapper" and normalize_lookup_key(row.get("business_model")) not in relevant_models:
            continue
        direct_signal = (
            _canonical_context_tokens(meta["informative_overlap"])
            | _canonical_context_tokens(meta.get("semantic_overlap", []))
            | _canonical_context_tokens(meta["model_overlap"])
        ) - _CATALOG_WEAK_CONTEXT_TOKENS
        if len(direct_signal) >= 2 and float(item["score"]) >= 32.0:
            add(item, "strong_cross_model_context_match")

    ordered_items = sorted(
        selected.values(),
        key=lambda item: (
            0 if normalize_lookup_key(item["row"].get("name")) in preferred else 1,
            0 if item["row"].get("required") else 1,
            0 if item["meta"]["role"] == "identity" else 1,
            0 if normalize_lookup_key(item["row"].get("business_model")) in model_scores else 1,
            -float(item["score"]),
            int(item["row"].get("depth", 0) or 0),
            item["index"],
            normalize_lookup_key(item["row"].get("name")),
        ),
    )
    truncated = max(0, len(ordered_items) - limit)
    selected_items = ordered_items[:limit]
    selected_rows = [dict(item["row"]) for item in selected_items]

    report = {
        "candidate_count": len(scored),
        "canonical_candidate_count": len(rows),
        "semantic_duplicate_rows_collapsed": max(0, len(catalog_rows or []) - len(rows)),
        "eligible_count_before_cap": len(ordered_items),
        "selected_count": len(selected_rows),
        "max_fields": limit,
        "capped": bool(truncated),
        "truncated_count": truncated,
        "selection_mode": "recall_first_domain_neutral_semantic_model_expansion",
        "direct_relevance_count": sum(
            1 for item in selected_items
            if item["meta"]["informative_overlap"] or item["meta"].get("semantic_overlap") or item["meta"]["model_overlap"]
        ),
        "relevant_models": sorted(relevant_models),
        "relationship_models": sorted(relationship_models),
        "primary_models": sorted(primary_models),
        "model_evidence": {key: sorted(value) for key, value in sorted(model_evidence.items())},
        "preferred_model_hits": {key: count for key, count in sorted(preferred_model_hits.items())},
        "selected_business_models": sorted({str(row.get("business_model") or row.get("model") or "") for row in selected_rows}),
        "privacy_sensitive_fields_excluded": sum(1 for item in scored if item["meta"].get("privacy_sensitive")),
        "technical_or_display_fields_excluded": sum(
            1 for item in scored
            if item["meta"]["metadata_penalty"] >= 30.0
            or _catalog_display_only(item["row"], context_tokens=context_tokens, preferred=preferred)
        ),
        "preferred_names_used": sorted({normalize_lookup_key(row.get("name")) for row in selected_rows} & preferred),
        "selected_names": [str(row.get("name") or "") for row in selected_rows],
        "excluded_names": sorted(excluded),
        "excluded_semantic_keys": sorted(excluded_semantics),
        "breadth_policy": {
            "prioritize_recall": True,
            "source_vocabulary_only": True,
            "semantic_wrapper_deduplication": True,
            "expand_relevant_business_models": True,
            "expand_source_defined_relationships": True,
            "exclude_transport_display_metadata": True,
            "exclude_requested_privacy_fields": True,
            "preserve_source_contract": True,
            "llm_is_not_final_breadth_gate": True,
        },
    }
    return selected_rows, report


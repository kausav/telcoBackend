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

from core import lexicon
from core.variable_semantics import normalize_variable_name, variable_semantic_aliases
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


_STORED_KEYS_TTL_SECONDS = 60.0
_stored_keys_cache: dict[str, Any] = {"at": float("-inf"), "keys": ()}


def _stored_industry_keys() -> tuple[str, ...]:
    """Industry keys of the active source documents (cached briefly; empty when the database is unreachable)."""
    now = monotonic()
    if now - _stored_keys_cache["at"] < _STORED_KEYS_TTL_SECONDS:
        return _stored_keys_cache["keys"]
    try:
        keys = tuple(sorted(str(k) for k in _collection().distinct("industry_key", {"active": True}) if k))
    except Exception:
        return ()
    _stored_keys_cache.update(at=now, keys=keys)
    return keys


def normalize_industry_key(value: str | None) -> str:
    """Resolve an industry label to the key the source documents are stored under.

    A label the vocabulary knows maps to its key; otherwise the label is matched against the stored industry
    keys (equal, or one is the start of the other: ``telecommunications`` -> ``telecom``), so a request does
    not have to spell the industry exactly the way it was registered.
    """
    raw = str(value or "").strip()
    if not raw:
        return "generic"
    compact = normalize_lookup_key(raw)
    known = lexicon.load().industry_labels.get(compact)
    if known:
        return known
    stored = _stored_industry_keys()
    if compact in stored:
        return compact
    close = [k for k in stored if len(k) >= 4 and len(compact) >= 4 and (compact.startswith(k) or k.startswith(compact))]
    return min(close, key=lambda k: (-len(k), k)) if close else (compact or "generic")


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
    if name.endswith("_ref") or name == "related_party":
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
    properties = model_schema.get("properties") if isinstance(model_schema, dict) else {}
    if normalized in _AUXILIARY_MODEL_MARKERS or any(marker in normalized for marker in ("error", "event_subscription")):
        return "auxiliary"
    if normalized.endswith("_ref") or normalized.endswith("_ref_or_value") or normalized.endswith("_or_value") or _is_reference_model_name(model_name, model_schema):
        return "support_reference"
    if _is_event_model_name(model_name):
        return "event_wrapper"
    if _is_crud_wrapper_model_name(model_name):
        return "crud_wrapper"
    # Query/task wrappers are transport/query surfaces around a concrete business resource. If the
    # concrete resource definition is present in the same source, keep the query wrapper out of the
    # primary synthetic feature catalog so query metadata cannot crowd out the underlying resource.
    if normalized.startswith("query_") or normalized.endswith("_query"):
        return "support"
    if "abstract resource" in description or normalized == "addressable":
        return "abstract"
    normalized_roots = {_normalize_model_name(item) for item in api_root_models}
    if normalized in normalized_roots or not api_root_models:
        return "resource"

    # Some TMF APIs expose a query/task resource at the endpoint while returning the actual business
    # resource only through a referenced property (for example QueryUsageConsumption -> UsageConsumption).
    # The old path treated that referenced business resource as generic support and consequently made
    # every one of its scalar leaves invisible to selection. Detect such concrete resource definitions
    # structurally: an id/href identity pair plus resource/task/business-purpose prose. Reference/value
    # schemas are already excluded above, so this does not promote ProductRefOrValue/BucketRefOrValue.
    if isinstance(properties, dict) and {"id", "href"}.issubset({str(key) for key in properties}):
        resource_markers = (
            "resource", "entity", "task", "enables", "allows", "provides", "represents",
            "used to", "used for", "can be ", "manage the", "retrieve the",
        )
        if any(marker in description for marker in resource_markers):
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


def _source_owner_family(value: Any) -> str:
    """Canonicalize a referenced schema owner into its business concept family.

    Reference/value wrappers such as ``ProductRefOrValue`` and CRUD/event wrappers are structural
    containers, not different business concepts. Removing those wrappers here lets duplicate
    detection operate on the schema definition that actually owns the scalar leaf.
    """
    name = _normalize_model_name(value)
    if not name:
        return ""
    name = _strip_schema_wrappers(name)
    changed = True
    while changed:
        changed = False
        for suffix in ("_ref_or_value", "_or_value", "_ref"):
            if name.endswith(suffix):
                name = name[: -len(suffix)].rstrip("_")
                changed = True
                break
    return name or _normalize_model_name(value)


def _normalize_source_relative_path(value: Any) -> str:
    """Normalize the scalar path relative to its owning schema definition."""
    parts = [
        _normalize_model_name(part.replace("[]", ""))
        for part in str(value or "").split(".")
        if _normalize_model_name(part.replace("[]", ""))
    ]
    if not parts:
        return ""
    return ".".join(parts)


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
    source_owner_model: str | None = None,
    source_owner_kind: str | None = None,
    source_owner_relation: str | None = None,
    source_owner_relative_path: str | None = None,
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
        "source_owner_model": _source_owner_family(source_owner_model or model),
        "source_owner_kind": str(source_owner_kind or model_kind or "").strip(),
        # Relationship roles use the same wrapper canonicalization as owner models. This turns
        # ``product_ref_or_value`` into the reusable ``product`` relationship without collapsing
        # genuinely different roles such as ``alternate_product`` or ``user``.
        "source_owner_relation": _source_owner_family(source_owner_relation or source_owner_model or model),
        "source_owner_relative_path": _normalize_source_relative_path(source_owner_relative_path or field_name),
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
        owner_model: str | None = None,
        owner_kind: str | None = None,
        owner_relation: str | None = None,
        owner_relative_prefix: str = "",
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
                        target_kind = _model_kind(
                            item_ref,
                            schemas.get(item_ref) if isinstance(schemas.get(item_ref), dict) else item_schema,
                            api_root_models,
                        )
                        visit(
                            item_ref, item_schema, path + "[]", is_required, stack + (item_ref,), depth + 1,
                            linked_business_models,
                            owner_model=_source_owner_family(item_ref),
                            owner_kind=target_kind,
                            owner_relation=str(field_name),
                            owner_relative_prefix="",
                        )
                        continue
                    if isinstance(item_schema.get("properties"), dict):
                        visit(
                            None, item_schema, path + "[]", is_required, stack, depth + 1, linked_business_models,
                            owner_model=owner_model or model_name,
                            owner_kind=owner_kind or root_kind,
                            owner_relation=owner_relation or owner_model or model_name,
                            owner_relative_prefix=(
                                f"{owner_relative_prefix}.{field_name}" if owner_relative_prefix else str(field_name)
                            ),
                        )
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
                target_schema_name = target_schema_name or ref
                target_kind = _model_kind(
                    target_schema_name,
                    schemas.get(target_schema_name) if isinstance(schemas.get(target_schema_name), dict) else prop,
                    api_root_models,
                )
                visit(
                    ref, prop, path, is_required, stack + (ref,), depth + 1, next_links,
                    owner_model=_source_owner_family(target_schema_name),
                    owner_kind=target_kind,
                    owner_relation=str(field_name),
                    owner_relative_prefix="",
                )
                continue
            if isinstance(prop.get("properties"), dict):
                visit(
                    None, prop, path, is_required, stack, depth + 1, linked_business_models,
                    owner_model=owner_model or model_name,
                    owner_kind=owner_kind or root_kind,
                    owner_relation=owner_relation or owner_model or model_name,
                    owner_relative_prefix=(
                        f"{owner_relative_prefix}.{field_name}" if owner_relative_prefix else str(field_name)
                    ),
                )
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
                source_owner_model=owner_model or model_name,
                source_owner_kind=owner_kind or root_kind,
                source_owner_relation=owner_relation or model_name,
                source_owner_relative_path=(
                    f"{owner_relative_prefix}.{field_name}" if owner_relative_prefix else str(field_name)
                ),
            )
            if spec is not None and linked_business_models:
                spec["linked_business_models"] = list(linked_business_models)
            if spec is not None:
                rows.append(spec)

    visit(
        model_name, model_schema, model_name, True, (model_name,), 0,
        owner_model=model_name, owner_kind=root_kind, owner_relation=model_name, owner_relative_prefix="",
    )
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
        source_owner_models: list[str] = []
        source_owner_kinds: list[str] = []
        source_owner_relations: list[str] = []
        source_owner_relative_paths: list[str] = []
        linked_models: list[str] = []
        conflicts: dict[str, list[Any]] = {}
        for candidate in candidates:
            for field, target in (
                ("source_ids", source_ids),
                ("source_paths", source_paths),
                ("source_models", source_models),
                ("source_names", source_names),
                ("source_owner_models", source_owner_models),
                ("source_owner_kinds", source_owner_kinds),
                ("source_owner_relations", source_owner_relations),
                ("source_owner_relative_paths", source_owner_relative_paths),
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
        if source_owner_models:
            base["source_owner_models"] = source_owner_models
            base["source_owner_model"] = source_owner_models[0]
        if source_owner_kinds:
            base["source_owner_kinds"] = source_owner_kinds
            base["source_owner_kind"] = source_owner_kinds[0]
        if source_owner_relations:
            base["source_owner_relations"] = source_owner_relations
            base["source_owner_relation"] = source_owner_relations[0]
        if source_owner_relative_paths:
            base["source_owner_relative_paths"] = source_owner_relative_paths
            base["source_owner_relative_path"] = source_owner_relative_paths[0]
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

    Prefer explicit source semantic identity when available. Otherwise normalize the executable
    variable name using shared structural rules (synonyms + adjacent repeated-token collapse).
    No generic suffix alias such as ``*_id`` is introduced.
    """
    data = variable if isinstance(variable, dict) else {}
    explicit = data.get("semantic_key") or data.get("_json_source_semantic_key")
    if explicit:
        return normalize_variable_name(str(explicit).replace("[]", "").replace(".", "_"))
    raw = normalize_variable_name(data.get("name") or variable)
    return raw


def _source_redundancy_signature(row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a structural business-concept signature for source-backed duplicate suppression.

    The key is based on the schema definition that owns the scalar leaf rather than the flattened
    resource path. This is important for TMF documents because the same reusable ``Product``,
    ``Price`` or ``RelatedParty`` concept can appear through many resource wrappers. At the same
    time, the owning concept and relationship role remain in the signature so unrelated fields such
    as ``Bucket.status`` and ``TopupBalance.status`` do not collapse merely because their contracts
    are identical.

    This is deliberately structural: no industry-specific keep/remove field list is used.
    """
    if not isinstance(row, dict):
        return None

    owner = _source_owner_family(
        row.get("source_owner_model")
        or row.get("model")
        or row.get("business_model")
    )
    relation = _normalize_model_name(
        row.get("source_owner_relation")
        or owner
        or row.get("business_model")
    )
    relative_path = _normalize_source_relative_path(
        row.get("source_owner_relative_path")
        or row.get("field")
        or row.get("name")
    )
    if not owner or not relative_path:
        return None

    # Executable dtype/enum/constraint differences are deliberately NOT part of structural identity.
    # TMF reference/value wrappers frequently restate the same business property with weaker or
    # incomplete constraints (for example BucketRefOrValue.status has no enum while Bucket.status
    # carries the authoritative vocabulary). When owner + role + relative property path are identical,
    # they describe the same source concept and one canonical representative must win. The selection
    # layer separately prefers the direct owning business resource over a nested reference wrapper.
    #
    # This also handles repeated scalar concepts such as RelatedParty.id, Money.taxIncludedAmount.value,
    # ProductTerm.duration.amount, and recurring period fields without maintaining a field-name list.
    return (owner, relation, relative_path)


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
            "The LLM may provide relevance hints, but deterministic selection is scenario-relevance-first; source models and scalar leaves are never emitted solely because they exist in the source catalog.",
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


def validate_catalog_selection(
    variables: Iterable[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    business_context: str = "",
    preferred_names: set[str] | None = None,
    industry_type: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate LLM-selected names against the canonical source catalog and request context.

    Validation is intentionally conservative: an LLM cannot reintroduce transport wrappers,
    unrelated secondary metadata branches, or presentation-only fields merely by naming them.
    The rule is structural/contextual, not a DU-01 field allowlist, so pricing/agreement/note/etc.
    become eligible automatically when the request genuinely asks for those concepts.
    """
    vocab = lexicon.load(normalize_industry_key(industry_type) if industry_type else None)
    context_tokens = _canonical_context_tokens(_catalog_text_tokens(business_context), vocab)
    preferred = {
        normalize_lookup_key(value) for value in (preferred_names or set())
        if normalize_lookup_key(value)
    }
    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen_semantics: set[str] = set()
    seen_redundancy: set[tuple[Any, ...]] = set()
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
        # Never allow an LLM candidate to bypass the same structural eligibility rules used by
        # deterministic source selection. Support/reference/auxiliary schemas are not independent
        # business resources; their scalar leaves are either wrapper transport or reusable nested
        # concepts owned by a concrete resource. Event wrappers are admitted only when the request
        # explicitly asks for event/history-style data.
        model_kind = str(spec.get("model_kind") or "").strip().lower()
        if model_kind in {"support", "support_reference", "abstract", "auxiliary", "crud_wrapper"}:
            rejected.append(name)
            continue
        if model_kind == "event_wrapper" and not (context_tokens & {"event", "history", "audit", "timeline", "log", "ledger", "notification"}):
            rejected.append(name)
            continue
        if context_tokens:
            if not _catalog_secondary_branch_relevant(spec, context_tokens=context_tokens, vocab=vocab):
                rejected.append(name)
                continue
            if _catalog_display_only(spec, context_tokens=context_tokens, preferred=preferred):
                rejected.append(name)
                continue
        redundancy = _source_redundancy_signature(spec)
        if redundancy is not None and redundancy in seen_redundancy:
            continue
        if semantic:
            seen_semantics.add(semantic)
        if redundancy is not None:
            seen_redundancy.add(redundancy)
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
    """Classify a scalar field by the leaf property and its description.

    Ancestor path tokens are intentionally excluded from broad role heuristics: a field such as
    ``TopupBalance.product.name`` contains ``balance`` in its flattened path but is a descriptive
    product name, not a numeric measurement. Looking at the leaf property plus its description also
    lets lifecycle words (status/state/result/reason/decision) win over numeric/container terms.
    """
    raw_name = str(spec.get("name") or "")
    raw_field = str(spec.get("field") or "")
    raw_path = str(spec.get("path") or "")
    leaf_name = raw_field or raw_name.rsplit(".", 1)[-1] or raw_name.rsplit("_", 1)[-1]
    description = str(spec.get("description") or "").lower()
    leaf = _snake_case(leaf_name).lower()
    semantic_text = f"{leaf} {description}"
    semantic_tokens = _catalog_text_tokens(semantic_text)
    dtype = _canonical_catalog_dtype(spec.get("dtype"), bool(spec.get("enum_values")))

    if dtype in {"datetime", "date"} or bool(semantic_tokens & {
        "timestamp", "date", "time", "effective", "expiry", "expiration"
    }):
        return "timing"

    lifecycle_signals = {"status", "state", "reason", "outcome", "result", "decision", "action"}
    if semantic_tokens & lifecycle_signals:
        return "status"

    categorical_signals = {"type", "category", "class", "code", "method", "channel", "mode", "role"}
    if bool(spec.get("enum_values")) or semantic_tokens & categorical_signals:
        return "categorical"

    if leaf.endswith(("_id", "_key")) or raw_path.lower().endswith(".id"):
        return "identity"

    measurement_signals = {
        "amount", "balance", "value", "quantity", "count", "rate", "score", "limit", "price",
        "percentage", "duration", "remaining", "consumed", "usage", "velocity", "volume",
    }
    if dtype in {"integer", "float", "number"} or semantic_tokens & measurement_signals:
        return "measurement"

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


_CATALOG_SECONDARY_BRANCH_TOKENS = {
    # Structural branches that are often legitimate source metadata but should not enter an
    # executable scenario merely because an LLM selected the parent resource. They become eligible
    # when the request itself contains evidence for that branch. This is intentionally generic: the
    # rule is about source structure (notes, agreements, pricing, places, etc.), not DU-01 field names.
    "note", "agreement", "agreement_item", "place", "serial", "serial_number", "category",
    "termination", "error", "price", "pricing", "tax", "alteration", "contact_medium",
    "product", "product_offering",
}

def _catalog_context_branch_tokens(row: dict[str, Any]) -> set[str]:
    """Return nested business-branch tokens from the complete source path.

    The deepest scalar owner is not sufficient to identify an excluded branch: a pricing leaf such as
    ``Product.productPrice[].price.dutyFreeAmount.value`` is ultimately owned by ``Money``. Looking only
    at the deepest owner would therefore lose the fact that the value lives under ``price``. The full
    source path retains that structural evidence while the root business-model tokens are removed.
    """
    root_model = _source_owner_family(row.get("business_model") or row.get("model"))
    root_tokens = _catalog_text_tokens(root_model)
    path_tokens = _catalog_text_tokens(row.get("path"))
    owner_tokens = _catalog_text_tokens(
        row.get("source_owner_model") or row.get("model") or row.get("business_model")
    )
    relation_tokens = _catalog_text_tokens(
        row.get("source_owner_relation") or row.get("source_owner_model") or row.get("relation")
    )
    return (path_tokens | owner_tokens | relation_tokens) - root_tokens


def _catalog_secondary_branch_relevant(
    row: dict[str, Any],
    *,
    context_tokens: set[str],
    vocab: lexicon.Lexicon | None = None,
) -> bool:
    """Return whether a secondary source branch has request evidence to enter the schema."""
    branch_tokens = _catalog_context_branch_tokens(row)
    secondary = branch_tokens & _CATALOG_SECONDARY_BRANCH_TOKENS
    if not secondary:
        return True
    effective = set(context_tokens or set())
    for token in list(effective):
        effective.update((vocab or lexicon.load()).branch_expansions.get(token, ()))
    # A compound nested branch is eligible only when every excluded branch component is supported by
    # the request. For example, `product + offering` is valid for an upsell request, while
    # `product + price` is not unless pricing was requested explicitly. This prevents a broad parent
    # concept such as `product` from accidentally authorizing unrelated pricing/notes/agreement leaves.
    return secondary <= effective


def _catalog_display_only(row: dict[str, Any], *, context_tokens: set[str], preferred: set[str]) -> bool:
    """Identify source leaves that primarily add transport/display noise instead of business signal."""
    name = normalize_lookup_key(row.get("name"))
    if not name or name in preferred:
        return False

    # Flattened TMF names end in the final scalar leaf, so checking only the last underscore token
    # cannot recognize nested display labels such as ``remaining_value.name``. Match structural
    # suffixes instead; these rules are schema-shape rules, not scenario-specific field names.
    if name.endswith((
        "_href", "_url", "_uri", "_schema_location", "_base_type", "_referred_type",
        "_display_label", "_display_name", "_formatted", "_remaining_value_name",
        "_logical_resource_name", "_party_account_name", "_related_party_name",
    )):
        return True

    leaf = name.rsplit("_", 1)[-1]
    if leaf == "description":
        return "description" not in context_tokens

    if leaf == "name":
        # Keep names only when the name belongs to a business concept explicitly represented in the
        # request. The root/branch identity matters: a request mentioning `product` must not pull
        # ``bucket_name`` merely because both are strings named `name`.
        root_model = _source_owner_family(row.get("business_model") or row.get("model"))
        root_tokens = _catalog_text_tokens(root_model)
        branch_tokens = _catalog_context_branch_tokens(row)
        name_context = {
            "product", "product_offering", "offering", "offer", "plan", "channel",
            "payment_method", "service", "resource", "resource_name", "named", "name",
        }
        return not bool((root_tokens | branch_tokens) & context_tokens & name_context)

    return False


_CATALOG_GENERIC_CONTEXT_TOKENS = {
    "generate", "dataset", "synthetic", "data", "type", "transactional", "aggregational",
    "industry", "normal", "scenario",
    "country", "in", "use", "case", "high", "fidelity", "sensitive", "without",
    "exposing", "customer", "customers", "pii", "personal", "privacy", "the", "and", "or",
    "for", "to", "of", "in", "on", "with", "an", "a", "is", "are", "be",
}

# Terms such as `balance`, `status`, `amount`, and `id` occur in almost every source schema.
# They should never establish model relevance by themselves. More domain-specific words can.
_CATALOG_WEAK_CONTEXT_TOKENS = {
    "balance", "status", "state", "amount", "value", "id", "key", "operation", "resource",
    "date", "time", "timestamp", "type", "name", "description", "reason", "code", "unit",
    "units", "reference", "customer",
}

# Generic model terms are useful for explaining a schema, but they do not identify a business
# resource by themselves. This prevents descriptions such as "original <Resource>" from making
# AdjustBalance/ReserveBalance/TransferBalance sibling resources relevant to a single-resource scenario.
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


_CATALOG_CONTEXT_STOPWORDS = {
    "to", "in", "of", "the", "and", "or", "for", "with", "without", "on", "an", "a",
    "is", "are", "be", "this", "that", "from", "by", "as", "it", "its", "can", "will",
    "used", "use", "using", "more", "high", "fidelity", "synthetic", "generate", "dataset",
    "scenario", "industry", "country", "type", "case", "normal", "transactional", "data",
    "sensitive", "exposing", "customer", "customers", "pii", "personal", "privacy",
}


def _canonical_context_tokens(tokens: Iterable[str], vocab: lexicon.Lexicon | None = None) -> set[str]:
    vocab = vocab or lexicon.load()
    result: set[str] = set()
    for token in tokens or []:
        value = str(token or "").casefold()
        if not value or value in _CATALOG_CONTEXT_STOPWORDS or value in vocab.stopwords:
            continue
        canonical = vocab.aliases.get(value, value)
        result.add(canonical)
        result.update(vocab.expansions.get(value, ()))
        result.update(vocab.expansions.get(canonical, ()))
    return result


def semantic_exclusion_aliases(variable: dict[str, Any]) -> set[str]:
    """Return conservative aliases for DB/source duplicate suppression.

    Aliases are intentionally local to the full variable concept. In particular, do not add the
    last two tokens (for example ``product_id``) to arbitrary fields such as ``order_product_id``;
    that historically caused false-positive cross-entity suppression.
    """
    explicit = ""
    if isinstance(variable, dict):
        explicit = str(variable.get("semantic_key") or variable.get("_json_source_semantic_key") or "").strip()
    raw = explicit.replace("[]", "").replace(".", "_") if explicit else (variable.get("name") if isinstance(variable, dict) else variable)
    aliases = variable_semantic_aliases(raw)
    canonical = canonical_variable_semantic_key(variable)
    if canonical:
        aliases.add(canonical)

    # Preserve the small domain-neutral balance structural shortcuts already supported by
    # persisted legacy variables, but only when the full owner path remains present. These aliases
    # never reduce a variable to a generic leaf such as ``status`` or ``id``.
    if canonical:
        parts = canonical.split("_")
        if len(parts) >= 3 and parts[-2:] == ["balance", "amount"]:
            owner = "_".join(parts[:-2]).rstrip("_")
            if owner:
                aliases.add(f"{owner}_amount")
        if len(parts) >= 3 and parts[-2:] == ["balance", "status"]:
            owner = "_".join(parts[:-2]).rstrip("_")
            if owner:
                aliases.add(f"{owner}_status")

    return {normalize_lookup_key(alias) for alias in aliases if normalize_lookup_key(alias)}



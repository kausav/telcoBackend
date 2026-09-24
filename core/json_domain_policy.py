"""Machine-readable domain grounding policies.

The Low Balance & Top-up domain is grounded in the bundled TM Forum Swagger/OpenAPI
artifacts. Standard fields are resolved from the runtime registry; this module exposes
those exact source artifacts to the intent agent and keeps provenance explicit.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOW_BALANCE_DIR = ROOT / "resources" / "telecom" / "domain_sources" / "low_balance_topup"

LOW_BALANCE_DOMAIN_ALIASES = {
    "low balance & top-up",
    "low balance and top up",
    "low balance and top-up",
    "prepay balance",
    "prepay balance management",
    "top up",
    "top-up",
    "recharge balance",
}

LOW_BALANCE_SOURCE_IDS = ("tmf654_v4", "tmf629_v4")
LOW_BALANCE_MAIN_MODEL_IDS = (
    "tmf654_v4__bucket",
    "tmf654_v4__topup_balance",
    "tmf629_v4__customer",
)
LOW_BALANCE_SOURCE_NAMES = {
    "tmf654_v4": "TMF654 Prepay Balance Management API v4.0.0 Swagger",
    "tmf629_v4": "TMF629 Customer Management API v4.0.0 Swagger",
}
LOW_BALANCE_SOURCE_FILES = {
    "tmf654_v4": "TMF654_Prepay_Balance_Management_API_v4.0.0_swagger.json",
    "tmf629_v4": "TMF629_Customer_Management_API_v4.0.0_swagger.json",
}


def normalize_domain(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def is_json_grounded_domain(value: str | None) -> bool:
    text = normalize_domain(value)
    return text in LOW_BALANCE_DOMAIN_ALIASES or (
        "low balance" in text and any(token in text for token in ("top", "recharge"))
    )


def _load(source_id: str) -> dict[str, Any]:
    path = LOW_BALANCE_DIR / LOW_BALANCE_SOURCE_FILES[source_id]
    return json.loads(path.read_text(encoding="utf-8"))



LOW_BALANCE_MAIN_MODELS = {
    "tmf654_v4": (("Bucket", "bucket", "entity"), ("TopupBalance", "topupbalance", "transaction")),
    "tmf629_v4": (("Customer", "customer", "entity"),),
}

_IGNORED_METADATA_FIELDS = {"@baseType", "@schemaLocation", "@type"}


def _snake_case(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)


def _schema_scalar_spec(definitions: dict[str, Any], schema: dict[str, Any], *, source_id: str, model_name: str, field_name: str, path: str, required: bool, depth: int) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    ref = str(schema.get("$ref") or "").strip()
    if ref.startswith("#/definitions/"):
        target_name = ref.split("/", 2)[-1]
        target = definitions.get(target_name)
        if not isinstance(target, dict) or isinstance(target.get("properties"), dict):
            return None
        schema = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
    schema_type = schema.get("type")
    enum_values = list(schema.get("enum") or [])
    if schema_type in {"object", "array"} and not enum_values:
        return None
    if schema_type is None and not enum_values:
        return None
    return {
        "source_id": source_id,
        "model": model_name,
        "field": field_name,
        "path": path,
        "name": _snake_case(path),
        "dtype": schema_type or "string",
        "description": str(schema.get("description") or ""),
        "enum_values": enum_values,
        "format": schema.get("format"),
        "required": bool(required),
        "depth": depth,
    }


def _expand_model_scalars(definitions: dict[str, Any], source_id: str, model_name: str, prefix: str, grain: str, *, max_depth: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(def_name: str, path_prefix: str, parent_required: bool, stack: tuple[str, ...], depth: int) -> None:
        if depth > max_depth or def_name in stack:
            return
        model = definitions.get(def_name)
        if not isinstance(model, dict):
            return
        required_names = set(model.get("required") or [])
        for field_name, prop in (model.get("properties") or {}).items():
            if field_name in _IGNORED_METADATA_FIELDS or not isinstance(prop, dict):
                continue
            path = f"{path_prefix}.{field_name}" if path_prefix else field_name
            child_required = parent_required and field_name in required_names
            ref = str(prop.get("$ref") or "").strip()
            if ref.startswith("#/definitions/"):
                target_name = ref.split("/", 2)[-1]
                target = definitions.get(target_name)
                if isinstance(target, dict) and isinstance(target.get("properties"), dict):
                    visit(target_name, path, child_required, stack + (def_name,), depth + 1)
                    continue
            if prop.get("type") == "array":
                continue
            spec = _schema_scalar_spec(definitions, prop, source_id=source_id, model_name=model_name, field_name=field_name, path=path, required=child_required, depth=depth)
            if spec is not None:
                spec["model_grain"] = grain
                rows.append(spec)

    visit(model_name, prefix, True, (), 0)
    return rows


def expanded_scalar_catalog() -> list[dict[str, Any]]:
    """Return every safe scalar leaf from the supplied primary TMF Swagger models."""
    rows: list[dict[str, Any]] = []
    for source_id, model_specs in LOW_BALANCE_MAIN_MODELS.items():
        document = _load(source_id)
        definitions = document.get("definitions") or {}
        for model_name, prefix, grain in model_specs:
            rows.extend(_expand_model_scalars(definitions, source_id, model_name, prefix, grain))
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or "")
        if name and name not in seen:
            seen.add(name)
            result.append(row)
    return result


def catalog_for_request() -> dict[str, Any]:
    """Return the broad source catalog exposed to the Low Balance intent agent."""
    payload = expanded_scalar_catalog()
    for row in payload:
        row["model_role"] = {
            "Bucket": "Balance bucket state",
            "TopupBalance": "Prepay top-up/recharge operation",
            "Customer": "Customer/account relationship context",
        }.get(row["model"], row["model"])
    return {
        "source_policy": "bundled_official_swagger_only",
        "sources": source_manifest(),
        "models": payload,
        "notes": [
            "The catalog includes all materializable scalar leaves from TMF654 Bucket/TopupBalance and TMF629 Customer, including scalar leaves inside referenced objects.",
            "One-to-many array relationships are intentionally excluded from the flat row contract rather than collapsed into a fake scalar.",
            "Swagger metadata fields beginning with @ are excluded because they are implementation/type-system metadata rather than useful business dimensions.",
            "The executable variable boundary is strict: a variable must be an exact scalar leaf from the supplied TMF654/TMF629 Swagger catalog or be explicitly supplied from MongoDB. The LLM may select/review variables but may not create executable variables.",
        ],
    }

def source_manifest() -> list[dict[str, str]]:
    result = []
    for source_id in LOW_BALANCE_SOURCE_IDS:
        filename = LOW_BALANCE_SOURCE_FILES[source_id]
        path = LOW_BALANCE_DIR / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        result.append({
            "source_id": source_id,
            "name": LOW_BALANCE_SOURCE_NAMES[source_id],
            "filename": filename,
            "sha256": digest,
        })
    return result

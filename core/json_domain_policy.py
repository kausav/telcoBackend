"""Machine-readable domain grounding policies.

The Low Balance & Top-up domain is grounded in the bundled TM Forum Swagger/OpenAPI
artifacts. Standard fields are resolved from the runtime registry; this module exposes
those exact source artifacts to the intent agent and keeps provenance explicit.
"""
from __future__ import annotations

import hashlib
import json
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


def _resolve(definitions: dict[str, Any], ref: str) -> dict[str, Any] | None:
    if not ref.startswith("#/definitions/"):
        return None
    name = ref.split("/", 2)[-1]
    value = definitions.get(name)
    return value if isinstance(value, dict) else None


def _dtype(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")
    if schema_type in {"integer", "int", "number", "float", "double", "decimal"}:
        return "integer" if schema_type in {"integer", "int"} else "float"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "string":
        fmt = str(schema.get("format") or "").lower()
        if fmt in {"date-time", "datetime", "timestamp"}:
            return "datetime"
        if fmt == "date":
            return "date"
        return "categorical" if schema.get("enum") else "string"
    return "string"


def _flatten_model(definitions: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    model = definitions.get(model_name)
    if not isinstance(model, dict):
        return []
    rows: list[dict[str, Any]] = []
    required = set(model.get("required") or [])
    for name, prop in (model.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        resolved = _resolve(definitions, str(prop.get("$ref") or ""))
        effective = dict(resolved or {})
        effective.update({k: v for k, v in prop.items() if k != "$ref"})
        # Nested reference/collection structures are intentionally not flattened into
        # fake scalar values. Their scalar child models can still be referenced by the
        # model catalog when separately relevant.
        if effective.get("type") == "array" or (resolved and resolved.get("properties")):
            continue
        rows.append({
            "model": model_name,
            "field": name,
            "dtype": _dtype(effective),
            "required": name in required,
            "description": prop.get("description") or effective.get("description") or "",
            "enum_values": list(prop.get("enum") or effective.get("enum") or []),
            "format": effective.get("format"),
        })
    return rows


def catalog_for_request() -> dict[str, Any]:
    """Return the exact flat scalar model catalog exposed to the Low Balance agent."""
    payload: list[dict[str, Any]] = []
    for source_id in LOW_BALANCE_SOURCE_IDS:
        document = _load(source_id)
        definitions = document.get("definitions") or {}
        models = (
            ("Bucket", "Balance bucket state"),
            ("TopupBalance", "Prepay top-up/recharge operation"),
            ("Customer", "Customer/account relationship context"),
        )
        for model_name, model_role in models:
            for field in _flatten_model(definitions, model_name):
                field["model_role"] = model_role
                payload.append({"source_id": source_id, **field})
    return {
        "source_policy": "bundled_official_swagger_only",
        "sources": source_manifest(),
        "models": payload,
        "notes": [
            "Only scalar fields from TMF654 Bucket/TopupBalance and TMF629 Customer are exposed to flat synthetic-data generation.",
            "Nested object/array references are not converted into fake scalar values.",
            "Scenario-specific analytical fields may be proposed when they are not standard attributes; those fields are marked SCENARIO_DERIVED and receive deterministic synthetic contracts.",
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

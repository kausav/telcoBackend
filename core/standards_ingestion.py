"""Normalize machine-readable telecom model artifacts into registry artifacts.

The runtime registry consumes normalized JSON artifacts. This module provides the
first ingestion adapter for TM Forum-style Swagger/OpenAPI files and can be
extended with other standards adapters without changing the runtime compiler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re


def _snake(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value


def _type_from_schema(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return "reference"
    typ = schema.get("type", "string")
    if typ == "integer":
        return "integer"
    if typ == "number":
        return "float"
    if typ == "boolean":
        return "boolean"
    if typ == "array":
        return "array"
    if typ == "object":
        return "object"
    return "string"


def _ref_target(ref: str) -> str | None:
    if not ref or "/" not in ref:
        return None
    return ref.rsplit("/", 1)[-1]


def normalize_openapi_document(document: dict[str, Any], source_url: str = "", organization: str = "TM Forum") -> dict[str, Any]:
    """Convert Swagger 2/OpenAPI 3 schemas into INGENII normalized registry JSON.

    This adapter intentionally extracts only semantic structure. Synthetic generation
    behavior is supplied separately through generation profiles.
    """
    is_swagger = document.get("swagger")
    is_openapi = document.get("openapi")
    if not is_swagger and not is_openapi:
        raise ValueError("Document is not Swagger 2.x or OpenAPI 3.x")

    definitions = document.get("definitions") or document.get("components", {}).get("schemas", {})
    info = document.get("info") or {}
    title = info.get("title") or "Imported Standards Artifact"
    version = info.get("version")
    artifact_id = _snake(title) or "imported_artifact"

    entities: list[dict[str, Any]] = []
    for schema_name, schema in definitions.items():
        if not isinstance(schema, dict) or schema.get("type") != "object":
            continue
        cid = _snake(schema_name)
        required = set(schema.get("required") or [])
        attributes: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        properties = schema.get("properties") or {}
        for field_name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            enum_values = prop.get("enum") or []
            attr = {
                "name": field_name,
                "dtype": _type_from_schema(prop),
                "required": field_name in required,
                "nullable": bool(prop.get("nullable", False)),
                "description": prop.get("description", ""),
                "enum_values": enum_values,
            }
            attributes.append(attr)
            ref = _ref_target(prop.get("$ref", ""))
            if ref:
                relationships.append({
                    "target": _snake(ref),
                    "relation": "references",
                    "cardinality": "1:N" if prop.get("type") == "array" else "N:1",
                    "required": field_name in required,
                    "description": f"Reference from {schema_name}.{field_name} to {ref}",
                })
            items = prop.get("items") or {}
            ref = _ref_target(items.get("$ref", "")) if isinstance(items, dict) else None
            if ref:
                relationships.append({
                    "target": _snake(ref),
                    "relation": "contains",
                    "cardinality": "1:N",
                    "required": field_name in required,
                    "description": f"Collection reference from {schema_name}.{field_name} to {ref}",
                })

        entities.append({
            "canonical_id": cid,
            "name": schema_name,
            "aliases": [schema_name],
            "domain": "telecom",
            "description": schema.get("description", ""),
            "sources": [{
                "standard": organization,
                "artifact": title,
                "version": version,
                "reference": f"{title}::{schema_name}",
                "url": source_url,
                "source_role": "machine-readable-source",
            }],
            "attributes": attributes,
            "relationships": relationships,
        })

    return {
        "artifact": {
            "artifact_id": artifact_id,
            "organization": organization,
            "title": title,
            "artifact_version": version,
            "status": "imported",
            "source_kind": "openapi",
            "source_url": source_url,
        },
        "entities": entities,
    }


def normalize_openapi_file(input_path: str | Path, output_path: str | Path, source_url: str = "", organization: str = "TM Forum") -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    document = json.loads(input_path.read_text(encoding="utf-8"))
    normalized = normalize_openapi_document(document, source_url=source_url, organization=organization)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return output_path

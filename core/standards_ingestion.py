"""Normalize official telecom machine-readable model artifacts.

Supported inputs:
- TM Forum Swagger 2 / OpenAPI 3 JSON
- MEF JSON Schema / YAML product schemas, including cross-file ``$ref``/``allOf``
- 3GPP ASN.1 packages containing named SEQUENCE/SET/CHOICE definitions

The normalized representation is an internal runtime index only. Every entity and
attribute keeps the originating standard, version, source URL and source reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import re

ASN1_NORMALIZER_VERSION = "2.1"

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _snake(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value or "model"


def _namespaced(source_id: str, value: str) -> str:
    return f"{_snake(source_id)}__{_snake(value)}"


def _clean_ref_basename(ref: str) -> str | None:
    """Return an entity-like name from local or relative JSON/YAML $refs."""
    if not ref:
        return None
    path_part = str(ref).split("#", 1)[0].replace("\\", "/")
    fragment = str(ref).split("#", 1)[1] if "#" in str(ref) else ""
    if path_part:
        name = Path(path_part).name
        name = re.sub(r"\.(json|yaml|yml)$", "", name, flags=re.I)
        if name:
            return name
    if fragment:
        bits = [bit for bit in fragment.split("/") if bit]
        if bits:
            return bits[-1]
    return None


def _ref_fragment_name(ref: str) -> str | None:
    if not ref or "#" not in ref:
        return None
    fragment = ref.split("#", 1)[1]
    bits = [bit for bit in fragment.split("/") if bit]
    return bits[-1] if bits else None


def _type_from_schema(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return "reference"
    typ = schema.get("type", "string")
    if isinstance(typ, list):
        non_null = [item for item in typ if item != "null"]
        typ = non_null[0] if non_null else "string"
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
    fmt = schema.get("format")
    if fmt == "date-time":
        return "datetime"
    if fmt == "date":
        return "date"
    return "string"


def _ref_target(ref: str) -> str | None:
    """Resolve a local OpenAPI reference to its model name."""
    return _ref_fragment_name(ref) or _clean_ref_basename(ref)


def _definitions_from_openapi(document: dict[str, Any]) -> dict[str, Any]:
    return document.get("definitions") or document.get("components", {}).get("schemas", {}) or {}


def _merge_all_of(schema: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    merged = dict(schema)
    properties = dict(merged.get("properties") or {})
    required = list(merged.get("required") or [])
    for item in schema.get("allOf", []) or []:
        if not isinstance(item, dict):
            continue
        ref = item.get("$ref")
        if ref:
            target = _ref_fragment_name(ref)
            if target and isinstance(definitions.get(target), dict):
                parent = _merge_all_of(definitions[target], definitions)
                properties = {**parent.get("properties", {}), **properties}
                required = list(dict.fromkeys([*parent.get("required", []), *required]))
        else:
            properties = {**properties, **(item.get("properties") or {})}
            required = list(dict.fromkeys([*required, *(item.get("required") or [])]))
    if properties:
        merged["properties"] = properties
    if required:
        merged["required"] = required
    return merged


def _build_entity(
    schema_name: str,
    schema: dict[str, Any],
    *,
    source_id: str,
    organization: str,
    artifact: str,
    version: str,
    source_url: str,
    source_page: str,
    reference_prefix: str,
    definitions: dict[str, Any] | None = None,
    ref_entity_resolver: Any | None = None,
) -> dict[str, Any]:
    schema = _merge_all_of(schema, definitions or {}) if definitions is not None else schema
    cid = _namespaced(source_id, schema_name)
    required = set(schema.get("required") or [])
    attributes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    properties = schema.get("properties") or {}

    for field_name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        enum_values = prop.get("enum") or []
        dtype = _type_from_schema(prop)
        attr = {
            "name": str(field_name),
            "dtype": dtype,
            "required": field_name in required,
            "nullable": bool(prop.get("nullable", False) or (isinstance(prop.get("type"), list) and "null" in prop.get("type", []))),
            "description": prop.get("description", ""),
            "enum_values": enum_values,
            "params": {
                key: prop[key]
                for key in ("format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems")
                if key in prop
            },
        }
        attributes.append(attr)

        ref = prop.get("$ref")
        if ref:
            target = ref_entity_resolver(ref) if ref_entity_resolver else _ref_target(ref)
            if target:
                relationships.append({
                    "target": _namespaced(source_id, target),
                    "relation": "references",
                    "cardinality": "N:1",
                    "required": field_name in required,
                    "description": f"Reference from {schema_name}.{field_name} to {target}",
                })
        items = prop.get("items") or {}
        if isinstance(items, dict) and items.get("$ref"):
            target = ref_entity_resolver(items["$ref"]) if ref_entity_resolver else _ref_target(items["$ref"])
            if target:
                relationships.append({
                    "target": _namespaced(source_id, target),
                    "relation": "contains",
                    "cardinality": "1:N",
                    "required": field_name in required,
                    "description": f"Collection reference from {schema_name}.{field_name} to {target}",
                })

    return {
        "canonical_id": cid,
        "name": schema_name,
        "aliases": [schema_name, f"{artifact} {schema_name}"],
        "domain": "telecom",
        "description": schema.get("description", ""),
        "sources": [{
            "standard": organization,
            "artifact": artifact,
            "version": version,
            "reference": f"{reference_prefix}::{schema_name}",
            "url": source_url,
            "source_role": "official-machine-readable-model",
            "source_page": source_page,
        }],
        "attributes": attributes,
        "relationships": relationships,
    }


def normalize_openapi_document(
    document: dict[str, Any],
    *,
    source_url: str = "",
    organization: str = "TM Forum",
    artifact: str | None = None,
    version: str | None = None,
    source_page: str | None = None,
    source_id: str = "tmforum_import",
) -> dict[str, Any]:
    """Convert Swagger/OpenAPI model definitions into the internal registry format."""
    if not document.get("swagger") and not document.get("openapi"):
        raise ValueError("Document is not Swagger 2.x or OpenAPI 3.x")
    definitions = _definitions_from_openapi(document)
    info = document.get("info") or {}
    title = artifact or info.get("title") or "Imported Standards Artifact"
    resolved_version = version or info.get("version")
    entities = [
        _build_entity(
            schema_name,
            schema,
            source_id=source_id,
            organization=organization,
            artifact=title,
            version=resolved_version,
            source_url=source_url,
            source_page=source_page or source_url,
            reference_prefix=f"{title}",
            definitions=definitions,
        )
        for schema_name, schema in definitions.items()
        if isinstance(schema, dict) and (schema.get("type") == "object" or schema.get("properties") or schema.get("allOf"))
    ]
    return {
        "artifact": {
            "artifact_id": _snake(source_id),
            "organization": organization,
            "title": title,
            "artifact_version": resolved_version,
            "status": "official-source-sync",
            "source_kind": "official-machine-readable-model",
            "source_url": source_url,
            "source_page": source_page or source_url,
        },
        "entities": entities,
    }


def _schema_document(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("type") == "object" or value.get("properties") or value.get("allOf"):
        return value
    # MEF product schemas may be definition containers.
    definitions = value.get("definitions")
    if isinstance(definitions, dict):
        # handled in the multi-file normalizer; single-file helper returns no root entity here
        return None
    for key in ("schema", "model"):
        if isinstance(value.get(key), dict) and (value[key].get("type") == "object" or value[key].get("properties") or value[key].get("allOf")):
            return value[key]
    return None


def _load_yaml_documents(paths: list[Path]) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to ingest official MEF YAML schemas")
    documents: dict[str, Any] = {}
    for path in paths:
        try:
            documents[path.as_posix()] = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return documents


def _make_yaml_resolver(source_root: Path, documents: dict[str, Any], current_path: Path):
    current_key = current_path.as_posix()
    def resolve_ref(ref: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
        if not ref:
            return None, None, None
        ref_text = str(ref)
        path_part, _, fragment = ref_text.partition("#")
        if path_part:
            target_path = (current_path.parent / path_part).resolve()
            try:
                rel = target_path.relative_to(source_root.resolve()).as_posix()
            except ValueError:
                rel = target_path.as_posix()
            doc = documents.get(rel) or documents.get(str(target_path))
        else:
            doc = documents.get(current_key)
        if not isinstance(doc, dict):
            return None, None, None
        node: Any = doc
        for bit in [x for x in fragment.split("/") if x]:
            bit = bit.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or bit not in node:
                return None, None, None
            node = node[bit]
        target_name = Path(path_part).stem if path_part else None
        if not target_name:
            target_name = _ref_fragment_name(ref_text)
        return node if isinstance(node, dict) else None, target_name, ref_text
    return resolve_ref


def _merge_mef_schema(schema: dict[str, Any], resolve_ref, *, depth: int = 0) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    if depth > 12:
        return dict(schema), []
    merged = dict(schema)
    props = dict(schema.get("properties") or {})
    required = list(schema.get("required") or [])
    relations: list[tuple[str, str]] = []
    for entry in schema.get("allOf", []) or []:
        if not isinstance(entry, dict):
            continue
        ref = entry.get("$ref")
        if ref:
            parent, target_name, _ = resolve_ref(ref)
            if parent is not None:
                expanded, parent_relations = _merge_mef_schema(parent, resolve_ref, depth=depth + 1)
                props = {**expanded.get("properties", {}), **props}
                required = list(dict.fromkeys([*expanded.get("required", []), *required]))
            if target_name:
                relations.append((ref, target_name))
        else:
            props = {**props, **(entry.get("properties") or {})}
            required = list(dict.fromkeys([*required, *(entry.get("required") or [])]))
    merged.pop("allOf", None)
    if props:
        merged["properties"] = props
    if required:
        merged["required"] = required
    return merged, relations


def normalize_json_schema_documents(
    files: list[Path],
    *,
    source_root: Path,
    source_url: str,
    source_page: str,
    organization: str,
    artifact: str,
    version: str,
    source_id: str,
) -> dict[str, Any]:
    """Normalize an entire MEF-style schema set with cross-file refs and inherited fields."""
    documents = _load_yaml_documents(files)
    schema_entries: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for path in files:
        doc = documents.get(path.as_posix())
        if not isinstance(doc, dict):
            continue
        definitions = doc.get("definitions")
        if isinstance(definitions, dict):
            for name, schema in definitions.items():
                if isinstance(schema, dict):
                    schema_entries[f"{path.as_posix()}#{name}"] = (path, str(name), schema)
        else:
            schema = _schema_document(doc)
            if schema is not None:
                logical_name = str(schema.get("title") or path.stem)
                schema_entries[path.as_posix()] = (path, logical_name, schema)

    entity_by_file: dict[str, str] = {}
    entity_by_name: dict[str, str] = {}
    for key, (path, logical_name, _) in schema_entries.items():
        # Prefer filename stem as canonical entity for whole-file schemas; definition names
        # receive their own model entity when they are explicitly present in a definitions map.
        canonical_name = logical_name if "#" in key else path.stem
        entity_by_file[path.resolve().as_posix()] = canonical_name
        entity_by_name.setdefault(logical_name, canonical_name)

    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, (path, logical_name, original_schema) in sorted(schema_entries.items(), key=lambda item: item[0]):
        resolver = _make_yaml_resolver(source_root, documents, path)
        merged, inherited_relations = _merge_mef_schema(original_schema, resolver)
        canonical_name = logical_name if "#" in key else path.stem
        cid = _namespaced(source_id, canonical_name)
        if cid in seen:
            continue
        seen.add(cid)
        required = set(merged.get("required") or [])
        attributes: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        for field_name, prop in (merged.get("properties") or {}).items():
            if not isinstance(prop, dict):
                continue
            enum_values = prop.get("enum") or []
            dtype = _type_from_schema(prop)
            attributes.append({
                "name": str(field_name),
                "dtype": dtype,
                "required": field_name in required,
                "nullable": bool(prop.get("nullable", False) or (isinstance(prop.get("type"), list) and "null" in prop.get("type", []))),
                "description": prop.get("description", ""),
                "enum_values": enum_values,
                "params": {
                    key: prop[key]
                    for key in ("format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems")
                    if key in prop
                },
            })
            ref = prop.get("$ref")
            if ref:
                target = _resolve_schema_target_name(ref, path, source_root, entity_by_file, entity_by_name)
                if target:
                    relationships.append({
                        "target": _namespaced(source_id, target),
                        "relation": "references",
                        "cardinality": "N:1",
                        "required": field_name in required,
                        "description": f"Reference from {logical_name}.{field_name} to {target}",
                    })
            items = prop.get("items") or {}
            if isinstance(items, dict) and items.get("$ref"):
                target = _resolve_schema_target_name(items["$ref"], path, source_root, entity_by_file, entity_by_name)
                if target:
                    relationships.append({
                        "target": _namespaced(source_id, target),
                        "relation": "contains",
                        "cardinality": "1:N",
                        "required": field_name in required,
                        "description": f"Collection reference from {logical_name}.{field_name} to {target}",
                    })
        for ref, target_name in inherited_relations:
            target = _resolve_schema_target_name(ref, path, source_root, entity_by_file, entity_by_name) or target_name
            if target:
                relationships.append({
                    "target": _namespaced(source_id, target),
                    "relation": "inherits",
                    "cardinality": "N:1",
                    "required": False,
                    "description": f"Schema inheritance from {logical_name} to {target}",
                })
        entities.append({
            "canonical_id": cid,
            "name": logical_name,
            "aliases": [logical_name, path.stem, f"{artifact} {logical_name}"],
            "domain": "telecom",
            "description": str(merged.get("description") or ""),
            "sources": [{
                "standard": organization,
                "artifact": artifact,
                "version": version,
                "reference": f"{path.relative_to(source_root).as_posix()}::{logical_name}",
                "url": source_url,
                "source_role": "official-machine-readable-model",
                "source_page": source_page,
            }],
            "attributes": attributes,
            "relationships": relationships,
        })
    return {
        "artifact": {
            "artifact_id": _snake(source_id),
            "organization": organization,
            "title": artifact,
            "artifact_version": version,
            "status": "official-source-sync",
            "source_kind": "official-machine-readable-model",
            "source_url": source_url,
        },
        "entities": entities,
    }


def _resolve_schema_target_name(ref: str, current_path: Path, source_root: Path, by_file: dict[str, str], by_name: dict[str, str]) -> str | None:
    path_part, _, fragment = str(ref).partition("#")
    fragment_name = _ref_fragment_name(str(ref))
    # When a definitions container is referenced with #/definitions/X, the named
    # definition is the strongest identity and must win over the source file's root.
    if fragment_name and fragment_name in by_name:
        return by_name[fragment_name]
    if path_part:
        target_path = (current_path.parent / path_part).resolve()
        target = by_file.get(target_path.as_posix())
        if target:
            return target
    cleaned = _clean_ref_basename(str(ref))
    return cleaned


def _asn1_field_type(type_expr: str) -> str:
    lowered = type_expr.strip().lower()
    if lowered.startswith("integer") or lowered in {"integer32", "integer64"}:
        return "integer"
    if "real" in lowered or "double" in lowered:
        return "float"
    if lowered in {"boolean", "bool"}:
        return "boolean"
    if "generalizedtime" in lowered or "utctime" in lowered or "timestamp" in lowered:
        return "datetime"
    if "octet string" in lowered or "ia5string" in lowered or "utf8string" in lowered or "printablestring" in lowered:
        return "string"
    return "reference"


def normalize_asn1_documents(
    files: list[Path],
    *,
    source_url: str,
    source_page: str,
    organization: str,
    artifact: str,
    version: str,
    source_id: str,
) -> dict[str, Any]:
    """Index the composite ASN.1 model in a pinned 3GPP package."""
    combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in files)
    entity_pattern = re.compile(
        r"(?ms)^\s*([A-Za-z][A-Za-z0-9-]*)\s*::=\s*(SEQUENCE|SET|CHOICE)\s*\{(.*?)^\s*\}\s*;?"
    )
    enum_pattern = re.compile(
        r"(?ms)^\s*([A-Za-z][A-Za-z0-9-]*)\s*::=\s*ENUMERATED\s*\{(.*?)\}\s*;?"
    )
    # 3GPP ASN.1 record fields commonly look like either
    # ``[7] recordOpeningTime TimeStamp`` or, in the released 32.298 sources,
    # ``recordOpeningTime [7] TimeStamp``. Accept tags on either side of the
    # official field name so the runtime index does not silently lose fields.
    field_pattern = re.compile(
        r"(?m)^\s*(?:\[[^\]]+\]\s*)*"
        r"([A-Za-z][A-Za-z0-9-]*)\s*"
        r"(?:\[[^\]]+\]\s*)*"
        r"([^,\n]+?)"
        r"(?:\s+OPTIONAL|\s+DEFAULT\s+[^,\n]+)?\s*(?:,|$)"
    )
    entity_names = {m.group(1) for m in entity_pattern.finditer(combined)}
    named_enums: dict[str, list[str]] = {}
    for enum_match in enum_pattern.finditer(combined):
        values = [item.strip().split("(")[0].strip() for item in enum_match.group(2).split(",") if item.strip()]
        if values:
            named_enums[enum_match.group(1)] = values
    entities: list[dict[str, Any]] = []
    for match in entity_pattern.finditer(combined):
        type_name, kind, body = match.group(1), match.group(2), match.group(3)
        cid = _namespaced(source_id, type_name)
        attrs: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        for field_match in field_pattern.finditer(body):
            field_name, type_expr = field_match.group(1), field_match.group(2).strip()
            context = body[field_match.start():field_match.end()]
            optional = bool(re.search(r"\bOPTIONAL\b", context, re.I))
            cleaned_type = re.sub(r"\b(OPTIONAL|DEFAULT\s+[^,]+)$", "", type_expr, flags=re.I).strip()
            dtype = _asn1_field_type(cleaned_type)
            enum_values: list[str] = []
            named_type = re.match(r"([A-Za-z][A-Za-z0-9-]*)", cleaned_type)
            if named_type and named_type.group(1) in named_enums:
                enum_values = list(named_enums[named_type.group(1)])
                dtype = "string"
            inline = re.search(r"ENUMERATED\s*\{([^}]*)\}", cleaned_type, re.I | re.S)
            if inline:
                enum_values = [item.strip().split("(")[0].strip() for item in inline.group(1).split(",") if item.strip()]
                dtype = "string"
            attrs.append({
                "name": field_name,
                "dtype": dtype,
                "required": not optional,
                "nullable": optional,
                "description": f"3GPP ASN.1 {kind} field {field_name} of {type_name}.",
                "enum_values": enum_values,
                "params": {},
            })
            target_match = re.match(r"([A-Za-z][A-Za-z0-9-]*)", cleaned_type)
            target = target_match.group(1) if target_match else None
            if target in entity_names and target != type_name:
                relationships.append({
                    "target": _namespaced(source_id, target),
                    "relation": "references",
                    "cardinality": "N:1",
                    "required": not optional,
                    "description": f"ASN.1 reference from {type_name}.{field_name} to {target}.",
                })
        if attrs:
            entities.append({
                "canonical_id": cid,
                "name": type_name,
                "aliases": [type_name, f"3GPP {type_name}"],
                "domain": "telecom",
                "description": f"3GPP ASN.1 {kind} data structure.",
                "sources": [{
                    "standard": organization,
                    "artifact": artifact,
                    "version": version,
                    "reference": f"ASN.1::{type_name}",
                    "url": source_url,
                    "source_role": "official-3gpp-asn1-model",
                    "source_page": source_page,
                }],
                "attributes": attrs,
                "relationships": relationships,
            })
    return {
        "artifact": {
            "artifact_id": _snake(source_id),
            "organization": organization,
            "title": artifact,
            "artifact_version": version,
            "status": "official-source-sync",
            "source_kind": "official-asn1-model",
            "normalizer_version": ASN1_NORMALIZER_VERSION,
            "source_url": source_url,
            "source_page": source_page,
        },
        "entities": entities,
    }


def normalize_official_file(
    paths: Iterable[Path],
    output_dir: Path,
    *,
    organization: str,
    artifact: str,
    version: str,
    source_url: str,
    source_page: str,
    source_id: str,
    parser: str = "auto",
) -> list[Path]:
    """Normalize one pinned official source into one runtime cache document."""
    files = sorted(set(Path(p) for p in paths if Path(p).is_file()), key=lambda p: p.as_posix())
    output_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        return []

    if parser in {"asn1", "asn1_zip"} or any(p.suffix.lower() in {".asn", ".asn1", ".asn1p"} for p in files):
        doc = normalize_asn1_documents(
            files,
            source_url=source_url,
            source_page=source_page,
            organization=organization,
            artifact=artifact,
            version=version,
            source_id=source_id,
        )
    elif parser in {"json_schema", "mef_schema"}:
        import os
        # Use the common parent directory so relative $refs resolve against the
        # extracted MEF source tree rather than against an individual YAML file.
        common_root = Path(os.path.commonpath([str(p.parent) for p in files]))
        doc = normalize_json_schema_documents(
            files,
            source_root=common_root,
            source_url=source_url,
            source_page=source_page,
            organization=organization,
            artifact=artifact,
            version=version,
            source_id=source_id,
        )
    else:
        # TM Forum single OpenAPI JSON source.
        candidate = files[0]
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Unable to parse official JSON source {candidate}: {exc}") from exc
        doc = normalize_openapi_document(
            value,
            source_url=source_url,
            organization=organization,
            artifact=artifact,
            version=version,
            source_page=source_page,
            source_id=source_id,
        )

    if not doc.get("entities"):
        raise ValueError(f"No model entities could be extracted from official source {source_id}")
    destination = output_dir / f"{_snake(source_id)}.json"
    destination.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return [destination]


def normalize_openapi_file(input_path: str | Path, output_path: str | Path, source_url: str = "", organization: str = "TM Forum") -> Path:
    """Backward-compatible single-file OpenAPI ingestion helper."""
    input_path = Path(input_path)
    normalized = normalize_openapi_document(
        json.loads(input_path.read_text(encoding="utf-8")),
        source_url=source_url,
        organization=organization,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return output_path

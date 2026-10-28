"""API behaviour used by the FastAPI router (``er.api``), which only forwards to these plain functions.

Every handler takes plain data and returns ``(status, payload)``; the export handler returns ``(status, bytes, filename)``.
Request body for validate, generate and export::

    {"model": {...ER model...}, "seed": 42 (optional), "counts": {"customer": 100} (optional, root entities only)}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from er import export
from er.engine import generate
from er.generators import GenerationError
from er.integrity import check
from er.model import SEED_MAX, Model, check_budget, check_counts, describe, parse

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


# Failures use the application's canonical error envelope (core/errors.py: {success:false, error:{code,message,details,request_id}}),
# so the front end's one error reader (getApiErrorMessage) shows them like any other error. The flat "errors" and "warnings" lists
# stay beside it for simple clients (the front end's ER screen reads them for a rejected generate).
_CODES = {400: "BAD_REQUEST", 404: "NOT_FOUND", 411: "BAD_REQUEST", 413: "BAD_REQUEST", 422: "VALIDATION_ERROR", 500: "INTERNAL_SERVER_ERROR"}
_PATH_PREFIX = re.compile(r"^([A-Za-z_][\w\[\]\.]*): (.+)$", re.S)


def failure(status: int, errors: list[str], warnings: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    """The error response for ``status``. ``request_id`` is empty here; the HTTP layer fills in the request's own id."""
    details = []
    for text in errors:
        m = _PATH_PREFIX.match(text)                      # "entities[1].columns[2].gen.type: unknown generator" -> field + message
        details.append({"field": m.group(1) if m else None, "message": m.group(2) if m else text})
    if status == 422:
        message = "The ER model is not valid" if len(errors) == 1 else f"The ER model has {len(errors)} problems"
    else:
        message = errors[0] if errors else "Request failed"
    return status, {
        "success": False,
        "error": {"code": _CODES.get(status, "HTTP_ERROR"), "message": message, "details": details if status == 422 else None, "request_id": ""},
        "errors": errors,
        "warnings": warnings or [],
    }


SAMPLE_ORDER = ("telecom_family", "banking_holders", "retail_snowflake")         # shown first, in this order; any other sample follows alphabetically


def _samples() -> dict[str, dict[str, Any]]:
    paths = sorted(SAMPLES_DIR.glob("*.json"), key=lambda p: (SAMPLE_ORDER.index(p.stem) if p.stem in SAMPLE_ORDER else len(SAMPLE_ORDER), p.stem))
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}


def samples() -> tuple[int, dict[str, Any]]:
    return 200, {"success": True, "samples": [
        {"id": sid, "name": m.get("name", sid), "description": m.get("description", ""),
         "entities": len(m.get("entities", [])), "relationships": len(m.get("relationships", []))}
        for sid, m in _samples().items()]}


def sample(sample_id: str) -> tuple[int, dict[str, Any]]:
    found = _samples().get(sample_id)
    if found is None:
        return failure(404, [f"no sample named '{sample_id}'"])
    return 200, {"success": True, "id": sample_id, "model": found}


def _prepare(body: Any) -> tuple[Model | None, list[str], list[str], dict[str, int] | None, int | None]:
    if not isinstance(body, dict) or "model" not in body:
        return None, ['body: send {"model": {...}} with the ER model inside "model"'], [], None, None
    errors: list[str] = []
    counts, seed = body.get("counts"), body.get("seed")
    if counts is not None and not isinstance(counts, dict):
        errors.append("counts: must be an object of entity name -> number")
        counts = None
    if seed is not None and not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= SEED_MAX):
        errors.append(f"seed: must be a whole number from 0 to {SEED_MAX}")
        seed = None
    model, parse_errors, warnings = parse(body["model"])
    errors += parse_errors
    if model is not None and not errors:
        errors += check_counts(model, counts) or check_budget(model, counts)
    return (model if not errors else None), errors, warnings, counts, seed


def validate(body: Any) -> tuple[int, dict[str, Any]]:
    model, errors, warnings, counts, _ = _prepare(body)
    if model is None:
        return 200, {"success": True, "valid": False, "errors": errors, "warnings": warnings, "plan": None}
    return 200, {"success": True, "valid": True, "errors": [], "warnings": warnings, "plan": describe(model, counts)}


def _table_view(model: Model, name: str, table: dict[str, Any]) -> dict[str, Any]:
    entity = model.entities[name]
    return {
        "fields": table["fields"],
        "records": table["records"],
        "total_records": len(table["records"]),
        "primary_key": entity.key,
        "foreign_keys": [{"column": f.column, "references": f"{f.parent}.{f.parent_key}", "relationship": f.relationship} for f in entity.fks],
    }


def _run(body: Any):
    model, errors, warnings, counts, seed = _prepare(body)
    if model is None:
        return None, None, None, warnings, errors
    try:
        result = generate(model, seed=seed, counts=counts)
    except GenerationError as exc:
        return None, None, None, warnings, [str(exc)]
    return model, result, check(model, result["tables"]), warnings, []


def generate_tables(body: Any) -> tuple[int, dict[str, Any]]:
    model, result, report, warnings, errors = _run(body)
    if model is None:
        return failure(422, errors, warnings)
    tables = {name: _table_view(model, name, result["tables"][name]) for name in result["order"]}
    return 200, {
        "success": True,
        "model_name": model.name,
        "seed": result["seed"],
        "order": result["order"],
        "total_records": sum(t["total_records"] for t in tables.values()),
        "elapsed_ms": result["elapsed_ms"],
        "warnings": warnings,
        "tables": tables,
        "integrity_report": report,
    }


def export_zip(body: Any) -> tuple[int, Any, str]:
    model, result, report, warnings, errors = _run(body)
    if model is None:
        return (*failure(422, errors, warnings), "")
    data = export.zip_bytes(model.raw, result["tables"], report)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model.name)[:40].strip("_") or "er_model"
    return 200, data, f"{safe}_seed{result['seed']}.zip"

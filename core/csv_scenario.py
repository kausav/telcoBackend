"""CSV scenario-definition parser.

The CSV is the source of truth for the schema.  A row is treated as an
entity/aggregational field when ``fields`` is empty.  A row is treated as an
event-owned field when ``fields`` contains one or more field names.  This makes
``record_type`` optional metadata rather than the mechanism used to distinguish
transactional from aggregational input.

Data type and generator names used by client CSVs are normalized to the
backend's internal generator contract.  The parser accepts both JSON params
(the legacy format) and the compact ``key=value;key=value`` format used by the
client sample.
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from agents.data_generation_agent import get_known_generator_types

ALLOWED_DTYPES = {
    "string", "int", "integer", "float", "decimal", "categorical",
    "datetime", "date", "bool", "boolean", "uuid",
}
_TRUE_STRINGS = {"true", "1", "yes", "y"}


def _split_list(value: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;|,]", value or "") if x.strip()]


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text or text.upper() == "NULL":
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", text):
            return float(text)
    except ValueError:
        pass
    return text


def _parse_params(raw: str, row_number: int, name: str) -> dict[str, Any]:
    """Parse legacy JSON or client's compact semicolon-separated parameters."""
    text = (raw or "").strip()
    if not text or text.upper() == "NULL":
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Row {row_number} ('{name}'): 'params' must be valid JSON or key=value pairs: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Row {row_number} ('{name}'): 'params' must be a JSON object")
        return parsed

    params: dict[str, Any] = {}
    # Client params use semicolons both as key/value separators and inside
    # unquoted choice lists (for example ``A;B;C``).  Split only at a semicolon
    # that starts another ``key=`` pair, so choice values are preserved.
    matches = list(re.finditer(r"(?:^|;)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", text))
    if matches:
        prefix = text[:matches[0].start()].strip(" ;")
        if prefix:
            params["value"] = _parse_scalar(prefix)
        for index, match in enumerate(matches):
            key = match.group(1).strip()
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[value_start:value_end].strip(" ;")
            params[key] = _parse_scalar(value)
    else:
        # Bare params are commonly used for constant/choice generators.
        params["value"] = _parse_scalar(text)

    # Normalize common compact encodings.
    for key in ("choices", "values", "country_codes"):
        if isinstance(params.get(key), str):
            params[key] = _split_list(str(params[key]))
    if isinstance(params.get("weights"), str):
        params["weights"] = [float(x) for x in _split_list(str(params["weights"]))]
    return params


def _range_from_text(value: Any) -> tuple[int | float, int | float] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*", value)
    if not match:
        return None
    a, b = match.groups()
    def number(s: str):
        return float(s) if "." in s else int(s)
    return number(a), number(b)


def _normalize_dtype(dtype: str) -> str:
    value = dtype.strip().lower()
    if value.startswith("decimal"):
        return "float"
    if value == "integer":
        return "int"
    if value == "bool":
        return "boolean"
    if value == "uuid":
        return "string"
    return value


def _normalize_generator(gen: str, params: dict[str, Any], depends_on: list[str], dtype: str, formula: str) -> tuple[str, dict[str, Any]]:
    """Map client-facing generator vocabulary into executable generators."""
    g = (gen or "").strip().lower()
    p = dict(params)

    if g == "unique_id":
        if p.get("prefix") is not None and p.get("digits") is not None:
            return "prefixed_int", p
        if p.get("prefix") is not None:
            return "prefixed_uuid", p
        return "prefixed_uuid", {"prefix": ""}

    if g == "indian_msisdn":
        cc = p.get("country_code", 91)
        cc_text = str(cc)
        if not cc_text.startswith("+"):
            cc_text = "+" + cc_text
        return "e164_phone", {"country_codes": [cc_text]}

    if g in {"uuid", "prefixed_uuid"}:
        return "prefixed_uuid", {"prefix": str(p.get("prefix", ""))}

    if g in {"choice", "weighted_choice"}:
        choices = p.get("choices")
        if choices is None and "value" in p:
            choices = _split_list(str(p.get("value", "")))
        if isinstance(choices, str):
            choices = _split_list(choices)
        elif choices is None:
            choices = []
        elif not isinstance(choices, list):
            choices = [choices]
        p["choices"] = choices
        return "weighted_choice", p

    if g == "range":
        return "uniform" if dtype == "float" else "uniform", p

    if g == "dependent_range":
        # dependent_range(min=0;max=controller) -> uniform_bounded.
        max_value = p.get("max")
        if isinstance(max_value, str) and max_value:
            p["hi_field"] = max_value
        return "uniform_bounded", p

    if g == "derived_distribution":
        # This is a stochastic distribution, not an authoritative formula.
        # The formula column is retained separately for documentation/rules.
        return "uniform", p

    if g in {"timestamp", "recent_timestamp"}:
        return "recent_datetime", p

    if g == "derived_timestamp":
        # Client convention: delay_seconds=5-120 and first dependency is the base timestamp.
        rng = _range_from_text(p.get("delay_seconds"))
        if rng:
            p["min_sec"], p["max_sec"] = int(rng[0]), int(rng[1])
        else:
            p.setdefault("min_sec", int(p.get("min_sec", 1)))
            p.setdefault("max_sec", int(p.get("max_sec", p["min_sec"])))
        if depends_on:
            p["base_field"] = depends_on[0]
        return "ts_offset", p

    if g == "derived":
        # A deterministic formula is executable when supplied.  Without a formula,
        # fall back to a bounded numeric distribution.
        if formula:
            return "formula", p
        return "uniform", p

    # Existing internal generators remain unchanged.
    return g, p


def _infer_transactional(rows: list[dict[str, str]]) -> bool:
    """Transactional iff at least one row has non-empty ``fields`` data."""
    return any(bool(_split_list(row.get("fields", ""))) for row in rows)


def infer_type_of_data(csv_text: str) -> str:
    """Infer output type solely from whether the CSV contains event-owned fields."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty (no header row found)")
    return "transactional" if _infer_transactional([
        {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        for row in reader
    ]) else "aggregational"


def parse_definition_csv(csv_text: str, type_of_data: str | None = None) -> tuple[list[dict], list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty (no header row found)")

    headers = [str(h).strip().lower() for h in reader.fieldnames if h is not None]
    if len(headers) != len(set(headers)):
        raise ValueError("CSV contains duplicate column names")
    required = {"name", "dtype", "gen"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

    raw_rows = []
    for raw in reader:
        normalized = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not any(normalized.values()):
            continue
        raw_rows.append(normalized)
    inferred_type = "transactional" if _infer_transactional(raw_rows) else "aggregational"
    normalized_requested = str(type_of_data or "").strip().lower()
    if normalized_requested and normalized_requested not in {"transactional", "aggregational"}:
        raise ValueError("typeOfData must be 'transactional' or 'aggregational'")
    # CSV structure is authoritative. If a caller sends the old form field, reject
    # mismatches rather than silently interpreting a transactional CSV as aggregate.
    if normalized_requested and normalized_requested != inferred_type:
        raise ValueError(
            f"typeOfData '{normalized_requested}' does not match the CSV. "
            f"The CSV is detected as '{inferred_type}' because fields data is "
            f"{'present' if inferred_type == 'transactional' else 'absent'} in event rows."
        )
    detected_type = inferred_type

    known_gens = get_known_generator_types() | {
        "unique_id", "indian_msisdn", "uuid", "timestamp", "recent_timestamp",
        "choice", "range", "dependent_range", "derived_distribution", "derived_timestamp", "derived",
    }
    variables: list[dict] = []
    field_order: list[str] = []
    event_groups: dict[tuple[str, int], dict[str, Any]] = {}
    seen_names: set[str] = set()
    all_names: set[str] = set()

    # First pass: validate/normalize variables.  Dependencies are allowed to point
    # forward because client schemas may use formulas whose source field appears later.
    for row_number, row in enumerate(raw_rows, start=2):
        name = row.get("name", "").strip()
        if not name:
            raise ValueError(f"Row {row_number}: 'name' is empty")
        if name in seen_names:
            raise ValueError(f"Row {row_number}: duplicate variable name '{name}'")
        seen_names.add(name)
        all_names.add(name)

        dtype_raw = row.get("dtype", "").strip().lower()
        if dtype_raw not in ALLOWED_DTYPES and not dtype_raw.startswith("decimal"):
            raise ValueError(f"Row {row_number} ('{name}'): invalid dtype '{dtype_raw}'")
        dtype = _normalize_dtype(dtype_raw)

        depends_on = _split_list(row.get("depends_on", ""))
        params = _parse_params(row.get("params", ""), row_number, name)
        formula = row.get("formula", "").strip()
        if formula.upper() == "NULL":
            formula = ""

        fields = _split_list(row.get("fields", ""))
        is_event_field = bool(fields)
        if detected_type == "aggregational" and is_event_field:
            raise ValueError(f"Row {row_number} ('{name}'): fields data is not allowed in an aggregational CSV")

        gen_raw = row.get("gen", "").strip().lower()
        if gen_raw not in known_gens:
            raise ValueError(f"Row {row_number} ('{name}'): unknown gen type '{gen_raw}'")
        gen, params = _normalize_generator(gen_raw, params, depends_on, dtype, formula)
        if gen not in get_known_generator_types():
            raise ValueError(f"Row {row_number} ('{name}'): generator '{gen}' is not executable")

        variable = {
            "name": name,
            "dtype": dtype,
            "description": row.get("description", ""),
            "gen": gen,
            "params": params,
            "depends_on": depends_on,
            "nullable": row.get("nullable", "").lower() in _TRUE_STRINGS,
        }
        if formula:
            variable["formula"] = formula
        variables.append(variable)
        field_order.append(name)

        if is_event_field:
            event_type = (row.get("event_type") or "").strip().upper().replace(" ", "_")
            if not event_type:
                raise ValueError(f"Row {row_number} ('{name}'): event row requires 'event_type'")
            try:
                sequence = int(row.get("sequence", "") or len(event_groups) + 1)
            except ValueError as exc:
                raise ValueError(f"Row {row_number} ('{name}'): sequence must be an integer") from exc
            if sequence < 1:
                raise ValueError(f"Row {row_number} ('{name}'): sequence must be >= 1")
            key = (event_type, sequence)
            event = event_groups.setdefault(key, {
                "event_type": event_type,
                "sequence": sequence,
                "fields": [],
                "description": row.get("description", ""),
                "min_occurrences": row.get("min_occurrences", "1") or "1",
                "max_occurrences": row.get("max_occurrences", "10") or "10",
            })
            # Client sample repeats the complete event field list on every event row.
            # Preserve declared order, and ensure the current variable is included.
            for field in fields + [name]:
                if field not in event["fields"]:
                    event["fields"].append(field)

    if not variables:
        raise ValueError("CSV must contain at least one variable row")

    unknown_dependencies = sorted({dep for v in variables for dep in v["depends_on"] if dep not in all_names})
    if unknown_dependencies:
        raise ValueError(f"depends_on references undefined variable(s): {unknown_dependencies}")

    events: list[dict] = []
    if detected_type == "transactional":
        for event in sorted(event_groups.values(), key=lambda e: (e["sequence"], e["event_type"])):
            missing = [f for f in event["fields"] if f not in all_names]
            if missing:
                raise ValueError(f"Event '{event['event_type']}' references undefined variable(s): {missing}")
            try:
                min_occ = max(1, int(event["min_occurrences"]))
                max_occ = max(min_occ, min(1000, int(event["max_occurrences"])))
            except ValueError as exc:
                raise ValueError(f"Event '{event['event_type']}': min_occurrences/max_occurrences must be integers") from exc
            events.append({
                "event_type": event["event_type"],
                "sequence": event["sequence"],
                "fields": event["fields"],
                "description": event["description"],
                "min_occurrences": min_occ,
                "max_occurrences": max_occ,
            })
        if not events:
            raise ValueError("Transactional CSV must contain at least one row with non-empty fields data")

    return variables, field_order, events

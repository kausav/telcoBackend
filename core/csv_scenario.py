"""
Import a variable catalog directly from an industry-supplied CSV instead of
having the Scenario Designer Agent (LLM) invent one. Useful when the business/
industry team already has their own spec (e.g. an Excel/CSV export of their
data dictionary) and wants that used verbatim for generation.

Expected CSV columns (header row required):
  name         (required) — field name, e.g. "subscriber_id"
  dtype        (required) — one of: string, int, float, categorical, datetime, bool
  gen          (required) — a generator type known to agents/generator_agent.py
                             (see get_known_generator_types()), e.g. "uniform"
  params       (optional) — JSON object string, e.g. {"min": 0, "max": 100}
                             (default: {})
  description  (optional) — plain-English description of the field
  depends_on   (optional) — semicolon-separated list of field names this one
                             depends on, e.g. "subscriber_id;event_timestamp"
  nullable     (optional) — true/false (default: false)
  formula      (optional) — required instead of params when gen == "formula"

Rows are treated as already being in the correct dependency order (same
contract as the LLM-produced field_order) — the CSV's row order becomes the
generation order, so a field must appear AFTER anything it depends on.
"""
from __future__ import annotations
import csv
import io
import json

from agents.generator_agent import get_known_generator_types

REQUIRED_COLUMNS = {"name", "dtype", "gen"}
ALLOWED_DTYPES = {"string", "int", "float", "categorical", "datetime", "bool", "boolean"}
_TRUE_STRINGS = {"true", "1", "yes", "y"}


def parse_variables_csv(csv_text: str) -> tuple[list[dict], list[str]]:
    """Parse and validate an industry-supplied CSV into (variables, field_order).
    Raises ValueError with a clear, row-numbered message on any problem —
    callers should surface that directly to the API caller."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("CSV appears to be empty (no header row found)")

    header = {h.strip().lower() for h in reader.fieldnames}
    missing = REQUIRED_COLUMNS - header
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

    known_gens = get_known_generator_types()
    variables: list[dict] = []
    seen_names: set[str] = set()

    for i, raw_row in enumerate(reader, start=2):  # row 1 is the header
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw_row.items()}

        name = row.get("name", "")
        if not name:
            raise ValueError(f"Row {i}: 'name' column is empty")
        if name in seen_names:
            raise ValueError(f"Row {i}: duplicate variable name '{name}'")
        seen_names.add(name)

        dtype = row.get("dtype", "").lower()
        if dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"Row {i} ('{name}'): invalid dtype '{dtype}'. Must be one of {sorted(ALLOWED_DTYPES)}"
            )

        gen = row.get("gen", "")
        if gen not in known_gens:
            raise ValueError(
                f"Row {i} ('{name}'): unknown gen type '{gen}'. Known types: {sorted(known_gens)}"
            )

        depends_on = [d.strip() for d in row.get("depends_on", "").split(";") if d.strip()]
        for dep in depends_on:
            if dep not in seen_names:
                raise ValueError(
                    f"Row {i} ('{name}'): depends_on references '{dep}', which must be "
                    f"defined in an EARLIER row (CSV row order = generation order)"
                )

        params_raw = row.get("params", "").strip()
        if params_raw:
            try:
                params = json.loads(params_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Row {i} ('{name}'): 'params' is not valid JSON: {exc}") from exc
        else:
            params = {}

        variable: dict = {
            "name": name,
            "dtype": "boolean" if dtype == "bool" else dtype,
            "gen": gen,
            "params": params,
            "description": row.get("description", ""),
            "depends_on": depends_on,
            "nullable": row.get("nullable", "").lower() in _TRUE_STRINGS,
        }
        if gen == "formula":
            formula = row.get("formula", "").strip()
            if not formula:
                raise ValueError(f"Row {i} ('{name}'): gen=='formula' requires a 'formula' column value")
            variable["formula"] = formula

        variables.append(variable)

    if not variables:
        raise ValueError("CSV contained a header but no data rows")

    field_order = [v["name"] for v in variables]
    return variables, field_order
# ---------------------------------------------------------------------------
# Sample-data CSV import
# ---------------------------------------------------------------------------

def _infer_dtype(values: list[str]) -> str:
    vals = [v.strip() for v in values if v.strip()]
    if not vals:
        return "string"
    low = [v.lower() for v in vals]
    if all(v in {"true", "false", "1", "0", "yes", "no"} for v in low):
        return "boolean"
    try:
        for v in vals:
            int(v)
        return "int"
    except ValueError:
        pass
    try:
        for v in vals:
            float(v)
        return "float"
    except ValueError:
        pass
    # Conservative ISO datetime detection.
    from datetime import datetime
    try:
        for v in vals[:20]:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        return "datetime"
    except ValueError:
        return "string"


def _sample_generator(dtype: str, values: list[str], field_name: str = "") -> tuple[str, dict]:
    vals = [v.strip() for v in values if v.strip()]
    # ID-like fields should not be generated as a finite weighted choice: the
    # transactional generator needs fresh entity identifiers. Preserve a useful
    # prefix/digit shape when possible.
    if field_name.lower().endswith("_id") or field_name.lower() in {"id", "customer", "subscriber", "account"}:
        import re
        if vals:
            m = re.match(r"^(.*?)(\d+)$", vals[0])
            if m:
                return "prefixed_int", {"prefix": m.group(1), "digits": len(m.group(2))}
        return "prefixed_uuid", {"prefix": (field_name.upper().replace("_", "-") + "-")}
    if dtype == "int":
        nums = [int(v) for v in vals]
        return "uniform", {"min": min(nums), "max": max(nums), "integer": True}
    if dtype == "float":
        nums = [float(v) for v in vals]
        return "uniform", {"min": min(nums), "max": max(nums)}
    if dtype == "boolean":
        choices = list(dict.fromkeys(vals)) or ["true", "false"]
        return "weighted_choice", {"choices": choices, "weights": [1] * len(choices)}
    if dtype == "datetime":
        return "recent_datetime", {"days": 30}
    choices = list(dict.fromkeys(vals))
    if choices and len(choices) <= 50:
        return "weighted_choice", {"choices": choices, "weights": [1] * len(choices)}
    return "constant", {"value": choices[0] if choices else ""}


def _infer_entity_key(fieldnames: list[str], rows: list[dict[str, str]], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"entityKey '{explicit}' is not present in the CSV header")
        return explicit
    candidates = []
    for name in fieldnames:
        lname = name.lower()
        if name.lower() in {"event_type", "event_sequence", "event_timestamp"}:
            continue
        vals = [r.get(name, "") for r in rows if r.get(name, "")]
        if vals and len(set(vals)) < len(vals):
            # A repeated field is unlikely to be the entity key in transactional data.
            continue
        score = 0
        if lname.endswith("_id") or lname in {"id", "customer", "subscriber", "account"}:
            score += 10
        if "subscriber" in lname or "customer" in lname or "account" in lname:
            score += 5
        if vals and len(set(vals)) == len(vals):
            score += 3
        candidates.append((score, name))
    if not candidates:
        raise ValueError("Unable to infer entityKey from transactional CSV; supply entityKey explicitly")
    return max(candidates, key=lambda x: x[0])[1]


def parse_sample_csv(csv_text: str, type_of_data: str, entity_key: str | None = None) -> tuple[list[dict], list[str], list[dict], str | None]:
    """Infer a proposal from ordinary sample-data CSV rows.

    Returns (variables, field_order, events, inferred_entity_key).
    This is deliberately deterministic: no LLM is needed for CSV imports.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty (no header row found)")
    fieldnames = [str(h).strip() for h in reader.fieldnames if h is not None and str(h).strip()]
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("CSV contains duplicate column names")
    rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]
    if not rows:
        raise ValueError("CSV contains no data rows")

    transactional = type_of_data == "transactional"
    if transactional and "event_type" not in fieldnames:
        raise ValueError("Transactional CSV must contain an 'event_type' column")

    inferred_key = _infer_entity_key(fieldnames, rows, entity_key) if transactional else None
    variables: list[dict] = []
    field_order: list[str] = []

    for name in fieldnames:
        vals = [r.get(name, "") for r in rows]
        dtype = _infer_dtype(vals)
        gen, params = _sample_generator(dtype, vals, name)
        # System transaction metadata is generated by the backend and should not be
        # treated as ordinary scenario variables.
        if transactional and name in {"event_type", "event_sequence", "event_timestamp", "transaction_id", "journey_id", "event_occurrence"}:
            continue
        variables.append({
            "name": name,
            "dtype": dtype,
            "description": f"Imported from CSV column '{name}'",
            "gen": gen,
            "params": params,
            "depends_on": [],
            "nullable": any(not r.get(name, "") for r in rows),
        })
        field_order.append(name)

    events: list[dict] = []
    if transactional:
        grouped: dict[str, dict] = {}
        for row in rows:
            et = str(row.get("event_type", "")).strip().upper().replace(" ", "_")
            if not et:
                continue
            if et not in grouped:
                grouped[et] = {"event_type": et, "fields": [], "sequence": None, "min_occurrences": 1, "max_occurrences": 10}
            for name in fieldnames:
                if name in {"event_type", "event_sequence", "event_timestamp", inferred_key}:
                    continue
                if row.get(name, "") and name not in grouped[et]["fields"]:
                    grouped[et]["fields"].append(name)
            seq = row.get("event_sequence", "")
            if seq:
                try:
                    n = int(seq)
                    grouped[et]["sequence"] = n if grouped[et]["sequence"] is None else min(grouped[et]["sequence"], n)
                except ValueError:
                    pass
        ordered = sorted(grouped.values(), key=lambda e: (e["sequence"] is None, e["sequence"] or 0, e["event_type"]))
        for i, event in enumerate(ordered, start=1):
            event["sequence"] = i
            events.append(event)
        if not events:
            raise ValueError("Transactional CSV contains no usable event_type values")

    return variables, field_order, events, inferred_key

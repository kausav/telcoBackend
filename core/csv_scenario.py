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
"""
Parse a scenario-definition CSV for /importCsv instead of asking the Scenario
Designer Agent (LLM) to invent variables/events. The uploaded CSV contains
only definitions of the data the user wants generated; it is never treated as
sample/output records.

Variable rows use these columns (header row required):
  name         (required) — field name, e.g. "subscriber_id"
  dtype        (required) — one of: string, int, float, categorical, datetime, bool
  gen          (required) — a generator type known to agents/generator_agent.py
                             (see get_known_generator_types()), e.g. "uniform"
  params       (optional) — JSON object string, e.g. {"min": 0, "max": 100}
                             (default: {})
  description  (optional) — plain-English description of the field
  depends_on   (optional) — separated list of field names this one
                             depends on, e.g. "subscriber_id;event_timestamp"
  nullable     (optional) — true/false (default: false)
  formula      (optional) — required instead of params when gen == "formula"

Rows are treated as already being in the correct dependency order (same
contract as the LLM-produced field_order) — the CSV's row order becomes the
generation order, so a field must appear AFTER anything it depends on.
"""
from __future__ import annotations
import csv
import ast
import io
import json
import re

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

        depends_on = [d.strip() for d in re.split(r"[;|,]", row.get("depends_on", "")) if d.strip()]
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

def parse_definition_csv(
    csv_text: str,
    type_of_data: str,
) -> tuple[list[dict], list[str], list[dict], list[dict], dict[str, object]]:
    """Parse a scenario-definition CSV for /importCsv.

    IMPORTANT: this parser does *not* read sample/output records. Every row is a
    definition telling the system what the user wants in the eventual synthetic
    dataset.

    Supported rows:
      record_type=variable — normal variable definition
      record_type=event — transactional event definition
      record_type=edge_case_variable — edge-case override/condition for a normal variable
      record_type=metadata — optional scenario metadata such as edge_case_percentage

    Variable rows use the same generator contract as /scenario/propose:
      name,dtype,gen,params,description,depends_on,nullable,formula

    Transactional event rows use:
      record_type,event_type,sequence,fields,min_occurrences,max_occurrences

    Edge-case rows use the normal variable columns plus:
      edge_case_name,edge_case_description,condition
    They MUST reference a variable already defined by a ``variable`` row. Multiple
    edge-case rows may share the same edge_case_name.

    Metadata rows may set:
      edge_case_percentage — finite float in [0, 1]; 0.02 means 2%.

    ``fields`` is a semicolon/pipe/comma-separated list of variable names owned by
    the event. Event fields and edge-case variable names are validated against the
    variables declared in the same CSV. Aggregational CSVs may contain variable,
    edge_case_variable and metadata rows, but not event rows.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty (no header row found)")

    headers = [str(h).strip().lower() for h in reader.fieldnames if h is not None]
    if len(headers) != len(set(headers)):
        raise ValueError("CSV contains duplicate column names")
    if "record_type" not in headers:
        raise ValueError("CSV is missing required column 'record_type'. Use 'variable' or 'event' rows.")

    known_gens = get_known_generator_types()
    variables: list[dict] = []
    field_order: list[str] = []
    events_raw: list[dict] = []
    edge_case_variables_raw: list[dict] = []
    metadata: dict[str, object] = {}
    seen_names: set[str] = set()

    for row_number, raw_row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw_row.items()}
        record_type = row.get("record_type", "").lower()
        if record_type not in {"variable", "event", "edge_case_variable", "metadata"}:
            raise ValueError(
                f"Row {row_number}: record_type must be 'variable', 'event', "
                "'edge_case_variable', or 'metadata'"
            )

        if record_type == "metadata":
            raw_pct = row.get("edge_case_percentage", "").strip()
            if raw_pct == "":
                # Also accept name=edgeCasePercentage/value=<number> for easier CSV authoring.
                if str(row.get("name", "")).strip().lower() in {"edgecasepercentage", "edge_case_percentage"}:
                    raw_pct = row.get("value", "").strip()
            if raw_pct == "":
                continue
            try:
                pct = float(raw_pct)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row {row_number}: edge_case_percentage must be a finite float between 0 and 1"
                ) from exc
            import math
            if not math.isfinite(pct) or not 0.0 <= pct <= 1.0:
                raise ValueError(
                    f"Row {row_number}: edge_case_percentage must be between 0 and 1"
                )
            metadata["edge_case_percentage"] = pct
            continue

        if record_type in {"variable", "edge_case_variable"}:
            name = row.get("name", "")
            if not name:
                raise ValueError(f"Row {row_number}: variable 'name' is empty")
            # Normal variables must be unique. Edge-case rows intentionally reuse a
            # normal variable name because they describe an override/condition for
            # that existing field and may repeat the same field across edge cases.
            if record_type == "variable" and name in seen_names:
                raise ValueError(f"Row {row_number}: duplicate variable name '{name}'")
            if record_type == "edge_case_variable" and name not in seen_names:
                raise ValueError(
                    f"Row {row_number} ('{name}'): edge-case variable must reference a "
                    "variable defined by an earlier 'variable' row"
                )

            base_variable = next((v for v in variables if v.get("name") == name), None)
            dtype = row.get("dtype", "").lower()
            gen = row.get("gen", "")
            if record_type == "edge_case_variable" and base_variable is not None:
                # Edge-case rows may omit schema columns; omitted values inherit the
                # normal variable definition. This makes CSV authoring concise.
                dtype = dtype or str(base_variable.get("dtype", "string"))
                gen = gen or str(base_variable.get("gen", ""))
            if dtype not in ALLOWED_DTYPES:
                raise ValueError(
                    f"Row {row_number} ('{name}'): invalid dtype '{dtype}'. Must be one of {sorted(ALLOWED_DTYPES)}"
                )

            if gen not in known_gens:
                raise ValueError(
                    f"Row {row_number} ('{name}'): unknown gen type '{gen}'. Known types: {sorted(known_gens)}"
                )

            depends_on_raw = row.get("depends_on", "")
            if record_type == "edge_case_variable" and not depends_on_raw and base_variable is not None:
                depends_on = list(base_variable.get("depends_on", []) or [])
            else:
                depends_on = [d.strip() for d in re.split(r"[;|,]", depends_on_raw) if d.strip()]
            for dep in depends_on:
                if dep not in seen_names:
                    raise ValueError(
                        f"Row {row_number} ('{name}'): depends_on references '{dep}', which must be defined in an earlier variable row"
                    )

            params_raw = row.get("params", "")
            if params_raw:
                try:
                    params = json.loads(params_raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Row {row_number} ('{name}'): 'params' must be valid JSON: {exc}") from exc
                if not isinstance(params, dict):
                    raise ValueError(f"Row {row_number} ('{name}'): 'params' must be a JSON object")
            elif record_type == "edge_case_variable" and base_variable is not None:
                params = dict(base_variable.get("params") or {})
            else:
                params = {}

            variable = {
                "name": name,
                "dtype": "boolean" if dtype == "bool" else dtype,
                "description": row.get("description", ""),
                "gen": gen,
                "params": params,
                "depends_on": depends_on,
                "nullable": row.get("nullable", "").lower() in _TRUE_STRINGS,
            }
            if gen == "formula":
                formula = row.get("formula", "").strip()
                if not formula and record_type == "edge_case_variable" and base_variable is not None:
                    formula = str(base_variable.get("formula", ""))
                if not formula:
                    raise ValueError(f"Row {row_number} ('{name}'): gen='formula' requires 'formula'")
                variable["formula"] = formula

            if record_type == "variable":
                variables.append(variable)
                field_order.append(name)
                seen_names.add(name)
            else:
                edge_name = row.get("edge_case_name", "").strip()
                edge_description = row.get("edge_case_description", "").strip()
                condition = row.get("condition", "").strip()
                if not edge_name:
                    raise ValueError(f"Row {row_number} ('{name}'): edge_case_name is required")
                if not condition:
                    raise ValueError(f"Row {row_number} ('{name}'): condition is required")
                try:
                    tree = ast.parse(condition, mode="eval")
                    allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not,
                               ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
                               ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Name,
                               ast.Constant, ast.List, ast.Tuple, ast.UnaryOp, ast.USub,
                               ast.UAdd, ast.Load)
                    if any(not isinstance(n, allowed) for n in ast.walk(tree)):
                        raise ValueError("condition contains unsupported expression syntax")
                    refs = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                    declared_names = {v["name"] for v in variables}
                    if not refs.issubset(declared_names):
                        missing = sorted(refs - declared_names)
                        raise ValueError(f"condition references undefined variable(s): {missing}")
                except SyntaxError as exc:
                    raise ValueError(f"Row {row_number} ('{name}'): condition is not a valid expression: {exc}") from exc
                edge_item = dict(variable)
                edge_item["edge_case_name"] = edge_name
                edge_item["edge_case_description"] = edge_description
                edge_item["condition"] = condition
                edge_case_variables_raw.append(edge_item)
            continue

        # Event row
        if type_of_data != "transactional":
            raise ValueError(f"Row {row_number}: event rows are only allowed when typeOfData='transactional'")
        event_type = (row.get("event_type") or row.get("name", "")).strip().upper().replace(" ", "_")
        if not event_type:
            raise ValueError(f"Row {row_number}: event row requires 'event_type'")

        sequence_raw = row.get("sequence", "")
        try:
            sequence = int(sequence_raw) if sequence_raw else len(events_raw) + 1
        except ValueError as exc:
            raise ValueError(f"Row {row_number} ('{event_type}'): sequence must be an integer") from exc
        if sequence < 1:
            raise ValueError(f"Row {row_number} ('{event_type}'): sequence must be >= 1")

        fields = [f.strip() for f in re.split(r"[;|,]", row.get("fields", "")) if f.strip()]
        events_raw.append({
            "event_type": event_type,
            "sequence": sequence,
            "fields": fields,
            "min_occurrences": row.get("min_occurrences", "1"),
            "max_occurrences": row.get("max_occurrences", "10"),
        })

    if not variables:
        raise ValueError("CSV must contain at least one variable row")

    # Validate event references only after all variable rows have been read so
    # event rows can appear anywhere in the CSV without creating a dependency
    # on CSV row ordering. Variable dependencies themselves remain ordered.
    if type_of_data == "aggregational" and events_raw:
        raise ValueError("Aggregational CSV cannot contain event rows")

    if type_of_data == "transactional":
        seen_events: set[str] = set()
        events: list[dict] = []
        for event in sorted(events_raw, key=lambda e: (e["sequence"], e["event_type"])):
            event_type = event["event_type"]
            if event_type in seen_events:
                raise ValueError(f"Duplicate event_type '{event_type}'")
            seen_events.add(event_type)
            missing_fields = [f for f in event["fields"] if f not in seen_names]
            if missing_fields:
                raise ValueError(
                    f"Event '{event_type}' references undefined variable(s): {missing_fields}"
                )
            try:
                min_occ = max(1, int(event["min_occurrences"]))
                max_occ = max(min_occ, min(1000, int(event["max_occurrences"])))
            except ValueError as exc:
                raise ValueError(
                    f"Event '{event_type}': min_occurrences/max_occurrences must be integers"
                ) from exc
            events.append({
                "event_type": event_type,
                "sequence": event["sequence"],
                "fields": event["fields"],
                "min_occurrences": min_occ,
                "max_occurrences": max_occ,
            })
        if not events:
            raise ValueError("Transactional CSV must contain at least one event row")
        # Normalize sequences to 1..N while preserving requested order.
        for index, event in enumerate(events, start=1):
            event["sequence"] = index
        return variables, field_order, events, edge_case_variables_raw, metadata

    return variables, field_order, [], edge_case_variables_raw, metadata

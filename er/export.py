"""CSV and zip output for generated tables."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value                      # a spreadsheet would run a text that starts like a formula
    return value


def csv_text(fields: list[str], records: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    for record in records:
        writer.writerow([_cell(record.get(f)) for f in fields])
    return buffer.getvalue()


def zip_bytes(model: dict[str, Any], tables: dict[str, dict[str, Any]], report: dict[str, Any]) -> bytes:
    """One CSV per table, the integrity report and the model that produced them."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, table in tables.items():
            z.writestr(f"{name}.csv", csv_text(table["fields"], table["records"]))
        z.writestr("integrity_report.json", json.dumps(report, indent=2))
        z.writestr("model.json", json.dumps(model, indent=2))
    return out.getvalue()

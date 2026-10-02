"""Projection: spec rows <-> the flat columns the API exposes.

The engine speaks column ids; the API speaks the approved column names. This module is the only place that knows
both, and it formats values (timestamp format and numeric precision) in one consistent way for every column.
A column keeps the timestamp format its own definition declares; otherwise timestamps are RFC 3339 with the
scenario's UTC offset, whole seconds.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from synth.clock import get_tz
from synth.spec import GenerationSpec, SpecColumn

RFC3339 = "rfc3339"
# The layout every timestamp column uses unless its own definition (or the spec's output) names another: the one the
# service has always delivered, so clients keep reading the same strings.
DEFAULT_STYLE = "%d/%m/%Y %I:%M %p"

# Names a definition may use for a timestamp layout, mapped to strftime patterns (anything else is taken as strftime).
_ALIASES = {
    "dd/mm/yyyy hh:mm a": "%d/%m/%Y %I:%M %p", "dd/mm/yyyy hh:mm am/pm": "%d/%m/%Y %I:%M %p",
    "dd/mm/yyyy hh:mm:ss a": "%d/%m/%Y %I:%M:%S %p", "yyyy-mm-dd hh:mm:ss": "%Y-%m-%d %H:%M:%S",
    "yyyy-mm-dd hh:mm": "%Y-%m-%d %H:%M", "yyyy-mm-dd": "%Y-%m-%d",
}
_RFC_NAMES = {"", "rfc3339", "iso", "iso-8601", "date-time", "datetime", "timestamp"}


def layout(declared: Any) -> str | None:
    """strftime pattern for a declared timestamp format, or None for RFC 3339."""
    text = str(declared or "").strip()
    if text.lower() in _RFC_NAMES:
        return None
    return _ALIASES.get(text.lower(), text)


def style_of(spec: GenerationSpec, column: SpecColumn | None = None) -> str | None:
    """strftime pattern a timestamp column is delivered in, or None for RFC 3339 (an explicit choice, never the default)."""
    declared = (column.timestamp_format if column is not None else None) or spec.output.get("timestamp_format")
    if not declared:
        return DEFAULT_STYLE
    return layout(declared)


def time_resolution(spec: GenerationSpec) -> int:
    """Seconds between distinguishable delivered timestamps: 60 when no datetime column shows seconds, else 1.

    The engine rounds its times to this, so that "A is after B" is decided on exactly the strings a client will read.
    """
    styles = [style_of(spec, c) for c in spec.columns.values() if c.dtype == "datetime" and c.kind != "latent"]
    return 60 if styles and all(st is not None and "%S" not in st for st in styles) else 1


def format_datetime(value: datetime, spec: GenerationSpec, column: SpecColumn | None = None) -> str:
    local = value.astimezone(get_tz(spec.timezone))
    if column is not None and column.dtype == "date" and not (column.timestamp_format or spec.output.get("timestamp_format")):
        return local.date().isoformat()
    style = style_of(spec, column)
    if style is None:
        return local.date().isoformat() if column is not None and column.dtype == "date" else local.isoformat(timespec="seconds")
    return local.strftime(style)


def format_value(value: Any, column: SpecColumn, spec: GenerationSpec) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return format_datetime(value, spec, column)
    if column.dtype == "float" and column.precision is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), column.precision)
    return value


def project(rows: list[dict[str, Any]], spec: GenerationSpec, delivered: dict[str, str]) -> list[dict[str, Any]]:
    """spec rows -> column rows, in the spec's column order (the order the variables were approved in)."""
    order = [cid for cid in spec.columns if cid in delivered]
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({delivered[cid]: format_value(row.get(cid), spec.columns[cid], spec) for cid in order})
    return out


def parse_datetime(value: Any, spec: GenerationSpec, column: SpecColumn | None = None) -> datetime | None:
    """Best-effort parse of a delivered timestamp into an aware UTC datetime (None if empty)."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        dt = None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            style = style_of(spec, column)
            if style is not None:
                try:
                    dt = datetime.strptime(text, style)
                except ValueError:
                    pass
        if dt is None:
            raise ValueError(f"unparseable timestamp {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_tz(spec.timezone))
    return dt.astimezone(timezone.utc)


def parse_columns(rows: list[dict[str, Any]], spec: GenerationSpec, delivered: dict[str, str]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """column rows -> spec rows (types restored). Returns ``(rows, parse_errors)``."""
    errors: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        rec: dict[str, Any] = {}
        for cid, col in delivered.items():
            value = row.get(col)
            column = spec.columns[cid]
            try:
                if value in ("", None):
                    value = None
                elif column.dtype in ("datetime", "date"):
                    value = parse_datetime(value, spec, column)
                elif column.dtype == "float":
                    value = float(value)
                elif column.dtype == "integer":
                    value = int(value)
                elif column.dtype == "boolean" and not isinstance(value, bool):
                    value = str(value).strip().lower() in ("true", "1", "yes")
            except (ValueError, TypeError) as exc:
                errors.append({"row": i, "column": col, "value": str(value)[:40], "error": str(exc)})
                value = None
            rec[cid] = value
        out.append(rec)
    return out, errors

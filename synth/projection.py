"""Projection: concept rows <-> the flat columns the API exposes.

The engine speaks concepts; the API speaks the approved column names. This module is the only place that knows both, and it formats values
(timestamp format and numeric precision come from the pack) in one consistent way for every column.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from synth.clock import get_tz
from synth.pack import BehaviorPack

RFC3339 = "rfc3339"


def format_datetime(value: datetime, pack: BehaviorPack) -> str:
    """``output.timestamp_format``: ``rfc3339`` (with explicit offset) or any ``strftime`` pattern."""
    local = value.astimezone(get_tz(pack.timezone))
    style = str(pack.output.get("timestamp_format", RFC3339))
    return local.isoformat(timespec="seconds") if style == RFC3339 else local.strftime(style)


def format_value(value: Any, concept: Any, pack: BehaviorPack) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return format_datetime(value, pack)
    if concept.dtype == "float" and concept.precision is not None and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        return round(float(value), concept.precision)
    return value


def project(rows: list[dict[str, Any]], pack: BehaviorPack, concept_columns: dict[str, str]) -> list[dict[str, Any]]:
    """concept rows -> column rows, in pack concept order (stable, readable)."""
    order = [c for c in pack.concepts if c in concept_columns]
    out: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for c in order:
            concept = pack.concepts[c]
            value = format_value(row.get(c), concept, pack)
            if value is None and concept.null_token is not None:
                value = concept.null_token
            rec[concept_columns[c]] = value
        out.append(rec)
    return out


def parse_datetime(value: Any, pack: BehaviorPack) -> datetime | None:
    """Best-effort parse of a delivered timestamp into an aware UTC datetime (None if unparseable)."""
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
            style = str(pack.output.get("timestamp_format", RFC3339))
            if style != RFC3339:
                try:
                    dt = datetime.strptime(text, style)
                except ValueError:
                    pass
        if dt is None:
            raise ValueError(f"unparseable timestamp {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_tz(pack.timezone))
    return dt.astimezone(timezone.utc)


def parse_columns(
    rows: list[dict[str, Any]], pack: BehaviorPack, concept_columns: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """column rows -> concept rows (types restored). Returns ``(rows, parse_errors)``."""
    errors: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        rec: dict[str, Any] = {}
        for cid, col in concept_columns.items():
            value = row.get(col)
            dtype = pack.concepts[cid].dtype
            try:
                if value in ("", None):
                    value = None
                elif dtype in ("datetime", "date"):
                    value = parse_datetime(value, pack)
                elif dtype == "float":
                    value = float(value)
                elif dtype == "integer":
                    value = int(value)
                elif dtype == "boolean" and not isinstance(value, bool):
                    value = str(value).strip().lower() in ("true", "1", "yes")
            except (ValueError, TypeError) as exc:
                errors.append({"row": i, "column": col, "value": str(value)[:40], "error": str(exc)})
                value = None
            rec[cid] = value
        out.append(rec)
    return out, errors



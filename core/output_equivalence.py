"""Generic executable-output equivalence for scenario variables.

This module answers a narrow question:

    "Will these two declared variables deterministically produce the same value
     for the same generated record context?"

It deliberately does *not* treat equal generators, equal ranges, or equal
probability distributions as duplicate outputs. Two independent random fields
using the same distribution are still different business variables.

The helper is source/domain neutral and is used as a deterministic final
schema guard. Persistence/source precedence is handled by the caller.
"""
from __future__ import annotations

import json
import re
from typing import Any


_DEFAULT_TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M %p"


def _runtime_field_name(value: Any) -> str:
    """Preserve the runtime field spelling used by the executable generator."""
    return str(value or "").strip()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical_value(value[k]) for k in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _serialized_signature(kind: str, value: Any) -> tuple[str, str]:
    return kind, json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), default=str)


def _dtype_key(variable: dict[str, Any]) -> str:
    dtype = str(variable.get("dtype") or "string").strip().lower()
    aliases = {
        "bool": "boolean",
        "int": "integer",
        "bigint": "integer",
        "smallint": "integer",
        "number": "float",
        "double": "float",
        "decimal": "float",
        "numeric": "float",
        "timestamp": "datetime",
        "date-time": "datetime",
    }
    return aliases.get(dtype, dtype)


def _choices(params: dict[str, Any]) -> list[Any]:
    raw = params.get("choices", params.get("values", []))
    if isinstance(raw, str):
        return [item.strip() for item in re.split(r"[;,|]", raw) if item.strip()]
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _fixed_number(params: dict[str, Any]) -> Any | None:
    """Return a single guaranteed numeric output, otherwise None."""
    # Field-driven bounds are data-dependent even when static fallback min/max are equal.
    if params.get("lo_field") or params.get("hi_field"):
        return None
    lo = params.get("min", params.get("lo"))
    hi = params.get("max", params.get("hi"))
    if lo is None or hi is None:
        return None
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except (TypeError, ValueError):
        return None
    if lo_f != hi_f:
        return None
    if params.get("integer"):
        return int(round(lo_f))
    precision = params.get("precision")
    try:
        precision_i = int(precision) if precision is not None else 12
    except (TypeError, ValueError):
        precision_i = 12
    return round(lo_f, max(0, min(12, precision_i)))


def _fixed_output_signature(dtype: str, value: Any) -> tuple[Any, ...]:
    """Canonical signature shared by all generators that emit one fixed value."""
    normalized = value
    if dtype == "integer":
        try:
            normalized = int(round(float(value)))
        except (TypeError, ValueError, OverflowError):
            pass
    elif dtype == "float":
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError):
            pass
    elif dtype == "boolean":
        if isinstance(value, bool):
            normalized = value
        elif isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "1", "yes", "y", "enabled", "on"}:
                normalized = True
            elif token in {"false", "0", "no", "n", "disabled", "off"}:
                normalized = False
    return _serialized_signature("fixed_output", (dtype, normalized))


def _effective_timestamp_format(params: dict[str, Any], generator: str) -> str:
    """Mirror the final serialization precision relevant to equivalence."""
    raw = params.get("timestamp_format", params.get("format"))
    fmt = str(raw).strip() if raw is not None and str(raw).strip() else _DEFAULT_TIMESTAMP_FORMAT
    if generator == "ts_add_field" and "%S" not in fmt:
        return "%d/%m/%Y %I:%M:%S %p"
    if generator == "ts_offset" and "%S" not in fmt:
        try:
            min_sec = int(params.get("min_sec", params.get("min_seconds", 0)) or 0)
            max_sec = int(params.get("max_sec", params.get("max_seconds", min_sec)) or min_sec)
            min_sec, max_sec = min(min_sec, max_sec), max(min_sec, max_sec)
            if min_sec != max_sec or min_sec % 60 != 0:
                return "%d/%m/%Y %I:%M:%S %p"
        except (TypeError, ValueError):
            return "%d/%m/%Y %I:%M:%S %p"
    return fmt


def output_equivalence_signature(variable: dict[str, Any] | Any) -> tuple[Any, ...] | None:
    """Return a signature only when runtime output equivalence is deterministic.

    ``None`` means "do not deduplicate based on executable output".
    """
    if not isinstance(variable, dict):
        if hasattr(variable, "model_dump"):
            variable = variable.model_dump()
        else:
            return None

    dtype = _dtype_key(variable)
    gen = str(variable.get("gen") or "").strip().lower()
    params = variable.get("params") if isinstance(variable.get("params"), dict) else {}
    formula = str(variable.get("formula") or "").strip()

    if formula:
        # Whitespace does not alter this constrained formula language. Preserve identifier
        # spelling because runtime field names are case-sensitive.
        normalized_formula = re.sub(r"\s+", "", formula)
        return ("formula", dtype, normalized_formula)

    if gen == "constant" and "value" in params:
        return _fixed_output_signature(dtype, params.get("value"))

    if gen == "dependent_choice":
        mapping = params.get("mapping")
        depends = _runtime_field_name(params.get("depends_on_field"))
        if isinstance(mapping, dict) and mapping and depends:
            values = list(mapping.values())
            if len({_serialized_signature("value", value)[1] for value in values}) == 1:
                return _fixed_output_signature(dtype, values[0])
            return _serialized_signature("dependent_choice", (dtype, depends, mapping))
        return None

    # id_mirror has a random fallback whenever its dependency has no usable numeric suffix,
    # therefore it is not guaranteed to produce an identical value and is intentionally excluded.
    if gen == "id_mirror":
        return None

    if gen == "date_offset":
        base_field = _runtime_field_name(params.get("base_field"))
        if base_field and params.get("days") is not None:
            return _serialized_signature("date_offset", (dtype, base_field, params.get("days")))
        return None

    if gen == "ts_add_field":
        base_field = _runtime_field_name(params.get("base_field"))
        add_field = _runtime_field_name(params.get("add_seconds_field"))
        if base_field and add_field:
            return _serialized_signature(
                "ts_add_field",
                (dtype, base_field, add_field, _effective_timestamp_format(params, gen)),
            )
        return None

    if gen == "ts_offset":
        base_field = _runtime_field_name(params.get("base_field") or params.get("source_field"))
        min_sec = params.get("min_sec", params.get("min_seconds"))
        max_sec = params.get("max_sec", params.get("max_seconds", min_sec))
        if base_field and min_sec is not None and max_sec is not None:
            try:
                min_i = int(min_sec)
                max_i = int(max_sec)
                if min_i == max_i:
                    return _serialized_signature(
                        "ts_offset",
                        (dtype, base_field, min_i, _effective_timestamp_format(params, gen)),
                    )
            except (TypeError, ValueError):
                pass
        return None

    if gen == "date_offset_range":
        base_field = _runtime_field_name(params.get("base_field"))
        min_days = params.get("min_days")
        max_days = params.get("max_days", min_days)
        if base_field and min_days is not None and max_days is not None:
            try:
                min_i = int(min_days)
                max_i = int(max_days)
                if min_i == max_i:
                    return _serialized_signature("date_offset", (dtype, base_field, min_i))
            except (TypeError, ValueError):
                pass
        return None

    if gen in {"uniform", "uniform_int", "uniform_bounded", "generic", "semantic_event", "semantic_string"}:
        fixed = _fixed_number(params)
        if fixed is not None:
            return _fixed_output_signature(dtype, fixed)
        if "value" in params and params.get("value") is not None:
            return _fixed_output_signature(dtype, params.get("value"))
        choices = _choices(params)
        if len(choices) == 1:
            return _fixed_output_signature(dtype, choices[0])
        return None

    if gen == "weighted_choice":
        choices = _choices(params)
        if len(choices) == 1:
            return _fixed_output_signature(dtype, choices[0])
        return None

    if gen == "weighted_bucket":
        buckets = params.get("buckets") or []
        normalized = []
        for bucket in buckets:
            if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
                continue
            try:
                lo = float(bucket[0])
                hi = float(bucket[1])
            except (TypeError, ValueError):
                continue
            normalized.append((min(lo, hi), max(lo, hi)))
        if len(normalized) == 1 and normalized[0][0] == normalized[0][1]:
            return _fixed_output_signature(dtype, normalized[0][0])
        return None

    return None

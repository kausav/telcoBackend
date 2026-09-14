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

import ast
import csv
import io
import json
import re
from typing import Any

from agents.data_generation_agent import get_known_generator_types

ALLOWED_DTYPES = {
    "string", "int", "integer", "float", "decimal", "categorical",
    "datetime", "date", "timestamp", "bool", "boolean", "uuid",
    # Common external aliases from pandas/warehouse exports.
    "object", "str", "text", "varchar", "number", "numeric",
    # Common schema/tooling aliases.
    "enum", "category", "double", "long", "short",
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
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:-|\.\.)\s*(-?\d+(?:\.\d+)?)\s*", value)
    if not match:
        return None
    a, b = match.groups()
    def number(s: str):
        return float(s) if "." in s else int(s)
    return number(a), number(b)


def _infer_prefixed_id_params(params: dict[str, Any]) -> dict[str, Any]:
    p = dict(params)
    example = str(p.get("example", "")).strip()
    prefix = str(p.get("prefix", ""))

    if example:
        match = re.fullmatch(r"([^0-9]*?)(\d+)", example)
        if match:
            example_prefix, digits_text = match.groups()
            if example_prefix:
                prefix = example_prefix
            p.setdefault("digits", len(digits_text))

    if "digits" not in p and "length" in p:
        try:
            length = int(p.get("length"))
            prefix_len = len(prefix)
            digits = max(1, length - prefix_len) if prefix_len and length > prefix_len else max(1, length)
            p["digits"] = digits
        except Exception:
            pass

    if prefix:
        p["prefix"] = prefix
    return p


def _choose_temporal_base_field(depends_on: list[str]) -> str | None:
    for name in reversed(depends_on or []):
        lowered = str(name).lower()
        if "timestamp" in lowered or lowered.endswith("_date") or lowered == "date":
            return str(name)
    return str(depends_on[-1]) if depends_on else None


def _normalize_dtype(dtype: str) -> str:
    value = dtype.strip().lower()
    if value == "timestamp":
        return "datetime"
    if value in {"enum", "category"}:
        return "categorical"
    if value in {"object", "str", "text", "varchar"}:
        return "string"
    if value in {"double"}:
        return "float"
    if value in {"long", "short"}:
        return "int"
    if value in {"number", "numeric"}:
        return "float"
    if value.startswith("decimal"):
        return "float"
    if value == "integer":
        return "int"
    if value == "bool":
        return "boolean"
    if value == "uuid":
        return "string"
    if value in ALLOWED_DTYPES:
        return value
    # Tolerant fallback for unknown client dtypes.
    return "string"


def _coerce_formula_text(formula: str) -> str:
    """Best-effort normalize client formula text to Python-like expressions.

    Returns an empty string when no safe parseable expression can be produced.
    """
    text = (formula or "").strip()
    if not text:
        return ""
    # Excel formulas often begin with '='. Strip it before normalization.
    if text.startswith("="):
        text = text[1:].strip()

    # Normalize common SQL/natural-language operators.
    expr = re.sub(r"\bAND\b", "and", text, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "or", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bNOT\b", "not", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bNULL\b", "None", expr, flags=re.IGNORECASE)
    expr = re.sub(r"<>", "!=", expr)
    expr = re.sub(r"\^", "**", expr)

    # Convert IN lists to quoted string tuples when items look like enum tokens.
    def _in_repl(match: re.Match[str]) -> str:
        items = [x.strip() for x in match.group(1).split(",") if x.strip()]
        quoted = [f"'{item}'" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item) else item for item in items]
        return "in (" + ", ".join(quoted) + ")"
    expr = re.sub(r"\bIN\s*\(([^)]*)\)", _in_repl, expr, flags=re.IGNORECASE)

    # Convert single '=' comparisons into '=='.
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)

    # Convert "X when cond" to "X if cond else None".
    m_when = re.fullmatch(r"(.+?)\s+when\s+(.+)", expr, flags=re.IGNORECASE)
    if m_when:
        lhs = m_when.group(1).strip()
        cond = m_when.group(2).strip()
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", lhs):
            lhs = f"'{lhs}'"
        expr = f"({lhs} if {cond} else None)"

    # Convert two-part ternary-like forms separated by ';'.
    parts = [p.strip() for p in expr.split(";") if p.strip()]
    if len(parts) == 2:
        m1 = re.fullmatch(r"(.+?)\s+if\s+(.+)", parts[0], flags=re.IGNORECASE)
        m2 = re.fullmatch(r"otherwise\s+(.+)", parts[1], flags=re.IGNORECASE)
        if m1 and m2:
            a, c = m1.group(1).strip(), m1.group(2).strip()
            b = m2.group(1).strip()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", a):
                a = f"'{a}'"
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", b):
                b = f"'{b}'"
            expr = f"({a} if {c} else {b})"

    # Convert chained "A if cond1; B if cond2" into nested ternary.
    if len(parts) == 2:
        m1 = re.fullmatch(r"(.+?)\s+if\s+(.+)", parts[0], flags=re.IGNORECASE)
        m2 = re.fullmatch(r"(.+?)\s+if\s+(.+)", parts[1], flags=re.IGNORECASE)
        if m1 and m2:
            a, c1 = m1.group(1).strip(), m1.group(2).strip()
            b, c2 = m2.group(1).strip(), m2.group(2).strip()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", a):
                a = f"'{a}'"
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", b):
                b = f"'{b}'"
            expr = f"({a} if {c1} else ({b} if {c2} else None))"

    try:
        ast.parse(expr, mode="eval")
        return expr
    except Exception:
        return ""


def _normalize_generator(gen: str, params: dict[str, Any], depends_on: list[str], dtype: str, formula: str) -> tuple[str, dict[str, Any]]:
    """Map client-facing generator vocabulary into executable generators."""
    g = (gen or "").strip().lower()
    p = dict(params)

    if g == "unique_id":
        p = _infer_prefixed_id_params(p)
        # When a unique id depends on another id-like field, preserve the source
        # suffix while applying this field's own prefix (e.g. SUB77668 -> EVT77668).
        source_dep = next((str(dep) for dep in depends_on if str(dep).strip().lower().endswith("_id")), None)
        if source_dep and p.get("prefix") is not None:
            return "id_mirror", {
                **p,
                "source_field": source_dep,
                "source_prefix": str(p.get("source_prefix", "")),
            }
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
        choices = p.get("choices", p.get("values"))
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

    if g in {"timestamp", "recent_timestamp", "datetime"}:
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

    if g == "fixed_config":
        if "default" in p:
            return "constant", {"value": p.get("default")}
        if "value" in p:
            return "constant", {"value": p.get("value")}
        min_value = p.get("min", p.get("lo"))
        max_value = p.get("max", p.get("hi", min_value))
        if min_value is not None or max_value is not None:
            lo = _to_numeric_param(min_value, 0)
            hi = _to_numeric_param(max_value, lo)
            midpoint = lo if hi is None else (lo + hi) / 2
            if dtype in {"int", "integer"}:
                midpoint = int(round(midpoint))
            elif dtype in {"float", "decimal", "number", "numeric"}:
                midpoint = round(float(midpoint), int(p.get("precision", 2) or 2))
            return "constant", {"value": midpoint}
        return "constant", {"value": "configured"}

    if g in {"random_integer", "integer_random"}:
        return "uniform_int", {
            "min": p.get("min", p.get("lo", 0)),
            "max": p.get("max", p.get("hi", 100)),
        }

    if g == "derived_date":
        base_field = _choose_temporal_base_field(depends_on)
        rng = _range_from_text(p.get("cooling_off_days")) or _range_from_text(p.get("days"))
        if base_field and rng:
            return "date_offset_range", {"base_field": base_field, "min_days": int(rng[0]), "max_days": int(rng[1])}
        if base_field and p.get("days") is not None:
            return "date_offset", {"base_field": base_field, "days": int(p.get("days"))}
        return "recent_datetime", {"days_back": 30}

    # Compatibility: some CSVs put dtype-like values in the `gen` column.
    if g in {"categorical", "category"}:
        choices = p.get("choices", p.get("values"))
        if choices is None and "value" in p:
            raw = p.get("value")
            choices = _split_list(str(raw)) if isinstance(raw, str) else ([raw] if raw is not None else [])
        if isinstance(choices, str):
            choices = _split_list(choices)
        if isinstance(choices, list) and choices:
            if len(choices) == 1:
                return "constant", {"value": choices[0]}
            p["choices"] = choices
            return "weighted_choice", p
        return "constant", {"value": "UNKNOWN"}

    if g in {"string", "text", "object", "varchar"}:
        if "value" in p:
            return "constant", {"value": p.get("value")}
        choices = p.get("choices", p.get("values"))
        if isinstance(choices, str):
            choices = _split_list(choices)
        if isinstance(choices, list) and choices:
            return "weighted_choice", {"choices": choices}
        return "constant", {"value": ""}

    if g in {"int", "integer", "float", "decimal", "number", "numeric"}:
        min_value = p.get("min", p.get("lo", 0))
        max_value = p.get("max", p.get("hi", min_value if min_value is not None else 0))
        if dtype in {"int", "integer"}:
            return "uniform_int", {"min": min_value, "max": max_value}
        return "uniform", {"min": min_value, "max": max_value}

    if g in {"bool", "boolean"}:
        if "value" in p:
            token = str(p.get("value", "")).strip().lower()
            if token in {"true", "1", "yes", "y", "on", "enabled"}:
                return "constant", {"value": True}
            if token in {"false", "0", "no", "n", "off", "disabled"}:
                return "constant", {"value": False}
        return "weighted_choice", {"choices": [True, False], "weights": [0.5, 0.5]}

    if g in {"date"}:
        days_back = p.get("days_back", 30)
        return "recent_datetime", {"days_back": days_back}

    if g == "synthetic_event":
        # Legacy/client alias used in some CSV templates.
        # Prefer explicit choices/value when provided; otherwise create a safe constant.
        choices = p.get("choices", p.get("values"))
        if choices is None and "value" in p:
            raw = p.get("value")
            if isinstance(raw, str):
                choices = _split_list(raw) or [raw]
            elif isinstance(raw, list):
                choices = raw
            elif raw is not None:
                choices = [raw]

        if isinstance(choices, str):
            choices = _split_list(choices)
        if isinstance(choices, list) and choices:
            if len(choices) == 1:
                return "constant", {"value": choices[0]}
            p["choices"] = choices
            return "weighted_choice", p

        if dtype in {"bool", "boolean"}:
            return "constant", {"value": True}
        return "constant", {"value": "event"}

    if g == "configuration":
        # Client alias for fixed or enumerated config values.
        choices = p.get("choices", p.get("values"))
        if choices is None and "value" in p:
            raw = p.get("value")
            if isinstance(raw, str):
                choices = _split_list(raw) or [raw]
            elif isinstance(raw, list):
                choices = raw
            elif raw is not None:
                choices = [raw]

        if isinstance(choices, str):
            choices = _split_list(choices)
        if isinstance(choices, list) and choices:
            if len(choices) == 1:
                return "constant", {"value": choices[0]}
            p["choices"] = choices
            return "weighted_choice", p

        if any(k in p for k in ("min", "max", "lo", "hi")):
            min_value = p.get("min", p.get("lo", 0))
            max_value = p.get("max", p.get("hi", min_value if min_value is not None else 0))
            return "uniform", {"min": min_value, "max": max_value}

        if "value" in p:
            return "constant", {"value": p.get("value")}

        return "constant", {"value": "configured"}

    if g == "derived_state":
        # Client alias for a state inferred from other fields.
        # Prefer an explicit formula, then enumerated choices, then a fallback constant.
        if formula:
            return "formula", p

        choices = p.get("choices", p.get("values"))
        if choices is None and "value" in p:
            raw = p.get("value")
            if isinstance(raw, str):
                choices = _split_list(raw) or [raw]
            elif isinstance(raw, list):
                choices = raw
            elif raw is not None:
                choices = [raw]

        if isinstance(choices, str):
            choices = _split_list(choices)
        if isinstance(choices, list) and choices:
            if len(choices) == 1:
                return "constant", {"value": choices[0]}
            p["choices"] = choices
            return "weighted_choice", p

        return "constant", {"value": "unknown_state"}

    if g == "derived_event":
        # Client alias for event-level outcomes derived from context.
        # Reuse the same compatibility behavior as derived_state.
        if formula:
            return "formula", p

        choices = p.get("choices", p.get("values"))
        if choices is None and "value" in p:
            raw = p.get("value")
            if isinstance(raw, str):
                choices = _split_list(raw) or [raw]
            elif isinstance(raw, list):
                choices = raw
            elif raw is not None:
                choices = [raw]

        if isinstance(choices, str):
            choices = _split_list(choices)
        if isinstance(choices, list) and choices:
            if len(choices) == 1:
                return "constant", {"value": choices[0]}
            p["choices"] = choices
            return "weighted_choice", p

        return "constant", {"value": "unknown_event"}

    # Additional compatibility aliases commonly seen in client templates.
    if g in {"random_numeric", "random_number", "random_float", "numeric_random"}:
        min_value = p.get("min", p.get("lo", 0))
        max_value = p.get("max", p.get("hi", 100))
        return "uniform", {"min": min_value, "max": max_value}

    # Smart fallback for previously unseen client gen names.
    # Prefer preserving deterministic intent over rejecting imports.
    if formula:
        return "formula", p

    choices = p.get("choices", p.get("values"))
    if choices is None and "value" in p and isinstance(p.get("value"), str):
        tokenized = _split_list(str(p.get("value")))
        if len(tokenized) > 1:
            choices = tokenized
    if isinstance(choices, str):
        choices = _split_list(choices)
    if isinstance(choices, list) and choices:
        if len(choices) == 1:
            return "constant", {"value": choices[0]}
        return "weighted_choice", {"choices": choices}

    if dtype in {"datetime", "date"} or any(t in g for t in ("date", "time", "timestamp")):
        return "recent_datetime", {"days_back": int(p.get("days_back", 30) or 30)}

    if dtype in {"bool", "boolean"}:
        token = str(p.get("value", "")).strip().lower()
        if token in {"true", "1", "yes", "y", "on", "enabled"}:
            return "constant", {"value": True}
        if token in {"false", "0", "no", "n", "off", "disabled"}:
            return "constant", {"value": False}
        return "weighted_choice", {"choices": [True, False], "weights": [0.5, 0.5]}

    if dtype in {"int", "integer", "float", "decimal", "number", "numeric"} or any(t in g for t in ("numeric", "number", "range", "amount")):
        min_value = p.get("min", p.get("lo", 0))
        max_value = p.get("max", p.get("hi", 100))
        if dtype in {"int", "integer"}:
            return "uniform_int", {"min": min_value, "max": max_value}
        return "uniform", {"min": min_value, "max": max_value}

    if "value" in p:
        return "constant", {"value": p.get("value")}

    # Existing internal generators remain unchanged.
    return g, p


def _coerce_executable_generator(gen: str, params: dict[str, Any], dtype: str, formula: str) -> tuple[str, dict[str, Any]]:
    """Guarantee an executable generator for low-quality CSV inputs.

    This is a final safety net for badly formed templates: if a normalized
    generator is still unknown to the backend, choose a deterministic fallback
    from dtype/params/formula semantics.
    """
    known = get_known_generator_types()
    if gen in known:
        return gen, params

    p = dict(params or {})
    g = str(gen or "").strip().lower()
    d = str(dtype or "string").strip().lower()

    if formula:
        return "formula", p

    # Event/container style aliases should become stable textual markers.
    if any(token in g for token in ("event", "container", "state", "stage", "node")):
        if "value" in p:
            return "constant", {"value": p.get("value")}
        return "constant", {"value": "event"}

    choices = p.get("choices", p.get("values"))
    if isinstance(choices, str):
        choices = _split_list(choices)
    if choices is None and "value" in p and isinstance(p.get("value"), str):
        tokenized = _split_list(str(p.get("value")))
        if len(tokenized) > 1:
            choices = tokenized
    if isinstance(choices, list) and choices:
        if len(choices) == 1:
            return "constant", {"value": choices[0]}
        return "weighted_choice", {"choices": list(choices)}

    if d in {"datetime", "date"}:
        return "recent_datetime", {"days_back": int(p.get("days_back", 30) or 30)}

    if d in {"bool", "boolean"}:
        token = str(p.get("value", "")).strip().lower()
        if token in {"true", "1", "yes", "y", "on", "enabled"}:
            return "constant", {"value": True}
        if token in {"false", "0", "no", "n", "off", "disabled"}:
            return "constant", {"value": False}
        return "weighted_choice", {"choices": [True, False], "weights": [0.5, 0.5]}

    if d in {"int", "integer", "float", "decimal", "number", "numeric"}:
        min_value = p.get("min", p.get("lo", 0))
        max_value = p.get("max", p.get("hi", 100))
        return "uniform", {"min": min_value, "max": max_value}

    if "value" in p:
        return "constant", {"value": p.get("value")}

    return "constant", {"value": "unknown"}


def _to_numeric_param(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _infer_transactional(rows: list[dict[str, str]]) -> bool:
    """Transactional iff at least one row has non-empty ``fields`` data."""
    return any(bool(_split_list(row.get("fields", ""))) for row in rows)


def _resolve_dependency_aliases(
    variables: list[dict[str, Any]],
    all_names: set[str],
    event_type_names: set[str],
) -> None:
    """Normalize client depends_on aliases to concrete variable names in-place.

    Some client templates use logical anchors (e.g. ``subscriber``) or control/event
    tokens (e.g. ``suppression_rule``) in ``depends_on``. Keep strict validation for
    true typos, but resolve common aliases and drop non-field control tokens.
    """
    lowered = {name.lower(): name for name in all_names}
    id_by_stem: dict[str, str] = {}
    for name in all_names:
        if name.lower().endswith("_id"):
            id_by_stem[name[:-3].lower()] = name

    for var in variables:
        normalized: list[str] = []
        for dep in var.get("depends_on", []) or []:
            token = str(dep).strip()
            if not token:
                continue
            key = token.lower()

            if token in all_names:
                if token not in normalized:
                    normalized.append(token)
                continue

            mapped = None
            # Common entity stem alias: subscriber -> subscriber_id, customer -> customer_id, etc.
            if key in id_by_stem:
                mapped = id_by_stem[key]
            # Exact lowercase match fallback for case/format drift.
            elif key in lowered:
                mapped = lowered[key]
            # Event/control token: not a variable dependency, keep out of strict check.
            elif key in {x.lower() for x in event_type_names}:
                mapped = None

            if mapped and mapped not in normalized:
                normalized.append(mapped)
        var["depends_on"] = normalized


def _align_id_mirror_sources(variables: list[dict[str, Any]], all_names: set[str]) -> None:
    """Align id_mirror source_field with normalized depends_on names.

    unique_id rows may have depends_on aliases (e.g. subscriber -> subscriber_id).
    After alias normalization, keep source_field in sync with the resolved id field.
    """
    for var in variables:
        if str(var.get("gen", "")).strip().lower() != "id_mirror":
            continue
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        depends_on = [str(x) for x in (var.get("depends_on") or []) if str(x)]
        source_field = str(params.get("source_field", "")).strip()
        if source_field in all_names:
            continue
        candidate = next((dep for dep in depends_on if dep in all_names and dep.lower().endswith("_id")), None)
        if candidate:
            params["source_field"] = candidate
            var["params"] = params


def _merge_duplicate_variable(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Merge repeated variable definitions when they are compatible.

    Returns True when the incoming row can be absorbed into existing, otherwise False.
    """
    if existing.get("dtype") != incoming.get("dtype"):
        return False
    if existing.get("formula", "") != incoming.get("formula", ""):
        return False

    ex_dep = list(existing.get("depends_on") or [])
    in_dep = list(incoming.get("depends_on") or [])
    merged_dep = list(ex_dep)
    for dep in in_dep:
        if dep not in merged_dep:
            merged_dep.append(dep)

    ex_params = existing.get("params") if isinstance(existing.get("params"), dict) else {}
    in_params = incoming.get("params") if isinstance(incoming.get("params"), dict) else {}
    ex_gen = str(existing.get("gen", ""))
    in_gen = str(incoming.get("gen", ""))

    # Same generator: merge deps and merge params conservatively.
    if ex_gen == in_gen:
        merged_params = dict(ex_params)
        for key, value in in_params.items():
            if key not in merged_params:
                merged_params[key] = value
                continue
            cur = merged_params[key]
            if isinstance(cur, list) and isinstance(value, list):
                for item in value:
                    if item not in cur:
                        cur.append(item)
                merged_params[key] = cur
            elif cur == value:
                continue
            elif key in {"choices", "values"}:
                left = cur if isinstance(cur, list) else _split_list(str(cur))
                right = value if isinstance(value, list) else _split_list(str(value))
                merged = list(left)
                for item in right:
                    if item not in merged:
                        merged.append(item)
                merged_params[key] = merged
            else:
                # Prefer an explicit existing value; keep import deterministic.
                pass

        existing["depends_on"] = merged_dep
        existing["params"] = merged_params
        existing["nullable"] = bool(existing.get("nullable", False)) and bool(incoming.get("nullable", False))
        if not existing.get("description") and incoming.get("description"):
            existing["description"] = incoming["description"]
        return True

    # Compatible categorical forms may vary between constant (single value)
    # and weighted_choice (multi-value). Promote to weighted_choice when needed.
    categorical_forms = {"weighted_choice", "constant"}
    if ex_gen in categorical_forms and in_gen in categorical_forms:
        def _to_choices(gen_name: str, params: dict[str, Any]) -> list[Any]:
            if gen_name == "weighted_choice":
                raw = params.get("choices", params.get("values", []))
                if isinstance(raw, list):
                    return list(raw)
                if isinstance(raw, str):
                    return _split_list(raw)
                return []
            if "value" in params:
                return [params.get("value")]
            return []

        ex_choices = _to_choices(ex_gen, ex_params)
        in_choices = _to_choices(in_gen, in_params)
        merged_choices = list(ex_choices)
        for choice in in_choices:
            if choice not in merged_choices:
                merged_choices.append(choice)

        if not merged_choices:
            return False

        if len(merged_choices) == 1:
            existing["gen"] = "constant"
            existing["params"] = {"value": merged_choices[0]}
        else:
            existing["gen"] = "weighted_choice"
            existing["params"] = {"choices": merged_choices}
        existing["depends_on"] = merged_dep
        existing["nullable"] = bool(existing.get("nullable", False)) and bool(incoming.get("nullable", False))
        if not existing.get("description") and incoming.get("description"):
            existing["description"] = incoming["description"]
        return True

    return False


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
        "datetime",
        "synthetic_event",
        "configuration",
        "derived_state",
        "derived_event",
        "categorical", "category",
        "string", "text", "object", "varchar",
        "int", "integer", "float", "decimal", "number", "numeric",
        "bool", "boolean", "date",
    }
    variables: list[dict] = []
    field_order: list[str] = []
    event_groups: dict[tuple[str, int], dict[str, Any]] = {}
    event_type_names: set[str] = set()
    variable_by_name: dict[str, dict[str, Any]] = {}
    all_names: set[str] = set()

    # First pass: validate/normalize variables. Dependencies are allowed to point
    # forward because client schemas may use formulas whose source field appears later.
    # Some client templates repeat the same variable row across multiple event rows.
    # We allow duplicate names only when the normalized definition is equivalent.
    for row_number, row in enumerate(raw_rows, start=2):
        name = row.get("name", "").strip()
        if not name:
            raise ValueError(f"Row {row_number}: 'name' is empty")

        dtype_raw = row.get("dtype", "").strip().lower()
        if not dtype_raw:
            raise ValueError(f"Row {row_number} ('{name}'): dtype is empty")
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
        gen, params = _normalize_generator(gen_raw, params, depends_on, dtype, formula)
        gen, params = _coerce_executable_generator(gen, params, dtype, formula)

        variable = {
            "name": name,
            "dtype": dtype,
            "description": row.get("description", ""),
            "gen": gen,
            "params": params,
            "depends_on": depends_on,
            "nullable": row.get("nullable", "").lower() in _TRUE_STRINGS,
        }
        normalized_formula = _coerce_formula_text(formula) if formula else ""
        if normalized_formula:
            # A parseable sheet formula is authoritative regardless of the original
            # CSV gen label, and still coexists with deterministic constraints.
            variable["formula"] = normalized_formula
        elif gen == "formula":
            # Invalid formula text should not halt imports.
            fallback_gen, fallback_params = _coerce_executable_generator("", params, dtype, "")
            variable["gen"] = fallback_gen
            variable["params"] = fallback_params
            if formula:
                variable.setdefault("params", {})["formula_text"] = formula
        elif formula:
            # Preserve non-parseable formula prose as metadata only.
            variable.setdefault("params", {})["formula_text"] = formula

        existing = variable_by_name.get(name)
        if existing is None:
            variable_by_name[name] = variable
            variables.append(variable)
            field_order.append(name)
            all_names.add(name)
        else:
            if not _merge_duplicate_variable(existing, variable):
                raise ValueError(
                    f"Row {row_number}: duplicate variable name '{name}' has conflicting definition"
                )

        if is_event_field:
            event_type = (row.get("event_type") or "").strip().upper().replace(" ", "_")
            if not event_type:
                raise ValueError(f"Row {row_number} ('{name}'): event row requires 'event_type'")
            event_type_names.add(event_type)
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

    _resolve_dependency_aliases(variables, all_names, event_type_names)
    _align_id_mirror_sources(variables, all_names)

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
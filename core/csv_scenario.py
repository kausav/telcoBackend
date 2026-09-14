"""CSV scenario-definition parser.

The CSV is the source of truth for the schema.  A row is treated as an
The CSV is a variable-level schema. Transactional history is derived from the
entity key and the first history timestamp field; no user-facing scope metadata is required.
``record_type`` is optional metadata rather than the mechanism used to distinguish
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
    """Normalize flexible unique-id encodings into safe executable parameters.

    Supported examples include:
      digits=7
      digits=0000001-9999999
      range=0000001-9999999
      example=SUB-0000001
      length=10

    The old implementation attempted ``int(digits)`` later in the generator,
    which made a perfectly reasonable range expression fail every record.
    """
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

    # ``digits`` is sometimes used as a numeric range rather than a count.
    # Accept both meanings and preserve zero padding width from the tokens.
    digits_raw = p.get("digits")
    if isinstance(digits_raw, str):
        digits_text = digits_raw.strip()
        rng = _range_from_text(digits_text)
        if rng is not None:
            lo, hi = int(rng[0]), int(rng[1])
            parts = re.fullmatch(r"\s*(\d+)\s*(?:-|\.\.)\s*(\d+)\s*", digits_text)
            if parts:
                lo_token, hi_token = parts.groups()
                width = max(len(lo_token), len(hi_token))
            else:
                width = max(len(str(abs(lo))), len(str(abs(hi))))
            p["digits"] = max(1, width)
            p.setdefault("number_min", lo)
            p.setdefault("number_max", hi)
        else:
            try:
                p["digits"] = max(1, int(float(digits_text)))
            except Exception:
                p.pop("digits", None)

    if "digits" not in p and "length" in p:
        try:
            length = int(p.get("length"))
            prefix_len = len(prefix)
            digits = max(1, length - prefix_len) if prefix_len and length > prefix_len else max(1, length)
            p["digits"] = digits
        except Exception:
            pass

    # Accept min/max, lo/hi, or range for the numeric suffix.
    for lo_key, hi_key in (("min", "max"), ("lo", "hi"), ("number_min", "number_max")):
        if lo_key in p or hi_key in p:
            try:
                if lo_key in p and p.get("number_min") is None:
                    p["number_min"] = int(float(p[lo_key]))
                if hi_key in p and p.get("number_max") is None:
                    p["number_max"] = int(float(p[hi_key]))
            except Exception:
                pass
            break

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
    """Map common SQL/Python/pandas/warehouse dtypes into generator families.

    Parameterized forms such as ``decimal(10,2)`` and ``varchar(255)`` are
    accepted. Unknown types intentionally degrade to ``string`` so a new client
    dtype cannot crash generation merely because the backend does not have a
    dedicated semantic type for it.
    """
    value = re.sub(r"\s+", " ", dtype.strip().lower())
    base = re.sub(r"\s*\(.*\)$", "", value).strip()

    integer_aliases = {
        "int", "integer", "bigint", "smallint", "tinyint", "mediumint",
        "long", "short", "byte", "int8", "int16", "int32", "int64",
        "uint", "uint8", "uint16", "uint32", "uint64", "unsigned integer",
    }
    float_aliases = {
        "float", "double", "double precision", "real", "number", "numeric",
        "float16", "float32", "float64", "decimal", "money", "smallmoney",
    }
    string_aliases = {
        "string", "object", "str", "text", "varchar", "nvarchar", "char",
        "nchar", "clob", "ntext", "character", "character varying",
        "json", "jsonb", "xml", "variant", "sql_variant", "binary", "varbinary",
        "blob", "bytes", "bytea", "base64", "uri", "url", "email",
    }
    datetime_aliases = {
        "timestamp", "timestamp with time zone", "timestamp without time zone",
        "timestamptz", "datetime", "datetime2", "datetime64",
    }
    date_aliases = {"date", "date32", "date64"}
    boolean_aliases = {"bool", "boolean", "bit"}

    if value in {"enum", "category"}:
        return "categorical"
    if base in integer_aliases:
        return "int"
    if base in float_aliases:
        return "float"
    if base in string_aliases:
        return "string"
    if base in datetime_aliases:
        return "datetime"
    if base in date_aliases:
        return "date"
    if base in boolean_aliases:
        return "boolean"
    if base == "uuid":
        return "string"
    if base in {"time", "time with time zone", "timetz", "interval", "duration"}:
        # Preserve these generic temporal values safely as strings unless a
        # dedicated generator is explicitly supplied.
        return "string"
    if base in {"array", "list", "struct", "map", "record", "dict", "dictionary"}:
        return "string"
    if base in ALLOWED_DTYPES:
        return base
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


def _normalize_generator(gen: str, params: dict[str, Any], depends_on: list[str], dtype: str, formula: str, name: str = "") -> tuple[str, dict[str, Any]]:
    """Map client-facing generator vocabulary into executable generators."""
    g = (gen or "").strip().lower()
    p = dict(params)

    if g == "unique_id":
        p = _infer_prefixed_id_params(p)
        # A dependency means linkage, not necessarily ID mirroring.  Only mirror
        # when the CSV explicitly asks for it, or for the conventional account-id
        # pattern where a stable account key commonly shares the subscriber suffix.
        explicit_source = str(p.get("source_field", "")).strip()
        mirror_flag = str(p.get("mirror", "")).strip().lower() in {"true", "1", "yes", "y"}
        field_name = str(name or "").strip().lower()
        source_dep = next((str(dep) for dep in depends_on if str(dep).strip().lower().endswith("_id")), None)
        should_mirror = bool(explicit_source or mirror_flag or (field_name.endswith("account_id") and source_dep))
        if should_mirror and p.get("prefix") is not None:
            source_field = explicit_source or source_dep
            if source_field:
                return "id_mirror", {
                    **p,
                    "source_field": source_field,
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
        # Allow bounds to reference another field, e.g.
        # ``min=recharge_count_30d;max=20``.
        lo_ref = p.get("min", p.get("lo"))
        hi_ref = p.get("max", p.get("hi"))
        field_ref = None
        if isinstance(lo_ref, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lo_ref):
            field_ref = ("lo_field", lo_ref)
        elif isinstance(hi_ref, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", hi_ref):
            field_ref = ("hi_field", hi_ref)
        if field_ref:
            out = dict(p)
            out[field_ref[0]] = field_ref[1]
            out["integer"] = dtype == "int"
            return "uniform_bounded", out
        return "uniform_int" if dtype == "int" else "uniform", p

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
        # Deprecated compatibility alias. Treat it as a generic categorical choice,
        # not as a separate event model.
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
        return "constant", {"value": "unknown"}

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
        # Deprecated compatibility alias. Treat it as a generic derived value.
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

        return "constant", {"value": "unknown"}

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


def _resolve_dependency_aliases(
    variables: list[dict[str, Any]],
    all_names: set[str],
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
    """Infer data grain from an explicit CSV metadata column when present.

    When no type metadata is present, aggregational is the safe default. API callers
    may pass typeOfData explicitly.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty (no header row found)")
    headers = {(h or "").strip().lower() for h in reader.fieldnames}
    for meta_key in ("type_of_data", "typeofdata", "data_type"):
        if meta_key in headers:
            for row in reader:
                raw = (row.get(meta_key) or "").strip().lower()
                if raw in {"transactional", "aggregational"}:
                    return raw
    return "aggregational"


def parse_definition_csv(csv_text: str, type_of_data: str | None = None) -> tuple[list[dict], list[str]]:
    """Parse a CSV schema definition with no event or scope model.

    Every row describes one variable. Transactional history grouping is inferred
    later from the configured entity key and the first history timestamp field.
    Aggregational scenarios use the same schema as a flat record definition.
    """
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

    raw_rows=[]
    # DictReader can silently shift every column when a parameterized dtype such
    # as DECIMAL(10,2) is supplied without CSV quotes. Recover that common export
    # mistake before normalizing fields; valid quoted CSV continues to behave as-is.
    rows_reader = csv.reader(io.StringIO(csv_text))
    next(rows_reader, None)
    expected_columns = len(reader.fieldnames)
    for values in rows_reader:
        values = list(values)
        if not any(str(v).strip() for v in values):
            continue
        if len(values) > expected_columns and expected_columns >= 2:
            dtype_parts = [values[1]]
            idx = 1
            while idx + 1 < len(values) and ")" not in dtype_parts[-1]:
                idx += 1
                dtype_parts.append(values[idx])
            if ")" in dtype_parts[-1] and len(dtype_parts) > 1:
                values = [values[0], ",".join(dtype_parts)] + values[idx + 1:]
        if len(values) < expected_columns:
            values.extend([""] * (expected_columns - len(values)))
        elif len(values) > expected_columns:
            # Preserve the fixed leading schema columns and fold any remaining
            # unquoted tail into the final formula column rather than dropping data.
            values = values[:expected_columns - 1] + [",".join(values[expected_columns - 1:])]
        normalized = {str(k).strip().lower(): str(v or "").strip() for k, v in zip(reader.fieldnames, values)}
        raw_rows.append(normalized)

    requested=str(type_of_data or "").strip().lower()
    if requested and requested not in {"transactional","aggregational"}:
        raise ValueError("typeOfData must be 'transactional' or 'aggregational'")
    detected_type=requested or infer_type_of_data(csv_text)

    known_gens=get_known_generator_types() | {
        "unique_id","indian_msisdn","uuid","timestamp","recent_timestamp","choice","range",
        "dependent_range","derived_distribution","derived_timestamp","derived","datetime",
        "synthetic_event","configuration","derived_state","derived_event","categorical","category",
        "string","text","object","varchar","int","integer","float","decimal","number","numeric",
        "bool","boolean","date",
    }

    variables=[]; field_order=[]; variable_by_name={}; all_names=set()
    for row_number,row in enumerate(raw_rows,start=2):
        name=row.get("name","").strip()
        if not name:
            raise ValueError(f"Row {row_number}: 'name' is empty")
        dtype_raw=row.get("dtype","").strip().lower()
        if not dtype_raw:
            raise ValueError(f"Row {row_number} ('{name}'): dtype is empty")
        dtype=_normalize_dtype(dtype_raw)
        depends_on=_split_list(row.get("depends_on",""))
        params=_parse_params(row.get("params",""),row_number,name)
        formula=row.get("formula","").strip()
        if formula.upper()=="NULL": formula=""

        gen_raw=row.get("gen","").strip().lower()
        # ``gen`` is optional for generic CSV producers. Infer a safe default
        # from dtype/params instead of forcing every producer to know our internal
        # generator vocabulary.
        if not gen_raw:
            if formula:
                gen_raw = "derived"
            elif "choices" in params or "values" in params:
                gen_raw = "choice"
            elif "value" in params:
                gen_raw = "constant"
            elif dtype == "int":
                gen_raw = "range"
            elif dtype == "float":
                gen_raw = "range"
            elif dtype in {"datetime", "date"}:
                gen_raw = "timestamp"
            elif dtype == "boolean":
                gen_raw = "boolean"
            else:
                gen_raw = "string"
        gen,params=_normalize_generator(gen_raw,params,depends_on,dtype,formula,name)
        gen,params=_coerce_executable_generator(gen,params,dtype,formula)
        # Numeric categorical/choice values must remain numeric so downstream formulas
        # (for example recharge_amount -> balance_after) can perform arithmetic.
        if dtype in {"int", "float"} and isinstance(params.get("choices"), list):
            converted=[]
            for choice in params["choices"]:
                try:
                    num=float(choice)
                    if dtype=="int" and num.is_integer(): num=int(num)
                    converted.append(num)
                except (TypeError,ValueError):
                    converted.append(choice)
            params["choices"]=converted
        if dtype in {"int", "float"} and "value" in params:
            try:
                num=float(params["value"])
                params["value"] = int(num) if dtype=="int" and num.is_integer() else num
            except (TypeError,ValueError):
                pass
        variable={
            "name":name,"dtype":dtype,"description":row.get("description",""),"gen":gen,
            "params":params,"depends_on":depends_on,"nullable":row.get("nullable","").lower() in _TRUE_STRINGS,
        }
        normalized_formula=_coerce_formula_text(formula) if formula else ""
        if normalized_formula:
            variable["formula"]=normalized_formula
        elif gen=="formula":
            fallback_gen,fallback_params=_coerce_executable_generator("",params,dtype,"")
            variable["gen"]=fallback_gen; variable["params"]=fallback_params
            if formula: variable.setdefault("params",{})["formula_text"]=formula
        elif formula:
            variable.setdefault("params",{})["formula_text"]=formula

        existing=variable_by_name.get(name)
        if existing is None:
            variable_by_name[name]=variable; variables.append(variable); field_order.append(name); all_names.add(name)
        else:
            if not _merge_duplicate_variable(existing,variable):
                raise ValueError(f"Row {row_number}: duplicate variable name '{name}' has conflicting definition")

    if not variables:
        raise ValueError("CSV must contain at least one variable row")

    _resolve_dependency_aliases(variables,all_names)
    _align_id_mirror_sources(variables,all_names)
    unknown_dependencies=sorted({dep for v in variables for dep in v["depends_on"] if dep not in all_names})
    if unknown_dependencies:
        raise ValueError(f"depends_on references undefined variable(s): {unknown_dependencies}")

    # Transactional definitions must have a usable entity/user key; API import can
    # infer it separately, so the parser only validates that variables exist.
    return variables, field_order

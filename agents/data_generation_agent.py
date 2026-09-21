"""
Agent 4 — Data Generation Agent
Generates records purely algorithmically (no LLM call per record), then
validates them through an internal QA layer. Implemented as a 2-node
LangGraph subgraph: generate -> qa_validate.
"""
from __future__ import annotations
import json
import logging
import math
import os
import ast
import re
import random
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from core.dynamic_scenarios import resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState
from core.generator_contracts import SUPPORTED_GENERATORS
from core.scenario_semantics import temporal_delay_limit_seconds as _temporal_delay_limit_seconds, temporal_role as _temporal_role
from core.deterministic_rules import build_deterministic_rules

logger = logging.getLogger(__name__)

# Canonical numeric dtypes used by the scenario contract and generation QA layer.
# Scenario-definition normalization canonicalizes integer/decimal/uuid types before they
# reach the generation agent, but keeping the aliases here makes the helper
# safe for both normalized and direct callers.
_NUMERIC_DTYPES = {"int", "integer", "float", "decimal", "number", "numeric"}

# Optional LLM QA configuration.  Deterministic QA remains the default; these
# constants are only used when QA_LLM_MODE=full is explicitly enabled.
_CHUNK = max(1, int(os.getenv("QA_LLM_CHUNK_SIZE", "10")))
_QA_SYSTEM = """You are the final QA validator for generated synthetic data.
Validate each supplied record against the FULL confirmed scenario contract and supplied
business/cross-field rules. The CSV is authoritative. Preserve every declared literal
choice/value, numeric min/max, bucket interval, weight distribution, precision, currency,
date/time semantics, dependency, and formula. Never invent a category or normalize a value
into a synonym not declared by params. Field descriptions are semantic constraints and must
not be contradicted. Preserve valid values, repair only clear deterministic violations, and
do not invent fields that are not in the schema. For related datetime fields, enforce the
causal sequence and any declared min/max delay; never accept a child event before its parent
or an absurd gap when the schema says the events are part of one workflow. Also check obvious
state/amount/count/formula contradictions described by the schema. Return JSON with keys:
valid_records, dropped_records, fixes_applied, issues_found.

Schema/business rules:
{rules}

Cross-field rules:
{cross_field_rules}
"""


def _boolean_semantic(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1", "enabled", "on"}:
            return True
        if token in {"false", "no", "n", "0", "disabled", "off"}:
            return False
    return None


def _safe_number(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _semantic_placeholder(name: str, params: dict, rec: dict) -> str:
    """Generate a deterministic semantic value for a generic event/object field.

    Never emit an accidental ``unknown`` solely because a client schema used
    ``gen=event``/``object`` without parameters.
    """
    p = params or {}
    if "value" in p and p.get("value") is not None:
        return str(p.get("value"))
    choices = p.get("choices", p.get("values"))
    if isinstance(choices, (list, tuple)) and choices:
        return str(choices[0])
    field = str(name or rec.get("__current_field__") or "event").strip()
    token = re.sub(r"[^A-Za-z0-9]+", "_", field).strip("_").upper()
    if token.lower().endswith("_id"):
        return _prefixed_int({"prefix": f"{token[:-3]}-", "digits": 10}, rec)
    return f"EVENT_{token[:32]}_{random.randint(1000, 9999)}" if token else f"EVENT_{random.randint(1000, 9999)}"


def _semantic_string(var: dict, rec: dict) -> str | None:
    """Generate a useful synthetic string for legacy/semantic fields without echoing field names."""
    p = dict(var.get("params") or {})
    name = str(var.get("name") or rec.get("__current_field__") or "").strip()
    n = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    if "choices" in p or "values" in p:
        vals = p.get("choices", p.get("values"))
        if isinstance(vals, str):
            vals = [x.strip() for x in re.split(r"[;,|]", vals) if x.strip()]
        if vals:
            return str(random.choice(list(vals)))
    if p.get("value") is not None:
        return str(p["value"])

    if n.endswith("_id") or n == "id" or n.endswith("_key"):
        return _prefixed_int({"prefix": f"{n[:-3].upper()}-" if n.endswith("_id") else "ID-", "digits": 10}, rec)
    if "phone" in n or "mobile" in n or n == "msisdn":
        country = str(p.get("country") or "IN").upper()
        dial_codes = {
            "IN":"+91","US":"+1","CA":"+1","GB":"+44","AU":"+61",
            "AE":"+971","SG":"+65","DE":"+49","FR":"+33","IT":"+39",
        }
        dial = dial_codes.get(country, "+91")
        return _e164_phone({"country_codes":[dial], "country":country}, rec)
    if "email" in n:
        return f"user{random.randint(100000, 999999)}@example.test"
    if any(token in n for token in ("href", "url", "schemalocation", "resourcepath", "path")):
        return f"https://example.test/telecom/{uuid.uuid4().hex[:12]}"
    if "channel" in n:
        return random.choice(["APP", "SMS", "WEB", "USSD", "WHATSAPP", "IVR", "RETAIL"])
    if "payment" in n and ("method" in n or "instrument" in n):
        return random.choice(["UPI", "CREDIT_CARD", "DEBIT_CARD", "WALLET", "CASH", "AUTO_DEBIT"])
    if "reason" in n:
        return random.choice(["LOW_BALANCE", "DATA_EXHAUSTED", "VALIDITY_EXPIRY", "CUSTOMER_REQUEST"])
    if n in {"stateorprovince", "province", "state"} or n.endswith("_state_or_province"):
        return random.choice(["Haryana", "Punjab", "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat"])
    if n in {"city"} or n.endswith("_city"):
        return random.choice(["Delhi", "Gurugram", "Ludhiana", "Chandigarh", "Mumbai", "Bengaluru", "Pune"])
    if n in {"postcode", "postalcode", "postal_code"}:
        return str(random.randint(110001, 999999))
    if "network" in n and "capability" in n:
        return random.choice(["2G", "3G", "4G", "5G"])
    if "status" in n or n.endswith("_state") or n in {"state"}:
        return random.choice(["PENDING", "COMPLETED", "FAILED"])
    if "offer" in n:
        return random.choice(["EXTRA_DATA", "CASH_BACK", "VALIDITY_BOOSTER", "DISCOUNT_VOUCHER"])
    if "segment" in n:
        return random.choice(["ULTRA_LOW", "MASS", "MID_TIER", "HIGH_VALUE"])
    if n in {"role", "partyroletype"} or n.endswith("_role"):
        return random.choice(["subscriber", "customer", "agent", "system"])
    if n == "name" or n.endswith("_name"):
        return random.choice(["Recharge Plan", "Data Booster", "Talktime Pack", "Retention Offer"])
    if n == "title" or n.endswith("_title"):
        return random.choice(["Low Balance Alert", "Recharge Event", "Top-up Request"])
    if n == "description" or n.endswith("_description"):
        return random.choice(["Recharge operation", "Balance adjustment", "Retention intervention"])
    if n in {"type", "basetype", "referredtype"} or n.endswith("_type"):
        return random.choice(["PrepayBalance", "BalanceTopup", "ProductOffering", "Subscriber"])
    if "code" in n:
        return f"CODE-{random.randint(100000, 999999)}"
    if "country" in n:
        return str(p.get("country") or "IN").upper()
    if "currency" in n:
        return str(p.get("currency") or "INR").upper()

    # Last-resort semantic token: it is intentionally not the field name and is clearly synthetic.
    label = re.sub(r"_+", "_", n.upper()).strip("_") or "VALUE"
    return f"SYN_{label[:24]}_{random.randint(1000, 9999)}"


def _generic_value(var: dict, rec: dict):
    """Best-effort generic generator for client vocab not known to the core."""
    p = dict(var.get("params") or {})
    dtype = str(var.get("dtype") or "string").lower()
    name = str(var.get("name") or rec.get("__current_field__") or "")
    if var.get("formula"):
        return _formula(var, rec)
    if "choices" in p or "values" in p:
        vals = p.get("choices", p.get("values"))
        if isinstance(vals, str):
            vals = [x.strip() for x in re.split(r"[;,|]", vals) if x.strip()]
        if vals:
            return random.choice(list(vals))
    if "value" in p and p.get("value") is not None:
        return p.get("value")
    if dtype in _NUMERIC_DTYPES:
        lo = _safe_number(p.get("min", p.get("lo")), 0.0)
        hi = _safe_number(p.get("max", p.get("hi")), 100.0)
        if dtype in {"int", "integer"}:
            return random.randint(int(round(min(lo, hi))), int(round(max(lo, hi))))
        return round(random.uniform(min(lo, hi), max(lo, hi)), int(p.get("precision", 2) or 2))
    if dtype == "boolean":
        return random.choice([True, False])
    if dtype == "datetime":
        return _recent_datetime(p, rec)
    if dtype == "date":
        return _recent_datetime(p, rec)[:10]
    if dtype == "array":
        choices = p.get("choices", p.get("values"))
        if isinstance(choices, (list, tuple)) and choices:
            size = random.randint(0, min(3, len(choices)))
            return random.sample(list(choices), size) if size else []
        return None
    if dtype == "object":
        # Empty objects are not valid synthetic data. A flat dataset cannot invent nested
        # structure safely, so callers must omit/nullable this field or reject the record.
        return None
    return _semantic_string(var, rec)

# ── Generator functions ────────────────────────────────────────────────────────

def _prefixed_int(params: dict, _rec: dict) -> str:
    """Generate a prefixed numeric ID from flexible digit/range encodings.

    Never allow a malformed ``digits`` value to abort the whole transactional
    user. Examples like ``0000001-9999999`` are interpreted as a suffix range
    with seven-character zero padding.
    """
    prefix = str(params.get("prefix", ""))
    raw_digits = params.get("digits", 8)
    digits = 8
    number_min = params.get("number_min", 0)
    number_max = params.get("number_max")

    try:
        digits = max(1, int(float(raw_digits)))
    except (TypeError, ValueError):
        text = str(raw_digits or "").strip()
        match = re.fullmatch(r"(\d+)\s*(?:-|\.\.)\s*(\d+)", text)
        if match:
            lo_text, hi_text = match.groups()
            number_min = int(lo_text)
            number_max = int(hi_text)
            digits = max(len(lo_text), len(hi_text))

    if number_max is None:
        number_max = (10 ** digits) - 1
    try:
        number_min = int(float(number_min))
    except (TypeError, ValueError):
        number_min = 0
    try:
        number_max = int(float(number_max))
    except (TypeError, ValueError):
        number_max = (10 ** digits) - 1

    lo, hi = sorted((number_min, number_max))
    number = random.randint(lo, hi)
    return f"{prefix}{str(number).zfill(digits)}"


def _e164_phone(params: dict, _rec: dict) -> str:
    cc = random.choice(params["country_codes"])
    if cc == "+91":
        # India: 10-digit mobile number, first digit must be 6-9 (TRAI numbering plan)
        first = random.choice("6789")
        rest = "".join(random.choice("0123456789") for _ in range(9))
        return f"{cc}{first}{rest}"
    if cc == "+44":
        # UK: mobile numbers start 7, followed by 9 digits
        rest = "".join(random.choice("0123456789") for _ in range(9))
        return f"{cc}7{rest}"
    if cc == "+971":
        # UAE: mobile prefixes 50/52/54/55/56/58 + 7 digits
        prefix = random.choice(["50", "52", "54", "55", "56", "58"])
        rest = "".join(random.choice("0123456789") for _ in range(7))
        return f"{cc}{prefix}{rest}"
    # Default / US (NANP): NPA (200-999) + NXX (200-999) + 4-digit line number
    npa = random.randint(200, 999)
    nxx = random.randint(200, 999)
    xxxx = random.randint(1000, 9999)
    return f"{cc}{npa}{nxx}{xxxx}"


def _constant(params: dict, _rec: dict):
    return params["value"]


def _dependent_choice(params: dict, rec: dict):
    """Choose a value from a mapping keyed by another generated field."""
    mapping = params.get("mapping") or {}
    depends_on = str(params.get("depends_on_field") or "").strip()
    if not isinstance(mapping, dict) or not mapping:
        return None
    if depends_on and rec.get(depends_on) in mapping:
        return mapping[rec[depends_on]]
    # Deterministic fallback for a malformed/missing dependency while still staying
    # inside the registry-declared mapping.
    return next(iter(mapping.values()))


def _weighted_choice(params: dict, _rec: dict):
    choices = list(params.get("choices", []))
    raw_weights = params.get("weights", [])
    if not choices:
        return None
    try:
        weights = [float(x) for x in (list(raw_weights) if isinstance(raw_weights, (list, tuple)) else [])]
    except (TypeError, ValueError):
        weights = []
    if len(weights) != len(choices) or any((not math.isfinite(w) or w < 0) for w in weights) or sum(weights) <= 0:
        weights = [1.0] * len(choices)
    return random.choices(choices, weights=weights, k=1)[0]


def _weighted_bucket(params: dict, _rec: dict):
    """Choose a numeric bucket by weight, then sample within that bucket."""
    buckets = params.get("buckets") or []
    if not buckets:
        return None
    normalized: list[tuple[float, float]] = []
    for bucket in buckets:
        try:
            lo, hi = bucket
            normalized.append((float(min(lo, hi)), float(max(lo, hi))))
        except (TypeError, ValueError):
            continue
    if not normalized:
        return None
    weights = params.get("weights") or []
    try:
        weights = [float(w) for w in weights]
    except (TypeError, ValueError):
        weights = []
    if len(weights) != len(normalized) or any((not math.isfinite(w) or w < 0) for w in weights) or sum(weights) <= 0:
        weights = [1.0] * len(normalized)
    lo, hi = random.choices(normalized, weights=weights, k=1)[0]
    precision = int(params.get("precision", 2) or 0)
    if precision > 0:
        scale = 10 ** precision
        lo_tick = int(math.ceil(lo * scale))
        hi_tick = int(math.floor(hi * scale))
        if hi_tick < lo_tick:
            return round(lo, precision)
        return random.randint(lo_tick, hi_tick) / scale
    return random.randint(int(math.ceil(lo)), int(math.floor(hi))) if lo.is_integer() and hi.is_integer() else random.uniform(lo, hi)


def _uniform(params: dict, _rec: dict) -> float:
    precision = int(params.get("precision", 2) or 2)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    if lo is None:
        lo = 0.0
    if hi is None:
        hi = lo
    hi = max(lo, hi)
    if precision > 0:
        scale = 10 ** precision
        lo_tick = int(math.ceil(lo * scale))
        hi_tick = int(math.floor(hi * scale))
        if hi_tick >= lo_tick:
            tick = random.randint(lo_tick, hi_tick)
            # Prefer non-integer decimal values when representable at this precision.
            if hi_tick > lo_tick and tick % scale == 0:
                if tick + 1 <= hi_tick:
                    tick += 1
                elif tick - 1 >= lo_tick:
                    tick -= 1
            return tick / scale
    return round(float(random.uniform(lo, hi)), precision)


def _uniform_int(params: dict, _rec: dict) -> int:
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    lo_int = int(round(lo if lo is not None else 0.0))
    hi_int = int(round(hi if hi is not None else lo_int))
    return random.randint(min(lo_int, hi_int), max(lo_int, hi_int))


def _lognormal(params: dict, _rec: dict) -> float:
    precision = int(params.get("precision", 2) or 2)
    mu = _to_finite_float(params.get("mu"), 0.0)
    sigma = _to_finite_float(params.get("sigma"), 1.0)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    raw = math.exp(random.gauss(mu if mu is not None else 0.0, sigma if sigma is not None else 1.0))
    clipped = max(lo if lo is not None else 0.0, min(hi if hi is not None else raw, raw))
    return round(float(clipped), precision)


def _lognormal_int(params: dict, _rec: dict) -> int:
    mu = _to_finite_float(params.get("mu"), 0.0)
    sigma = _to_finite_float(params.get("sigma"), 1.0)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    raw = int(math.exp(random.gauss(mu if mu is not None else 0.0, sigma if sigma is not None else 1.0)))
    return int(max(lo if lo is not None else 0.0, min(hi if hi is not None else raw, raw)))


def _beta(params: dict, _rec: dict) -> float:
    return round(random.betavariate(params["alpha"], params["beta"]), 4)


def _segment_range(params: dict, rec: dict) -> float:
    precision = int(params.get("precision", 4) or 4)
    controller = params.get("field") or params.get("segment_field") or ""
    key = rec.get(controller) if controller else None
    rng = params.get(key) if key in params else params.get("default")
    if not isinstance(rng, dict):
        numeric_ranges = [v for v in params.values() if isinstance(v, dict) and "min" in v and "max" in v]
        rng = numeric_ranges[0] if numeric_ranges else {"min": 0, "max": 1}
    return round(float(random.uniform(float(rng.get("min", 0)), float(rng.get("max", 1)))), precision)


def _to_finite_float(value, default: float | None = None) -> float | None:
    """Coerce numeric-looking values safely for dependent generators.

    Scenario definitions are persisted as confirmed contracts, so numeric params may be
    strings. Dependent fields can also be represented as strings (for example
    ``"20.0"``). Never pass a raw string into random.uniform/max.
    """
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _uniform_bounded(params: dict, rec: dict):
    """Generate a value with one or both bounds optionally coming from fields."""
    precision = int(params.get("precision", 2) or 2)
    hi = _to_finite_float(rec.get(params.get("hi_field")), None) if params.get("hi_field") else None
    lo = _to_finite_float(rec.get(params.get("lo_field")), None) if params.get("lo_field") else None
    if hi is None:
        hi = _to_finite_float(params.get("hi", params.get("max")), 1.00)
    if lo is None:
        lo = _to_finite_float(params.get("lo", params.get("min")), 0.00)
    if lo is None:
        lo = 0.00
    if hi is None:
        hi = lo
    lo, hi = min(lo, hi), max(lo, hi)
    if params.get("integer"):
        return random.randint(int(math.ceil(lo)), int(math.floor(hi)))
    return round(float(random.uniform(lo, hi)), precision)


def _recent_datetime(params: dict, _rec: dict) -> str:
    days_back = int(params.get("days_back", 0) or 0)
    base = datetime.now(timezone.utc) - timedelta(
        days=random.randint(0, max(0, days_back)),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return _format_datetime(base, params)


def _ts_offset(params: dict, rec: dict) -> str:
    base_str = rec.get(params["base_field"], datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    offset = timedelta(seconds=random.randint(params["min_sec"], params["max_sec"]))
    return _format_datetime(base + offset, params)


def _ts_add_field(params: dict, rec: dict) -> str:
    base_str = rec.get(params["base_field"], datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    seconds = int(rec.get(params["add_seconds_field"], 60))
    return _format_datetime(base + timedelta(seconds=seconds), params)


def _date_offset(params: dict, rec: dict) -> str:
    base_str = rec.get(params["base_field"], datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    return (base + timedelta(days=params["days"])).date().isoformat()


def _date_offset_range(params: dict, rec: dict) -> str:
    base_str = rec.get(params["base_field"], datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    min_days = int(params.get("min_days", 0))
    max_days = int(params.get("max_days", min_days))
    offset_days = random.randint(min(min_days, max_days), max(min_days, max_days))
    return (base + timedelta(days=offset_days)).date().isoformat()


def _id_mirror(params: dict, rec: dict) -> str:
    """Copy the numeric suffix from source_field and attach a new prefix."""
    prefix = str(params.get("prefix", ""))
    source_field = str(params.get("source_field", ""))
    source_prefix = str(params.get("source_prefix", ""))
    source = str(rec.get(source_field, "") or "")

    # Prefer a trailing numeric suffix so dependent IDs stay aligned.
    number = ""
    match = re.search(r"(\d+)$", source)
    if match:
        number = match.group(1)
    elif source_prefix and source.startswith(source_prefix):
        number = source[len(source_prefix):]

    if not number:
        digits = int(params.get("digits", 8) or 8)
        number = str(random.randint(0, max(0, 10**digits - 1))).zfill(digits)
    return f"{prefix}{number}"


def _prefixed_uuid(params: dict, _rec: dict) -> str:
    raw = str(uuid.uuid4())           # 8-4-4-4-12
    suffix = raw[len(params["prefix"]):]
    return params["prefix"] + suffix


def _tx_id(params: dict, rec: dict) -> str:
    ts = rec.get("record_timestamp", datetime.now(timezone.utc).isoformat())
    date_part = ts[:10].replace("-", "")
    rand_part = random.randint(1_000_000, 9_999_999)
    return f"{params['prefix']}{date_part}-{rand_part}"


def _formula(var: dict, rec: dict):
    """Evaluate a constrained formula language against the current record."""
    expr = str(var.get("formula", "") or "").strip()
    if not expr:
        return None
    return _safe_formula(expr, rec)


DEFAULT_TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M %p"

_TIMESTAMP_FORMAT_ALIASES = {
    "dd/mm/yyyy hh:mm a": DEFAULT_TIMESTAMP_FORMAT,
    "dd/mm/yyyy hh:mm am/pm": DEFAULT_TIMESTAMP_FORMAT,
    "dd/mm/yyyy hh:mm:ss a": "%d/%m/%Y %I:%M:%S %p",
    "yyyy-mm-dd hh:mm:ss": "%Y-%m-%d %H:%M:%S",
    "yyyy-mm-dd hh:mm": "%Y-%m-%d %H:%M",
    "iso": "ISO",
    "iso-8601": "ISO",
    "date-time": DEFAULT_TIMESTAMP_FORMAT,
    "datetime": DEFAULT_TIMESTAMP_FORMAT,
    "timestamp": DEFAULT_TIMESTAMP_FORMAT,
}

def _normalize_timestamp_format(params: dict | None = None) -> str:
    params = params if isinstance(params, dict) else {}
    raw = params.get("timestamp_format", params.get("format"))
    if raw is None or not str(raw).strip():
        return DEFAULT_TIMESTAMP_FORMAT
    text = str(raw).strip()
    # Match both exact strftime tokens and client-friendly case-insensitive aliases.
    if text in {"ISO", "ISO-8601"}:
        return "ISO"
    return _TIMESTAMP_FORMAT_ALIASES.get(text.lower(), text)


def _parse_timestamp_text(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    iso_text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_text)
    except Exception:
        pass
    for fmt in (
        DEFAULT_TIMESTAMP_FORMAT,
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %I:%M %p",
        "%m/%d/%Y %I:%M %p",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None

def _format_datetime(dt: datetime, params: dict | None = None) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    fmt = _normalize_timestamp_format(params)
    if fmt.upper() in {"ISO", "ISO-8601"}:
        return dt.isoformat()
    try:
        return dt.strftime(fmt)
    except (TypeError, ValueError):
        return dt.strftime(DEFAULT_TIMESTAMP_FORMAT)

def _format_datetime_fields(rec: dict, variables: list[dict]) -> dict:
    """Serialize all datetime fields according to their confirmed contract format.

    Generation and formula evaluation may use real datetime objects internally;
    this helper is the single deterministic boundary that converts them to the
    client-facing representation for both raw_records and final_records.
    """
    out = dict(rec)
    for var in variables:
        name = str(var.get("name", ""))
        if not name or name not in out or out.get(name) is None:
            continue
        if str(var.get("dtype", "")).strip().lower() != "datetime":
            continue
        dt = _qa_parse_dt(out.get(name))
        if dt is not None:
            out[name] = _format_datetime(dt, var.get("params") or {})
    return out

def _parse_dt(s: str) -> datetime:
    """Parse ISO-8601 and common human-readable timestamps to timezone-aware datetime."""
    parsed = _parse_timestamp_text(str(s))
    if parsed is None:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ── Dispatch table ─────────────────────────────────────────────────────────────

_GENERATORS = {
    "prefixed_int":   lambda v, rec: _prefixed_int(v["params"], rec),
    "id_mirror":      lambda v, rec: _id_mirror(v["params"], rec),
    "e164_phone":     lambda v, rec: _e164_phone(v["params"], rec),
    "constant":       lambda v, rec: _constant(v["params"], rec),
    "weighted_choice":lambda v, rec: _weighted_choice(v["params"], rec),
    "dependent_choice":lambda v, rec: _dependent_choice(v["params"], rec),
    "weighted_bucket":lambda v, rec: _weighted_bucket(v["params"], rec),
    "uniform":        lambda v, rec: _uniform(v["params"], rec),
    "uniform_int":    lambda v, rec: _uniform_int(v["params"], rec),
    "lognormal":      lambda v, rec: _lognormal(v["params"], rec),
    "lognormal_int":  lambda v, rec: _lognormal_int(v["params"], rec),
    "beta":           lambda v, rec: _beta(v["params"], rec),
    "segment_range":  lambda v, rec: _segment_range(v["params"], rec),
    "uniform_bounded":lambda v, rec: _uniform_bounded(v["params"], rec),
    "recent_datetime":lambda v, rec: _recent_datetime(v["params"], rec),
    "ts_offset":      lambda v, rec: _ts_offset(v["params"], rec),
    "ts_add_field":   lambda v, rec: _ts_add_field(v["params"], rec),
    "date_offset":    lambda v, rec: _date_offset(v["params"], rec),
    "date_offset_range": lambda v, rec: _date_offset_range(v["params"], rec),
    "prefixed_uuid":  lambda v, rec: _prefixed_uuid(v["params"], rec),
    "tx_id":          lambda v, rec: _tx_id(v["params"], rec),
    "formula":        lambda v, rec: _formula(v, rec),
    "generic":        lambda v, rec: _generic_value(v, rec),
    "semantic_event": lambda v, rec: _semantic_placeholder(v.get("name"), v.get("params") or {}, rec),
    "semantic_string": lambda v, rec: _semantic_string(v, rec),
}



def _rule_constraint_for(field_name: str, rules: dict | None) -> dict:
    """Return machine-readable generation constraints for a field.

    SchemaAgent produces these once per confirmed scenario. Keeping this lookup
    deterministic means use-case/business-context rules influence generation
    without making an LLM call for every record.
    """
    if not isinstance(rules, dict):
        return {}
    constraints = rules.get("generation_constraints", {})
    if not isinstance(constraints, dict):
        return {}
    value = constraints.get(field_name, {})
    return value if isinstance(value, dict) else {}


def _coerce_rule_values(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # SchemaAgent normally returns a JSON list, but tolerate compact text.
        for sep in ("|", ";", ","):
            if sep in text:
                return [x.strip() for x in text.split(sep) if x.strip()]
        return [text]
    return []


def _declared_param_options(params: dict) -> list[Any]:
    """Return authoritative value options declared in params, if any."""
    if not isinstance(params, dict):
        return []
    choices = params.get("choices")
    if isinstance(choices, list) and choices:
        return list(choices)
    values = params.get("values")
    if isinstance(values, list) and values:
        return list(values)
    if values is not None and not isinstance(values, list):
        return [values]
    if "value" in params:
        return [params.get("value")]
    return []


def _matches_declared_option(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return _boolean_semantic(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        actual_num = _to_finite_float(actual, None)
        return actual_num is not None and math.isclose(actual_num, float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return _normalize(actual) == _normalize(expected)


def _event_like_field(var: dict) -> bool:
    name = str(var.get("name") or "").strip().lower()
    dtype = str(var.get("dtype") or "").strip().lower()
    desc = str(var.get("description") or "").strip().lower()
    return (
        dtype in {"event", "event_type", "event_container"}
        or name.endswith("_event")
        or " event " in f" {desc} "
    )


def _event_fallback_value(var: dict) -> str:
    name = str(var.get("name") or "event").strip()
    token = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return token or "EVENT"


def _apply_generation_constraint(var: dict, value, rec: dict, rules: dict | None):
    """Apply safe machine-readable SchemaAgent constraints to a generated value."""
    constraint = _rule_constraint_for(str(var.get("name", "")), rules)
    if value is None:
        return value

    params = var.get("params") if isinstance(var.get("params"), dict) else {}
    dtype = str(var.get("dtype", "")).strip().lower()
    precision = int(params.get("precision", 2) or 2)

    # Never allow the generic/LLM constraint layer to reintroduce the literal
    # placeholder "unknown" for semantic event fields. Event fields without an
    # explicit value are represented by a deterministic event token instead.
    if _event_like_field(var) and isinstance(value, str) and value.strip().lower() == "unknown":
        value = _event_fallback_value(var)

    # Params are authoritative: if explicit values are provided, do not emit
    # anything outside that set. Preserve the original token/casing from params.
    declared_options = _declared_param_options(params)
    if declared_options:
        for opt in declared_options:
            if _matches_declared_option(value, opt):
                return opt
        # Confirmed choices are authoritative as the allowed set, while scenario-derived
        # preferred_values select the semantically appropriate member of that set.
        preferred = _coerce_rule_values(constraint.get("preferred_values")) if constraint else []
        if _event_like_field(var):
            preferred = [x for x in preferred if str(x).strip().lower() != "unknown"]
        if preferred:
            preferred_allowed = [
                opt for opt in declared_options
                if any(_matches_declared_option(opt, wanted) for wanted in preferred)
            ]
            if preferred_allowed:
                return preferred_allowed[0]
        return random.choice(declared_options)

    if dtype in {"float", "decimal", "number", "numeric"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = round(float(value), precision)

    if not constraint:
        return value

    numeric_param_bounded = (
        dtype in {"float", "decimal", "number", "numeric", "int", "integer"}
        and any(k in params for k in ("min", "max", "lo", "hi"))
    )
    if numeric_param_bounded:
        return value

    allowed = _coerce_rule_values(constraint.get("preferred_values"))
    if not allowed:
        allowed = _coerce_rule_values(constraint.get("valid_values"))
    if _event_like_field(var):
        allowed = [x for x in allowed if str(x).strip().lower() != "unknown"]
    if allowed and not numeric_param_bounded:
        # Match case/format while preserving the canonical value supplied by rules.
        norm = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        matches = [x for x in allowed if str(x).strip().lower().replace("-", "_").replace(" ", "_") == norm]
        if matches:
            return matches[0]
        # If the generator produced a value outside an authoritative categorical
        # constraint, choose from the constrained set instead of leaking invalid data.
        return random.choice(allowed)

    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lo = constraint.get("min")
            hi = constraint.get("max")
            if lo is not None:
                value = max(value, float(lo))
            if hi is not None:
                value = min(value, float(hi))
            if isinstance(value, float):
                value = round(value, 2)
    except (TypeError, ValueError):
        pass
    return value


def _apply_scenario_semantics(rec: dict, rules: dict | None) -> dict:
    """Apply schema-driven scenario-type guardrails after generic generation."""
    if not isinstance(rules, dict):
        return rec
    semantics = rules.get("scenario_semantics")
    if not isinstance(semantics, dict):
        return rec
    for field in semantics.get("force_true_fields", []) or []:
        if field in rec:
            rec[field] = True
    for field in semantics.get("force_false_fields", []) or []:
        if field in rec:
            rec[field] = False
    for field, preferred in (semantics.get("preferred_values") or {}).items():
        if field in rec and isinstance(preferred, list) and preferred:
            rec[field] = preferred[0]
    return rec


def _apply_conditional_rules(rec: dict, rules: dict | None) -> dict:
    """Apply structured cross-field rules deterministically.

    Rules are intentionally industry-agnostic. SchemaAgent may return:
      {"when": {"field": value}, "then": {"dependent": value_or_list}}
    or the equivalent ``conditions``/``set`` keys. Multiple passes allow chained
    dependencies to settle. A condition is only applied when every referenced
    controller field is present and matches semantically.
    """
    if not isinstance(rules, dict):
        return rec
    raw = rules.get("conditional_rules") or rules.get("relationship_rules") or []
    if not isinstance(raw, list):
        return rec

    def norm(v):
        return str(v).strip().lower().replace("-", "_").replace(" ", "_")

    def matches(actual, expected):
        if isinstance(expected, list):
            return any(matches(actual, x) for x in expected)
        if isinstance(expected, bool):
            return _boolean_semantic(actual) is expected
        try:
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                return _to_finite_float(actual, None) == float(expected)
        except Exception:
            pass
        return norm(actual) == norm(expected)

    def condition_matches(when):
        if not isinstance(when, dict):
            return False
        for field, expected in when.items():
            if field not in rec or rec.get(field) is None or not matches(rec.get(field), expected):
                return False
        return True

    def set_dependent(field, desired):
        if field not in rec:
            return False
        if isinstance(desired, dict):
            if "value" in desired:
                desired = desired["value"]
            elif "values" in desired:
                desired = desired["values"]
            elif "valid_values" in desired:
                desired = desired["valid_values"]
        if isinstance(desired, list):
            if not desired:
                return False
            if not any(matches(rec.get(field), x) for x in desired):
                rec[field] = random.choice(desired)
                return True
            return False
        if not matches(rec.get(field), desired):
            rec[field] = desired
            return True
        return False

    for _ in range(max(2, len(raw) * 2)):
        changed = False
        for rule in raw:
            if not isinstance(rule, dict):
                continue
            when = rule.get("when") or rule.get("conditions")
            then = rule.get("then") or rule.get("set") or rule.get("dependent_values")
            if condition_matches(when) and isinstance(then, dict):
                for field, desired in then.items():
                    changed = set_dependent(str(field), desired) or changed
        if not changed:
            break
    return rec





def _formula_from_rules(field_name: str, rules: dict | None):
    if not isinstance(rules, dict):
        return None
    for item in rules.get("formula_rules", []) or []:
        if isinstance(item, dict) and str(item.get("field", "")) == field_name and item.get("expression"):
            return str(item["expression"])
    return None


def _formula_dependencies(expression: str) -> set[str]:
    """Extract field names referenced by a simple formula expression."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                and node.id not in {"round", "min", "max", "abs", "sum", "True", "False", "None"}}
    except Exception:
        return set()


def _variable_dependency_order(
    variables: list[dict],
    selected_names: set[str] | None = None,
    rules: dict | None = None,
) -> tuple[list[dict], set[str]]:
    """Return variables in dependency order and identify dependency cycles.

    Scenario dependencies may point forward.  A topological pass makes those definitions
    executable regardless of row order.  Cycles cannot be solved deterministically;
    callers can generate a seed value for cyclic formula fields and let downstream
    formulas derive from it.
    """
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    selected = set(selected_names or by_name) & set(by_name)
    changed = True
    while changed:
        changed = False
        for name in tuple(selected):
            dep_names = list(by_name[name].get("depends_on", []) or [])
            expr = by_name[name].get("formula") or _formula_from_rules(name, rules)
            if expr:
                for dep in _formula_dependencies(str(expr)):
                    if dep not in dep_names:
                        dep_names.append(dep)
            for dep in dep_names:
                if dep in by_name and dep not in selected:
                    selected.add(dep)
                    changed = True

    indegree = {name: 0 for name in selected}
    outgoing = {name: [] for name in selected}
    for name in selected:
        dep_names = list(by_name[name].get("depends_on", []) or [])
        expr = by_name[name].get("formula") or _formula_from_rules(name, rules)
        if expr:
            for dep in _formula_dependencies(str(expr)):
                if dep not in dep_names:
                    dep_names.append(dep)
        for dep in dep_names:
            if dep in selected:
                indegree[name] += 1
                outgoing[dep].append(name)
    queue = [name for name in selected if indegree[name] == 0]
    queue.sort(key=lambda n: list(by_name).index(n))
    ordered_names: list[str] = []
    while queue:
        name = queue.pop(0)
        ordered_names.append(name)
        for child in outgoing[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
        queue.sort(key=lambda n: list(by_name).index(n))
    cyclic = selected - set(ordered_names)
    ordered = [by_name[name] for name in ordered_names]
    # Keep cyclic variables at the end so their generators can provide a seed.
    ordered.extend(by_name[name] for name in by_name if name in cyclic)
    return ordered, cyclic


def _generate_record(variables: list[dict], rules: dict | None = None) -> dict:
    """Generate one record in dependency order, while safely handling cycles."""
    rec: dict = {}
    ordered, cyclic = _variable_dependency_order(variables, rules=rules)
    known_fields = {str(v.get("name")) for v in variables if v.get("name")}
    for var in ordered:
        gen_type = var["gen"]
        effective_var = var
        # A formula is authoritative when it can be evaluated.  For a dependency
        # cycle, seed the cyclic field from its declared generator so the remaining
        # fields can still be generated and QA can evaluate any resolvable formulas.
        rule_formula = None if var["name"] in cyclic else (var.get("formula") or _formula_from_rules(var["name"], rules))
        if rule_formula:
            deps = _formula_dependencies(str(rule_formula))
            available = known_fields | set(rec.keys())
            if deps and any(dep not in available for dep in deps):
                # Keep the declared generator when formula text references symbolic
                # tokens that are not schema fields (common in human-readable scenario definitions).
                rule_formula = None
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        generator = _GENERATORS.get(effective_var.get("gen"))
        if generator:
            helper_rec = dict(rec)
            helper_rec["__current_field__"] = var["name"]
            value = generator(effective_var, helper_rec)
        else:
            value = None
        rec[var["name"]] = _apply_generation_constraint(var, value, rec, rules)
    rec = _apply_conditional_rules(rec, rules)
    rec = _apply_scenario_semantics(rec, rules)
    rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
    rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
    rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
    rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
    rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
    rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
    return _format_datetime_fields(rec, variables)


def _generate_selected_record(
    variables: list[dict],
    selected_names: set[str],
    base: dict | None = None,
    rules: dict | None = None,
) -> dict:
    """Generate selected variables plus dependencies in dependency order."""
    rec = dict(base or {})
    ordered, cyclic = _variable_dependency_order(variables, selected_names, rules=rules)
    known_fields = {str(v.get("name")) for v in variables if v.get("name")}
    for var in ordered:
        name = var["name"]
        if name in rec:
            continue
        effective_var = var
        rule_formula = None if name in cyclic else (var.get("formula") or _formula_from_rules(name, rules))
        if rule_formula:
            deps = _formula_dependencies(str(rule_formula))
            available = known_fields | set(rec.keys())
            if deps and any(dep not in available for dep in deps):
                rule_formula = None
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        generator = _GENERATORS.get(effective_var.get("gen"))
        if generator:
            helper_rec = dict(rec)
            helper_rec["__current_field__"] = name
            value = generator(effective_var, helper_rec)
        else:
            value = None

        declared_dtype = str(var.get("dtype", "")).strip().lower()
        try:
            if declared_dtype in {"int", "integer"} and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = int(round(value))
            elif declared_dtype in {"float", "decimal", "number", "numeric"} and isinstance(value, (int, float)) and not isinstance(value, bool):
                precision = int((var.get("params") or {}).get("precision", 2) or 2)
                value = round(float(value), precision)
        except (TypeError, ValueError, OverflowError):
            pass
        rec[name] = _apply_generation_constraint(var, value, rec, rules)
    rec = _apply_conditional_rules(rec, rules)
    rec = _apply_scenario_semantics(rec, rules)
    rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
    rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
    rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
    rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
    rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
    rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
    return _format_datetime_fields(rec, variables)


# ── Transactional/user-history generation helpers ─────────────────────────────

def _pick_timestamp_field(variables: list[dict]) -> str | None:
    preferred=("record_timestamp","transaction_timestamp","timestamp","created_at","updated_at")
    names={str(v.get("name")):v for v in variables if v.get("name")}
    for name in preferred:
        if name in names and str(names[name].get("dtype","")).lower() in {"datetime","date"}:
            return name
    for v in variables:
        if str(v.get("dtype","")).lower()=="datetime" and v.get("name"):
            return str(v["name"])
    return None


def _enforce_mandatory_telecom_identity(
    user_context: dict,
    variables: list[dict],
    *,
    country: str | None = None,
    used_values: dict[str, set[str]] | None = None,
) -> dict:
    """Repair mandatory telecom identity anchors before transactional grouping.

    Confirmed scenarios can outlive compiler changes and may still contain legacy generic
    generators that emit placeholders such as ``SUBSCRIBER_ID``, ``MSISDN`` or ``{}``.
    These fields are part of the API contract, so generation must normalize them regardless
    of the generator stored in the older confirmed draft.
    """
    out=dict(user_context)
    names={str(v.get("name")) for v in variables if v.get("name")}
    used_values = used_values if used_values is not None else {
        "subscriber_id": set(), "account_id": set(), "msisdn": set()
    }

    def unique_prefixed(prefix: str, digits: int, field: str) -> str:
        for _ in range(1000):
            candidate=_prefixed_int({"prefix":prefix, "digits":digits}, out)
            if candidate not in used_values[field]:
                used_values[field].add(candidate)
                return candidate
        raise RuntimeError(f"Unable to generate a unique {field}")

    if "subscriber_id" in names:
        current=str(out.get("subscriber_id") or "")
        if not re.fullmatch(r"SUB-[0-9]+", current) or current in used_values["subscriber_id"]:
            current=unique_prefixed("SUB-", 10, "subscriber_id")
        else:
            used_values["subscriber_id"].add(current)
        out["subscriber_id"]=current

    if "account_id" in names:
        subscriber=str(out.get("subscriber_id") or "")
        suffix=re.search(r"([0-9]+)$", subscriber)
        current=f"ACC-{suffix.group(1)}" if suffix else ""
        if (not current) or current in used_values["account_id"]:
            current=unique_prefixed("ACC-", 10, "account_id")
        else:
            used_values["account_id"].add(current)
        out["account_id"]=current

    if "msisdn" in names:
        iso=str(country or "IN").strip().upper()
        dial_codes={
            "IN":"+91","US":"+1","CA":"+1","GB":"+44","AU":"+61",
            "AE":"+971","SG":"+65","DE":"+49","FR":"+33","IT":"+39",
        }
        dial=dial_codes.get(iso, iso if iso.startswith("+") else "+"+iso)
        current=str(out.get("msisdn") or "")
        # Require an E.164-like value for the public MSISDN contract.
        valid=bool(re.fullmatch(r"\+[1-9][0-9]{6,14}", current))
        if (not valid) or current in used_values["msisdn"]:
            for _ in range(1000):
                candidate=_e164_phone({"country_codes":[dial], "country":iso}, out)
                if candidate not in used_values["msisdn"]:
                    current=candidate
                    break
            else:
                raise RuntimeError("Unable to generate a unique msisdn")
        used_values["msisdn"].add(current)
        out["msisdn"]=current

    return out


def _transactional_records(compiled, user_count: int, records_per_user: int = 10,
                            rules: dict | None = None, record_errors_out: list[dict] | None = None,
                            country: str | None = None) -> list[dict]:
    """Generate a fixed-length recent history for each user/entity.

    Stable user-context variables are generated once and copied into each row.
    History variables are regenerated for every historical row. The output remains flat so
    downstream QA operates on ordinary records; the API groups those records by
    entity_key after generation.
    """
    variables=list(compiled.variables); entity_key=compiled.entity_key
    records_per_user=max(1,min(50,int(records_per_user or 10)))
    timestamp_field=_pick_timestamp_field(variables)
    generated=[]
    used_entity_keys=set()
    used_identity_values={name:set() for name in ("subscriber_id", "account_id", "msisdn")}

    for user_index in range(user_count):
        try:
            user_context=_generate_selected_record(variables,set(compiled.user_fields),rules=rules)
            if entity_key and entity_key not in user_context:
                # Ensure the entity key is generated even if inferred user context omitted it.
                key_var=compiled.variable_by_name.get(entity_key)
                if key_var:
                    user_context=_generate_selected_record(variables,{entity_key},base=user_context,rules=rules)
            # Repair application-level telecom identity anchors before grouping. This is
            # intentionally independent of the confirmed draft's stored generators so old
            # scenarios cannot collapse all requested users into a single subscriber.
            user_context=_enforce_mandatory_telecom_identity(
                user_context,
                variables,
                country=country,
                used_values=used_identity_values,
            )

            if entity_key and entity_key in user_context:
                attempts=0
                while str(user_context[entity_key]) in used_entity_keys and attempts < 100:
                    key_var=compiled.variable_by_name.get(entity_key)
                    if key_var:
                        user_context=_generate_selected_record(variables,{entity_key},base=user_context,rules=rules)
                    attempts+=1
                used_entity_keys.add(str(user_context.get(entity_key)))

            # Enforce unique stable telecom identity anchors at the user/entity level.
            # This prevents response grouping from collapsing multiple requested users into
            # one object when a legacy/edited draft contains constant placeholder generators.
            variable_names={str(v.get("name")) for v in variables if v.get("name")}
            for identity_name in ("subscriber_id", "account_id", "msisdn"):
                if identity_name not in variable_names:
                    continue
                identity_value=str(user_context.get(identity_name) or "")
                attempts=0
                while identity_value in used_identity_values[identity_name] and attempts < 100:
                    if identity_name == "account_id":
                        user_context=_generate_selected_record(
                            variables,{"subscriber_id","account_id"},base=user_context,rules=rules
                        )
                    else:
                        user_context=_generate_selected_record(
                            variables,{identity_name},base=user_context,rules=rules
                        )
                    identity_value=str(user_context.get(identity_name) or "")
                    attempts+=1
                if identity_value:
                    used_identity_values[identity_name].add(identity_value)
        except Exception as exc:
            err={"user_index":user_index,"error":str(exc),"record":{}}
            if record_errors_out is not None: record_errors_out.append(err)
            logger.warning("[DataGeneration] Skipping transactional user %d: %s",user_index,exc)
            continue

        # Generate timestamps for the history window first, then sort newest-first
        # at the API boundary. Keep the period bounded to the last 90 days.
        end_ts=datetime.now(timezone.utc)
        span=max(records_per_user,1)-1
        if span:
            step_seconds=random.randint(6*3600, 72*3600)
            start_ts=end_ts-timedelta(seconds=step_seconds*span)
        else:
            start_ts=end_ts
        timestamps=[]
        if timestamp_field:
            for i in range(records_per_user):
                jitter=random.randint(0,max(60,min(6*3600,step_seconds if span else 60)))
                ts=start_ts+timedelta(seconds=(step_seconds*i if span else 0)+jitter)
                ts=min(ts,end_ts)
                timestamps.append(ts)
            timestamps=sorted(timestamps)

        for record_index in range(records_per_user):
            try:
                base=dict(user_context)
                if timestamp_field:
                    base[timestamp_field]=timestamps[record_index].isoformat()
                row=_generate_selected_record(variables,set(compiled.record_fields),base=base,rules=rules)
                if timestamp_field and timestamp_field not in row:
                    row[timestamp_field]=timestamps[record_index].isoformat()
                generated.append(row)
            except Exception as exc:
                err={"user_index":user_index,"record_index":record_index,"error":str(exc),"record":dict(user_context)}
                if record_errors_out is not None: record_errors_out.append(err)
                logger.warning("[DataGeneration] Skipping transactional record user=%d record=%d: %s",user_index,record_index,exc)
    return generated


def _default_for_dtype(dtype: str):
    if dtype in _NUMERIC_DTYPES:
        return 0
    if dtype == "boolean":
        return False
    return ""


def _qa_parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = _parse_timestamp_text(value)
        if dt is None:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt



def _safe_formula(expr: str, rec: dict):
    """Evaluate the same small arithmetic expression language used by the generator."""
    def DATE(value):
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        parsed = _qa_parse_dt(value)
        if parsed is not None:
            return parsed.date()
        try:
            return date.fromisoformat(str(value)[:10])
        except Exception:
            return None

    allowed_funcs = {"round": round, "min": min, "max": max, "abs": abs, "DATE": DATE}
    names = {}
    for k, v in rec.items():
        if k == "__current_field__" or v is None:
            continue
        if isinstance(v, str):
            parsed = _qa_parse_dt(v)
            names[k] = parsed if parsed is not None else v
        else:
            names[k] = v
    try:
        tree = ast.parse(expr, mode="eval")
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
            ast.Name, ast.Call, ast.Load, ast.Tuple, ast.List,
            ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.BoolOp, ast.And, ast.Or, ast.Not,
            ast.IfExp,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Name) and node.id not in names and node.id not in allowed_funcs:
                return None
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs):
                return None
        result = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {**names, **allowed_funcs})
        if isinstance(result, timedelta):
            return result.total_seconds()
        return result
    except Exception:
        return None


def _normalize(v: Any) -> str:
    return str(v).strip().lower().replace(" ", "_").replace("-", "_")





def _infer_temporal_relationships(
    variables: list[dict], rules: dict | None = None
) -> list[tuple[str, str, int | None, int]]:
    """Return authoritative/derived parent -> child datetime relationships.

    Sources, in order of authority:
      1. explicit SchemaAgent temporal_rules;
      2. confirmed-contract depends_on edges between datetime fields.

    A temporal rule can therefore protect a relationship even when the CSV expresses
    it in its description/business semantics rather than as depends_on.
    """
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    out: list[tuple[str, str, int | None, int]] = []
    seen: set[tuple[str, str]] = set()

    for item in (rules or {}).get("temporal_rules", []) or []:
        if not isinstance(item, dict):
            continue
        parent = str(item.get("before", ""))
        child = str(item.get("after", ""))
        if not parent or not child or parent == child:
            continue
        parent_var = by_name.get(parent)
        child_var = by_name.get(child)
        if not parent_var or not child_var:
            continue
        if str(parent_var.get("dtype", "")).strip().lower() != "datetime":
            continue
        if str(child_var.get("dtype", "")).strip().lower() != "datetime":
            continue
        max_gap = None
        min_gap = 0
        try:
            if item.get("max_delay_seconds") is not None:
                max_gap = max(0, int(float(item.get("max_delay_seconds"))))
        except (TypeError, ValueError):
            max_gap = None
        try:
            if item.get("min_delay_seconds") is not None:
                min_gap = max(0, int(float(item.get("min_delay_seconds"))))
        except (TypeError, ValueError):
            min_gap = 0
        key = (parent, child)
        if key not in seen:
            seen.add(key)
            out.append((parent, child, max_gap, min_gap))

    # CSV dependency edges are causal by construction for datetime dependencies.
    for child_name, child in by_name.items():
        if str(child.get("dtype", "")).strip().lower() != "datetime":
            continue
        for dep in child.get("depends_on", []) or []:
            parent_name = str(dep)
            parent = by_name.get(parent_name)
            if not parent or str(parent.get("dtype", "")).strip().lower() != "datetime":
                continue
            key = (parent_name, child_name)
            if key in seen:
                continue
            seen.add(key)
            out.append((parent_name, child_name, _temporal_delay_limit_seconds(child, parent), 0))

    return out


def _enforce_temporal_consistency(
    rec: dict, variables: list[dict], rules: dict | None = None
) -> tuple[dict, list[str]]:
    """Repair causal timestamp contradictions deterministically.

    The pass is intentionally iterative because chains such as
    offer -> decision -> processing -> completion can require more than one repair.
    Explicit min/max delay windows are respected; otherwise CSV datetime dependencies
    receive conservative, domain-neutral ceilings to prevent absurd month/year gaps.
    Unrelated timestamps are never reordered.
    """
    rec = dict(rec)
    issues: list[str] = []
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    relations = _infer_temporal_relationships(variables, rules=rules)
    if not relations:
        return rec, issues

    max_passes = max(2, len(relations) * 2)
    for _ in range(max_passes):
        changed = False
        for parent_name, child_name, max_gap, min_gap in relations:
            parent_dt = _qa_parse_dt(rec.get(parent_name))
            child_dt = _qa_parse_dt(rec.get(child_name))
            if parent_dt is None or child_dt is None:
                continue

            desired = child_dt
            lower_bound = parent_dt + timedelta(seconds=min_gap)
            upper_bound = parent_dt + timedelta(seconds=max_gap) if max_gap is not None else None

            if desired < lower_bound:
                desired = lower_bound
                reason = f"{child_name} moved after {parent_name} by declared causal delay"
            elif upper_bound is not None and desired > upper_bound:
                desired = upper_bound
                reason = f"{child_name} capped to declared/reasonable causal gap from {parent_name}"
            else:
                continue

            child_var = by_name.get(child_name, {})
            rec[child_name] = _format_datetime(desired, child_var.get("params") or {})
            issues.append(reason)
            changed = True
        if not changed:
            break
    return rec, issues


def _collect_formula_specs(variables: list[dict], rules: dict | None) -> list[tuple[str, str]]:
    """Collect authoritative formulas, preferring explicit variable definitions."""
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for var in variables:
        field = str(var.get("name", ""))
        expr = var.get("formula")
        if field and expr and field not in seen:
            specs.append((field, str(expr)))
            seen.add(field)
    for item in (rules or {}).get("formula_rules", []) or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field", ""))
        expr = item.get("expression")
        if field and expr and field not in seen:
            specs.append((field, str(expr)))
            seen.add(field)
    return specs


def _declared_numeric_bounds(params: dict) -> tuple[float | None, float | None]:
    lo = _to_finite_float(params.get("min", params.get("lo")), None)
    hi = _to_finite_float(params.get("max", params.get("hi")), None)
    return lo, hi


def _numeric_in_bucket(value: Any, bucket: tuple[float, float], precision: int = 2) -> bool:
    number = _to_finite_float(value, None)
    if number is None:
        return False
    lo, hi = bucket
    # Compare at the schema's declared precision so decimal serialization does not
    # create false negatives at exact bucket edges.
    rounded = round(number, precision)
    return rounded >= round(float(lo), precision) and rounded <= round(float(hi), precision)


def _enforce_authoritative_formulas(
    rec: dict, variables: list[dict], rules: dict | None = None
) -> tuple[dict, list[str]]:
    """Recalculate explicit CSV formulas after semantic/conditional mutations."""
    rec = dict(rec)
    issues: list[str] = []
    variable_by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    for field, expr in _collect_formula_specs(variables, rules):
        if field not in rec:
            continue
        field_def = variable_by_name.get(field) or {}
        if str(field_def.get("gen", "")).strip().lower() in {"derived_timestamp", "ts_offset"}:
            if (field_def.get("params") or {}).get("delay_seconds") is not None:
                continue
        deps = _formula_dependencies(expr)
        if any(rec.get(dep) is None for dep in deps):
            continue
        expected = _safe_formula(expr, rec)
        if expected is None:
            continue
        actual = rec.get(field)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            mismatch = not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=0.01)
        else:
            mismatch = str(actual) != str(expected)
        if mismatch:
            rec[field] = expected
            issues.append(f"{field} recalculated from authoritative CSV formula")
    return rec, issues


def _enforce_csv_contract(
    rec: dict, variables: list[dict], rules: dict | None = None, repair: bool = True
) -> tuple[dict, list[str]]:
    """Apply the confirmed CSV contract after *all* semantic mutations.

    Confirmed contract params are hard constraints. Gemini/business rules can only select among
    declared values; they can never introduce a new categorical value, leave a
    numeric field outside declared bounds/buckets, or change declared precision.
    """
    rec = dict(rec)
    issues: list[str] = []
    for var in variables:
        name = str(var.get("name", ""))
        if not name or name not in rec or rec.get(name) is None:
            continue
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        dtype = str(var.get("dtype", "")).strip().lower()
        value = rec.get(name)

        declared = _declared_param_options(params)
        if declared and not any(_matches_declared_option(value, opt) for opt in declared):
            if not repair:
                continue
            # Prefer a semantic preference only when it is inside the declared set.
            constraint = _rule_constraint_for(name, rules)
            preferred = _coerce_rule_values(constraint.get("preferred_values"))
            selected = next((opt for opt in declared if any(_matches_declared_option(opt, p) for p in preferred)), None)
            rec[name] = selected if selected is not None else declared[0]
            value = rec[name]
            issues.append(f"{name} restored to declared CSV value")

        precision_raw = params.get("precision")
        if precision_raw is not None and dtype in _NUMERIC_DTYPES and isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                precision = max(0, int(precision_raw))
                rounded = round(float(value), precision)
                if rounded != value:
                    rec[name] = rounded
                    value = rounded
                    issues.append(f"{name} rounded to declared precision={precision}")
            except (TypeError, ValueError):
                pass

        buckets = params.get("buckets")
        if buckets and dtype in _NUMERIC_DTYPES:
            try:
                precision = int(params.get("precision", 2) or 0)
                if not any(_numeric_in_bucket(value, (float(b[0]), float(b[1])), precision) for b in buckets):
                    # Re-sample from the declared bucket distribution rather than
                    # clamping into an arbitrary bucket. This preserves both range
                    # membership and the requested probability model.
                    rec[name] = _weighted_bucket(params, rec)
                    value = rec[name]
                    issues.append(f"{name} regenerated inside declared bucket distribution")
            except Exception:
                pass

        lo, hi = _declared_numeric_bounds(params)
        if dtype in _NUMERIC_DTYPES and isinstance(value, (int, float)) and not isinstance(value, bool):
            if lo is not None and float(value) < lo and repair:
                rec[name] = lo if dtype in {"float", "decimal", "number", "numeric"} else int(math.ceil(lo))
                issues.append(f"{name} raised to declared minimum")
            if hi is not None and float(rec[name]) > hi and repair:
                rec[name] = hi if dtype in {"float", "decimal", "number", "numeric"} else int(math.floor(hi))
                issues.append(f"{name} lowered to declared maximum")
    return rec, issues


def _is_placeholder_value(field_name: str, value: Any, var: dict) -> bool:
    """Reject obvious schema/sample placeholders while preserving legitimate declared values."""
    if isinstance(value, dict) and not value:
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    declared = _declared_param_options(var.get("params") or {})
    if declared and any(_matches_declared_option(text, opt) for opt in declared):
        return False

    norm_value = re.sub(r"[^a-z0-9]+", "", text.lower())
    norm_field = re.sub(r"[^a-z0-9]+", "", str(field_name or "").lower())
    known = {
        "date-time", "datetime", "timestamp", "string", "object", "array",
        "null", "none", "unknown", "placeholder",
    }
    if text.lower() in known:
        return True
    if norm_field and norm_value == norm_field:
        return True
    # Catch generated placeholders such as LOW_BALANCE_EVENT_ID for low_balance_event_id.
    return bool(norm_field and norm_value == norm_field.replace("scenario", ""))

def _strip_unusable_placeholders(rec: dict, variables: list[dict]) -> tuple[dict, list[str]]:
    out = dict(rec)
    issues: list[str] = []
    for var in variables:
        name = str(var.get("name") or "")
        if name not in out:
            continue
        if _is_placeholder_value(name, out.get(name), var):
            out.pop(name, None)
            issues.append(f"{name} removed as placeholder")
    return out, issues


def _enforce_obvious_semantic_consistency(rec: dict, variables: list[dict]) -> tuple[dict, list[str]]:
    """Repair deterministic cross-field contradictions that are obvious from field semantics."""
    rec = dict(rec)
    issues: list[str] = []
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}

    def set_value(name: str, value: Any, reason: str) -> None:
        if name in rec and rec.get(name) != value:
            rec[name] = value
            issues.append(reason)

    def dt(name: str) -> datetime | None:
        return _qa_parse_dt(rec.get(name)) if name in rec else None

    # Generic paired numeric bounds: lower/min cannot exceed upper/max.
    for low, high in (
        ("numberRelOfferLowerLimit", "numberRelOfferUpperLimit"),
        ("minCardinality", "maxCardinality"),
        ("minValue", "maxValue"),
        ("lowerLimit", "upperLimit"),
    ):
        if low in rec and high in rec and isinstance(rec.get(low), (int, float)) and isinstance(rec.get(high), (int, float)):
            if float(rec[low]) > float(rec[high]):
                low_val, high_val = rec[high], rec[low]
                set_value(low, low_val, f"{low} corrected to be <= {high}")
                set_value(high, high_val, f"{high} corrected to be >= {low}")

    # Low-balance/top-up causal chain.
    trigger = dt("low_balance_trigger_timestamp")
    recharge = dt("recharge_timestamp")
    if trigger is not None and recharge is not None and "hours_to_recharge_after_trigger" in rec:
        hours_params = by_name.get("hours_to_recharge_after_trigger", {}).get("params") or {}
        precision = int(hours_params.get("precision", 2) or 2)
        lo_hours = _to_finite_float(hours_params.get("min", hours_params.get("lo")), 0.0) or 0.0
        hi_hours = _to_finite_float(hours_params.get("max", hours_params.get("hi")), None)
        try:
            requested_hours = float(rec.get("hours_to_recharge_after_trigger"))
        except (TypeError, ValueError):
            requested_hours = 1.0
        if hi_hours is not None:
            requested_hours = min(requested_hours, hi_hours)
        requested_hours = max(lo_hours, requested_hours)

        elapsed_hours = max(0.0, (recharge - trigger).total_seconds() / 3600.0)
        # The timestamp pair and duration are one contract. Choose a timestamp consistent with
        # the bounded duration, rather than clamping the duration after the fact.
        if abs(elapsed_hours - requested_hours) > (0.5 / (10 ** max(0, precision))):
            repaired_recharge = trigger + timedelta(hours=requested_hours)
            params = by_name.get("recharge_timestamp", {}).get("params") or {}
            set_value("recharge_timestamp", _format_datetime(repaired_recharge, params),
                      "recharge_timestamp aligned to hours_to_recharge_after_trigger")
            recharge = repaired_recharge
            elapsed_hours = requested_hours
        set_value("hours_to_recharge_after_trigger", round(elapsed_hours, precision),
                  "hours_to_recharge_after_trigger recalculated from timestamps")
    elif trigger is not None and recharge is not None and recharge < trigger:
        params = by_name.get("recharge_timestamp", {}).get("params") or {}
        repaired_recharge = trigger + timedelta(hours=1)
        set_value("recharge_timestamp", _format_datetime(repaired_recharge, params),
                  "recharge_timestamp moved after low_balance_trigger_timestamp")
        recharge = repaired_recharge

    if "pre_trigger_core_balance_inr" in rec and "threshold_breach_limit_inr" in rec:
        pre = rec.get("pre_trigger_core_balance_inr")
        threshold = rec.get("threshold_breach_limit_inr")
        if isinstance(pre, (int, float)) and isinstance(threshold, (int, float)) and pre > threshold:
            set_value("threshold_breach_limit_inr", round(float(pre), 2),
                      "threshold_breach_limit_inr raised to contain pre-trigger balance")

    if "zero_balance_outage_duration_hours" in rec and "hours_to_recharge_after_trigger" in rec:
        outage = rec.get("zero_balance_outage_duration_hours")
        elapsed = rec.get("hours_to_recharge_after_trigger")
        if isinstance(outage, (int, float)) and isinstance(elapsed, (int, float)) and outage > elapsed:
            max_outage = max(0, int(math.floor(float(elapsed))))
            set_value("zero_balance_outage_duration_hours", max_outage,
                      "zero_balance_outage_duration_hours capped by recharge delay")

    if all(k in rec for k in ("pre_trigger_core_balance_inr", "credited_monetary_amount_inr", "emergency_credit_deducted_inr", "post_recharge_core_balance_inr")):
        pre = rec.get("pre_trigger_core_balance_inr")
        credited = rec.get("credited_monetary_amount_inr")
        emergency = rec.get("emergency_credit_deducted_inr")
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (pre, credited, emergency)):
            target_params = by_name.get("post_recharge_core_balance_inr", {}).get("params") or {}
            post_lo = _to_finite_float(target_params.get("min", target_params.get("lo")), 0.0) or 0.0
            post_hi = _to_finite_float(target_params.get("max", target_params.get("hi")), 100.0)
            raw_expected = float(pre) + float(credited) - float(emergency)
            if post_hi is not None and raw_expected > post_hi:
                adjusted_credited = max(0.0, min(float(credited), float(post_hi) - float(pre) + float(emergency)))
                set_value("credited_monetary_amount_inr", round(adjusted_credited, 2),
                          "credited_monetary_amount_inr adjusted to preserve post-balance bounds")
                credited = adjusted_credited
                raw_expected = float(pre) + float(credited) - float(emergency)
            elif raw_expected < post_lo:
                adjusted_emergency = max(0.0, min(float(emergency), float(pre) + float(credited) - float(post_lo)))
                set_value("emergency_credit_deducted_inr", round(adjusted_emergency, 2),
                          "emergency_credit_deducted_inr adjusted to preserve post-balance bounds")
                emergency = adjusted_emergency
                raw_expected = float(pre) + float(credited) - float(emergency)
            expected = round(max(post_lo, min(post_hi if post_hi is not None else raw_expected, raw_expected)), 2)
            set_value("post_recharge_core_balance_inr", expected,
                      "post_recharge_core_balance_inr recalculated from balance movement")

    # Intervention fields: a conversion cannot occur without an intervention being sent.
    if rec.get("proactive_nudge_sent_flag") is False:
        for name in ("nudge_channel", "nudge_offer_type"):
            if name in rec and by_name.get(name, {}).get("nullable", True):
                set_value(name, None, f"{name} cleared because proactive nudge was not sent")
        if "intervention_conversion_flag" in rec:
            set_value("intervention_conversion_flag", False,
                      "intervention_conversion_flag forced false because proactive nudge was not sent")

    # Obvious lifecycle ordering for commonly named datetime pairs. Iterate because repairing
    # one edge can legitimately move a shared timestamp and therefore require a second pass.
    date_pairs = (
        ("startDate", "terminationDate"),
        ("startDateTime", "endDateTime"),
        ("orderDate", "startDate"),
        ("requestedDate", "confirmationDate"),
        ("creationDate", "lastUpdate"),
        ("effectiveDate", "lastUpdate"),
    )
    for _ in range(max(2, len(date_pairs))):
        changed = False
        for start, end in date_pairs:
            a, b = dt(start), dt(end)
            if a is not None and b is not None and b < a and end in by_name:
                params = by_name[end].get("params") or {}
                new_value = _format_datetime(a, params)
                if rec.get(end) != new_value:
                    rec[end] = new_value
                    issues.append(f"{end} moved after {start}")
                    changed = True
        if not changed:
            break

    return rec, issues


def _validate_record(
    rec: dict,
    variables: list[dict],
    field_order: list[str],
    transactional: bool,
    rules: dict | None = None,
) -> tuple[dict, list[str]]:
    """Validate and repair one record using only the confirmed CSV schema."""
    rec = dict(rec)
    issues: list[str] = []
    active_fields = set(field_order)

    rec, placeholder_issues = _strip_unusable_placeholders(rec, variables)
    issues.extend(placeholder_issues)

    for var in variables:
        name = var["name"]
        if name not in rec or rec[name] is None:
            # Do not silently replace a failed generation with a dtype default
            # (e.g. 0 for integers). Regenerate from the declared CSV schema first,
            # including any dependencies required by that field.
            regenerated = None
            try:
                regenerated = _generate_selected_record(
                    variables, {name}, base=rec, rules=rules
                ).get(name)
            except Exception as exc:
                logger.debug("[QA] regeneration failed for %s: %s", name, exc)
            if regenerated is not None:
                rec[name] = regenerated
                issues.append(f"{name} regenerated from declared schema")
            elif var.get("nullable"):
                continue
            else:
                raise ValueError(f"Non-nullable field '{name}' has no valid executable generation value")
            if rec.get(name) is None:
                continue
        value = rec[name]
        dtype = var.get("dtype", "string")
        params = var.get("params") or {}
        if _event_like_field(var) and isinstance(value, str) and value.strip().lower() in {"unknown", ""}:
            rec[name] = _event_fallback_value(var)
            value = rec[name]
            issues.append(f"{name} repaired from placeholder event value")
        precision = int(params.get("precision", 2) or 2)
        try:
            if dtype == "int" and not isinstance(value, bool) and not isinstance(value, int):
                rec[name] = int(float(value)); issues.append(f"{name} coerced to int")
            elif dtype == "float" and not isinstance(value, bool) and not isinstance(value, (int, float)):
                rec[name] = float(value); issues.append(f"{name} coerced to float")
            elif dtype == "boolean" and not isinstance(value, bool):
                token = str(value).strip().lower()
                if token in {"true", "1", "yes"}: rec[name] = True
                elif token in {"false", "0", "no"}: rec[name] = False
                else: raise ValueError("invalid boolean")
                issues.append(f"{name} coerced to boolean")
            elif dtype == "datetime" and _qa_parse_dt(value) is None:
                rec[name] = _format_datetime(datetime.now(timezone.utc), params); issues.append(f"{name} repaired as datetime")
            elif dtype == "date":
                try: date.fromisoformat(str(value)[:10])
                except Exception: rec[name] = date.today().isoformat(); issues.append(f"{name} repaired as date")
        except Exception:
            rec[name] = _default_for_dtype(dtype)
            issues.append(f"{name} repaired from invalid type")

        if dtype in {"float", "decimal", "number", "numeric"} and isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool):
            rec[name] = round(float(rec[name]), precision)

        declared_options = _declared_param_options(params)
        if declared_options and rec.get(name) is not None:
            if not any(_matches_declared_option(rec.get(name), opt) for opt in declared_options):
                rec[name] = declared_options[0]
                issues.append(f"{name} corrected to declared schema params")
            else:
                for opt in declared_options:
                    if _matches_declared_option(rec.get(name), opt):
                        if rec.get(name) != opt:
                            rec[name] = opt
                        break

        if isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool):
            val = float(rec[name])

            def resolve_bound(raw):
                # CSV bounds may be literal numbers (min=0) or references to
                # another generated field (min=recharge_count_30d). Never call
                # float() directly on a field name; resolve it from the record.
                if raw is None:
                    return None
                if isinstance(raw, str):
                    text = raw.strip()
                    if text in rec:
                        return _to_finite_float(rec.get(text), None)
                return _to_finite_float(raw, None)

            lo = resolve_bound(params.get("min", params.get("lo")))
            hi = resolve_bound(params.get("max", params.get("hi")))
            if lo is not None and val < lo:
                rec[name] = int(lo) if dtype in {"int", "integer"} else float(lo); issues.append(f"{name} raised to minimum")
            if hi is not None and val > hi:
                rec[name] = int(hi) if dtype in {"int", "integer"} else float(hi); issues.append(f"{name} lowered to maximum")
            if "pct" in name.lower() or "percent" in name.lower() or "percentage" in name.lower():
                old = rec[name]; rec[name] = max(0, min(100, old))
                if old != rec[name]: issues.append(f"{name} clamped to percentage range")

    variable_by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    for field, expr in _collect_formula_specs(variables, rules):
        if field not in active_fields:
            continue
        # A derived timestamp formula can be descriptive in the confirmed contract. The executable
        # source of truth is its delay_seconds range plus the declared base dependency.
        field_def = variable_by_name.get(field) or {}
        if str(field_def.get("gen", "")).strip().lower() in {"derived_timestamp", "ts_offset"}:
            if (field_def.get("params") or {}).get("delay_seconds") is not None:
                continue
        deps = _formula_dependencies(expr)
        if any(rec.get(dep) is None for dep in deps):
            issues.append(f"{field} formula could not be evaluated; missing dependencies")
            continue
        expected = _safe_formula(expr, rec)
        if expected is None:
            issues.append(f"{field} formula could not be evaluated")
            continue
        actual = rec.get(field)
        mismatch = (not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=0.01)
                    if isinstance(expected, (int, float)) and isinstance(actual, (int, float))
                    else str(actual) != str(expected))
        if mismatch:
            rec[field] = expected; issues.append(f"{field} corrected from formula")

    before_semantics = dict(rec)
    rec = _apply_conditional_rules(rec, rules)
    rec = _apply_scenario_semantics(rec, rules)
    for name in rec:
        if name in before_semantics and rec.get(name) != before_semantics.get(name):
            issues.append(f"{name} corrected to scenario semantics")

    # Confirmed scenario contract is the final authority after every semantic mutation. This prevents an
    # LLM-derived conditional rule from reintroducing a value not declared by params.
    rec, contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(contract_issues)

    # Re-validate authoritative formulas after semantic/conditional repairs and CSV contract enforcement.
    for field, expr in _collect_formula_specs(variables, rules):
        if field not in active_fields:
            continue
        field_def = variable_by_name.get(field) or {}
        if str(field_def.get("gen", "")).strip().lower() in {"derived_timestamp", "ts_offset"}:
            if (field_def.get("params") or {}).get("delay_seconds") is not None:
                continue
        deps = _formula_dependencies(expr)
        if any(rec.get(dep) is None for dep in deps):
            continue
        expected = _safe_formula(expr, rec)
        if expected is None:
            continue
        actual = rec.get(field)
        mismatch = (not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=0.01)
                    if isinstance(expected, (int, float)) and isinstance(actual, (int, float))
                    else str(actual) != str(expected))
        if mismatch:
            rec[field] = expected; issues.append(f"{field} re-corrected after scenario semantics")

    # Final contract pass guarantees both raw-style deterministic generation and final QA
    # output satisfy the confirmed CSV params after formula corrections.
    rec, final_contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(final_contract_issues)

    rec, temporal_issues = _enforce_temporal_consistency(rec, variables, rules=rules)
    issues.extend(temporal_issues)
    # Recalculate explicit formulas after temporal repair because a moved parent timestamp
    # can legitimately change a dependent duration/formula field.
    rec, temporal_formula_issues = _enforce_authoritative_formulas(rec, variables, rules=rules)
    issues.extend(temporal_formula_issues)
    rec, final_contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(final_contract_issues)

    timestamp_field = next((name for name in ("record_timestamp", "transaction_timestamp", "timestamp", "created_at", "updated_at") if name in rec), None)
    base_ts = _qa_parse_dt(rec.get(timestamp_field)) if timestamp_field else None
    dispatch_ts = _qa_parse_dt(rec.get("notification_dispatch_ts"))
    response_ts = _qa_parse_dt(rec.get("customer_response_ts"))
    if base_ts and dispatch_ts and dispatch_ts < base_ts:
        dispatch_var = next((v for v in variables if str(v.get("name")) == "notification_dispatch_ts"), {})
        rec["notification_dispatch_ts"] = _format_datetime(base_ts, dispatch_var.get("params") or {})
        issues.append("notification_dispatch_ts corrected")
    if dispatch_ts and response_ts and response_ts < dispatch_ts:
        response_var = next((v for v in variables if str(v.get("name")) == "customer_response_ts"), {})
        rec["customer_response_ts"] = _format_datetime(dispatch_ts, response_var.get("params") or {})
        issues.append("customer_response_ts corrected")

    rec, semantic_issues = _enforce_obvious_semantic_consistency(rec, variables)
    issues.extend(semantic_issues)
    rec, final_contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(final_contract_issues)

    bad_placeholders = []
    for var in variables:
        name = str(var.get("name") or "")
        if name in rec and _is_placeholder_value(name, rec.get(name), var):
            bad_placeholders.append(name)
    if bad_placeholders:
        raise ValueError(f"Unresolved placeholder values remain: {bad_placeholders}")

    # Presentation format is a deterministic CSV concern, not an LLM concern.
    # Internally timestamps are parsed as real datetimes for formulas/comparisons;
    # the record returned to callers uses each field's declared timestamp_format.
    rec = _format_datetime_fields(rec, variables)

    # Recalculate displayed-duration fields after timestamp serialization. The public format is
    # minute-based by default, so computing the duration before formatting can differ by a
    # fraction of a minute from the timestamps the client actually receives.
    final_trigger = _qa_parse_dt(rec.get("low_balance_trigger_timestamp"))
    final_recharge = _qa_parse_dt(rec.get("recharge_timestamp"))
    if final_trigger is not None and final_recharge is not None and "hours_to_recharge_after_trigger" in rec:
        hour_params = next((v.get("params") or {} for v in variables if v.get("name") == "hours_to_recharge_after_trigger"), {})
        hour_precision = int(hour_params.get("precision", 2) or 2)
        final_hours = round(max(0.0, (final_recharge - final_trigger).total_seconds() / 3600.0), hour_precision)
        lo = _to_finite_float(hour_params.get("min", hour_params.get("lo")), 0.0) or 0.0
        hi = _to_finite_float(hour_params.get("max", hour_params.get("hi")), None)
        if final_hours < lo:
            final_hours = lo
        if hi is not None and final_hours > hi:
            final_hours = hi
        rec["hours_to_recharge_after_trigger"] = final_hours

    allowed = set(field_order)
    return {k: v for k, v in rec.items() if k in allowed}, issues


def run_deterministic_agentic_generation(
    scenario: str,
    count: int,
    industry: str,
    country: str | None,
    type_of_data: str,
    scenario_context: dict[str, Any],
    records_per_user: int = 10,
) -> WorkflowState:
    """Fast path for confirmed agentic scenarios.

    It deliberately avoids GeminiClient construction, Orchestrator/SchemaAgent/DataGeneration
    LangGraph construction, and all LLM calls. The confirmed schema plus complete scenario
    context are enough to build deterministic semantic guardrails and generate/QA the data.
    """
    from core.compiled_schema import compile_scenario

    state = WorkflowState(
        scenario=scenario,
        count=max(1, int(count)),
        industry=industry,
        country=country,
        type_of_data=type_of_data,
        records_per_user=max(1, min(50, int(records_per_user or 10))),
        domain=str(scenario_context.get("domain") or ""),
        business_scenario=str(scenario_context.get("business_scenario") or ""),
        business_response=scenario_context.get("business_response"),
        expected_outcome=scenario_context.get("expected_outcome"),
        scenario_type=scenario_context.get("scenario_type"),
        use_case=scenario_context.get("use_case"),
        entity_key=scenario_context.get("entity_key"),
        scenario_context=dict(scenario_context),
    )
    dyn = resolve_variables(scenario)
    if dyn is None:
        state.errors.append(f"Unknown scenario '{scenario}'")
        return state
    variables, field_order = dyn
    state.field_order = list(field_order)
    state.rules = build_deterministic_rules(state, variables)

    if type_of_data == "transactional":
        compiled = compile_scenario(scenario)
        state.raw_records = _transactional_records(
            compiled,
            state.count,
            state.records_per_user,
            rules=state.rules,
            record_errors_out=state.record_errors,
            country=state.country,
        )
    else:
        for index in range(state.count):
            try:
                state.raw_records.append(_generate_record(variables, rules=state.rules))
            except Exception as exc:
                state.record_errors.append({"record_index": index, "error": str(exc), "record": {}})

    checked: list[dict] = []
    fixes = 0
    for record_index, record in enumerate(state.raw_records):
        try:
            repaired, issues = _validate_record(
                record,
                variables,
                state.field_order,
                type_of_data == "transactional",
                rules=state.rules,
            )
            checked.append(repaired)
            fixes += len(issues)
        except Exception as exc:
            state.record_errors.append({"record_index": record_index, "error": str(exc), "record": dict(record)})

    state.final_records = checked
    state.validation_report = {
        "total_input": len(state.raw_records),
        "total_valid": len(checked),
        "total_dropped": len(state.record_errors),
        "record_errors": len(state.record_errors),
        "recovered": 0,
        "algo_fixes": fixes,
        "llm_fixes": 0,
        "llm_issues": 0,
        "deterministic_checks": [
            "schema_and_type", "declared_ranges_and_choices", "formula_and_arithmetic",
            "scenario_semantics", "cross_field_semantics", "timestamp_relationships", "user_history_consistency",
        ],
    }
    return state


class DataGenerationAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("generate", self._generate)
        graph.add_node("qa_validate", self._qa_validate)
        graph.set_entry_point("generate")
        graph.add_edge("generate", "qa_validate")
        graph.add_edge("qa_validate", END)
        return graph.compile()

    def run(self, state: WorkflowState) -> WorkflowState:
        result = self._graph.invoke(state, config={"recursion_limit": 10})
        return result if isinstance(result, WorkflowState) else WorkflowState.model_validate(result)

    def _generate(self, state: WorkflowState) -> WorkflowState:
        dyn = resolve_variables(state.scenario)
        if dyn is None:
            raise ValueError(f"Unknown scenario '{state.scenario}'")
        variables, csv_field_order = dyn

        if state.type_of_data == "transactional":
            from core.compiled_schema import compile_scenario
            compiled = compile_scenario(state.scenario)
            records = _transactional_records(
                compiled, state.count, state.records_per_user,
                rules=state.rules, record_errors_out=state.record_errors,
            )
            state.field_order = list(csv_field_order)
        else:
            state.field_order = list(csv_field_order)
            records = []
            for index in range(state.count):
                rec = {}
                try:
                    rec = _generate_record(variables, rules=state.rules)
                    records.append(rec)
                except Exception as exc:
                    state.record_errors.append({"record_index": index, "error": str(exc), "record": dict(rec)})
                    logger.warning("[DataGeneration] Skipping aggregational record %d: %s", index, exc)

        state.raw_records = records
        return state

    def _qa_validate(self, state: WorkflowState) -> WorkflowState:
        records = state.raw_records
        dyn = resolve_variables(state.scenario)
        if dyn is None:
            raise ValueError(f"Unknown scenario '{state.scenario}'")
        variables, _ = dyn
        transactional = state.type_of_data == "transactional"

        checked: list[dict] = []
        algo_fixed = 0
        for record_index, record in enumerate(records):
            try:
                repaired, issues = _validate_record(
                    record, variables, state.field_order, transactional,
                    rules=state.rules,
                )
                algo_fixed += len(issues)
                checked.append(repaired)
            except Exception as exc:
                state.record_errors.append({"record_index": record_index, "error": str(exc), "record": dict(record)})
                logger.warning("[QA] Skipping invalid record %d: %s", record_index, exc)

        qa_mode = os.getenv("QA_LLM_MODE", "off").strip().lower()
        llm_fixes = 0
        llm_issues = 0
        dropped_all: list[dict] = []
        if qa_mode == "full" and checked:
            valid_all: list[dict] = []
            dyn_variables = variables
            schema_contract = json.dumps([
                {
                    "name": v.get("name"), "dtype": v.get("dtype"),
                    "description": v.get("description", ""), "params": v.get("params", {}),
                    "depends_on": v.get("depends_on", []), "formula": v.get("formula", ""),
                    "nullable": v.get("nullable", False),
                } for v in dyn_variables
            ], default=str, sort_keys=True)
            system_prompt = _QA_SYSTEM.format(
                rules="\n".join(f"- {r}" for r in state.rules.get("business_rules", [])) or "Apply the supplied CSV schema and deterministic checks.",
                cross_field_rules="\n".join(f"- {r}" for r in state.rules.get("cross_field_rules", [])) or "Validate declared mathematical, temporal, and business relationships.",
            ) + f"\n\nFULL SCENARIO SCHEMA CONTRACT (authoritative):\n{schema_contract}\n"
            for i in range(0, len(checked), _CHUNK):
                chunk = checked[i:i + _CHUNK]
                try:
                    result = self._llm.generate_json(
                        system_prompt,
                        f"Scenario: {state.scenario}\nIndustry: {state.industry}\nCountry: {state.country or 'GLOBAL'}\n"
                        f"Domain: {state.domain}\nBusiness scenario: {state.business_scenario}\n"
                        f"Business response: {state.business_response or ''}\nExpected outcome: {state.expected_outcome or ''}\n"
                        f"Use case: {state.use_case or ''}\nScenario type: {state.scenario_type or ''}\nRecords to validate:\n{json.dumps(chunk, default=str)}",
                        temperature=0.1,
                    )
                    validated = result.get("valid_records", chunk)
                    valid_all.extend([{k: r[k] for k in state.field_order if k in r} for r in validated if isinstance(r, dict)])
                    dropped = result.get("dropped_records", []) or []
                    dropped_all.extend(dropped)
                    llm_fixes += int(result.get("fixes_applied", 0))
                    llm_issues += int(result.get("issues_found", 0))
                except Exception as exc:
                    logger.warning("[QA] Chunk %d error: %s — deterministic validation retained", i, exc)
                    state.errors.append(f"QA chunk {i} error: {exc}")
                    valid_all.extend(chunk)
            checked = valid_all

        # Never trust an LLM QA response as the final authority. Re-run the complete
        # deterministic CSV/formula contract after any LLM repair so final_records are
        # guaranteed to remain inside the confirmed schema.
        if checked:
            deterministic_final: list[dict] = []
            for record_index, record in enumerate(checked):
                try:
                    repaired, final_issues = _validate_record(
                        record, variables, state.field_order, transactional, rules=state.rules
                    )
                    algo_fixed += len(final_issues)
                    deterministic_final.append(repaired)
                except Exception as exc:
                    state.record_errors.append({"record_index": record_index, "error": str(exc), "record": dict(record)})
                    logger.warning("[QA-final] Rejecting record %d after deterministic recheck: %s", record_index, exc)
            checked = deterministic_final

        state.final_records = checked
        state.validation_report = {
            "total_input": len(records),
            "total_valid": len(checked),
            "total_dropped": len(dropped_all) + len(state.record_errors),
            "record_errors": len(state.record_errors),
            "recovered": 0,
            "algo_fixes": algo_fixed,
            "llm_fixes": llm_fixes,
            "llm_issues": llm_issues,
            "deterministic_checks": [
                "schema_and_type", "declared_ranges_and_choices", "formula_and_arithmetic",
                "timestamp_relationships", "user_history_consistency",
            ],
        }
        return state
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
from dataclasses import dataclass
from functools import lru_cache
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

Validate every supplied record against the FULL confirmed scenario contract, official source
semantics, business/cross-field rules, and subscriber-history rules. The goal is not merely
to produce syntactically valid rows: every record must represent a logically possible business
event sequence. Treat the generated dataset as if it were emitted by a real telecom charging
and customer-management system.

The confirmed schema and supplied official TMF Swagger JSON are authoritative for structure,
field meaning, required/optional fields, datatype, enum vocabulary, nested-reference meaning,
amount/unit meaning, lifecycle timestamps, and relationship semantics. Never invent official
enum values or substitute synonyms. Never use a free-form value from one field as though it
were the semantic value of a different field.

For every record, validate ALL dimensions, not only timestamps:
1. Entity identity consistency: subscriber/account/customer/bucket identities and stable keys.
2. Reference integrity: ids, hrefs, names, roles, and @referredType semantics agree with the
   referenced resource type.
3. Enum/category fidelity: every choice is inside the declared/source value set.
4. Datatype/range/precision/bucket/weight constraints.
5. Amount and unit consistency: values describing the same balance or recharge agree.
6. Balance invariants: remaining/reserved/recharge quantities cannot contradict one another.
7. State-machine consistency: status, outcome, eligibility, suppression, decision, and execution
   state cannot describe mutually exclusive states simultaneously.
8. Auto-top-up consistency: recurrence fields are present only when auto-top-up is enabled and
   their period/count agree with that behavior.
9. Plan/validity consistency: a recharge validity window is tied to the recharge/plan event and
   is never an independently sampled unrelated date range.
10. Temporal causality: request <= confirmation, activation/start <= expiry/end, and dependent
    events follow their parent events within declared or domain-appropriate bounds.
11. Dependency consistency: values derived from another field must actually agree with that field.
12. Formula/arithmetic consistency: formulas and calculated values must match exactly within the
    declared precision.
13. Subscriber history consistency: repeated transactions for one entity must form a plausible
    chronological history; identity context must remain stable.
14. Scenario semantics: the requested scenarioType/business scenario must materially constrain
    the relevant state transitions and outcomes.

CRITICAL RULE: do not validate each field independently. First reason about the business event
and its dependencies, then validate the individual fields against that event. If a contradiction
is found, identify the root field/event and repair only the affected downstream values. Re-run
all dependent checks after every repair. Never repair a contradiction by changing an unrelated
field just to make a scalar check pass.

A record is INVALID if any material contradiction remains. Do not accept a record merely because
its JSON, datatype, or range checks pass. If deterministic validation can prove a contradiction,
that deterministic result overrides an LLM judgement. Never invent missing business facts to make
a record look valid.

Return JSON with keys: valid_records, dropped_records, fixes_applied, issues_found.

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
    """Generate a useful synthetic string from field name/description semantics.

    This is intentionally conservative: explicit params remain authoritative, then clear
    description examples and well-known telecom/domain semantics are used before the final
    synthetic fallback. The fallback never echoes the schema field name verbatim.
    """
    p = dict(var.get("params") or {})
    name = str(var.get("name") or rec.get("__current_field__") or "").strip()
    desc = str(var.get("description") or "").strip()
    n = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    d = desc.lower()

    if "choices" in p or "values" in p:
        vals = p.get("choices", p.get("values"))
        if isinstance(vals, str):
            vals = [x.strip() for x in re.split(r"[;,|]", vals) if x.strip()]
        if vals:
            return str(random.choice(list(vals)))
    if p.get("value") is not None:
        return str(p["value"])

    # Clear description examples are stronger than fuzzy field-name heuristics.
    example_bodies = []
    for pattern in (
        r"\bsuch as\s+(.+?)(?:\.|$)",
        r"\bfor example\s+(.+?)(?:\.|$)",
        r"\bpossible values (?:are|include)\s+(.+?)(?:\.|$)",
        r"\bvalid values (?:are|include)\s+(.+?)(?:\.|$)",
    ):
        match = re.search(pattern, desc, flags=re.IGNORECASE)
        if match:
            example_bodies.append(match.group(1).strip())
    for body in example_bodies:
        parts = [
            re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", item).strip()
            for item in re.split(r"\s*(?:,|;|\bor\b|\band\b)\s*", body, flags=re.IGNORECASE)
        ]
        parts = [x for x in parts if x and len(x) <= 80 and x.strip().lower() not in {"and so forth", "etc", "etc."}]
        if len(parts) >= 2:
            return str(random.choice(parts))

    # Identifier/reference semantics must be checked before broad words such as "offer",
    # "type", or "status", because many TMF fields are references to another entity.
    if (
        n.endswith("_id") or n == "id" or n.endswith("_key")
        or "identifier" in d or "reference id" in d or "unique reference" in d
    ):
        prefix_source = n[:-3] if n.endswith("_id") else n
        prefix = re.sub(r"[^A-Z0-9]+", "_", prefix_source.upper()).strip("_") or "REF"
        return _prefixed_int({"prefix": f"{prefix}-", "digits": 10}, rec)

    if "phone" in n or "mobile" in n or n == "msisdn":
        country = str(p.get("country") or "IN").upper()
        dial_codes = {
            "IN": "+91", "US": "+1", "CA": "+1", "GB": "+44", "AU": "+61",
            "AE": "+971", "SG": "+65", "DE": "+49", "FR": "+33", "IT": "+39",
        }
        dial = dial_codes.get(country, "+91")
        return _e164_phone({"country_codes": [dial], "country": country}, rec)
    if "email" in n:
        return f"user{random.randint(100000, 999999)}@example.test"
    if "uri" in d or "url" in d or any(token in n for token in ("href", "url", "schemalocation", "resourcepath", "path")):
        return f"https://example.test/telecom/{uuid.uuid4().hex[:12]}"
    if "reference" in d and not any(token in d for token in ("uri", "url", "documentation")):
        prefix = re.sub(r"[^A-Z0-9]+", "_", n.upper()).strip("_") or "REF"
        return _prefixed_int({"prefix": f"{prefix}-", "digits": 10}, rec)

    if "operating circle" in d or "regulatory service area" in d:
        return random.choice([
            "Delhi", "Haryana", "Punjab", "Rajasthan", "Uttar Pradesh East", "Uttar Pradesh West",
            "Maharashtra", "Mumbai", "Gujarat", "Karnataka", "Tamil Nadu", "Kerala",
            "Andhra Pradesh", "Telangana", "West Bengal", "Bihar", "Odisha", "Assam",
            "North East", "Himachal Pradesh", "Jammu Kashmir", "Madhya Pradesh", "Kolkata",
        ])
    if "medium" in n and "contact" in d:
        return random.choice(["email", "telephone", "postal_address"])
    if "contact medium" in d:
        return random.choice(["email", "telephone", "postal_address"])
    if "type of contact" in d:
        return random.choice(["mobile", "fixed_home", "fixed_office", "shipping_address"])
    if "payment plan" in d or n == "plantype":
        return random.choice(["prepaid", "postpaid", "hybrid"])
    if "consumption counter" in d:
        return random.choice(["used", "outOfBucket"])
    if "currency" in d or "iso4217" in d:
        return str(p.get("currency") or "INR").upper()
    if "currency" in n:
        return str(p.get("currency") or "INR").upper()
    if "billing time period" in d or "repeat the application of the price" in d:
        return random.choice(["week", "month", "quarter", "year"])
    if "frequency of" in d or "frequency" == n:
        return random.choice(["daily", "weekly", "monthly", "quarterly"])
    if "price" in n and ("recurring" in d or "discount" in d or "allowance" in d or "penalty" in d):
        return random.choice(["recurring", "discount", "allowance", "penalty"])
    if "catalog" in n and "catalog" in d:
        return random.choice(["product", "service", "resource"])
    if "relationship" in n and ("relationship" in d or "migration" in d or "substitution" in d):
        return random.choice(["override", "discount", "replace", "migrate"])
    if n == "value_type" or ("kind of value" in d and "numeric" in d and "text" in d):
        return random.choice(["numeric", "text"])
    if n == "range_interval" or "inclusion or exclusion" in d:
        return random.choice(["open", "closed", "closedBottom", "closedTop"])
    if n == "adjust_type" or "recurringcharge" in d or "onetimecharge" in d:
        return random.choice(["RecurringCharge", "OneTimeCharge"])
    if "format of the exported data" in d or n == "content_type":
        return random.choice(["application/json", "text/csv", "application/xml"])
    if "attachment mime type" in d:
        return random.choice(["application/pdf", "image/png", "video/mp4"])
    if n == "attachment_type" or "attachment type" in d:
        return random.choice(["document", "image", "video"])
    if "type of notification" in d or "type of the notification" in d:
        return random.choice(["LOW_BALANCE_ALERT", "RECHARGE_UPDATE", "PAYMENT_UPDATE", "SYSTEM_NOTIFICATION"])
    if "network" in n and "capability" in n:
        return random.choice(["2G", "3G", "4G", "5G"])
    if "channel" in n:
        return random.choice(["APP", "SMS", "WEB", "USSD", "WHATSAPP", "IVR", "RETAIL"])
    if "payment" in n and ("method" in n or "instrument" in n):
        return random.choice(["UPI", "CREDIT_CARD", "DEBIT_CARD", "WALLET", "CASH", "AUTO_DEBIT"])
    if "reason" in n:
        return random.choice(["LOW_BALANCE", "DATA_EXHAUSTED", "VALIDITY_EXPIRY", "CUSTOMER_REQUEST"])
    if n in {"stateorprovince", "province", "state"} or n.endswith("_state_or_province"):
        return random.choice(["Haryana", "Punjab", "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat"])
    if n == "city" or n.endswith("_city"):
        return random.choice(["Delhi", "Gurugram", "Ludhiana", "Chandigarh", "Mumbai", "Bengaluru", "Pune"])
    if n in {"postcode", "postalcode", "postal_code", "postcode"}:
        return str(random.randint(110001, 999999))
    if "status reason" in d or n == "status_reason":
        return random.choice(["CUSTOMER_REQUEST", "PAYMENT_FAILURE", "SYSTEM_ERROR", "POLICY_VIOLATION"])
    if "status" in n or n.endswith("_state") or n in {"state"}:
        return random.choice(["PENDING", "COMPLETED", "FAILED"])
    if "offer" in n and any(token in d for token in ("incentive", "recommendation", "cash back", "validity booster")):
        return random.choice(["EXTRA_DATA", "CASH_BACK", "VALIDITY_BOOSTER", "DISCOUNT_VOUCHER"])
    if "segment" in n or "market segment" in d:
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

# -- Generator functions --------------------------------------------------------

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


def _temporal_output_params(params: dict | None, generator: str) -> dict:
    """Keep enough serialized timestamp precision to preserve declared temporal offsets."""
    result = dict(params or {})
    fmt = _normalize_timestamp_format(result)
    if "%S" in fmt:
        return result
    gen = str(generator or "").strip().lower()
    if gen == "ts_offset":
        try:
            min_sec = int(result.get("min_sec", result.get("min_seconds", 0)) or 0)
            max_sec = int(result.get("max_sec", result.get("max_seconds", min_sec)) or min_sec)
        except (TypeError, ValueError):
            min_sec, max_sec = 0, 1
        min_sec, max_sec = min(min_sec, max_sec), max(min_sec, max_sec)
        # A variable/fractional-minute offset cannot be represented by minute-only output.
        if min_sec != max_sec or min_sec % 60 != 0:
            result["timestamp_format"] = "dd/mm/yyyy hh:mm:ss a"
    elif gen == "ts_add_field":
        # The runtime add_seconds field is data-dependent, so its precision is unknown here.
        result["timestamp_format"] = "dd/mm/yyyy hh:mm:ss a"
    return result


def _ts_offset(params: dict, rec: dict) -> str:
    base_field = str(params.get("base_field") or params.get("source_field") or "").strip()
    if not base_field:
        raise ValueError("ts_offset requires 'base_field' (or legacy alias 'source_field')")
    base_str = rec.get(base_field, datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    min_sec = int(params.get("min_sec", params.get("min_seconds", 0)))
    max_sec = int(params.get("max_sec", params.get("max_seconds", min_sec)))
    min_sec, max_sec = min(min_sec, max_sec), max(min_sec, max_sec)
    offset = timedelta(seconds=random.randint(min_sec, max_sec))
    return _format_datetime(base + offset, _temporal_output_params(params, "ts_offset"))


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


def _coerce_formula_result(value: Any, field_def: dict[str, Any] | None = None) -> Any:
    """Normalize formula results to the field contract.

    Datetime subtraction naturally yields timedelta; numeric duration fields must receive
    seconds so downstream validation and CSV serialization stay type-correct.
    """
    if not isinstance(value, timedelta):
        return value
    field_def = field_def or {}
    dtype = str(field_def.get("dtype") or "").strip().lower()
    if dtype in _NUMERIC_DTYPES:
        params = field_def.get("params") if isinstance(field_def.get("params"), dict) else {}
        try:
            precision = int(params.get("precision", 2) or 2)
        except (TypeError, ValueError):
            precision = 2
        return round(value.total_seconds(), precision)
    return value


def _formula(var: dict, rec: dict):
    """Evaluate a constrained formula language against the current record."""
    expr = str(var.get("formula", "") or "").strip()
    if not expr:
        return None
    return _coerce_formula_result(_safe_formula(expr, rec), var)


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


def _format_datetime_for_variable(dt: datetime, var: dict) -> str:
    """Format a datetime without discarding precision needed by declared temporal offsets.

    Standalone timestamps keep the configured client format (default: minute precision).
    Relational generators such as ``ts_offset``/``ts_add_field`` may encode sub-minute
    differences; in that case seconds are retained so formulas and temporal validators
    continue to match the serialized values exactly.
    """
    params = _temporal_output_params(var.get("params") or {}, str(var.get("gen") or ""))
    return _format_datetime(dt, params)


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
            out[name] = _format_datetime_for_variable(dt, var)
    return out

def _parse_dt(s: str) -> datetime:
    """Parse ISO-8601 and common human-readable timestamps to timezone-aware datetime."""
    parsed = _parse_timestamp_text(str(s))
    if parsed is None:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# -- Dispatch table -------------------------------------------------------------

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


@lru_cache(maxsize=4096)
def _formula_dependencies(expression: str) -> set[str]:
    """Extract field names referenced by a simple formula expression.

    Formula text is immutable for a confirmed scenario, so parsing/AST walking is cached
    across records instead of repeated for every row and every repair pass.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                and node.id not in {"round", "min", "max", "abs", "sum", "True", "False", "None", "DATE"}}
    except Exception:
        return set()


@dataclass(frozen=True)
class _GenerationPlan:
    ordered: tuple[dict, ...]
    cyclic: frozenset[str]
    known_fields: frozenset[str]
    formula_by_name: dict[str, str]


def _variable_dependency_order(
    variables: list[dict],
    selected_names: set[str] | None = None,
    rules: dict | None = None,
) -> tuple[list[dict], set[str]]:
    """Return variables in stable dependency order with O(n) position lookups."""
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    position = {name: idx for idx, name in enumerate(by_name)}
    selected = set(selected_names or by_name) & set(by_name)
    changed = True
    while changed:
        changed = False
        for name in tuple(selected):
            var = by_name[name]
            dep_names = list(var.get("depends_on", []) or [])
            expr = var.get("formula") or _formula_from_rules(name, rules)
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
        var = by_name[name]
        dep_names = list(var.get("depends_on", []) or [])
        expr = var.get("formula") or _formula_from_rules(name, rules)
        if expr:
            for dep in _formula_dependencies(str(expr)):
                if dep not in dep_names:
                    dep_names.append(dep)
        for dep in dep_names:
            if dep in selected:
                indegree[name] += 1
                outgoing[dep].append(name)

    queue = sorted((name for name in selected if indegree[name] == 0), key=position.__getitem__)
    ordered_names: list[str] = []
    while queue:
        name = queue.pop(0)
        ordered_names.append(name)
        for child in outgoing[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
        if len(queue) > 1:
            queue.sort(key=position.__getitem__)

    cyclic = selected - set(ordered_names)
    ordered = [by_name[name] for name in ordered_names]
    ordered.extend(by_name[name] for name in by_name if name in cyclic)
    return ordered, cyclic


def _build_generation_plan(
    variables: list[dict],
    selected_names: set[str] | None = None,
    rules: dict | None = None,
) -> _GenerationPlan:
    ordered, cyclic = _variable_dependency_order(variables, selected_names, rules)
    formula_by_name: dict[str, str] = {}
    for var in variables:
        name = str(var.get("name") or "")
        expr = var.get("formula") or _formula_from_rules(name, rules)
        if name and expr:
            formula_by_name[name] = str(expr)
    return _GenerationPlan(
        ordered=tuple(ordered),
        cyclic=frozenset(cyclic),
        known_fields=frozenset(str(v.get("name")) for v in variables if v.get("name")),
        formula_by_name=formula_by_name,
    )

def _generate_record(variables: list[dict], rules: dict | None = None, plan: _GenerationPlan | None = None, apply_repairs: bool = True) -> dict:
    """Generate one record in dependency order, while safely handling cycles."""
    rec: dict = {}
    plan = plan or _build_generation_plan(variables, rules=rules)
    ordered, cyclic, known_fields = plan.ordered, plan.cyclic, plan.known_fields
    for var in ordered:
        gen_type = var["gen"]
        effective_var = var
        # A formula is authoritative when it can be evaluated.  For a dependency
        # cycle, seed the cyclic field from its declared generator so the remaining
        # fields can still be generated and QA can evaluate any resolvable formulas.
        rule_formula = None if var["name"] in cyclic else plan.formula_by_name.get(var["name"])
        if rule_formula:
            deps = _formula_dependencies(str(rule_formula))
            if deps and any(dep not in known_fields and dep not in rec for dep in deps):
                # Keep the declared generator when formula text references symbolic
                # tokens that are not schema fields (common in human-readable scenario definitions).
                rule_formula = None
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        generator = _GENERATORS.get(effective_var.get("gen"))
        if generator:
            rec["__current_field__"] = var["name"]
            try:
                value = generator(effective_var, rec)
            finally:
                rec.pop("__current_field__", None)
        else:
            value = None
        rec[var["name"]] = _apply_generation_constraint(var, value, rec, rules)
    if apply_repairs:
        rec = _apply_conditional_rules(rec, rules)
        rec = _apply_scenario_semantics(rec, rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_low_balance_topup_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        return _format_datetime_fields(rec, variables)
    return rec


def _generate_selected_record(
    variables: list[dict],
    selected_names: set[str],
    base: dict | None = None,
    rules: dict | None = None,
    plan: _GenerationPlan | None = None,
    apply_repairs: bool = True,
) -> dict:
    """Generate selected variables plus dependencies using a precomputed dependency plan."""
    rec = dict(base or {})
    plan = plan or _build_generation_plan(variables, selected_names, rules)
    ordered, cyclic, known_fields = plan.ordered, plan.cyclic, plan.known_fields
    for var in ordered:
        name = var["name"]
        if name in rec:
            continue
        effective_var = var
        rule_formula = None if name in cyclic else plan.formula_by_name.get(name)
        if rule_formula:
            deps = _formula_dependencies(str(rule_formula))
            if deps and any(dep not in known_fields and dep not in rec for dep in deps):
                rule_formula = None
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        generator = _GENERATORS.get(effective_var.get("gen"))
        if generator:
            rec["__current_field__"] = name
            try:
                value = generator(effective_var, rec)
            finally:
                rec.pop("__current_field__", None)
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
    if apply_repairs:
        rec = _apply_conditional_rules(rec, rules)
        rec = _apply_scenario_semantics(rec, rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_low_balance_topup_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        return _format_datetime_fields(rec, variables)
    return rec


# -- Transactional/user-history generation helpers -----------------------------

def _pick_timestamp_field(variables: list[dict]) -> str | None:
    preferred=(
        "topup_requested_date_time", "topupbalance_requested_date", "topupbalance_requesteddate",
        "recharge_timestamp", "transaction_timestamp", "record_timestamp", "timestamp", "created_at", "updated_at",
    )
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
                            country: str | None = None, fixes_out: list[int] | None = None) -> list[dict]:
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
    used_resource_ids={name:set() for name in ("bucket_id", "topupbalance_id", "topup_transaction_id")}
    user_field_names = set(compiled.user_fields)
    record_field_names = set(compiled.record_fields)
    user_plan = _build_generation_plan(variables, user_field_names, rules)
    record_plan = _build_generation_plan(variables, record_field_names, rules)

    for user_index in range(user_count):
        try:
            user_context=_generate_selected_record(variables,user_field_names,rules=rules,plan=user_plan)
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
            # Establish stable entity-level domain context before transaction rows are created.
            user_context, _ = _enforce_low_balance_topup_consistency(user_context, variables, rules=rules)

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

            # Resource identifiers are real identifiers, not descriptive dimensions. Keep
            # bucket ids unique across subscribers so cross-subscriber references cannot collide.
            if "bucket_id" in variable_names:
                bucket_id_value = str(user_context.get("bucket_id") or "")
                if (not bucket_id_value) or bucket_id_value in used_resource_ids["bucket_id"]:
                    for _ in range(1000):
                        candidate = _prefixed_int({"prefix": "BUCKET-", "digits": 10}, user_context)
                        if candidate not in used_resource_ids["bucket_id"]:
                            bucket_id_value = candidate
                            break
                    else:
                        raise RuntimeError("Unable to generate a unique bucket_id")
                    user_context["bucket_id"] = bucket_id_value
                used_resource_ids["bucket_id"].add(bucket_id_value)
                # All bucket references are synchronized later, but the stable user context
                # should already carry the authoritative bucket id.
                user_context["balance_bucket_id"] = bucket_id_value if "balance_bucket_id" in variable_names else user_context.get("balance_bucket_id")
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
            last_exc: Exception | None = None
            for attempt in range(5):
                try:
                    base=dict(user_context)
                    if timestamp_field:
                        # One authoritative transaction timestamp anchors the complete row.
                        base[timestamp_field]=timestamps[record_index].isoformat()
                    row=_generate_selected_record(variables,record_field_names,base=base,rules=rules,plan=record_plan,apply_repairs=False)
                    if timestamp_field and timestamp_field not in row:
                        row[timestamp_field]=timestamps[record_index].isoformat()

                    # TopupBalance.id and topup_transaction_id identify concrete transaction
                    # resources. Guarantee uniqueness across the generated dataset instead of
                    # relying on the stochastic semantic_string generator.
                    for id_name,prefix in (("topupbalance_id","TOPUPBALANCE-"),("topup_transaction_id","TOPUP_TRANSACTION-")):
                        if id_name in row:
                            candidate=str(row.get(id_name) or "")
                            if (not candidate) or candidate in used_resource_ids[id_name]:
                                for _ in range(1000):
                                    generated_id=_prefixed_int({"prefix":prefix,"digits":10},row)
                                    if generated_id not in used_resource_ids[id_name]:
                                        candidate=generated_id
                                        break
                                else:
                                    raise RuntimeError(f"Unable to generate a unique {id_name}")
                                row[id_name]=candidate
                            # Keep the resource href synchronized when the stochastic id had to
                            # be replaced for uniqueness.
                            if id_name == "topupbalance_id" and "topupbalance_href" in row:
                                row["topupbalance_href"] = f"https://example.test/telecom/topupBalance/{candidate}"
                            used_resource_ids[id_name].add(candidate)

                    # If the schema exposes a RelatedTopupBalance reference, point it to a
                    # real earlier transaction in this subscriber's history. The first event has
                    # no earlier top-up and therefore leaves the optional reference null.
                    if "topupbalance_balance_topup_id" in row and "topupbalance_id" in row:
                        previous_topup_id = generated[-1].get("topupbalance_id") if generated and generated[-1].get("subscriber_id") == row.get("subscriber_id") else None
                        if previous_topup_id and str(previous_topup_id) != str(row.get("topupbalance_id")):
                            row["topupbalance_balance_topup_id"] = previous_topup_id
                            row["topupbalance_balance_topup_href"] = f"https://example.test/telecom/topupBalance/{previous_topup_id}" if "topupbalance_balance_topup_href" in row else row.get("topupbalance_balance_topup_href")
                            row["topupbalance_balance_topup_name"] = "Related Top-up" if "topupbalance_balance_topup_name" in row else row.get("topupbalance_balance_topup_name")
                            row["topupbalance_balance_topup_role"] = "child" if "topupbalance_balance_topup_role" in row else row.get("topupbalance_balance_topup_role")
                            row["topupbalance_balance_topup_referred_type"] = "TopupBalance" if "topupbalance_balance_topup_referred_type" in row else row.get("topupbalance_balance_topup_referred_type")
                        elif previous_topup_id is None:
                            # The first transaction has no real parent/related top-up. Clear all
                            # flattened RelatedTopupBalance leaves so no orphaned metadata remains.
                            for ref_name in (
                                "topupbalance_balance_topup_id",
                                "topupbalance_balance_topup_href",
                                "topupbalance_balance_topup_name",
                                "topupbalance_balance_topup_role",
                                "topupbalance_balance_topup_referred_type",
                            ):
                                if ref_name in row:
                                    row[ref_name] = None

                    # Run the full validator once per successful attempt. This replaces the
                    # previous strict-validation pass plus a second batch validation pass.
                    repaired, issues = _validate_record(
                        row, variables, list(compiled.field_order), True, rules=rules
                    )
                    generated.append(repaired)
                    if fixes_out is not None:
                        fixes_out.append(len(issues))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                err={"user_index":user_index,"record_index":record_index,"error":str(last_exc),"record":dict(user_context),"attempts":5}
                if record_errors_out is not None: record_errors_out.append(err)
                logger.warning("[DataGeneration] Unable to produce a valid transactional record user=%d record=%d after 5 attempts: %s",user_index,record_index,last_exc)
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



_FORMULA_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
    ast.Name, ast.Call, ast.Load, ast.Tuple, ast.List,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BoolOp, ast.And, ast.Or, ast.Not, ast.IfExp,
)
_ALLOWED_FORMULA_FUNCS = {"round", "min", "max", "abs", "sum", "DATE"}

@lru_cache(maxsize=4096)
def _compile_safe_formula(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _FORMULA_ALLOWED_NODES):
                return None
            if isinstance(node, ast.Name) and node.id not in _ALLOWED_FORMULA_FUNCS:
                # Actual field names are checked dynamically by _safe_formula.
                continue
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FORMULA_FUNCS
            ):
                return None
        return compile(tree, "<scenario-formula>", "eval")
    except Exception:
        return None


def _safe_formula(expr: str, rec: dict):
    """Evaluate a constrained formula language against the current record.

    The validated Python expression is compiled once per distinct formula and reused across
    every generated record. Field names and datetime coercion remain record-specific.
    """
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

    code = _compile_safe_formula(str(expr))
    if code is None:
        return None
    names: dict[str, Any] = {}
    for k, v in rec.items():
        if k == "__current_field__" or v is None:
            continue
        if isinstance(v, str):
            parsed = _qa_parse_dt(v)
            names[k] = parsed if parsed is not None else v
        else:
            names[k] = v
    names.update({
        "round": round,
        "min": min,
        "max": max,
        "abs": abs,
        "sum": sum,
    })
    names["DATE"] = DATE
    try:
        return eval(code, {"__builtins__": {}}, names)
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
        expected = _coerce_formula_result(_safe_formula(expr, rec), field_def)
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



def _lb_domain(rules: dict | None) -> bool:
    domain = str((rules or {}).get("domain") or "").strip().lower()
    return "low balance" in domain and any(token in domain for token in ("top up", "top-up", "recharge"))


def _lb_variables_by_name(variables: list[dict]) -> dict[str, dict]:
    return {str(v.get("name")): v for v in variables if v.get("name")}


def _lb_find_fields(variables: list[dict], *predicates) -> list[str]:
    names = []
    for var in variables:
        name = str(var.get("name") or "")
        if not name:
            continue
        text = f"{name} {var.get('description') or ''}".lower()
        if all(predicate(name.lower(), text, var) for predicate in predicates):
            names.append(name)
    return names


def _lb_first_name(variables: list[dict], candidates: tuple[str, ...], *fallback_tokens: str) -> str | None:
    by_name = _lb_variables_by_name(variables)
    for candidate in candidates:
        if candidate in by_name:
            return candidate
    for name in by_name:
        lowered = name.lower()
        if fallback_tokens and all(token in lowered for token in fallback_tokens):
            return name
    return None


def _lb_var(variables: list[dict], name: str | None) -> dict:
    if not name:
        return {}
    return _lb_variables_by_name(variables).get(name, {})


def _lb_exact_declared(var: dict, desired: Any) -> Any:
    choices = _declared_param_options(var.get("params") or {})
    if not choices:
        return desired
    for choice in choices:
        if _matches_declared_option(choice, desired):
            return choice
    normalized_desired = _normalize(desired)
    for choice in choices:
        if _normalize(choice) == normalized_desired:
            return choice
    return choices[0]


def _lb_set(rec: dict, variables: list[dict], name: str | None, value: Any, issues: list[str], reason: str) -> None:
    if not name or name not in rec:
        return
    if rec.get(name) != value:
        rec[name] = value
        issues.append(reason)


def _lb_set_declared(rec: dict, variables: list[dict], name: str | None, desired: Any, issues: list[str], reason: str) -> None:
    if not name or name not in rec:
        return
    value = _lb_exact_declared(_lb_var(variables, name), desired)
    _lb_set(rec, variables, name, value, issues, reason)


def _lb_bound_number(var: dict, current: Any, *, floor: float | None = None, ceiling: float | None = None, fallback: float = 0.0) -> float:
    params = var.get("params") or {}
    lo = _to_finite_float(params.get("min", params.get("lo")), floor)
    hi = _to_finite_float(params.get("max", params.get("hi")), ceiling)
    if floor is not None:
        lo = max(lo if lo is not None else floor, floor)
    if ceiling is not None:
        hi = min(hi if hi is not None else ceiling, ceiling)
    if lo is None:
        lo = 0.0
    if hi is None:
        hi = max(lo, fallback)
    if hi < lo:
        hi = lo
    value = _to_finite_float(current, None)
    if value is None or not math.isfinite(value):
        value = lo if lo == hi else random.uniform(lo, hi)
    value = max(lo, min(hi, value))
    dtype = str(var.get("dtype") or "").lower()
    precision = int((params.get("precision", 2) or 2))
    return float(round(value, precision) if dtype in _NUMERIC_DTYPES else value)


def _lb_unit_for_usage(usage: Any) -> str:
    token = _normalize(usage)
    return {
        "monetary": "INR",
        "voice": "MIN",
        "data": "GB",
        "sms": "SMS",
        "other": "UNIT",
    }.get(token, "INR")


def _lb_choose_usage(rec: dict, variables: list[dict]) -> str:
    names = _lb_find_fields(variables, lambda n, t, v: "usage_type" in n or "usagetype" in n)
    for name in names:
        choices = _declared_param_options(_lb_var(variables, name).get("params") or {})
        existing = rec.get(name)
        if existing is not None and (not choices or any(_matches_declared_option(existing, c) for c in choices)):
            return str(existing)
    return "monetary"


def _lb_history_timestamp_name(variables: list[dict]) -> str | None:
    # Prefer a top-up request timestamp because it is the natural anchor for a
    # transactional recharge history when the schema has no generic timestamp field.
    candidates = (
        "topup_requested_date_time", "topupbalance_requested_date", "topupbalance_requesteddate",
        "topup_request_timestamp", "recharge_timestamp", "transaction_timestamp", "record_timestamp", "timestamp",
    )
    for candidate in candidates:
        var = _lb_var(variables, candidate)
        if var and str(var.get("dtype", "")).lower() in {"datetime", "date"}:
            return candidate
    for var in variables:
        if str(var.get("dtype", "")).lower() == "datetime" and var.get("name"):
            name = str(var["name"]).lower()
            if any(token in name for token in ("request", "recharge", "transaction")):
                return str(var["name"])
    return _pick_timestamp_field(variables)


def _lb_plan_validity_days(rec: dict, variables: list[dict]) -> int:
    # The source Swagger models validFor but does not prescribe a plan-duration enumeration.
    # When the compiled scenario exposes the synthetic plan-duration field, it is authoritative
    # for the synthetic contract; otherwise the auto-top-up cadence supplies a sensible duration.
    plan_days_name = _lb_first_name(variables, ("recharge_plan_validity_days",), "plan", "validity", "days")
    if plan_days_name and rec.get(plan_days_name) is not None:
        try:
            candidate = int(float(rec.get(plan_days_name)))
            if candidate in {1, 7, 14, 28, 30, 56, 84}:
                return candidate
        except (TypeError, ValueError):
            pass
    period_name = _lb_first_name(
        variables,
        ("topupbalance_recurring_period", "topupbalance_recurringperiod"),
        "recurring", "period",
    )
    token = _normalize(rec.get(period_name)) if period_name else ""
    if token == "weekly":
        return 7
    if token == "fortnightly":
        return 14
    if token == "monthly":
        return 30
    return 28


def _lb_sync_reference_fields(rec: dict, variables: list[dict], issues: list[str]) -> None:
    by_name = _lb_variables_by_name(variables)
    def copy(source: str | None, targets: tuple[str, ...], reason: str) -> None:
        if not source or source not in rec or rec.get(source) is None:
            return
        for target in targets:
            if target in by_name and target in rec:
                _lb_set(rec, variables, target, rec[source], issues, reason)

    bucket_id = _lb_first_name(variables, ("bucket_id",), "bucket", "id")
    account_id = _lb_first_name(variables, ("account_id",))
    topup_id = _lb_first_name(variables, ("topupbalance_id", "topup_transaction_id"), "topup", "id")
    customer_id = _lb_first_name(variables, ("customer_id",), "customer", "id")
    copy(bucket_id, ("topupbalance_bucket_id", "balance_bucket_id"), "top-up/balance bucket references linked to the subscriber bucket")
    copy(bucket_id, ("topupbalance_bucket_href",), "top-up bucket href linked to bucket context")
    copy(account_id, ("bucket_party_account_id", "topupbalance_party_account_id"), "party-account references linked to the subscriber account")
    if account_id and account_id in rec and rec.get(account_id) is not None:
        account_value = str(rec.get(account_id))
        for target in ("bucket_party_account_href", "topupbalance_party_account_href"):
            if target in by_name and target in rec:
                _lb_set(rec, variables, target, f"https://example.test/telecom/party-account/{account_value}", issues, "party-account href linked to the subscriber account")
    if account_id and account_id in rec:
        _lb_set(rec, variables, "customer_engaged_party_id", rec.get("subscriber_id"), issues, "customer engaged party linked to subscriber")
    if customer_id and customer_id in rec and rec.get(customer_id) is None:
        _lb_set(rec, variables, customer_id, _prefixed_int({"prefix": "CUSTOMER-", "digits": 10}, rec), issues, "customer identity generated consistently for subscriber")
    if topup_id and topup_id in rec:
        topup_value = rec.get(topup_id)
        if topup_value is not None:
            _lb_set(rec, variables, "topup_transaction_id", topup_value, issues, "transaction id linked to TopupBalance resource")
    # Do not invent a RelatedTopupBalance resource. The transactional generator is responsible
    # for supplying a real earlier transaction id when one exists. Keeping a missing optional
    # reference null prevents orphaned ids and fabricated relationships.

    if "bucket_id" in rec:
        bucket_id_value = str(rec.get("bucket_id") or "")
        if bucket_id_value:
            _lb_set(rec, variables, "bucket_href", f"https://example.test/telecom/bucket/{bucket_id_value}", issues, "bucket href linked to bucket id")
    if "customer_id" in rec:
        customer_id_value = str(rec.get("customer_id") or "")
        if customer_id_value:
            _lb_set(rec, variables, "customer_href", f"https://example.test/telecom/customer/{customer_id_value}", issues, "customer href linked to customer id")
    if "topupbalance_id" in rec:
        topup_value = str(rec.get("topupbalance_id") or "")
        if topup_value:
            _lb_set(rec, variables, "topupbalance_href", f"https://example.test/telecom/topupBalance/{topup_value}", issues, "top-up href linked to top-up id")


def _lb_response_variant(rules: dict | None) -> str:
    """Return the explicit customer-response variant for decline/no-response scenarios."""
    if not isinstance(rules, dict):
        return "declined"
    scenario_type = str(rules.get("scenario_type") or "").strip().lower()
    sem = rules.get("scenario_semantics") if isinstance(rules.get("scenario_semantics"), dict) else {}
    expected = str(sem.get("expected_outcome") or "").strip().lower()
    context = re.sub(r"[^a-z0-9]+", " ", f"{scenario_type} {expected}")
    if "no response" in context and "decline" not in context:
        return "no_response"
    if "reject" in context and "decline" not in context:
        return "rejected"
    return "declined"


def _enforce_low_balance_topup_consistency(
    rec: dict, variables: list[dict], rules: dict | None = None
) -> tuple[dict, list[str]]:
    """Build and enforce a coherent Low Balance & Top-up business record.

    The official Swagger artifacts define the resource structure and field meanings, but not
    synthetic behavioral distributions. This layer therefore supplies deterministic modeling
    conventions while preserving source enum values and declared numeric constraints. It is
    applied to both newly generated and legacy-confirmed Low Balance scenarios.
    """
    rec = dict(rec)
    if not _lb_domain(rules):
        return rec, []

    issues: list[str] = []
    by_name = _lb_variables_by_name(variables)
    outcome_mode = str((rules or {}).get("scenario_mode") or (rules or {}).get("scenario_semantics", {}).get("outcome_mode") or "mixed").lower()

    # -------------------- entity identity/context --------------------
    _lb_sync_reference_fields(rec, variables, issues)

    usage_names = _lb_find_fields(variables, lambda n, t, v: "usage_type" in n or "usagetype" in n)
    usage = _lb_choose_usage(rec, variables)
    for name in usage_names:
        if name in rec:
            _lb_set_declared(rec, variables, name, usage, issues, f"{name} aligned to one coherent balance usage type")

    unit_names = _lb_find_fields(variables, lambda n, t, v: n.endswith("_unit") or n.endswith("_units") or "currency_unit" in n or "usage_unit" in n)
    unit = _lb_unit_for_usage(usage)
    for name in unit_names:
        if name in rec and str(by_name.get(name, {}).get("dtype", "")).lower() in {"string", "str", "text", "categorical"}:
            _lb_set(rec, variables, name, unit, issues, f"{name} aligned to usage-specific unit {unit}")

    # Entity-level lifecycle fields describe the same subscriber/customer/bucket context.
    customer_status = _lb_first_name(variables, ("customer_status",), "customer", "status")
    if customer_status:
        _lb_set_declared(rec, variables, customer_status, "active", issues, "customer lifecycle status aligned to an active prepaid customer")
    customer_status_reason = _lb_first_name(variables, ("customer_status_reason",), "customer", "status", "reason")
    if customer_status_reason and customer_status_reason in rec:
        _lb_set(rec, variables, customer_status_reason, "CURRENT_PREPAID_CUSTOMER", issues, "customer status reason aligned to the active prepaid lifecycle")
    bucket_status = _lb_first_name(variables, ("bucket_status",), "bucket", "status")
    if bucket_status:
        _lb_set_declared(rec, variables, bucket_status, "active", issues, "bucket lifecycle status aligned to active balance usage")
    bucket_name = _lb_first_name(variables, ("bucket_name",), "bucket", "name")
    if bucket_name:
        name_map = {"monetary": "Prepaid Wallet", "voice": "Voice Balance", "data": "Data Balance", "sms": "SMS Balance", "other": "Prepaid Balance"}
        _lb_set(rec, variables, bucket_name, name_map.get(_normalize(usage), "Prepaid Balance"), issues, "bucket name aligned to usage type")
    if "bucket_description" in by_name and "bucket_description" in rec:
        _lb_set(rec, variables, "bucket_description", f"Prepaid {str(usage).lower()} balance bucket", issues, "bucket description aligned to usage type")
    customer_label = "Prepaid Subscriber"
    suffix = re.search(r"([0-9]+)$", str(rec.get("subscriber_id") or ""))
    if suffix:
        customer_label = f"Prepaid Subscriber {suffix.group(1)}"
    if "customer_name" in by_name and "customer_name" in rec:
        _lb_set(rec, variables, "customer_name", customer_label, issues, "customer name linked to subscriber context")
    if "customer_id" in rec and rec.get("subscriber_id") is not None:
        customer_value = f"CUSTOMER-{suffix.group(1)}" if suffix else f"CUSTOMER-{random.randint(1000000000, 9999999999)}"
        _lb_set(rec, variables, "customer_id", customer_value, issues, "customer id deterministically linked to subscriber identity")
        if "customer_href" in rec:
            _lb_set(rec, variables, "customer_href", f"https://example.test/telecom/customer/{customer_value}", issues, "customer href synchronized with customer id")
        if "customer_engaged_party_id" in rec:
            _lb_set(rec, variables, "customer_engaged_party_id", rec.get("subscriber_id"), issues, "customer engaged party synchronized with subscriber identity")

    # Balance amounts are one snapshot, not independent random columns.
    remaining_names = [n for n in by_name if ("remaining" in n and "amount" in n) and str(by_name[n].get("dtype", "")).lower() in _NUMERIC_DTYPES]
    reserved_names = [n for n in by_name if ("reserved" in n and "amount" in n) and str(by_name[n].get("dtype", "")).lower() in _NUMERIC_DTYPES]
    threshold_name = _lb_first_name(variables, ("low_balance_trigger_threshold",), "low", "balance", "threshold")
    if threshold_name and threshold_name in rec:
        threshold = _lb_bound_number(by_name[threshold_name], rec.get(threshold_name), floor=1.0, fallback=300.0)
        _lb_set(rec, variables, threshold_name, threshold, issues, "low-balance threshold kept positive and within its declared bounds")
    else:
        threshold = 300.0
    balance_name = _lb_first_name(variables, ("balance_remaining_amount",), "balance", "remaining", "amount")
    if balance_name and balance_name in rec:
        balance_var = by_name[balance_name]
        balance = _lb_bound_number(balance_var, rec.get(balance_name), floor=0.0, ceiling=threshold, fallback=max(0.0, threshold * 0.5))
        # Keep trigger records meaningfully below threshold, not merely equal by accident.
        if balance >= threshold and threshold > 0:
            params = balance_var.get("params") or {}
            lo = _to_finite_float(params.get("min", params.get("lo")), 0.0) or 0.0
            balance = round(max(lo, threshold * random.uniform(0.15, 0.85)), int(params.get("precision", 2) or 2))
        _lb_set(rec, variables, balance_name, balance, issues, "remaining balance aligned with low-balance trigger threshold")
    else:
        balance = None

    # Keep source-backed bucket snapshot values coherent with the transactional balance when
    # they are actually generated at transaction grain.
    for name in remaining_names:
        var = by_name[name]
        if str(var.get("scope") or "").lower() == "transaction" and balance is not None:
            _lb_set(rec, variables, name, _lb_bound_number(var, balance, floor=0.0, ceiling=1000.0, fallback=balance), issues, f"{name} aligned with transactional remaining balance")
    reserved_cap = balance if balance is not None else None
    for name in reserved_names:
        var = by_name[name]
        current = rec.get(name)
        if reserved_cap is not None:
            _lb_set(rec, variables, name, _lb_bound_number(var, current, floor=0.0, ceiling=reserved_cap, fallback=max(0.0, reserved_cap * 0.2)), issues, f"{name} constrained not to exceed remaining balance")

    # -------------------- transaction timeline --------------------
    request_fields = [
        n for n in by_name
        if ("request" in n or "requested" in n) and ("topup" in n or "recharge" in n or n.startswith("bucket_"))
        and str(by_name[n].get("dtype", "")).lower() == "datetime"
    ]
    confirmation_fields = [
        n for n in by_name
        if ("confirm" in n or "completion" in n) and ("topup" in n or "recharge" in n or n.startswith("bucket_"))
        and str(by_name[n].get("dtype", "")).lower() == "datetime"
    ]
    request_name = _lb_history_timestamp_name(variables)
    request_dt = _qa_parse_dt(rec.get(request_name)) if request_name and request_name in rec else None
    if request_dt is None:
        # Use an existing request timestamp if available; otherwise create one once for the
        # complete transaction instead of independently sampling each request-like field.
        for name in request_fields:
            request_dt = _qa_parse_dt(rec.get(name))
            if request_dt is not None:
                request_name = name
                break
    if request_dt is None:
        request_dt = datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 72), minutes=random.randint(0, 59))

    # Scenario-specific intervention may happen before the transaction request.
    trigger_field = _lb_first_name(variables, ("low_balance_trigger_timestamp",), "low", "balance", "trigger", "timestamp")
    if trigger_field and trigger_field in rec:
        trigger_dt = request_dt - timedelta(minutes=random.randint(5, 120))
        _lb_set(rec, variables, trigger_field, _format_datetime(trigger_dt, by_name[trigger_field].get("params") or {}), issues, "low-balance trigger placed before recharge request")

    desired_status = "completed"
    if outcome_mode == "negative":
        desired_status = "failed"
    elif outcome_mode == "decline_or_no_response":
        desired_status = "cancelled"

    status_name = _lb_first_name(variables, ("topupbalance_status",), "topup", "status")
    execution_name = _lb_first_name(variables, ("topup_execution_status",), "topup", "execution", "status")
    outcome_name = _lb_first_name(variables, ("recharge_outcome",), "recharge", "outcome")
    _lb_set_declared(rec, variables, status_name, desired_status, issues, "TopupBalance status aligned to scenario lifecycle")
    _lb_set_declared(rec, variables, execution_name, desired_status, issues, "top-up execution status aligned to TopupBalance status")

    success_state = desired_status in {"completed", "approved", "accepted", "success"}
    confirmation_dt = request_dt + timedelta(minutes=random.randint(1, 60)) if success_state else None

    # All request-like and confirmation-like timestamps in the same transaction share one
    # authoritative event pair; they are never independently sampled.
    for name in request_fields:
        _lb_set(rec, variables, name, _format_datetime(request_dt, by_name[name].get("params") or {}), issues, f"{name} synchronized to the transaction request event")
    for name in confirmation_fields:
        if success_state and confirmation_dt is not None:
            _lb_set(rec, variables, name, _format_datetime(confirmation_dt, by_name[name].get("params") or {}), issues, f"{name} synchronized to the transaction confirmation event")
        elif by_name[name].get("nullable", True):
            _lb_set(rec, variables, name, None, issues, f"{name} cleared because the transaction did not complete")

    recharge_timestamp = _lb_first_name(variables, ("recharge_timestamp",), "recharge", "timestamp")
    if recharge_timestamp and recharge_timestamp in rec:
        recharge_dt = confirmation_dt or request_dt
        _lb_set(rec, variables, recharge_timestamp, _format_datetime(recharge_dt, by_name[recharge_timestamp].get("params") or {}), issues, "recharge timestamp linked to transaction lifecycle")

    topup_req = _lb_first_name(variables, ("topupbalance_requested_date", "topupbalance_requesteddate"), "topup", "requested")
    topup_conf = _lb_first_name(variables, ("topupbalance_confirmation_date", "topupbalance_confirmationdate"), "topup", "confirmation")
    if topup_req:
        _lb_set(rec, variables, topup_req, _format_datetime(request_dt, by_name[topup_req].get("params") or {}), issues, "TopupBalance requestedDate linked to transaction request")
    if topup_conf:
        if confirmation_dt is not None:
            _lb_set(rec, variables, topup_conf, _format_datetime(confirmation_dt, by_name[topup_conf].get("params") or {}), issues, "TopupBalance confirmationDate linked to transaction confirmation")
        elif by_name[topup_conf].get("nullable", True):
            _lb_set(rec, variables, topup_conf, None, issues, "TopupBalance confirmationDate cleared for unsuccessful transaction")

    # Plan/validity lifecycle. validFor is a period, not two independent random timestamps.
    plan_days_name = _lb_first_name(variables, ("recharge_plan_validity_days",), "plan", "validity", "days")
    plan_code_name = _lb_first_name(variables, ("recharge_plan_code",), "plan", "code")
    plan_days_choices = (1, 7, 14, 28, 30, 56, 84)
    recurring_name = _lb_first_name(variables, ("topupbalance_recurring_period", "topupbalance_recurringperiod"), "recurring", "period")
    periods_name = _lb_first_name(variables, ("topupbalance_number_of_periods", "topupbalance_numberofperiods"), "number", "period")
    period_token = _normalize(rec.get(recurring_name)) if recurring_name else ""
    default_plan_days = {"weekly": 7, "fortnightly": 14, "monthly": 30}.get(period_token, random.choice(plan_days_choices))
    if plan_days_name and plan_days_name in rec:
        current_days = rec.get(plan_days_name)
        try:
            current_days = int(float(current_days))
        except (TypeError, ValueError):
            current_days = default_plan_days
        if current_days not in plan_days_choices:
            current_days = default_plan_days
        _lb_set(rec, variables, plan_days_name, int(current_days), issues, "recharge plan validity duration normalized to a supported synthetic prepaid plan")
        default_plan_days = int(current_days)
    if plan_code_name and plan_code_name in rec:
        code = f"PREPAID_{default_plan_days}D"
        _lb_set_declared(rec, variables, plan_code_name, code, issues, "recharge plan code synchronized with plan validity duration")

    validity_pairs = []
    for prefix in ("topupbalance", "topup"):
        start = _lb_first_name(variables, (f"{prefix}_valid_for_start_date_time", f"{prefix}_validity_start_date_time"), prefix, "valid", "start")
        end = _lb_first_name(variables, (f"{prefix}_valid_for_end_date_time", f"{prefix}_validity_end_date_time"), prefix, "valid", "end")
        if start or end:
            validity_pairs.append((start, end))
    duration_days = _lb_plan_validity_days(rec, variables)
    for start_name, end_name in validity_pairs:
        if start_name and start_name in rec:
            if success_state and confirmation_dt is not None:
                start_dt = confirmation_dt
                end_dt = start_dt + timedelta(days=duration_days)
                _lb_set(rec, variables, start_name, _format_datetime(start_dt, by_name[start_name].get("params") or {}), issues, f"{start_name} derived from recharge confirmation")
                if end_name and end_name in rec:
                    _lb_set(rec, variables, end_name, _format_datetime(end_dt, by_name[end_name].get("params") or {}), issues, f"{end_name} derived from plan validity duration")
            elif by_name[start_name].get("nullable", True):
                _lb_set(rec, variables, start_name, None, issues, f"{start_name} cleared because no successful recharge validity was created")
                if end_name and by_name[end_name].get("nullable", True):
                    _lb_set(rec, variables, end_name, None, issues, f"{end_name} cleared because no successful recharge validity was created")
        elif end_name and end_name in rec and by_name[end_name].get("nullable", True):
            _lb_set(rec, variables, end_name, None, issues, f"{end_name} cleared because its validity start is unavailable")

    # Stable entity validity windows describe the entity itself and contain the operational history.
    now = datetime.now(timezone.utc)
    for token in ("bucket", "customer"):
        starts = [n for n in by_name if token in n and "valid_for_start" in n and str(by_name[n].get("dtype", "")).lower() == "datetime"]
        ends = [n for n in by_name if token in n and "valid_for_end" in n and str(by_name[n].get("dtype", "")).lower() == "datetime"]
        if starts and ends:
            stable_start = now - timedelta(days=365)
            stable_end = now + timedelta(days=365)
            for name in starts:
                if str(by_name[name].get("scope") or "").lower() == "entity" and name in rec:
                    _lb_set(rec, variables, name, _format_datetime(stable_start, by_name[name].get("params") or {}), issues, f"{name} aligned to stable entity validity window")
            for name in ends:
                if str(by_name[name].get("scope") or "").lower() == "entity" and name in rec:
                    _lb_set(rec, variables, name, _format_datetime(stable_end, by_name[name].get("params") or {}), issues, f"{name} aligned to stable entity validity window")

    # Monetary movement is one transaction-level quantity. Keep all amount aliases equal.
    amount_fields = [n for n in by_name if "topup" in n and "amount" in n and "unit" not in n and str(by_name[n].get("dtype", "")).lower() in _NUMERIC_DTYPES]
    canonical_amount = None
    for name in amount_fields:
        value = _to_finite_float(rec.get(name), None)
        if value is not None and math.isfinite(value) and value >= 0:
            canonical_amount = value
            break
    if canonical_amount is None:
        canonical_amount = round(random.uniform(10.0, 1000.0), 2)
    for name in amount_fields:
        var = by_name[name]
        value = _lb_bound_number(var, canonical_amount, floor=0.0, fallback=canonical_amount)
        _lb_set(rec, variables, name, value, issues, f"{name} aligned to the transaction recharge amount")

    # Same-transaction operational dimensions must share one context.
    auto_names = [n for n in by_name if "auto_topup" in n or "autotopup" in n or ("auto" in n and "topup" in n)]
    auto_value = None
    for name in auto_names:
        existing = _boolean_semantic(rec.get(name))
        if existing is not None:
            auto_value = existing
            break
    if auto_value is None:
        auto_value = False
    # A customer decline/no-response scenario represents an explicit customer interaction, not
    # an autonomous recharge. Keep auto-top-up off for that scenario.
    if outcome_mode == "decline_or_no_response":
        auto_value = False
    for name in auto_names:
        if name in rec:
            _lb_set(rec, variables, name, auto_value, issues, f"{name} synchronized to one auto-top-up state")

    if auto_value:
        if recurring_name and recurring_name in rec:
            recurring_choices = _declared_param_options(by_name[recurring_name].get("params") or {})
            recurring = next((c for c in recurring_choices if _normalize(c) in {"weekly", "fortnightly", "monthly"}), None) or (random.choice(recurring_choices) if recurring_choices else "monthly")
            _lb_set(rec, variables, recurring_name, recurring, issues, "recurring period synchronized with enabled auto-top-up")
        if periods_name and periods_name in rec:
            pv = by_name[periods_name]
            pp = pv.get("params") or {}
            lo = max(1, int(_to_finite_float(pp.get("min", pp.get("lo")), 1) or 1))
            hi_raw = _to_finite_float(pp.get("max", pp.get("hi")), None)
            hi = max(lo, int(hi_raw if hi_raw is not None else max(lo, 12)))
            _lb_set(rec, variables, periods_name, random.randint(lo, hi), issues, "auto-top-up period count made compatible with recurring behavior")
    else:
        for name in (recurring_name, periods_name):
            if name and name in rec and by_name[name].get("nullable", True):
                _lb_set(rec, variables, name, None, issues, f"{name} cleared because auto-top-up is disabled")

    # Low-balance warning/intervention semantics.
    warning_name = _lb_first_name(variables, ("low_balance_warning_sent_flag",), "low", "balance", "warning", "sent")
    suppression_name = _lb_first_name(variables, ("intervention_suppression_state",), "suppression", "state")
    eligibility_name = _lb_first_name(variables, ("intervention_eligibility",), "intervention", "eligibility")
    offer_name = _lb_first_name(variables, ("retention_intervention_offer_code",), "retention", "offer")
    decision_name = _lb_first_name(variables, ("customer_decision",), "customer", "decision")
    decline_reason_name = _lb_first_name(variables, ("decline_reason",), "decline", "reason")

    if outcome_mode == "suppression":
        if suppression_name:
            _lb_set_declared(rec, variables, suppression_name, "SUPPRESSED", issues, "suppression scenario explicitly marks intervention as suppressed")
        if eligibility_name:
            _lb_set(rec, variables, eligibility_name, False, issues, "suppressed intervention is not eligible")
        if warning_name:
            _lb_set(rec, variables, warning_name, False, issues, "suppressed intervention cannot send a warning")
        if offer_name and by_name[offer_name].get("nullable", True):
            _lb_set(rec, variables, offer_name, None, issues, "suppressed intervention has no retention offer")
    elif outcome_mode == "decline_or_no_response":
        response_variant = _lb_response_variant(rules)
        response_value = {"declined": "DECLINED", "rejected": "REJECTED", "no_response": "NO_RESPONSE"}[response_variant]
        if decision_name:
            _lb_set_declared(rec, variables, decision_name, response_value, issues, "customer response synchronized to the explicit scenario variant")
        if decline_reason_name:
            reason_value = "TIMEOUT" if response_variant == "no_response" else ("TRUST" if response_variant == "rejected" else "PRICE")
            declared_reasons = _declared_param_options(by_name.get(decline_reason_name, {}).get("params") or {})
            if declared_reasons and not any(_normalize(reason_value) == _normalize(x) for x in declared_reasons):
                reason_value = declared_reasons[0]
            _lb_set_declared(rec, variables, decline_reason_name, reason_value, issues, "customer response reason synchronized to the explicit scenario variant")
        if eligibility_name:
            _lb_set(rec, variables, eligibility_name, True, issues, "customer response scenario keeps the intervention eligible before the response")
        if warning_name:
            _lb_set(rec, variables, warning_name, True, issues, "customer response scenario requires an intervention presentation event")
        if offer_name and offer_name in rec:
            choices = _declared_param_options(by_name[offer_name].get("params") or {})
            if choices:
                _lb_set(rec, variables, offer_name, choices[0], issues, "customer response scenario includes a presented intervention")
    else:
        if eligibility_name:
            _lb_set(rec, variables, eligibility_name, True, issues, "active intervention scenario keeps the subscriber eligible")
        if warning_name and outcome_mode != "mixed":
            _lb_set(rec, variables, warning_name, True, issues, "low-balance intervention scenario requires a warning event")
        if offer_name and offer_name in rec and rec.get(offer_name) is None and not by_name[offer_name].get("nullable", True):
            choices = _declared_param_options(by_name[offer_name].get("params") or {})
            if choices:
                _lb_set(rec, variables, offer_name, choices[0], issues, "required intervention offer populated")

    channel_names = [n for n in by_name if "channel_name" in n or (n.endswith("_channel") and "referred_type" not in n)]
    channel = next((str(rec[n]) for n in channel_names if rec.get(n)), None)
    channel_choices = ["APP", "WEB", "USSD", "SMS", "WHATSAPP", "IVR", "RETAIL"]
    if not channel or channel not in channel_choices:
        channel = random.choice(channel_choices)
    for name in channel_names:
        if name in rec:
            _lb_set(rec, variables, name, channel, issues, f"{name} synchronized to one recharge channel")

    payment_names = [n for n in by_name if "payment_method_name" in n or ("paymentmethod" in n and n.endswith("_name"))]
    payment = next((str(rec[n]) for n in payment_names if rec.get(n)), None)
    payment_choices = ["UPI", "CREDIT_CARD", "DEBIT_CARD", "WALLET", "CASH", "AUTO_DEBIT"]
    if not payment or payment not in payment_choices:
        payment = random.choice(payment_choices)
    for name in payment_names:
        if name in rec:
            _lb_set(rec, variables, name, payment, issues, f"{name} synchronized to one recharge payment method")

    # A voucher is meaningful only for a voucher-based payment path. This source contract does
    # not declare VOUCHER as one of the generated payment choices above, so optional voucher
    # fields are cleared instead of carrying an unrelated identifier.
    for name in by_name:
        if "voucher" in name and name in rec and by_name[name].get("nullable", True):
            if _normalize(payment) != "voucher":
                _lb_set(rec, variables, name, None, issues, f"{name} cleared because payment method is not voucher-based")

    # The source model describes reason, trigger, channel and payment method as properties of
    # the same recharge. Keep the aliases synchronized rather than independently sampled.
    reason_name = _lb_first_name(variables, ("topupbalance_reason",), "topup", "reason")
    trigger_reason_name = _lb_first_name(variables, ("topup_trigger_reason",), "topup", "trigger", "reason")
    reason_choices = _declared_param_options(by_name.get(reason_name, {}).get("params") or {}) if reason_name else []
    trigger_choices = _declared_param_options(by_name.get(trigger_reason_name, {}).get("params") or {}) if trigger_reason_name else []
    coherent_reason = None
    if reason_name and rec.get(reason_name) is not None:
        coherent_reason = rec.get(reason_name)
    elif trigger_reason_name and rec.get(trigger_reason_name) is not None:
        coherent_reason = rec.get(trigger_reason_name)
    if coherent_reason is None:
        preferred = ["LOW_BALANCE", "DATA_EXHAUSTED", "VALIDITY_EXPIRY", "CUSTOMER_REQUEST"]
        candidates = reason_choices or trigger_choices or preferred
        coherent_reason = random.choice(candidates)
    if reason_name and reason_name in rec:
        _lb_set_declared(rec, variables, reason_name, coherent_reason, issues, "TopupBalance reason synchronized to the recharge trigger")
    if trigger_reason_name and trigger_reason_name in rec:
        _lb_set_declared(rec, variables, trigger_reason_name, coherent_reason, issues, "top-up trigger reason synchronized with TopupBalance reason")

    # Source-backed reference names and types must describe the same objects.
    bucket_name = _lb_first_name(variables, ("bucket_name",), "bucket", "name")
    account_name = f"Prepaid Account {str(rec.get('account_id') or '').replace('ACC-', '')}".strip() or "Prepaid Account"
    subscriber_label = f"Prepaid Subscriber {re.search(r'([0-9]+)$', str(rec.get('subscriber_id') or '')).group(1)}" if re.search(r'([0-9]+)$', str(rec.get('subscriber_id') or '')) else "Prepaid Subscriber"
    canonical_bucket_label = str(rec.get(bucket_name) or "Prepaid Balance") if bucket_name else "Prepaid Balance"
    for name in ("topupbalance_bucket_name",):
        if name in rec:
            _lb_set(rec, variables, name, canonical_bucket_label, issues, f"{name} linked to the authoritative bucket name")
    for name in ("bucket_party_account_name", "topupbalance_party_account_name"):
        if name in rec:
            _lb_set(rec, variables, name, account_name, issues, f"{name} linked to the subscriber account")
    for name in ("topupbalance_requestor_name",):
        if name in rec:
            _lb_set(rec, variables, name, "Auto Top-up Service" if auto_value else subscriber_label, issues, f"{name} linked to the actor performing the recharge")
    for name in ("customer_engaged_party_name",):
        if name in rec:
            _lb_set(rec, variables, name, subscriber_label, issues, f"{name} linked to subscriber identity")
    for name in ("bucket_party_account_status", "topupbalance_party_account_status"):
        if name in rec:
            _lb_set(rec, variables, name, "paid", issues, f"{name} aligned with prepaid account semantics")
    for name in ("bucket_remaining_value_name",):
        if name in rec and balance is not None:
            _lb_set(rec, variables, name, f"{balance:.2f} {unit}", issues, f"{name} formatted from the authoritative remaining balance")

    # Keep source-backed amount/unit aliases together.
    for name in ("bucket_remaining_value_units", "bucket_reserved_value_units", "topupbalance_amount_units", "topup_amount_currency_unit"):
        if name in rec:
            _lb_set(rec, variables, name, unit, issues, f"{name} aligned to the transaction usage unit")
    for name in ("bucket_is_shared", "balance_is_shared"):
        if name in rec:
            shared = _boolean_semantic(rec.get(name))
            if shared is None:
                shared = False
            _lb_set(rec, variables, name, shared, issues, f"{name} synchronized to one bucket sharing state")

    # Link the canonical top-up aliases.
    for source, targets in ((
        "topupbalance_channel_name", ("topup_channel_name",)),
        ("topupbalance_payment_method_name", ("topup_payment_method_name",)),
        ("topupbalance_is_auto_topup", ("is_auto_topup_enabled",)),
        ("topupbalance_amount_amount", ("topup_recharge_amount",)),
        ("topupbalance_valid_for_start_date_time", ("topup_validity_start_date_time",)),
        ("topupbalance_valid_for_end_date_time", ("topup_validity_end_date_time",)),
    ):
        if source in rec:
            for target in targets:
                if target in rec:
                    _lb_set(rec, variables, target, rec.get(source), issues, f"{target} synchronized to its source-backed top-up alias")

    # Reference role/type semantics are source-derived where descriptions define the vocabulary.
    # A related top-up reference is only populated when it points to a real, distinct prior
    # transaction. Otherwise the optional flattened reference fields stay null.
    related_topup_value = rec.get("topupbalance_balance_topup_id")
    current_topup_value = rec.get("topupbalance_id")
    if related_topup_value and current_topup_value and str(related_topup_value) != str(current_topup_value):
        _lb_set_declared(rec, variables, "topupbalance_balance_topup_role", "child", issues, "related TopupBalance role constrained to a real earlier child reference")
        _lb_set_declared(rec, variables, "topupbalance_balance_topup_referred_type", "TopupBalance", issues, "related TopupBalance reference type aligned")
    else:
        for _name in ("topupbalance_balance_topup_id", "topupbalance_balance_topup_href", "topupbalance_balance_topup_name", "topupbalance_balance_topup_role", "topupbalance_balance_topup_referred_type"):
            if _name in rec and _name in _lb_variables_by_name(variables) and _lb_variables_by_name(variables)[_name].get("nullable", True):
                _lb_set(rec, variables, _name, None, issues, "optional related TopupBalance reference cleared when no real related resource exists")
    _lb_set_declared(rec, variables, "topupbalance_bucket_referred_type", "Bucket", issues, "bucket reference type aligned")
    _lb_set_declared(rec, variables, "topupbalance_channel_referred_type", "Channel", issues, "channel reference type aligned")
    _lb_set_declared(rec, variables, "topupbalance_payment_method_referred_type", "PaymentMethod", issues, "payment-method reference type aligned")
    _lb_set_declared(rec, variables, "bucket_party_account_referred_type", "PartyAccount", issues, "bucket party-account reference type aligned")
    _lb_set_declared(rec, variables, "topupbalance_party_account_referred_type", "PartyAccount", issues, "top-up party-account reference type aligned")
    _lb_set_declared(rec, variables, "customer_engaged_party_role", "subscriber", issues, "customer engaged-party role aligned to subscriber")
    _lb_set_declared(rec, variables, "customer_engaged_party_referred_type", "Individual", issues, "customer engaged-party reference type aligned")
    _lb_set_declared(rec, variables, "topupbalance_requestor_role", "system" if auto_value else "subscriber", issues, "requestor role aligned to the recharge actor")
    _lb_set_declared(rec, variables, "topupbalance_requestor_referred_type", "Organization" if auto_value else "Individual", issues, "requestor reference type aligned to the recharge actor")

    # Deterministic reference ids/hrefs must describe the same underlying objects as the names/types.
    subscriber_suffix = re.search(r"([0-9]+)$", str(rec.get("subscriber_id") or ""))
    suffix_value = subscriber_suffix.group(1) if subscriber_suffix else str(random.randint(1000000000, 9999999999))
    if "topupbalance_channel_id" in rec:
        channel_id = f"CHANNEL-{str(channel or 'APP')}-{suffix_value}"
        _lb_set(rec, variables, "topupbalance_channel_id", channel_id, issues, "channel id linked to channel value")
        if "topupbalance_channel_href" in rec:
            _lb_set(rec, variables, "topupbalance_channel_href", f"https://example.test/telecom/channel/{channel_id}", issues, "channel href linked to channel id")
    if "topupbalance_payment_method_id" in rec:
        payment_id = f"PAYMENT_METHOD-{str(payment or 'UPI')}-{suffix_value}"
        _lb_set(rec, variables, "topupbalance_payment_method_id", payment_id, issues, "payment method id linked to payment method value")
        if "topupbalance_payment_method_href" in rec:
            _lb_set(rec, variables, "topupbalance_payment_method_href", f"https://example.test/telecom/payment-method/{payment_id}", issues, "payment method href linked to payment method id")
    if "topupbalance_requestor_id" in rec:
        requestor_id = f"REQUESTOR-{suffix_value}"
        _lb_set(rec, variables, "topupbalance_requestor_id", requestor_id, issues, "requestor id linked to subscriber actor")
        if "topupbalance_requestor_href" in rec:
            _lb_set(rec, variables, "topupbalance_requestor_href", f"https://example.test/telecom/requestor/{requestor_id}", issues, "requestor href linked to requestor id")
    if "customer_engaged_party_href" in rec:
        _lb_set(rec, variables, "customer_engaged_party_href", f"https://example.test/telecom/party/{rec.get('subscriber_id')}", issues, "engaged party href linked to subscriber")
    # Link a related TopupBalance only to a real prior transaction supplied by the history
    # generator. Never fabricate a RELATED_TOPUP id.
    related_id = rec.get("topupbalance_balance_topup_id")
    current_id = rec.get("topupbalance_id")
    if related_id and current_id and str(related_id) != str(current_id):
        if "topupbalance_balance_topup_role" in rec:
            _lb_set_declared(rec, variables, "topupbalance_balance_topup_role", "child", issues, "related TopupBalance role linked to an actual prior transaction")
        if "topupbalance_balance_topup_referred_type" in rec:
            _lb_set_declared(rec, variables, "topupbalance_balance_topup_referred_type", "TopupBalance", issues, "related TopupBalance type linked to an actual prior transaction")
        if "topupbalance_balance_topup_href" in rec:
            _lb_set(rec, variables, "topupbalance_balance_topup_href", f"https://example.test/telecom/topupBalance/{related_id}", issues, "related TopupBalance href linked to actual reference id")
        if "topupbalance_balance_topup_name" in rec:
            _lb_set(rec, variables, "topupbalance_balance_topup_name", "Related Top-up", issues, "related TopupBalance name describes the referenced resource")
    else:
        for _name in ("topupbalance_balance_topup_id", "topupbalance_balance_topup_href", "topupbalance_balance_topup_name", "topupbalance_balance_topup_role", "topupbalance_balance_topup_referred_type"):
            if _name in rec and _name in by_name and by_name[_name].get("nullable", True):
                _lb_set(rec, variables, _name, None, issues, "optional related TopupBalance fields cleared when no real reference exists")

    # Scenario outcome mirrors the authoritative execution state, using the scenario field's
    # own declared vocabulary rather than inventing a new enum.
    if outcome_name and outcome_name in rec:
        if outcome_mode == "negative":
            desired_outcome = "FAILED"
        elif outcome_mode == "decline_or_no_response":
            response_value = {"declined": "DECLINED", "rejected": "REJECTED", "no_response": "NO_RESPONSE"}[_lb_response_variant(rules)]
            desired_outcome = response_value
        elif desired_status == "completed":
            desired_outcome = "COMPLETED"
        else:
            desired_outcome = str(desired_status).upper()
        _lb_set_declared(rec, variables, outcome_name, desired_outcome, issues, "scenario outcome synchronized with transaction execution state")

    # Latency is derived from the actual timestamps, never independently sampled.
    latency_name = _lb_first_name(variables, ("recharge_latency_hours",), "recharge", "latency", "hours")
    if latency_name and latency_name in rec:
        end_dt = confirmation_dt or request_dt
        start_dt = _qa_parse_dt(rec.get(trigger_field)) if trigger_field else None
        if start_dt is None:
            start_dt = request_dt
        hours = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)
        params = by_name[latency_name].get("params") or {}
        lo = _to_finite_float(params.get("min", params.get("lo")), 0.0) or 0.0
        hi = _to_finite_float(params.get("max", params.get("hi")), None)
        if hi is not None:
            hours = min(hours, hi)
        hours = max(hours, lo)
        precision = int(params.get("precision", 2) or 2)
        latency_value = round(hours, precision)
        if str(by_name[latency_name].get("dtype") or "").lower() in {"int", "integer"}:
            latency_value = int(round(latency_value))
        _lb_set(rec, variables, latency_name, latency_value, issues, "recharge latency derived from trigger/request/confirmation timestamps")

    # Descriptive aliases should not contradict the same object.
    if "topupbalance_description" in rec:
        _lb_set(rec, variables, "topupbalance_description", "Prepaid recharge transaction", issues, "top-up description aligned to resource semantics")
    if "topupbalance_usagetype" in rec:
        _lb_set_declared(rec, variables, "topupbalance_usagetype", usage, issues, "TopupBalance usageType aligned to bucket usage")
    if "bucket_usagetype" in rec:
        _lb_set_declared(rec, variables, "bucket_usagetype", usage, issues, "bucket usageType aligned to top-up usage")

    # Generic paired start/end periods across the full schema.
    paired: list[tuple[str, str]] = []
    for name in by_name:
        lname = name.lower()
        if lname.endswith("_start_date_time"):
            candidate = name[:-len("_start_date_time")] + "_end_date_time"
        elif lname.endswith("_start_datetime"):
            candidate = name[:-len("_start_datetime")] + "_end_datetime"
        else:
            continue
        if candidate in by_name:
            paired.append((name, candidate))
    for start_name, end_name in paired:
        start_dt = _qa_parse_dt(rec.get(start_name))
        end_dt = _qa_parse_dt(rec.get(end_name))
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            _lb_set(rec, variables, end_name, _format_datetime(start_dt, by_name[end_name].get("params") or {}), issues, f"{end_name} corrected to be on/after {start_name}")

    return rec, issues


def _assert_low_balance_topup_consistency(rec: dict, variables: list[dict], rules: dict | None = None) -> None:
    """Fail closed on the complete Low Balance business contract after all repairs."""
    if not _lb_domain(rules):
        return
    by_name = _lb_variables_by_name(variables)
    errors: list[str] = []

    def dt(name: str) -> datetime | None:
        return _qa_parse_dt(rec.get(name)) if name in rec else None

    # Schema-level type/options/required checks are performed separately; here focus on
    # cross-field business invariants that cannot be expressed as a scalar parameter.
    for name, var in by_name.items():
        if bool(var.get("required")) and (name not in rec or rec.get(name) is None):
            errors.append(f"required field '{name}' is missing")
        if name in rec and rec.get(name) is not None:
            declared = _declared_param_options(var.get("params") or {})
            if declared and not any(_matches_declared_option(rec.get(name), opt) for opt in declared):
                errors.append(f"{name} is outside its declared choice set")

    # Every obvious validity period must be ordered.
    for start, end in (
        ("bucket_valid_for_start_date_time", "bucket_valid_for_end_date_time"),
        ("topupbalance_valid_for_start_date_time", "topupbalance_valid_for_end_date_time"),
        ("topup_validity_start_date_time", "topup_validity_end_date_time"),
        ("customer_valid_for_start_date_time", "customer_valid_for_end_date_time"),
    ):
        if start in rec and end in rec:
            a, b = dt(start), dt(end)
            if a is not None and b is not None and a > b:
                errors.append(f"{start} occurs after {end}")

    request_fields = [n for n in by_name if ("requested" in n or "request" in n) and str(by_name[n].get("dtype", "")).lower() == "datetime" and "topup" in n]
    confirmation_fields = [n for n in by_name if ("confirmation" in n or "confirm" in n or "completion" in n) and str(by_name[n].get("dtype", "")).lower() == "datetime" and "topup" in n]
    request_dt = next((dt(n) for n in request_fields if dt(n) is not None), None)
    confirmation_dt = next((dt(n) for n in confirmation_fields if dt(n) is not None), None)
    if request_dt is not None and confirmation_dt is not None and confirmation_dt < request_dt:
        errors.append("TopupBalance confirmation occurs before request")

    # Duplicate aliases must represent the same event.
    alias_groups = (
        ("topupbalance_requested_date", "topup_requested_date_time"),
        ("topupbalance_confirmation_date", "topup_confirmation_date_time"),
        ("topupbalance_amount_amount", "topup_recharge_amount"),
        ("bucket_id", "topupbalance_bucket_id", "balance_bucket_id"),
    )
    for group in alias_groups:
        present = [rec[n] for n in group if n in rec and rec.get(n) is not None]
        if len(present) >= 2 and len({_normalize(x) for x in present}) > 1:
            errors.append(f"related aliases disagree: {group}")

    # Monetary balance logic.
    rem = _to_finite_float(rec.get("balance_remaining_amount"), None)
    threshold = _to_finite_float(rec.get("low_balance_trigger_threshold"), None)
    reserved = _to_finite_float(rec.get("balance_reserved_amount"), None)
    if rem is not None and threshold is not None and rem > threshold + 1e-9:
        errors.append("remaining balance exceeds low-balance trigger threshold")
    if rem is not None and reserved is not None and reserved > rem + 1e-9:
        errors.append("reserved balance exceeds remaining balance")

    # Usage/unit coherence.
    usage_values = {str(rec[n]) for n in by_name if "usage_type" in n or "usagetype" in n if n in rec and rec.get(n) is not None}
    if len(usage_values) > 1:
        errors.append("usageType fields disagree")
    expected_unit = _lb_unit_for_usage(next(iter(usage_values), "monetary"))
    unit_values = {str(rec[n]) for n in by_name if (n.endswith("_unit") or n.endswith("_units") or "currency_unit" in n or "usage_unit" in n) and n in rec and rec.get(n) is not None}
    if unit_values and expected_unit not in unit_values:
        errors.append(f"unit fields are not aligned to usage type ({expected_unit})")

    # Same-event aliases must describe one transaction.
    alias_groups = (
        ("topupbalance_channel_name", "topup_channel_name"),
        ("topupbalance_payment_method_name", "topup_payment_method_name"),
        ("topupbalance_is_auto_topup", "is_auto_topup_enabled"),
        ("topupbalance_amount_amount", "topup_recharge_amount"),
        ("topupbalance_amount_units", "topup_amount_currency_unit"),
        ("topupbalance_valid_for_start_date_time", "topup_validity_start_date_time"),
        ("topupbalance_valid_for_end_date_time", "topup_validity_end_date_time"),
        ("bucket_name", "topupbalance_bucket_name"),
        ("account_id", "bucket_party_account_id", "topupbalance_party_account_id"),
        ("subscriber_id", "customer_engaged_party_id"),
    )
    for group in alias_groups:
        vals = [rec[n] for n in group if n in rec and rec.get(n) is not None]
        if len(vals) >= 2 and len({_normalize(v) for v in vals}) > 1:
            errors.append(f"related aliases disagree: {group}")

    # Top-up recurrence is conditional on the auto-top-up flag.
    auto_name = _lb_first_name(variables, ("topupbalance_is_auto_topup", "is_auto_topup_enabled"), "auto", "topup")
    auto = _boolean_semantic(rec.get(auto_name)) if auto_name else None
    rec_period_name = _lb_first_name(variables, ("topupbalance_recurring_period", "topupbalance_recurringperiod"), "recurring", "period")
    periods_name = _lb_first_name(variables, ("topupbalance_number_of_periods", "topupbalance_numberofperiods"), "number", "period")
    if auto is False:
        if rec_period_name and rec.get(rec_period_name) is not None:
            errors.append("recurring period is populated while auto-top-up is disabled")
        if periods_name and rec.get(periods_name) is not None:
            errors.append("number of periods is populated while auto-top-up is disabled")
    elif auto is True:
        if rec_period_name and rec.get(rec_period_name) is None:
            errors.append("auto-top-up is enabled but recurring period is missing")
        if periods_name and rec.get(periods_name) is not None and _to_finite_float(rec.get(periods_name), 0) < 1:
            errors.append("auto-top-up number of periods must be at least 1")

    # Reference-type semantics.
    expected_refs = {
        "topupbalance_bucket_referred_type": "bucket",
        "topupbalance_channel_referred_type": "channel",
        "topupbalance_payment_method_referred_type": "paymentmethod",
        "topupbalance_balance_topup_referred_type": "topupbalance",
    }
    for name, expected in expected_refs.items():
        if name in rec and rec.get(name) is not None and _normalize(rec.get(name)) != _normalize(expected):
            errors.append(f"{name} is not a valid reference type")
    if "topupbalance_balance_topup_role" in rec and rec.get("topupbalance_balance_topup_role") is not None:
        role_value = _normalize(rec.get("topupbalance_balance_topup_role"))
        if role_value not in {"parent", "child"}:
            errors.append("topupbalance_balance_topup_role is outside parent/child semantics")
        if "topupbalance_balance_topup_id" in rec and "topupbalance_id" in rec:
            related_id = rec.get("topupbalance_balance_topup_id")
            current_id = rec.get("topupbalance_id")
            if related_id is not None and current_id is not None and str(related_id) != str(current_id) and role_value != "child":
                errors.append("related TopupBalance reference must use role=child when it points to an earlier top-up")
            if related_id is not None and "topupbalance_balance_topup_href" in rec and rec.get("topupbalance_balance_topup_href") is not None:
                if str(related_id) not in str(rec.get("topupbalance_balance_topup_href")):
                    errors.append("topupbalance_balance_topup_href does not reference topupbalance_balance_topup_id")
    if "topupbalance_balance_topup_id" in rec and rec.get("topupbalance_balance_topup_id") is None:
        for _name in ("topupbalance_balance_topup_href", "topupbalance_balance_topup_name", "topupbalance_balance_topup_role"):
            if _name in rec and rec.get(_name) is not None:
                errors.append(f"{_name} is populated without a related TopupBalance id")

    # Reference identity and hrefs must agree with the object they reference.
    for ref_name, id_name in (
        ("topupbalance_bucket_href", "bucket_id"),
        ("bucket_href", "bucket_id"),
        ("customer_href", "customer_id"),
        ("topupbalance_href", "topupbalance_id"),
    ):
        if ref_name in rec and id_name in rec and rec.get(ref_name) is not None and rec.get(id_name) is not None:
            if str(rec[id_name]) not in str(rec[ref_name]):
                errors.append(f"{ref_name} does not reference {id_name}")
    if "account_id" in rec and "subscriber_id" in rec:
        account = str(rec.get("account_id") or "")
        subscriber = str(rec.get("subscriber_id") or "")
        if subscriber.startswith("SUB-") and account != f"ACC-{subscriber[4:]}" :
            errors.append("account_id does not mirror subscriber_id")
    if "msisdn" in rec and rec.get("msisdn") is not None:
        msisdn = str(rec["msisdn"])
        if not re.fullmatch(r"\+91[0-9]{10}", msisdn):
            errors.append("msisdn is not a valid synthetic India +91 number")
    if "customer_id" in rec and "subscriber_id" in rec:
        suffix = re.search(r"([0-9]+)$", str(rec.get("subscriber_id") or ""))
        expected_customer = f"CUSTOMER-{suffix.group(1)}" if suffix else None
        if expected_customer and str(rec.get("customer_id")) != expected_customer:
            errors.append("customer_id is not linked to subscriber_id")

    # Requestor must describe the same actor represented by the auto-top-up state.
    if "topupbalance_requestor_role" in rec and "topupbalance_requestor_referred_type" in rec:
        role = _normalize(rec.get("topupbalance_requestor_role"))
        ref_type = _normalize(rec.get("topupbalance_requestor_referred_type"))
        auto_name = _lb_first_name(variables, ("topupbalance_is_auto_topup", "is_auto_topup_enabled"), "auto", "topup")
        auto_value = _boolean_semantic(rec.get(auto_name)) if auto_name else None
        if auto_value is True and not (role == "system" and ref_type == "organization"):
            errors.append("auto-top-up requestor must be a system/organization actor")
        if auto_value is False and not (role == "subscriber" and ref_type == "individual"):
            errors.append("manual top-up requestor must be a subscriber/individual actor")

    # Status/outcome consistency.
    status = _normalize(rec.get("topupbalance_status")) if rec.get("topupbalance_status") is not None else None
    execution = _normalize(rec.get("topup_execution_status")) if rec.get("topup_execution_status") is not None else None
    outcome = _normalize(rec.get("recharge_outcome")) if rec.get("recharge_outcome") is not None else None
    if status and execution and status != execution:
        # Both are separate schema concepts but in this synthetic transaction contract they
        # represent the same processing lifecycle and must not contradict each other.
        errors.append("TopupBalance status and execution status disagree")
    if status in {"completed", "failed", "cancelled"} and outcome:
        mode = str((rules or {}).get("scenario_mode") or (rules or {}).get("scenario_semantics", {}).get("outcome_mode") or "mixed").lower()
        if status == "cancelled" and mode == "decline_or_no_response":
            if outcome not in {"declined", "no_response", "rejected"}:
                errors.append("decline scenario recharge_outcome is not a decline/no-response outcome")
        else:
            expected = {"completed": "completed", "failed": "failed", "cancelled": "cancelled"}[status]
            if outcome != expected:
                errors.append("recharge_outcome disagrees with the transaction status")
    if status == "completed":
        if request_dt is None or confirmation_dt is None:
            errors.append("completed top-up must have request and confirmation timestamps")
    if status in {"failed", "cancelled"} and confirmation_dt is not None:
        errors.append(f"{status} top-up must not have a successful confirmation timestamp")

    # Scenario semantics must remain coherent with intervention state.
    mode = str((rules or {}).get("scenario_mode") or (rules or {}).get("scenario_semantics", {}).get("outcome_mode") or "mixed").lower()
    if mode == "suppression":
        if rec.get("intervention_eligibility") not in (None, False):
            errors.append("suppressed intervention is still marked eligible")
        if rec.get("low_balance_warning_sent_flag") is True:
            errors.append("suppressed intervention emitted a warning")
        if rec.get("retention_intervention_offer_code") not in (None, ""):
            errors.append("suppressed intervention contains an offer")
    if mode == "decline_or_no_response":
        decision_norm = _normalize(rec.get("customer_decision")) if rec.get("customer_decision") is not None else None
        if decision_norm is not None and decision_norm not in {"declined", "no_response", "rejected"}:
            errors.append("customer_decision contradicts the decline/no-response scenario")
        expected_variant = _lb_response_variant(rules)
        if decision_norm is not None and decision_norm != expected_variant:
            errors.append(f"customer_decision does not match the explicit {expected_variant} scenario")
        outcome_norm = _normalize(rec.get("recharge_outcome")) if rec.get("recharge_outcome") is not None else None
        if outcome_norm is not None and outcome_norm != expected_variant:
            errors.append(f"recharge_outcome does not match the explicit {expected_variant} scenario")

    if errors:
        raise ValueError("Low Balance logical validation failed: " + "; ".join(errors))


def _strict_validate_record(rec: dict, variables: list[dict], rules: dict | None = None) -> None:
    """Final fail-closed validation for the complete confirmed contract."""
    by_name = _lb_variables_by_name(variables)
    for name, var in by_name.items():
        value = rec.get(name)
        if bool(var.get("required")) and value is None:
            raise ValueError(f"required field '{name}' is missing")
        if value is None:
            continue
        dtype = str(var.get("dtype") or "string").lower()
        if dtype in {"int", "integer"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} has invalid integer type")
        elif dtype in {"float", "decimal", "number", "numeric"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} has invalid numeric type")
        elif dtype == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{name} has invalid boolean type")
        elif dtype == "datetime":
            if _qa_parse_dt(value) is None:
                raise ValueError(f"{name} has invalid datetime value")
        elif dtype == "date":
            try:
                date.fromisoformat(str(value)[:10])
            except Exception:
                raise ValueError(f"{name} has invalid date value")

        params = var.get("params") or {}
        declared = _declared_param_options(params)
        if declared and not any(_matches_declared_option(value, choice) for choice in declared):
            raise ValueError(f"{name} is outside its declared value set")
        if dtype in _NUMERIC_DTYPES and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = _declared_numeric_bounds(params)
            if lo is not None and float(value) < lo - 1e-9:
                raise ValueError(f"{name} is below declared minimum")
            if hi is not None and float(value) > hi + 1e-9:
                raise ValueError(f"{name} is above declared maximum")

    # A non-null dependent field cannot exist without the dependencies it claims.
    for name, var in by_name.items():
        value = rec.get(name)
        if value is None:
            continue
        for dep in var.get("depends_on", []) or []:
            dep_name = str(dep)
            if dep_name in by_name and rec.get(dep_name) is None:
                raise ValueError(f"{name} has a value but dependency '{dep_name}' is missing")

        if str(var.get("gen") or "").lower() == "id_mirror":
            params = var.get("params") or {}
            source = str(params.get("source_field") or "")
            if source and rec.get(source) is not None:
                source_value = str(rec[source])
                source_prefix = str(params.get("source_prefix") or "")
                target_prefix = str(params.get("prefix") or "")
                expected = target_prefix + (source_value[len(source_prefix):] if source_prefix and source_value.startswith(source_prefix) else source_value)
                if str(value) != expected:
                    raise ValueError(f"{name} does not mirror {source}")

    # Generic date-pair safety net for every domain; the domain-specific validators may add
    # stricter business timelines.
    def _assert_order(start_name: str, end_name: str, label: str) -> None:
        if start_name in rec and end_name in rec:
            a, b = _qa_parse_dt(rec.get(start_name)), _qa_parse_dt(rec.get(end_name))
            if a is not None and b is not None and a > b:
                raise ValueError(f"{label}: {start_name} occurs after {end_name}")

    names = list(by_name)
    for name in names:
        low = name.lower()
        if low.endswith("_start_date_time"):
            suffix = "_start_date_time"
            end_name = name[:-len(suffix)] + "_end_date_time"
            if end_name in by_name:
                _assert_order(name, end_name, "validity period")
        elif low.endswith("_start_datetime"):
            suffix = "_start_datetime"
            end_name = name[:-len(suffix)] + "_end_datetime"
            if end_name in by_name:
                _assert_order(name, end_name, "validity period")
        if "requested" in low and low.endswith("_date_time"):
            confirmation_candidates = [n for n in names if n.lower().endswith("_confirmation_date_time") and n.lower().rsplit("_confirmation_date_time",1)[0] == name[:-len("_requested_date_time")]]
            for end_name in confirmation_candidates:
                _assert_order(name, end_name, "request/confirmation lifecycle")

    # Formula safety: a final validator may never return a numerically incorrect formula field.
    for field, expr in _collect_formula_specs(variables, rules):
        if field not in rec:
            continue
        deps = _formula_dependencies(expr)
        if any(rec.get(dep) is None for dep in deps):
            raise ValueError(f"formula field '{field}' is missing dependencies")
        field_def = by_name.get(field) or {}
        expected = _coerce_formula_result(_safe_formula(expr, rec), field_def)
        if expected is None:
            raise ValueError(f"formula field '{field}' could not be evaluated")
        actual = rec.get(field)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=0.01):
                raise ValueError(f"formula field '{field}' is inconsistent with its formula")
        elif str(actual) != str(expected):
            raise ValueError(f"formula field '{field}' is inconsistent with its formula")

    if _lb_domain(rules):
        _assert_low_balance_topup_consistency(rec, variables, rules=rules)

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
        expected = _coerce_formula_result(_safe_formula(expr, rec), field_def)
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
        expected = _coerce_formula_result(_safe_formula(expr, rec), field_def)
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
    rec, topup_issues = _enforce_low_balance_topup_consistency(rec, variables, rules=rules)
    issues.extend(topup_issues)
    # Domain semantics can move timestamps/amounts/statuses, so authoritative formulas must
    # be recalculated once more before the final contract/strict validation boundary.
    rec, post_domain_formula_issues = _enforce_authoritative_formulas(rec, variables, rules=rules)
    issues.extend(post_domain_formula_issues)
    rec, final_contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(final_contract_issues)

    # Fail closed after every repair/constraint pass. This prevents a future change to
    # one repair stage from silently reintroducing a contradiction into final_records.
    _assert_low_balance_topup_consistency(rec, variables, rules=rules)

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

    rec, final_formula_issues = _enforce_authoritative_formulas(rec, variables, rules=rules)
    issues.extend(final_formula_issues)
    rec = _format_datetime_fields(rec, variables)
    _strict_validate_record(rec, variables, rules=rules)

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
        transactional_fixes: list[int] = []
        state.raw_records = _transactional_records(
            compiled,
            state.count,
            state.records_per_user,
            rules=state.rules,
            record_errors_out=state.record_errors,
            country=state.country,
            fixes_out=transactional_fixes,
        )
    else:
        transactional_fixes = []
        aggregate_plan = _build_generation_plan(variables, rules=state.rules)
        for index in range(state.count):
            try:
                state.raw_records.append(_generate_record(variables, rules=state.rules, plan=aggregate_plan, apply_repairs=False))
            except Exception as exc:
                state.record_errors.append({"record_index": index, "error": str(exc), "record": {}})

    checked: list[dict] = []
    fixes = 0
    if type_of_data == "transactional":
        # _transactional_records now runs the complete validator exactly once for every
        # accepted row, including its final strict validation boundary. Re-validating the
        # same rows here only duplicated the most expensive work.
        checked = list(state.raw_records)
    else:
        for record_index, record in enumerate(state.raw_records):
            try:
                repaired, issues = _validate_record(
                    record,
                    variables,
                    state.field_order,
                    False,
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
        "algo_fixes": fixes + sum(transactional_fixes),
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
            aggregate_plan = _build_generation_plan(variables, rules=state.rules)
            for index in range(state.count):
                rec = {}
                try:
                    rec = _generate_record(variables, rules=state.rules, plan=aggregate_plan)
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

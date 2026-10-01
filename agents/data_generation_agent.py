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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import re
import random as _random_module
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from core.dynamic_scenarios import resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState
from core.scenario_semantics import temporal_role as _temporal_role
from core.deterministic_rules import build_deterministic_rules
from core.pattern_generator import PatternGenerationError, generate_regex_sample
from core.temporal_contract import (
    normalize_temporal_family,
    is_supported_temporal_rule,
    source_declared_max_delay_seconds,
    source_declared_min_delay_seconds,
)
from config.runtime import (
    AGENTIC_REQUIRE_CLEAN_RECORDS, AGENTIC_REQUIRE_EXACT_RECORD_COUNT, GENERATION_MAX_ATTEMPTS_PER_RECORD,
    OPEN_BOUND_SPAN_FLOAT, OPEN_BOUND_SPAN_INT,
)
from config.country_metadata import COUNTRY_BASE

logger = logging.getLogger(__name__)

_GENERATION_RNG: ContextVar[_random_module.Random | None] = ContextVar("generation_rng", default=None)


def _rng():
    """Return the request-scoped RNG when one is configured, otherwise the module RNG."""
    return _GENERATION_RNG.get() or _random_module


def _uuid4() -> uuid.UUID:
    """Generate UUIDv4 using the request RNG when seeded, preserving reproducibility."""
    rng = _GENERATION_RNG.get()
    if rng is None:
        return uuid.uuid4()
    value = rng.getrandbits(128)
    value &= ~(0xF << 76)
    value |= (4 << 76)
    value &= ~(0x3 << 62)
    value |= (2 << 62)
    return uuid.UUID(int=value)


@contextmanager
def generation_seed(seed: int | None):
    """Temporarily make stochastic generation reproducible for one request/test."""
    if seed is None:
        yield
        return
    token = _GENERATION_RNG.set(_random_module.Random(int(seed)))
    try:
        yield
    finally:
        _GENERATION_RNG.reset(token)

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
semantics, business/cross-field rules, and entity-history rules. The goal is not merely
to produce syntactically valid rows: every record must represent a logically possible business
event sequence. Treat the generated dataset as if it were emitted by the real business system
defined by the confirmed scenario and its approved industry/domain source documents.

The confirmed schema and supplied official source JSON are authoritative for structure, field
meaning, required/optional fields, datatype, enum vocabulary, nested-reference meaning,
amount/unit meaning, lifecycle timestamps, and relationship semantics. Never invent official
enum values or substitute synonyms. Never use a free-form value from one field as though it
were the semantic value of a different field.

For every record, validate ALL dimensions, not only timestamps:
1. Entity identity consistency: entity/account/customer/resource identities and stable keys.
2. Reference integrity: ids, hrefs, names, roles, and @referredType semantics agree with the
   referenced resource type.
3. Enum/category fidelity: every choice is inside the declared/source value set.
4. Datatype/range/precision/bucket/weight constraints.
5. Amount and unit consistency: values describing the same quantity agree.
6. Quantity invariants: remaining/reserved/added quantities cannot contradict one another.
7. State-machine consistency: status, outcome, eligibility, suppression, decision, and execution
   state cannot describe mutually exclusive states simultaneously.
8. Recurrence consistency: recurrence fields are present only when the recurring/automatic flag is enabled and
   their period/count agree with that behavior.
9. Validity consistency: a validity window is tied to the event that grants it and
   is never an independently sampled unrelated date range.
10. Temporal causality: request <= confirmation, activation/start <= expiry/end, and dependent
    events follow their parent events within declared or domain-appropriate bounds.
11. Dependency consistency: values derived from another field must actually agree with that field.
12. Formula/arithmetic consistency: formulas and calculated values must match exactly within the
    declared precision.
13. Entity history consistency: repeated transactions for one entity must form a plausible
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

The generator is expected to produce records that already satisfy the complete contract. Treat a non-zero
repair count as evidence that the generation policy missed a dependency or scenario invariant. Do not perform
creative repair merely to increase the number of accepted rows; any repair must be deterministic, contract-backed,
and limited to the affected downstream values.

Return JSON with keys: valid_records, dropped_records, fixes_applied, issues_found.
Each input record contains an internal `__qa_id`. Preserve that exact ID on every returned valid
record. Never create, modify, duplicate, or reuse a `__qa_id`. If a record must be dropped, put its
exact `__qa_id` in dropped_records. The deterministic application will reconcile IDs and reject
unknown or duplicated identifiers.

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
    return f"EVENT_{token[:32]}_{_rng().randint(1000, 9999)}" if token else f"EVENT_{_rng().randint(1000, 9999)}"


def _fit_string_length(value: str, params: dict) -> str:
    text = str(value)
    minimum = params.get("min_length")
    maximum = params.get("max_length")
    try:
        minimum_i = max(0, int(minimum)) if minimum is not None else None
    except (TypeError, ValueError):
        minimum_i = None
    try:
        maximum_i = max(0, int(maximum)) if maximum is not None else None
    except (TypeError, ValueError):
        maximum_i = None
    if maximum_i is not None and len(text) > maximum_i:
        text = text[:maximum_i]
    if minimum_i is not None and len(text) < minimum_i:
        padding = "X" if any(ch.isupper() for ch in text) else "x"
        text = text + (padding * (minimum_i - len(text)))
    return text


def _pattern_string(var: dict, _rec: dict) -> str:
    params = dict(var.get("params") or {})
    pattern = str(params.get("pattern") or "")
    if not pattern:
        raise ValueError("pattern_string requires params.pattern")
    try:
        minimum = max(0, int(params.get("min_length", 0) or 0))
        maximum = max(minimum, int(params.get("max_length", 256) or 256))
        return generate_regex_sample(pattern, _rng(), min_length=minimum, max_length=maximum)
    except PatternGenerationError as exc:
        raise ValueError(f"Unable to generate value for source pattern: {pattern!r}: {exc}") from exc


def _semantic_string(var: dict, rec: dict) -> str | None:
    """Generate a useful synthetic string from field name/description semantics.

    This is intentionally conservative: explicit params remain authoritative, then clear
    description examples and generic name semantics are used before the final
    synthetic fallback. The fallback never echoes the schema field name verbatim.
    """
    p = dict(var.get("params") or {})
    name = str(var.get("name") or rec.get("__current_field__") or "").strip()
    desc = str(var.get("description") or "").strip()
    n = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    d = desc.lower()
    fmt = str(p.get("format") or "").strip().lower()

    if p.get("pattern"):
        return _pattern_string(var, rec)

    # Standard formats are authoritative and must win over field-name heuristics.
    if fmt in {"uuid", "uuid4"}:
        return str(_uuid4())
    if fmt in {"email", "idn-email"}:
        return f"user{_rng().randint(100000, 999999)}@example.test"
    if fmt in {"uri", "uri-reference", "url"}:
        return f"https://example.test/resource/{_uuid4().hex[:12]}"
    if fmt == "ipv4":
        return ".".join(str(_rng().randint(1, 254) if i == 0 else _rng().randint(0, 255)) for i in range(4))
    if fmt == "ipv6":
        groups = [f"{_rng().randint(0, 65535):x}" for _ in range(8)]
        return ":".join(groups)

    if "choices" in p or "values" in p:
        vals = p.get("choices", p.get("values"))
        if isinstance(vals, str):
            vals = [x.strip() for x in re.split(r"[;,|]", vals) if x.strip()]
        if vals:
            return str(_rng().choice(list(vals)))
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
            return str(_rng().choice(parts))

    # Identifier/reference and phone semantics must be resolved before the generic source-contract
    # fallback. Otherwise every ``semantic_string`` ID/phone becomes a SYN_* placeholder even when
    # the field name itself contains enough information to generate a stable structural value.
    if (
        n.endswith("_id") or n == "id" or n.endswith("_key")
        or "identifier" in d or "reference id" in d or "unique reference" in d
    ):
        prefix_source = n[:-3] if n.endswith("_id") else n
        prefix = re.sub(r"[^A-Z0-9]+", "_", prefix_source.upper()).strip("_") or "REF"
        return _prefixed_int({"prefix": f"{prefix}-", "digits": 10}, rec)

    if "phone" in n or "mobile" in n or "msisdn" in n or ("number" in d and ("mobile" in d or "phone" in d)):
        # Country is request-scoped. The generator receives it through the internal execution
        # context so a US/GB/etc. request never silently falls back to India's +91 prefix.
        country = str(p.get("country") or rec.get("__country__") or "").strip().upper()
        country_code = str((COUNTRY_BASE.get(country) or COUNTRY_BASE.get("GLOBAL") or {}).get("phone_country_code") or "").strip()
        if country_code:
            return _e164_phone({"country_codes": [country_code]}, rec)

    # MongoDB source examples/defaults are the closest available value semantics for a free-form
    # source field. Prefer them before the generic synthetic fallback; they are dynamic source data,
    # not application/static vocabularies.
    source_examples = p.get("source_examples")
    if isinstance(source_examples, (list, tuple)):
        usable_examples = [
            str(value).strip() for value in source_examples
            if value is not None and str(value).strip()
        ]
        if usable_examples:
            return _fit_string_length(str(_rng().choice(usable_examples)), p)

    # Source-grounded free-form strings must not fall through into domain vocabularies. The authoritative source contract owns the value semantics. Pattern-backed
    # fields are handled by the dedicated pattern generator before this neutral fallback.
    if p.get("source_contract"):
        return _fit_string_length(f"SYN_{(n.upper() or 'VALUE')[:32]}_{_rng().randint(1000, 9999)}", p)

    if "email" in n:
        return f"user{_rng().randint(100000, 999999)}@example.test"
    if "uri" in d or "url" in d or any(token in n for token in ("href", "url", "schemalocation", "resourcepath", "path")):
        return f"https://example.test/resource/{_uuid4().hex[:12]}"
    if "reference" in d and not any(token in d for token in ("uri", "url", "documentation")):
        prefix = re.sub(r"[^A-Z0-9]+", "_", n.upper()).strip("_") or "REF"
        return _prefixed_int({"prefix": f"{prefix}-", "digits": 10}, rec)

    label = re.sub(r"_+", "_", n.upper()).strip("_") or "VALUE"
    return _fit_string_length(f"SYN_{label[:24]}_{_rng().randint(1000, 9999)}", p)


def _generic_value(var: dict, rec: dict):
    """Best-effort generic generator for client vocab not known to the core."""
    p = dict(var.get("params") or {})
    dtype = str(var.get("dtype") or "string").lower()
    if var.get("formula"):
        return _formula(var, rec)
    if "choices" in p or "values" in p:
        vals = p.get("choices", p.get("values"))
        if isinstance(vals, str):
            vals = [x.strip() for x in re.split(r"[;,|]", vals) if x.strip()]
        if vals:
            return _rng().choice(list(vals))
    if "value" in p and p.get("value") is not None:
        return p.get("value")
    if dtype in _NUMERIC_DTYPES:
        lo = _safe_number(p.get("min", p.get("lo")), 0.0)
        hi = _safe_number(p.get("max", p.get("hi")), 100.0)
        if dtype in {"int", "integer"}:
            return _rng().randint(int(round(min(lo, hi))), int(round(max(lo, hi))))
        return round(_rng().uniform(min(lo, hi), max(lo, hi)), int(p.get("precision", 2) or 2))
    if dtype == "boolean":
        return _rng().choice([True, False])
    if dtype == "datetime":
        return _recent_datetime(p, rec)
    if dtype == "date":
        return _recent_date(p, rec)
    if dtype == "array":
        choices = p.get("choices", p.get("values"))
        if isinstance(choices, (list, tuple)) and choices:
            size = _rng().randint(0, min(3, len(choices)))
            return _rng().sample(list(choices), size) if size else []
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
    number = _rng().randint(lo, hi)
    return f"{prefix}{str(number).zfill(digits)}"


def _e164_phone(params: dict, _rec: dict) -> str:
    cc = _rng().choice(params["country_codes"])
    if cc == "+91":
        # India: 10-digit mobile number, first digit must be 6-9 (TRAI numbering plan)
        first = _rng().choice("6789")
        rest = "".join(_rng().choice("0123456789") for _ in range(9))
        return f"{cc}{first}{rest}"
    if cc == "+44":
        # UK: mobile numbers start 7, followed by 9 digits
        rest = "".join(_rng().choice("0123456789") for _ in range(9))
        return f"{cc}7{rest}"
    if cc == "+971":
        # UAE: mobile prefixes 50/52/54/55/56/58 + 7 digits
        prefix = _rng().choice(["50", "52", "54", "55", "56", "58"])
        rest = "".join(_rng().choice("0123456789") for _ in range(7))
        return f"{cc}{prefix}{rest}"
    # Default / US (NANP): NPA (200-999) + NXX (200-999) + 4-digit line number
    npa = _rng().randint(200, 999)
    nxx = _rng().randint(200, 999)
    xxxx = _rng().randint(1000, 9999)
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
    if not choices:
        raise ValueError("weighted_choice requires at least one choice")
    if "weights" not in params:
        return _rng().choice(choices)
    raw_weights = params.get("weights")
    try:
        weights = [float(x) for x in (list(raw_weights) if isinstance(raw_weights, (list, tuple)) else [])]
    except (TypeError, ValueError) as exc:
        raise ValueError("weighted_choice contains non-numeric weights") from exc
    if (
        len(weights) != len(choices)
        or any((not math.isfinite(w) or w < 0) for w in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("weighted_choice weights must be finite, non-negative, non-zero, and match choices")
    return _rng().choices(choices, weights=weights, k=1)[0]


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
    if "weights" not in params:
        weights = None
    else:
        try:
            weights = [float(w) for w in params.get("weights") or []]
        except (TypeError, ValueError) as exc:
            raise ValueError("weighted_bucket contains non-numeric weights") from exc
        if (
            len(weights) != len(normalized)
            or any((not math.isfinite(w) or w < 0) for w in weights)
            or sum(weights) <= 0
        ):
            raise ValueError("weighted_bucket weights must be finite, non-negative, non-zero, and match buckets")
    lo, hi = _rng().choices(normalized, weights=weights, k=1)[0] if weights is not None else _rng().choice(normalized)
    precision = int(params.get("precision", 2) or 0)
    if precision > 0:
        scale = 10 ** precision
        lo_tick = int(math.ceil(lo * scale))
        hi_tick = int(math.floor(hi * scale))
        if hi_tick < lo_tick:
            return round(lo, precision)
        return _rng().randint(lo_tick, hi_tick) / scale
    return _rng().randint(int(math.ceil(lo)), int(math.floor(hi))) if lo.is_integer() and hi.is_integer() else _rng().uniform(lo, hi)


def _multiple_of_tick_bounds(lo: float, hi: float, multiple_of: float, scale: int) -> tuple[int, int, int] | None:
    """Return integer tick bounds for a positive multipleOf constraint.

    The returned tuple is ``(lo_index, hi_index, step)`` where the generated value is
    ``index * multiple_of`` after conversion to the requested decimal precision.
    """
    try:
        multiple = float(multiple_of)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(multiple) or multiple <= 0:
        return None
    step = max(1, int(round(multiple * scale)))
    if abs((step / scale) - multiple) > (1 / scale) * 0.51:
        return None
    lo_tick = int(math.ceil((lo * scale) / step))
    hi_tick = int(math.floor((hi * scale) / step))
    if hi_tick < lo_tick:
        return None
    return lo_tick, hi_tick, step


def _uniform(params: dict, _rec: dict) -> float:
    precision = int(params.get("precision", 2) or 2)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), None)
    if lo is None:
        lo = 0.0
    if hi is None:
        hi = lo + OPEN_BOUND_SPAN_FLOAT      # no declared upper bound: never collapse to a constant
    hi = max(lo, hi)
    if precision > 0:
        scale = 10 ** precision
        multiple = params.get("multiple_of")
        bounds = _multiple_of_tick_bounds(lo, hi, multiple, scale) if multiple is not None else None
        if bounds:
            lo_index, hi_index, step = bounds
            return (_rng().randint(lo_index, hi_index) * step) / scale
        lo_tick = int(math.ceil(lo * scale))
        hi_tick = int(math.floor(hi * scale))
        if hi_tick >= lo_tick:
            tick = _rng().randint(lo_tick, hi_tick)
            # Prefer non-integer decimal values when representable at this precision.
            if hi_tick > lo_tick and tick % scale == 0:
                if tick + 1 <= hi_tick:
                    tick += 1
                elif tick - 1 >= lo_tick:
                    tick -= 1
            return tick / scale
    return round(float(_rng().uniform(lo, hi)), precision)


def _uniform_int(params: dict, _rec: dict) -> int:
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), None)
    lo_int = int(math.ceil(lo if lo is not None else 0.0))
    hi_int = int(math.floor(hi)) if hi is not None else lo_int + OPEN_BOUND_SPAN_INT
    if hi_int < lo_int:
        hi_int = lo_int
    multiple = params.get("multiple_of")
    try:
        multiple_int = int(multiple) if multiple is not None else 0
    except (TypeError, ValueError):
        multiple_int = 0
    if multiple_int > 0:
        first = int(math.ceil(lo_int / multiple_int) * multiple_int)
        last = int(math.floor(hi_int / multiple_int) * multiple_int)
        if first <= last:
            return _rng().randrange(first, last + multiple_int, multiple_int)
    return _rng().randint(lo_int, hi_int)


def _lognormal(params: dict, _rec: dict) -> float:
    precision = int(params.get("precision", 2) or 2)
    mu = _to_finite_float(params.get("mu"), 0.0)
    sigma = _to_finite_float(params.get("sigma"), 1.0)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    raw = math.exp(_rng().gauss(mu if mu is not None else 0.0, sigma if sigma is not None else 1.0))
    clipped = max(lo if lo is not None else 0.0, min(hi if hi is not None else raw, raw))
    return round(float(clipped), precision)


def _lognormal_int(params: dict, _rec: dict) -> int:
    mu = _to_finite_float(params.get("mu"), 0.0)
    sigma = _to_finite_float(params.get("sigma"), 1.0)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    raw = int(math.exp(_rng().gauss(mu if mu is not None else 0.0, sigma if sigma is not None else 1.0)))
    return int(max(lo if lo is not None else 0.0, min(hi if hi is not None else raw, raw)))


def _beta(params: dict, _rec: dict) -> float:
    return round(_rng().betavariate(params["alpha"], params["beta"]), 4)


def _segment_range(params: dict, rec: dict) -> float:
    precision = int(params.get("precision", 4) or 4)
    controller = params.get("field") or params.get("segment_field") or ""
    key = rec.get(controller) if controller else None
    rng = params.get(key) if key in params else params.get("default")
    if not isinstance(rng, dict):
        numeric_ranges = [v for v in params.values() if isinstance(v, dict) and "min" in v and "max" in v]
        rng = numeric_ranges[0] if numeric_ranges else {"min": 0, "max": 1}
    return round(float(_rng().uniform(float(rng.get("min", 0)), float(rng.get("max", 1)))), precision)


def _to_finite_float(value, default: float | None = None) -> float | None:
    """Coerce numeric-looking values safely for dependent generators.

    Scenario definitions are persisted as confirmed contracts, so numeric params may be
    strings. Dependent fields can also be represented as strings (for example
    ``"20.0"``). Never pass a raw string into _rng().uniform/max.
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
    """Generate a bounded value while honoring integer and multipleOf constraints."""
    precision = max(0, int(params.get("precision", 2) or 2))
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
        lo_int = int(math.ceil(lo))
        hi_int = int(math.floor(hi))
        try:
            multiple_int = int(params.get("multiple_of") or params.get("multipleOf") or 0)
        except (TypeError, ValueError):
            multiple_int = 0
        if multiple_int > 1:
            first = int(math.ceil(lo_int / multiple_int)) * multiple_int
            last = int(math.floor(hi_int / multiple_int)) * multiple_int
            if last >= first:
                return _rng().randint(first // multiple_int, last // multiple_int) * multiple_int
        if hi_int < lo_int:
            return lo_int
        return _rng().randint(lo_int, hi_int)

    multiple = params.get("multiple_of", params.get("multipleOf"))
    if multiple is not None:
        bounds = _multiple_of_tick_bounds(lo, hi, float(multiple), 10 ** precision)
        if bounds:
            lo_index, hi_index, step = bounds
            return (_rng().randint(lo_index, hi_index) * step) / (10 ** precision)
    return round(float(_rng().uniform(lo, hi)), precision)


def _recent_datetime(params: dict, _rec: dict) -> str:
    days_back = int(params.get("days_back", 0) or 0)
    base = datetime.now(timezone.utc) - timedelta(
        days=_rng().randint(0, max(0, days_back)),
        hours=_rng().randint(0, 23),
        minutes=_rng().randint(0, 59),
        seconds=_rng().randint(0, 59),
    )
    return _format_datetime(base, params)


def _recent_date(params: dict, _rec: dict) -> str:
    """Generate an ISO-8601 calendar date for fields declared as JSON Schema ``format=date``."""
    days_back = int(params.get("days_back", 0) or 0)
    base = datetime.now(timezone.utc).date() - timedelta(
        days=_rng().randint(0, max(0, days_back))
    )
    return base.isoformat()


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


def _generate_temporal_child_value_for_constraints(
    var: dict,
    constraints: list[tuple[str, datetime, int | None, int]],
) -> str:
    """Generate one temporal child satisfying all currently available parent constraints.

    Lower bounds are hard causal ordering. Maximum-delay constraints are honored whenever the
    feasible interval is non-empty; if a set of optional bounds is internally contradictory, the
    safest executable behavior is to preserve causality and ignore only the conflicting maxima.
    The rule is generic and therefore works for any industry/resource naming scheme.
    """
    if not constraints:
        raise ValueError("temporal child generation requires at least one parent constraint")

    lower = max(parent_dt + timedelta(seconds=max(0, int(min_gap or 0)))
                for _parent, parent_dt, _max_gap, min_gap in constraints)
    uppers = [parent_dt + timedelta(seconds=int(max_gap))
              for _parent, parent_dt, max_gap, _min_gap in constraints
              if max_gap is not None]
    upper = min(uppers) if uppers else None

    # With no source-declared upper bound, retain causal ordering but avoid deterministic equality.
    # If the client contract is minute-only, use at least a one-minute gap so serialization cannot
    # collapse a valid sub-minute relation back into equal timestamps. If the contract explicitly
    # allows only a sub-minute window, seconds are retained for that relational field.
    output_params = dict(var.get("params") or {})
    minute_only = "%S" not in _normalize_timestamp_format(output_params) and str(var.get("dtype") or "").casefold() != "date"
    max_parent = max(parent_dt for _parent, parent_dt, _max_gap, _min_gap in constraints)
    positive_causal_gap = lower > max_parent
    if upper is None:
        delay_floor = 60 if minute_only and positive_causal_gap else 1
        desired = lower + timedelta(seconds=_rng().randint(delay_floor, 3600))
    elif upper < lower:
        # A model-generated rule set can contain overlapping maximum-delay hints that are
        # impossible to satisfy simultaneously. Do not fabricate a pre-parent event. Preserve
        # the non-negotiable ordering relation and use a conservative causal timestamp.
        desired = lower
    elif upper == lower:
        desired = lower
    else:
        span = int((upper - lower).total_seconds())
        min_delay = 0
        if minute_only and positive_causal_gap and span >= 60:
            min_delay = 60
        desired = lower + timedelta(seconds=_rng().randint(min_delay, max(min_delay, span)))

    if str(var.get("dtype") or "datetime").strip().lower() == "date":
        return desired.date().isoformat()
    # A positive causal lower bound that cannot be represented in minute precision requires
    # seconds in the serialized child. This branch is only used for sub-minute source-declared
    # windows; ordinary unconstrained request/confirmation pairs stay minute-formatted with >=60s gaps.
    if minute_only and positive_causal_gap and upper is not None and (upper - lower).total_seconds() < 60:
        output_params["timestamp_format"] = "dd/mm/yyyy hh:mm:ss a"
        return _format_datetime(desired, output_params)
    return _format_datetime_for_variable(desired, var)


def _ts_offset(params: dict, rec: dict) -> str:
    base_field = str(params.get("base_field") or params.get("source_field") or "").strip()
    if not base_field:
        raise ValueError("ts_offset requires 'base_field' (or legacy alias 'source_field')")
    if base_field not in rec or rec.get(base_field) in (None, ""):
        raise ValueError(f"ts_offset dependency '{base_field}' is missing")
    base = _parse_dt(rec[base_field])
    min_sec = int(params.get("min_sec", params.get("min_seconds", 0)))
    max_sec = int(params.get("max_sec", params.get("max_seconds", min_sec)))
    min_sec, max_sec = min(min_sec, max_sec), max(min_sec, max_sec)
    offset = timedelta(seconds=_rng().randint(min_sec, max_sec))
    return _format_datetime(base + offset, _temporal_output_params(params, "ts_offset"))


def _ts_add_field(params: dict, rec: dict) -> str:
    base_field = str(params.get("base_field") or "").strip()
    if not base_field or base_field not in rec or rec.get(base_field) in (None, ""):
        raise ValueError("ts_add_field requires a valid base_field dependency")
    base = _parse_dt(rec[base_field])
    seconds = int(rec.get(params["add_seconds_field"], 60))
    return _format_datetime(base + timedelta(seconds=seconds), _temporal_output_params(params, "ts_add_field"))


def _date_offset(params: dict, rec: dict) -> str:
    base_field = str(params.get("base_field") or "").strip()
    if not base_field or base_field not in rec or rec.get(base_field) in (None, ""):
        raise ValueError("date_offset requires a valid base_field dependency")
    base = _parse_dt(rec[base_field])
    return (base + timedelta(days=params["days"])).date().isoformat()


def _date_offset_range(params: dict, rec: dict) -> str:
    base_field = str(params.get("base_field") or "").strip()
    if not base_field or base_field not in rec or rec.get(base_field) in (None, ""):
        raise ValueError("date_offset_range requires a valid base_field dependency")
    base = _parse_dt(rec[base_field])
    min_days = int(params.get("min_days", 0))
    max_days = int(params.get("max_days", min_days))
    offset_days = _rng().randint(min(min_days, max_days), max(min_days, max_days))
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
        number = str(_rng().randint(0, max(0, 10**digits - 1))).zfill(digits)
    return f"{prefix}{number}"


def _prefixed_uuid(params: dict, _rec: dict) -> str:
    """Return the configured prefix followed by the complete UUID value."""
    prefix = str(params.get("prefix", ""))
    return prefix + str(_uuid4())


def _tx_id(params: dict, rec: dict) -> str:
    ts = rec.get("record_timestamp", datetime.now(timezone.utc).isoformat())
    date_part = ts[:10].replace("-", "")
    rand_part = _rng().randint(1_000_000, 9_999_999)
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
    """Parse an existing timestamp strictly; invalid values must never become ``now``."""
    parsed = _parse_timestamp_text(str(s))
    if parsed is None:
        raise ValueError(f"Invalid datetime value: {s!r}")
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
    "recent_date":    lambda v, rec: _recent_date(v["params"], rec),
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
    "uuid_string":    lambda v, rec: str(_uuid4()),
    "pattern_string": lambda v, rec: _pattern_string(v, rec),
    "email_string":   lambda v, rec: f"user{_rng().randint(100000, 999999)}@example.test",
    "uri_string":     lambda v, rec: f"https://example.test/resource/{_uuid4().hex[:12]}",
    "ipv4_string":    lambda v, rec: ".".join(str(_rng().randint(1, 254) if i == 0 else _rng().randint(0, 255)) for i in range(4)),
    "ipv6_string":    lambda v, rec: ":".join(f"{_rng().randint(0, 65535):x}" for _ in range(8)),
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
        return _rng().choice(declared_options)

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
        return _rng().choice(allowed)

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


def _temporal_field_is_absent_by_scenario(var: dict, rules: dict | None) -> bool:
    """Return True when scenario semantics say the temporal event did not occur.

    This is deliberately role-based.  It prevents a field such as
    ``customer_response_timestamp`` from being generated for a ``No Response`` scenario
    without maintaining a list of industry-specific field names.
    """
    if not isinstance(rules, dict):
        return False
    semantics = rules.get("scenario_semantics")
    if not isinstance(semantics, dict):
        return False
    absent_roles = {str(x).strip().lower() for x in semantics.get("absent_temporal_roles", []) or []}
    if not absent_roles:
        return False
    return _temporal_role(var) in absent_roles


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

    # Non-occurring temporal events are represented by null rather than a fabricated timestamp.
    # Field constraints retain the authoritative description/nullability without duplicating the
    # entire confirmed variable list inside the rules document.
    field_constraints = rules.get("field_constraints") if isinstance(rules.get("field_constraints"), dict) else {}
    for name, constraint in field_constraints.items():
        if not isinstance(constraint, dict):
            continue
        var = {"name": name, "description": constraint.get("description", ""), "dtype": constraint.get("dtype", "")}
        if _temporal_field_is_absent_by_scenario(var, rules) and name in rec and bool(constraint.get("nullable", True)):
            rec[name] = None
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
        if isinstance(expected, dict) and str(expected.get("op") or "").strip():
            op = str(expected.get("op") or "").strip().lower()
            operand = expected.get("value")
            if op in {"in", "not_in", "notin"}:
                values = operand if isinstance(operand, list) else [operand]
                result = any(matches(actual, item) for item in values)
                return result if op == "in" else not result
            if op in {"=", "==", "eq"}:
                return matches(actual, operand)
            if op in {"!=", "ne"}:
                return not matches(actual, operand)
            actual_num = _to_finite_float(actual, None)
            operand_num = _to_finite_float(operand, None)
            if actual_num is None or operand_num is None:
                return False
            if op in {"<", "lt"}:
                return actual_num < operand_num
            if op in {"<=", "lte"}:
                return actual_num <= operand_num
            if op in {">", "gt"}:
                return actual_num > operand_num
            if op in {">=", "gte"}:
                return actual_num >= operand_num
            return False
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
                rec[field] = _rng().choice(desired)
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
    temporal_relations: tuple[tuple[str, str, int | None, int], ...] = ()


def _variable_dependency_order(
    variables: list[dict],
    selected_names: set[str] | None = None,
    rules: dict | None = None,
) -> tuple[list[dict], set[str]]:
    """Return variables in stable dependency order with O(n) position lookups."""
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    position = {name: idx for idx, name in enumerate(by_name)}
    selected = set(selected_names or by_name) & set(by_name)

    # Temporal relationships are execution dependencies even when they are not persisted in
    # the source-backed ``depends_on`` list. This prevents lifecycle timestamps from being
    # independently sampled. The relationship remains runtime metadata only.
    temporal_relations = _infer_temporal_relationships(variables, rules=rules)
    temporal_parent_by_child: dict[str, list[str]] = {}
    for parent, child, _max_gap, _min_gap in temporal_relations:
        temporal_parent_by_child.setdefault(child, []).append(parent)

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
            for temporal_parent in temporal_parent_by_child.get(name, []):
                if temporal_parent not in dep_names:
                    dep_names.append(temporal_parent)
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
        for temporal_parent in temporal_parent_by_child.get(name, []):
            if temporal_parent not in dep_names:
                dep_names.append(temporal_parent)
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


def _validate_temporal_plan(relations: list[tuple[str, str, int | None, int]]) -> None:
    """Reject structurally contradictory executable temporal bounds before record generation."""
    by_child: dict[str, list[tuple[str, int | None, int]]] = {}
    for parent, child, max_gap, min_gap in relations:
        by_child.setdefault(child, []).append((parent, max_gap, min_gap))
    for child, edges in by_child.items():
        # Multiple explicit bounds can be valid, but a child cannot have two hard maxima that
        # are structurally impossible once the parents are generated independently. The runtime
        # generator handles their intersection per record; no static contradiction exists here.
        if len(edges) <= 1:
            continue
        # Duplicate parent edges are collapsed by the relation builder. Keep this function as a
        # centralized contract hook rather than adding scenario-specific exceptions.
        parents = [parent for parent, _max_gap, _min_gap in edges]
        if len(parents) != len(set(parents)):
            raise ValueError(f"Conflicting temporal rules for '{child}' contain duplicate parents")


def _public_record(rec: dict) -> dict:
    """Remove generator-only execution context before a record reaches the API/client contract."""
    return {key: value for key, value in rec.items() if not key.startswith("__") }


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
    temporal_relations_list = _infer_temporal_relationships(variables, rules=rules)
    _validate_temporal_plan(temporal_relations_list)
    temporal_relations = tuple(temporal_relations_list)
    return _GenerationPlan(
        ordered=tuple(ordered),
        cyclic=frozenset(cyclic),
        known_fields=frozenset(str(v.get("name")) for v in variables if v.get("name")),
        formula_by_name=formula_by_name,
        temporal_relations=temporal_relations,
    )

def _generate_record(variables: list[dict], rules: dict | None = None, plan: _GenerationPlan | None = None, apply_repairs: bool = True) -> dict:
    """Generate one record in dependency order, while safely handling cycles."""
    rec: dict = {}
    if isinstance(rules, dict) and str(rules.get("country") or "").strip():
        rec["__country__"] = str(rules.get("country")).strip().upper()
    plan = plan or _build_generation_plan(variables, rules=rules)
    ordered, cyclic, known_fields = plan.ordered, plan.cyclic, plan.known_fields
    temporal_child_rules = {}
    for parent, child, max_gap, min_gap in plan.temporal_relations:
        temporal_child_rules.setdefault(child, []).append((parent, max_gap, min_gap))
    for var in ordered:
        effective_var = var
        # A formula is authoritative when it can be evaluated.  For a dependency
        # cycle, seed the cyclic field from its declared generator so the remaining
        # fields can still be generated and QA can evaluate any resolvable formulas.
        rule_formula = None if var["name"] in cyclic else plan.formula_by_name.get(var["name"])
        if rule_formula:
            deps = _formula_dependencies(str(rule_formula))
            missing_deps = sorted(dep for dep in deps if dep not in known_fields and dep not in rec)
            if missing_deps:
                raise ValueError(
                    f"Formula for '{var['name']}' references unknown field(s): {', '.join(missing_deps)}"
                )
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        # Generate only events that occur in this scenario.  In a No Response scenario,
        # response timestamps are absent instead of being generated and later repaired away.
        if _temporal_field_is_absent_by_scenario(var, rules):
            rec[var["name"]] = None
            continue

        temporal_rules = temporal_child_rules.get(var["name"], [])
        original_generator = str(effective_var.get("gen") or "").strip().lower()
        if temporal_rules and not rule_formula and original_generator in {
            "recent_datetime", "recent_date", "timestamp", "datetime"
        }:
            available_constraints = []
            for parent_name, max_gap, min_gap in temporal_rules:
                parent_dt = _qa_parse_dt(rec.get(parent_name))
                if parent_dt is not None:
                    available_constraints.append((parent_name, parent_dt, max_gap, min_gap))
            value = (
                _generate_temporal_child_value_for_constraints(var, available_constraints)
                if available_constraints
                else None
            )
        else:
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
        rec, _ = _enforce_generic_business_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules, relations=plan.temporal_relations)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules, relations=plan.temporal_relations)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        return _public_record(_format_datetime_fields(rec, variables))
    return _public_record(rec)


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
    if isinstance(rules, dict) and str(rules.get("country") or "").strip() and "__country__" not in rec:
        rec["__country__"] = str(rules.get("country")).strip().upper()
    plan = plan or _build_generation_plan(variables, selected_names, rules)
    ordered, cyclic, known_fields = plan.ordered, plan.cyclic, plan.known_fields
    temporal_child_rules = {}
    for parent, child, max_gap, min_gap in plan.temporal_relations:
        temporal_child_rules.setdefault(child, []).append((parent, max_gap, min_gap))
    for var in ordered:
        name = var["name"]
        if name in rec:
            continue
        effective_var = var
        rule_formula = None if name in cyclic else plan.formula_by_name.get(name)
        if rule_formula:
            deps = _formula_dependencies(str(rule_formula))
            missing_deps = sorted(dep for dep in deps if dep not in known_fields and dep not in rec)
            if missing_deps:
                raise ValueError(
                    f"Formula for '{name}' references unknown field(s): {', '.join(missing_deps)}"
                )
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        # Do not fabricate timestamps for events the scenario says did not occur.
        if _temporal_field_is_absent_by_scenario(var, rules):
            rec[name] = None
            continue

        temporal_rules = temporal_child_rules.get(name, [])
        original_generator = str(effective_var.get("gen") or "").strip().lower()
        if temporal_rules and not rule_formula and original_generator in {
            "recent_datetime", "recent_date", "timestamp", "datetime"
        }:
            available_constraints = []
            for parent_name, max_gap, min_gap in temporal_rules:
                parent_dt = _qa_parse_dt(rec.get(parent_name))
                if parent_dt is not None:
                    available_constraints.append((parent_name, parent_dt, max_gap, min_gap))
            value = (
                _generate_temporal_child_value_for_constraints(var, available_constraints)
                if available_constraints
                else None
            )
        else:
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
        rec, _ = _enforce_generic_business_consistency(rec, variables, rules=rules)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules, relations=plan.temporal_relations)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        rec, _ = _enforce_temporal_consistency(rec, variables, rules=rules, relations=plan.temporal_relations)
        rec, _ = _enforce_authoritative_formulas(rec, variables, rules=rules)
        rec, _ = _enforce_csv_contract(rec, variables, rules=rules)
        return _public_record(_format_datetime_fields(rec, variables))
    return _public_record(rec)


# -- Transactional/user-history generation helpers -----------------------------

def _pick_timestamp_field(variables: list[dict]) -> str | None:
    """Choose the transaction-history anchor without accidentally anchoring on a completion/end date.

    A transactional history needs an event/request/start timestamp as its chronological anchor.
    Completion/confirmation/end/expiry timestamps are generated from their causal parent and
    must not become the user-history clock merely because they appear first in the source schema.
    """
    candidates=[]
    exact_preferred=(
        "event_timestamp", "event_datetime", "event_date_time",
        "transaction_timestamp", "transaction_datetime", "transaction_date_time",
        "record_timestamp", "record_datetime", "record_date_time",
        "requested_timestamp", "requested_datetime", "requested_date_time",
        "occurred_at", "occurred_timestamp", "created_at", "creation_date_time",
        "start_date_time", "start_datetime", "start_date",
    )
    names={str(v.get("name")):v for v in variables if v.get("name")}
    valid_dtypes={"datetime","date"}
    # A behaviour pack names its history clock explicitly; that beats any name-based guess.
    for name,var in names.items():
        if var.get("primary_timestamp") is True and str(var.get("dtype","")).strip().lower() in valid_dtypes:
            return name
    for name in exact_preferred:
        var=names.get(name)
        if var and str(var.get("dtype","")).strip().lower() in valid_dtypes:
            return name

    positive=("event", "transaction", "record", "occurred", "requested", "request", "start", "created", "creation", "timestamp")
    negative=("confirmation", "confirmed", "decision", "end", "expiry", "expiration", "expired", "updated", "update", "valid_for_end", "validity_end")
    for index,var in enumerate(variables):
        name=str(var.get("name") or "").strip()
        dtype=str(var.get("dtype") or "").strip().lower()
        if not name or dtype not in valid_dtypes:
            continue
        low=name.casefold()
        score=0
        score += sum(4 for token in positive if token in low)
        score -= sum(6 for token in negative if token in low)
        if low.endswith(("_timestamp", "_datetime", "_date_time")):
            score += 2
        if "valid_for" in low or "validity" in low:
            score -= 8
        candidates.append((score, -index, name))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    return None


def _history_event_tokens(name: str) -> set[str]:
    """Extract stable business-event tokens from a history-derived field name."""
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").casefold()).strip("_")
    parts = [p for p in text.split("_") if p]
    stop = {
        "last", "previous", "prior", "avg", "average", "interval", "days", "day", "hours", "hour",
        "minutes", "minute", "seconds", "second", "count", "number", "of", "the",
        "current", "latest", "message", "messages", "24h", "24", "flag", "timestamp",
        "datetime", "date", "time", "at", "on", "recent",
    }
    tokens = {p for p in parts if p not in stop and len(p) > 1}
    aliases = {
        "notifications": "message",
        "contacts": "communication",
    }
    return {aliases.get(p, p) for p in tokens}


def _history_qualifiers(name: str) -> set[str]:
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").casefold()).strip("_")
    parts = set(text.split("_"))
    return {token for token in ("successful", "accepted", "completed") if token in parts}


def _history_timestamp_candidates(rows: list[dict], event_tokens: set[str], require_tokens: set[str] | None = None) -> list[str]:
    """Find transaction datetime fields describing the same event family."""
    candidates: list[tuple[int, str]] = []
    for name in rows[0].keys() if rows else []:
        low = str(name).casefold()
        # History snapshots themselves are copied into each row for entity-level output. They
        # must never become the source of truth for recalculating history, otherwise reconciliation
        # can become circular and preserve an independently generated random value.
        if low.startswith(("last_", "previous_", "avg_")) or low.endswith("_count_24h"):
            continue
        if not any(token in low for token in ("timestamp", "datetime", "date_time", "_date", "_at")):
            continue
        name_tokens = _history_event_tokens(name)
        overlap = len(event_tokens & name_tokens) if event_tokens else 0
        if overlap <= 0:
            continue
        if require_tokens and not require_tokens.intersection(name_tokens):
            continue
        score = overlap * 10
        if any(token in low for token in ("requested", "occurred", "event", "transaction", "created", "impression", "presentation")):
            score += 4
        if "confirmation" in low or "completed" in low or "completion" in low:
            score += 2
        if any(token in low for token in ("valid_for", "validity", "expiry", "expiration", "expir")):
            score -= 20
        candidates.append((score, str(name)))
    return [name for _score, name in sorted(candidates, key=lambda item: (-item[0], item[1]))]


def _history_status_is_success(row: dict, event_tokens: set[str]) -> bool:
    """Return True when a row's status/outcome for an event is explicitly successful."""
    positive = {"completed", "complete", "success", "successful", "accepted", "approved", "done", "fulfilled", "settled", "authorized"}
    keys: list[tuple[int, str]] = []
    for name, value in row.items():
        low = str(name).casefold()
        if not any(token in low for token in ("status", "state", "outcome", "result")):
            continue
        name_tokens = _history_event_tokens(name)
        overlap = len(event_tokens & name_tokens) if event_tokens else 0
        if overlap:
            keys.append((overlap, str(value or "").casefold().replace("-", "_").replace(" ", "_")))
    if not keys:
        return False
    keys.sort(key=lambda item: -item[0])
    return any(value in positive for _overlap, value in keys[:3])


def _history_set_derived_fields(compiled, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Reconcile entity/history-derived fields against the transaction rows actually generated.

    This is deliberately semantic rather than domain-specific. It only derives fields whose names
    explicitly describe history (last_*, avg_*_interval_days, previous_*_count, *_count_24h) and only
    from matching transaction timestamps/statuses already present in the confirmed schema.
    """
    if not rows:
        return rows, []
    variables = list(compiled.variables)
    by_name = {str(v.get("name") or ""): v for v in variables if v.get("name")}
    entity_key = compiled.entity_key
    if not entity_key:
        return rows, []

    user_fields = set(compiled.user_fields)
    record_fields = set(compiled.record_fields)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get(entity_key) or "")
        if key:
            grouped.setdefault(key, []).append(row)

    changed: list[str] = []
    for entity_value, entity_rows in grouped.items():
        # Sort oldest -> newest for every history-derived calculation.
        anchor_name = _pick_timestamp_field(variables)
        ordered = sorted(
            entity_rows,
            key=lambda row: _qa_parse_dt(row.get(anchor_name)) if anchor_name and _qa_parse_dt(row.get(anchor_name)) is not None else datetime.min.replace(tzinfo=timezone.utc),
        )

        # Entity-level "last_*" timestamps are snapshots of the generated history, not independent random dates.
        for field_name in sorted(user_fields):
            field_var = by_name.get(field_name, {})
            dtype = str(field_var.get("dtype") or "").casefold()
            low_name = field_name.casefold()
            if dtype != "datetime" or not low_name.startswith("last_"):
                continue
            event_tokens = _history_event_tokens(field_name)
            qualifiers = _history_qualifiers(field_name)
            required_timestamp_tokens = {"accepted"} if "accepted" in qualifiers else None
            candidates = _history_timestamp_candidates(ordered, event_tokens, require_tokens=required_timestamp_tokens)
            if not candidates:
                continue
            if "successful" in qualifiers or "completed" in qualifiers:
                success_rows = [
                    row for row in ordered
                    if _history_status_is_success(row, event_tokens)
                    and any(row.get(c) is not None for c in candidates)
                ]
                if not success_rows:
                    continue
                target_rows = success_rows
            else:
                target_rows = ordered
            target_values = []
            for row in target_rows:
                # Choose one canonical event timestamp per transaction. Using both requested and
                # confirmation timestamps would turn a request/confirmation delay into an artificial
                # inter-transaction interval. Candidate ordering already prefers request/event times.
                for candidate in candidates:
                    dt = _qa_parse_dt(row.get(candidate))
                    if dt is not None:
                        target_values.append((dt, candidate))
                        break
            if target_values:
                target_dt = max(target_values, key=lambda item: item[0])[0]
                formatted = _format_datetime_for_variable(target_dt, field_var)
                if any(r.get(field_name) != formatted for r in ordered):
                    for row in ordered:
                        if field_name in row:
                            row[field_name] = formatted
                    changed.append(field_name)

        # Previous-count snapshots can be computed from the corresponding event timestamps/statuses.
        for field_name in sorted(user_fields):
            field_var = by_name.get(field_name, {})
            if str(field_var.get("dtype") or "").casefold() not in {"int", "integer"}:
                continue
            low_name = field_name.casefold()
            if not low_name.startswith("previous_") or not low_name.endswith("_count"):
                continue
            event_match_tokens = _history_event_tokens(field_name)
            candidates = _history_timestamp_candidates(ordered, event_match_tokens)
            if not candidates:
                continue

            def row_matches_event(row: dict) -> bool:
                # A matching timestamp establishes that the event family exists. For
                # domain-qualified counts (for example previous_data_events_count),
                # additionally require the row's explicit usage/category values to contain
                # the requested qualifier, preventing one kind of event from being counted as another.
                has_time = any(_qa_parse_dt(row.get(candidate)) is not None for candidate in candidates)
                if not has_time:
                    return False
                qualifiers = {
                    token
                    for key, value in row.items()
                    if any(token in str(key).casefold() for token in ("usage_type", "event_type", "transaction_type", "product_type", "category"))
                    for token in _history_event_tokens(str(value))
                }
                requested_specific = event_match_tokens - {t for c in candidates for t in _history_event_tokens(str(c))}
                if requested_specific and qualifiers and not requested_specific.intersection(qualifiers):
                    return False
                return True

            count = sum(1 for row in ordered[:-1] if row_matches_event(row))
            for row in ordered:
                if field_name in row and row.get(field_name) != count:
                    row[field_name] = count
                    changed.append(field_name)

        # Average interval fields are calculated from the matching event timestamps.
        for field_name in sorted(user_fields):
            field_var = by_name.get(field_name, {})
            if str(field_var.get("dtype") or "").casefold() not in {"float", "decimal", "number", "numeric", "int", "integer"}:
                continue
            low_name = field_name.casefold()
            if not low_name.startswith("avg_") or not low_name.endswith("_interval_days"):
                continue
            event_tokens = _history_event_tokens(field_name)
            candidates = _history_timestamp_candidates(ordered, event_tokens)
            timestamps = sorted(
                {
                    dt
                    for row in ordered
                    for dt in [next((_qa_parse_dt(row.get(candidate)) for candidate in candidates if _qa_parse_dt(row.get(candidate)) is not None), None)]
                    if dt is not None
                }
            )
            intervals = [max(0.0, (b - a).total_seconds() / 86400.0) for a, b in zip(timestamps, timestamps[1:])]
            avg_days = round(sum(intervals) / len(intervals), int((field_var.get("params") or {}).get("precision", 2) or 2)) if intervals else 0
            for row in ordered:
                if field_name in row and row.get(field_name) != avg_days:
                    row[field_name] = avg_days
                    changed.append(field_name)

        # *_count_24h fields are evaluated from communication/message/notification timestamps actually present.
        for field_name in sorted(record_fields):
            field_var = by_name.get(field_name, {})
            dtype = str(field_var.get("dtype") or "").casefold()
            if dtype not in {"int", "integer"} or not field_name.casefold().endswith("_count_24h"):
                continue
            event_tokens = _history_event_tokens(field_name)
            if not any(token in field_name.casefold() for token in ("message", "notification", "communication", "contact")):
                continue
            candidates = _history_timestamp_candidates(ordered, event_tokens | {"communication", "notification"})
            if not candidates:
                continue
            for row in ordered:
                anchor = _qa_parse_dt(row.get(anchor_name)) if anchor_name else None
                if anchor is None:
                    continue
                count = 0
                for other in ordered:
                    for candidate in candidates:
                        dt = _qa_parse_dt(other.get(candidate))
                        if dt is not None and timedelta(seconds=0) <= anchor - dt <= timedelta(hours=24):
                            count += 1
                            break
                if row.get(field_name) != count:
                    row[field_name] = count
                    changed.append(field_name)

    # Deduplicate change-report names while preserving order for diagnostics.
    changed = list(dict.fromkeys(changed))
    return rows, changed


def _history_interval_driver_seconds(variables: list[dict], user_context: dict) -> int | None:
    """Return a schema-driven inter-transaction interval when the contract exposes one.

    Values such as ``avg_event_interval_days`` are generated once per entity and should drive the
    history spacing instead of being generated as a disconnected snapshot. If the field is absent
    or non-positive, the generic generator retains its bounded default cadence.
    """
    candidates = []
    for var in variables:
        name = str(var.get("name") or "").strip()
        low = name.casefold()
        if not (low.startswith("avg_") and low.endswith("_interval_days")):
            continue
        value = _to_finite_float(user_context.get(name), None)
        if value is None or value <= 0:
            continue
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        minimum = _to_finite_float(params.get("min", params.get("lo")), None)
        maximum = _to_finite_float(params.get("max", params.get("hi")), None)
        if minimum is not None and value < minimum:
            continue
        if maximum is not None and value > maximum:
            continue
        candidates.append(value)
    if not candidates:
        return None
    # The first candidate follows the confirmed variable order, which is stable for a scenario.
    days = candidates[0]
    # Add modest multiplicative jitter while preserving the generated population's average interval.
    jitter = math.exp(_rng().gauss(0.0, 0.12))
    return max(15 * 60, int(days * 86400.0 * jitter))


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
    timestamp_var = next((v for v in variables if str(v.get("name") or "") == str(timestamp_field or "")), None)
    generated_records=[]
    used_entity_keys=set()
    user_field_names = set(compiled.user_fields)
    record_field_names = set(compiled.record_fields)
    user_plan = _build_generation_plan(variables, user_field_names, rules)
    record_plan = _build_generation_plan(variables, record_field_names, rules)

    for user_index in range(user_count):
        try:
            user_context=_generate_selected_record(variables,user_field_names,rules=rules,plan=user_plan)
            if not isinstance(user_context, dict):
                raise TypeError("Transactional user context generator returned a non-object")
            if entity_key and entity_key not in user_context:
                # Ensure the entity key is generated even if inferred user context omitted it.
                key_var=compiled.variable_by_name.get(entity_key)
                if key_var:
                    user_context=_generate_selected_record(variables,{entity_key},base=user_context,rules=rules)
            if entity_key and entity_key in user_context:
                attempts=0
                while str(user_context[entity_key]) in used_entity_keys and attempts < 100:
                    key_var=compiled.variable_by_name.get(entity_key)
                    if key_var:
                        user_context=_generate_selected_record(variables,{entity_key},base=user_context,rules=rules)
                    attempts+=1
                used_entity_keys.add(str(user_context.get(entity_key)))

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
            step_seconds=_history_interval_driver_seconds(variables, user_context)
            if step_seconds is None:
                step_seconds=_rng().randint(6*3600, 72*3600)
            start_ts=end_ts-timedelta(seconds=step_seconds*span)
        else:
            step_seconds=0
            start_ts=end_ts
        timestamps=[]
        if timestamp_field:
            for i in range(records_per_user):
                jitter=_rng().randint(0,max(60,min(6*3600,step_seconds if span else 60)))
                ts=start_ts+timedelta(seconds=(step_seconds*i if span else 0)+jitter)
                ts=min(ts,end_ts)
                timestamps.append(ts)
            timestamps=sorted(timestamps)

        for record_index in range(records_per_user):
            last_exc: Exception | None = None
            for attempt in range(GENERATION_MAX_ATTEMPTS_PER_RECORD):
                try:
                    base=dict(user_context)
                    if timestamp_field:
                        # One authoritative transaction timestamp anchors the complete row.
                        # Serialize it using the field's own confirmed contract before any dependent
                        # timestamp is generated. This avoids mixing an ISO+timezone anchor with a
                        # display-formatted child, which can shift comparisons by the local UTC offset.
                        if timestamp_var and str(timestamp_var.get("dtype") or "").strip().lower() == "date":
                            base[timestamp_field] = timestamps[record_index].date().isoformat()
                        elif timestamp_var:
                            base[timestamp_field] = _format_datetime_for_variable(timestamps[record_index], timestamp_var)
                        else:
                            base[timestamp_field] = timestamps[record_index].isoformat()
                    row=_generate_selected_record(variables,record_field_names,base=base,rules=rules,plan=record_plan,apply_repairs=False)
                    if not isinstance(row, dict):
                        raise TypeError("Transactional record generator returned a non-object")
                    if timestamp_field and timestamp_field not in row:
                        if timestamp_var and str(timestamp_var.get("dtype") or "").strip().lower() == "date":
                            row[timestamp_field] = timestamps[record_index].date().isoformat()
                        elif timestamp_var:
                            row[timestamp_field] = _format_datetime_for_variable(timestamps[record_index], timestamp_var)
                        else:
                            row[timestamp_field] = timestamps[record_index].isoformat()

                    # Apply confirmed behavioral rules during generation, before QA. This makes
                    # scenario relationships part of the generator rather than a repair side effect.
                    row = _apply_conditional_rules(row, rules)
                    row = _apply_scenario_semantics(row, rules)
                    row, _ = _enforce_generic_business_consistency(row, variables, rules=rules)
                    row, _ = _enforce_generic_business_consistency(row, variables, rules=rules)

                    # Apply scenario semantics before strict validation. Validation remains fail-closed.
                    strict_clean = bool((rules or {}).get("agentic")) and AGENTIC_REQUIRE_CLEAN_RECORDS
                    repaired, issues = _validate_record(
                        row,
                        variables,
                        list(compiled.field_order),
                        True,
                        rules=rules,
                        temporal_relations=record_plan.temporal_relations,
                    )
                    if not isinstance(repaired, dict):
                        raise TypeError("Transactional validator returned a non-object record")
                    if strict_clean and issues:
                        raise ValueError(
                            "agentic record required deterministic repair before delivery: "
                            + "; ".join(issues[:6])
                        )
                    generated_records.append(repaired)
                    if fixes_out is not None:
                        fixes_out.append(len(issues))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                err={"user_index":user_index,"record_index":record_index,"error":str(last_exc),"record":dict(user_context),"attempts":GENERATION_MAX_ATTEMPTS_PER_RECORD}
                if record_errors_out is not None: record_errors_out.append(err)
                logger.warning("[DataGeneration] Unable to produce a valid transactional record user=%d record=%d after %d attempts: %s",user_index,record_index,GENERATION_MAX_ATTEMPTS_PER_RECORD,last_exc)
    history_changes: list[str] = []
    if bool((rules or {}).get("agentic")):
        generated_records, history_changes = _history_set_derived_fields(compiled, generated_records)
    if history_changes:
        # History reconciliation is a generator-stage calculation, not a repair. Run a cheap
        # fail-closed contract/causal assertion after it so the client never receives an edited
        # snapshot that bypassed final QA.
        for record in generated_records:
            _assert_temporal_consistency(record, variables, relations=record_plan.temporal_relations)
            _assert_authoritative_formulas(record, variables, rules=rules)
            _strict_validate_record(record, variables, rules=rules)

    return generated_records


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
    """Infer safe, domain-neutral temporal relationships from the confirmed schema.

    Explicit temporal rules and datetime depends_on edges remain authoritative. In addition,
    the generator protects universally meaningful field-name pairs such as start/end and
    requested/confirmation. These inferred relations are deliberately narrow: unrelated
    timestamps are never reordered merely because they share a generic word like ``date``.
    """
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    out: list[tuple[str, str, int | None, int]] = []
    seen: set[tuple[str, str]] = set()

    def add_relation(parent: str, child: str, max_gap: int | None, min_gap: int = 0) -> None:
        if parent == child or parent not in by_name or child not in by_name:
            return
        if str(by_name[parent].get("dtype", "")).strip().lower() != "datetime":
            return
        if str(by_name[child].get("dtype", "")).strip().lower() != "datetime":
            return
        key = (parent, child)
        normalized_min = max(0, int(min_gap))
        if key not in seen:
            seen.add(key)
            out.append((parent, child, max_gap, normalized_min))
            return
        # Multiple evidence sources can describe the same edge. Preserve the strictest safe
        # lower bound and any explicitly declared upper bound instead of letting an earlier
        # dependency with min_gap=0 weaken the later universal request->confirmation invariant.
        for index, (old_parent, old_child, old_max, old_min) in enumerate(out):
            if old_parent == parent and old_child == child:
                effective_max = old_max if old_max is not None else max_gap
                if old_max is not None and max_gap is not None:
                    effective_max = min(old_max, max_gap)
                out[index] = (parent, child, effective_max, max(old_min, normalized_min))
                break

    # 1) Explicit rules supplied by the schema layer. These rules are executable only when the
    # confirmed schema itself provides structural evidence: an explicit datetime dependency or an
    # unambiguous same-resource relationship. Cross-resource LLM guesses are deliberately ignored.
    for item in (rules or {}).get("temporal_rules", []) or []:
        if not isinstance(item, dict):
            continue
        parent = str(item.get("before", ""))
        child = str(item.get("after", ""))
        if not parent or not child or parent not in by_name or child not in by_name:
            continue
        if not is_supported_temporal_rule(by_name[parent], by_name[child]):
            continue
        # LLM-provided bounds are hints, not executable constraints. Only source-declared
        # bounds can become hard temporal limits.
        max_gap = source_declared_max_delay_seconds(by_name[parent], by_name[child])
        min_gap = source_declared_min_delay_seconds(by_name[parent], by_name[child]) or 0
        add_relation(parent, child, max_gap, min_gap)

    # 2) Explicit depends_on edges between datetime fields.
    for child_name, child in by_name.items():
        if str(child.get("dtype", "")).strip().lower() != "datetime":
            continue
        for dep in child.get("depends_on", []) or []:
            parent_name = str(dep)
            parent = by_name.get(parent_name)
            if parent and is_supported_temporal_rule(parent, child):
                add_relation(
            parent_name,
            child_name,
            source_declared_max_delay_seconds(parent, child),
            source_declared_min_delay_seconds(parent, child) or 0,
        )

    def add_suffix_pair(
        start_suffixes: tuple[str, ...],
        end_suffixes: tuple[str, ...],
        max_gap: int | None,
        label_contains: tuple[str, ...] = (),
    ) -> None:
        """Pair lifecycle fields only when their canonical resource families agree.

        Exact prefix matches are preferred. A canonical-family fallback handles harmless naming
        variants such as ``resourcex_*`` vs ``resource_x_*`` without ever pairing different
        resources.
        """
        starts: list[tuple[str, str]] = []
        ends: list[tuple[str, str]] = []
        for name in by_name:
            low = name.casefold()
            if label_contains and not all(token in low for token in label_contains):
                continue
            start_suffix = next((suffix for suffix in start_suffixes if low.endswith(suffix)), None)
            end_suffix = next((suffix for suffix in end_suffixes if low.endswith(suffix)), None)
            if start_suffix:
                starts.append((name, low[:-len(start_suffix)]))
            if end_suffix:
                ends.append((name, low[:-len(end_suffix)]))

        for start_name, start_prefix in starts:
            # First prefer an exact normalized prefix match; this is unambiguous and cheapest.
            exact = [end_name for end_name, end_prefix in ends if end_prefix == start_prefix]
            if exact:
                add_relation(start_name, sorted(exact)[0], max_gap, 1 if any(suffix.endswith("requested_date") or suffix.endswith("requested_timestamp") or suffix.endswith("requested_datetime") or suffix.endswith("requested_date_time") for suffix in start_suffixes) else 0)
                continue

            family = normalize_temporal_family(start_name)
            family_matches = [
                end_name
                for end_name, _end_prefix in ends
                if normalize_temporal_family(end_name) == family
            ]
            # Only use the fallback when the family has a single target. If multiple target
            # timestamps exist, ambiguity is safer to leave unconstrained than to invent a link.
            if len(family_matches) == 1:
                add_relation(start_name, family_matches[0], max_gap, 1 if any("requested" in suffix for suffix in start_suffixes) else 0)

    # 3) Same-resource lifecycle pairs. Normalize common lifecycle suffixes before matching so
    # ``order_created_at`` and ``order_completed_at`` can be linked while unrelated resources
    # such as ``resource_a_*`` and ``resource_b_*`` remain independent.
    lifecycle_pairs = {
        ("start", "presentation"),
        ("start", "response"),
        ("start", "completion"),
        ("start", "end"),
        ("dispatch", "presentation"),
        ("dispatch", "response"),
        ("presentation", "response"),
        ("presentation", "completion"),
        ("response", "completion"),
    }
    datetime_vars = [v for v in variables if str(v.get("dtype", "")).strip().lower() == "datetime"]
    for parent in datetime_vars:
        for child in datetime_vars:
            if parent is child:
                continue
            if normalize_temporal_family(parent.get("name")) != normalize_temporal_family(child.get("name")):
                continue
            if _temporal_role(parent) == _temporal_role(child):
                continue
            if (_temporal_role(parent), _temporal_role(child)) in lifecycle_pairs:
                add_relation(
                    str(parent.get("name")),
                    str(child.get("name")),
                    source_declared_max_delay_seconds(parent, child),
                    source_declared_min_delay_seconds(parent, child) or 0,
                )

    # 4) Universal validity windows. Enforce ordering without imposing an arbitrary duration
    # when the source contract does not define one.
    add_suffix_pair(("_start_date_time", "_start_datetime", "_start_date", "_start_at"), ("_end_date_time", "_end_datetime", "_end_date", "_end_at"), None)

    # 5) Request/confirmation lifecycle. Only pair fields belonging to the SAME canonical
    # resource family. Shared words like request/confirmation are not evidence that two
    # different resources form one lifecycle. There is no arbitrary maximum delay here.
    add_suffix_pair(
        ("_requested_date_time", "_requested_datetime", "_requested_timestamp", "_requested_date"),
        ("_confirmation_date_time", "_confirmation_datetime", "_confirmation_timestamp", "_confirmation_date"),
        None,
    )

    # 6) Common presentation/decision pair. Again, only pair fields from the same canonical
    # business resource family. A generic seven-day ceiling caused unrelated offer/decision
    # timestamps to become hard constraints; ordering is the only universal invariant.
    presentation_fields = [
        name for name, var in by_name.items()
        if str(var.get("dtype", "")).strip().lower() == "datetime"
        and any(token in name.casefold() for token in ("offer_presented", "offer_presentation", "presented_at", "presented_timestamp"))
    ]
    decision_fields = [
        name for name, var in by_name.items()
        if str(var.get("dtype", "")).strip().lower() == "datetime"
        and any(token in name.casefold() for token in ("decision_timestamp", "decision_date", "decision_at"))
    ]
    if len(presentation_fields) == 1 and len(decision_fields) == 1:
        parent_name, child_name = presentation_fields[0], decision_fields[0]
        if normalize_temporal_family(parent_name) == normalize_temporal_family(child_name):
            add_relation(parent_name, child_name, None, 0)

    return out


def _enforce_temporal_consistency(
    rec: dict,
    variables: list[dict],
    rules: dict | None = None,
    relations: list[tuple[str, str, int | None, int]] | tuple[tuple[str, str, int | None, int], ...] | None = None,
) -> tuple[dict, list[str]]:
    """Repair causal timestamp contradictions deterministically.

    The pass is intentionally iterative because chains such as
    offer -> decision -> processing -> completion can require more than one repair.
    Explicit source-declared min/max delay windows are respected. When no source bound exists,
    only causal ordering is enforced. Unrelated timestamps are never reordered.
    """
    rec = dict(rec)
    issues: list[str] = []
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    relations = list(relations) if relations is not None else _infer_temporal_relationships(variables, rules=rules)
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
            rec[child_name] = _format_datetime_for_variable(desired, child_var)
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
        if _temporal_field_is_absent_by_scenario(field_def, rules) and rec.get(field) is None:
            continue
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


def _assert_authoritative_formulas(
    rec: dict, variables: list[dict], rules: dict | None = None
) -> None:
    """Fail closed when a final record no longer satisfies a confirmed formula contract."""
    variable_by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    for field, expr in _collect_formula_specs(variables, rules):
        if field not in rec:
            continue
        field_def = variable_by_name.get(field) or {}
        if _temporal_field_is_absent_by_scenario(field_def, rules) and rec.get(field) is None:
            continue
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
            raise ValueError(f"{field} violates confirmed formula '{expr}'")


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
            if selected is None:
                selected = _weighted_choice({"choices": declared, "weights": params.get("weights")}, rec)
            rec[name] = selected
            value = rec[name]
            issues.append(f"{name} resampled from declared CSV values")

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
    # Catch generated placeholders that echo the field name, e.g. FIELD_NAME_ID for field_name_id.
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


def _variables_by_name(variables: list[dict]) -> dict[str, dict]:
    return {str(v.get("name")): v for v in variables if v.get("name")}


def _strict_validate_record(rec: dict, variables: list[dict], rules: dict | None = None) -> None:
    """Final fail-closed validation for the complete confirmed contract."""
    by_name = _variables_by_name(variables)
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
                parsed_date = date.fromisoformat(str(value))
                if str(value) != parsed_date.isoformat():
                    raise ValueError("non-canonical date")
            except Exception:
                raise ValueError(f"{name} has invalid date value")

        params = var.get("params") or {}

        # Declared string constraints are part of the confirmed executable contract.
        # Enforce them regardless of provenance so they survive confirmation and reloads.
        if isinstance(value, str):
            try:
                min_length = int(params["min_length"]) if params.get("min_length") is not None else None
                max_length = int(params["max_length"]) if params.get("max_length") is not None else None
            except (TypeError, ValueError):
                min_length = max_length = None
            if min_length is not None and len(value) < min_length:
                raise ValueError(f"{name} is shorter than source minLength")
            if max_length is not None and len(value) > max_length:
                raise ValueError(f"{name} exceeds source maxLength")
            pattern = params.get("pattern")
            if pattern:
                try:
                    if re.fullmatch(str(pattern), value) is None:
                        raise ValueError(f"{name} does not match source pattern")
                except re.error:
                    # OpenAPI/JSON Schema regex dialects can differ from Python's. Keep the
                    # source contract intact rather than rejecting an otherwise valid document.
                    logger.warning("[QA] Skipping unsupported Python regex dialect for source-backed field %s", name)

        declared_multiple = params.get("multiple_of")
        if dtype in _NUMERIC_DTYPES and declared_multiple is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                multiple = float(declared_multiple)
            except (TypeError, ValueError):
                multiple = 0.0
            if multiple > 0:
                quotient = float(value) / multiple
                if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{name} violates source multipleOf={declared_multiple}")
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
        requested_suffixes = ("_requested_date_time", "_requested_datetime", "_requested_timestamp", "_requested_date")
        confirmation_suffixes = ("_confirmation_date_time", "_confirmation_datetime", "_confirmation_timestamp", "_confirmation_date")
        requested_suffix = next((suffix for suffix in requested_suffixes if low.endswith(suffix)), None)
        if requested_suffix:
            prefix = low[:-len(requested_suffix)]
            for confirmation_suffix in confirmation_suffixes:
                end_name = prefix + confirmation_suffix
                if end_name in by_name:
                    _assert_order(name, end_name, "request/confirmation lifecycle")
                    break

    presentation_fields = [
        name for name, var in by_name.items()
        if str(var.get("dtype", "")).strip().lower() == "datetime"
        and any(token in name.casefold() for token in ("offer_presented", "offer_presentation", "presented_at", "presented_timestamp"))
    ]
    decision_fields = [
        name for name, var in by_name.items()
        if str(var.get("dtype", "")).strip().lower() == "datetime"
        and any(token in name.casefold() for token in ("decision_timestamp", "decision_date", "decision_at"))
    ]
    if len(presentation_fields) == 1 and len(decision_fields) == 1:
        _assert_order(presentation_fields[0], decision_fields[0], "presentation/decision lifecycle")

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


def _enforce_generic_business_consistency(rec: dict, variables: list[dict], rules: dict | None = None) -> tuple[dict, list[str]]:
    """Enforce high-confidence cross-field relationships inferred from the confirmed contract.

    This layer is intentionally vocabulary-driven rather than industry/field-list driven. It handles
    common state machines such as request->confirmation, offer presentation->acceptance, auto-run
    configuration, and linked remaining/reserved values without inventing source semantics.
    """
    rec = dict(rec)
    issues: list[str] = []
    by_name = {str(v.get("name") or ""): v for v in variables if v.get("name")}

    def norm(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")

    def set_value(name: str, value: Any, reason: str) -> None:
        if name in rec and rec.get(name) != value:
            rec[name] = value
            issues.append(reason)

    def dt(name: str) -> datetime | None:
        return _qa_parse_dt(rec.get(name)) if name in rec else None

    names = list(by_name)
    # 1) Auto-run configuration is a single stateful contract. A disabled recurring process has no
    # recurring schedule/occurrence count; an enabled process needs both when the schema exposes them.
    auto_fields = [n for n in names if "auto" in norm(n) and any(t in norm(n) for t in ("recurring", "automatic", "renew"))]
    auto = auto_fields[0] if auto_fields else None
    if auto and isinstance(rec.get(auto), bool):
        period_count = next((n for n in names if "period" in norm(n) and ("number" in norm(n) or "count" in norm(n))), None)
        recurring = next((n for n in names if "period" in norm(n) and "recurr" in norm(n) and str(by_name[n].get("dtype") or "").lower() in {"categorical", "string"}), None)
        if rec.get(auto) is False:
            if period_count and by_name[period_count].get("nullable", True):
                set_value(period_count, None, f"{period_count} cleared because {auto} is false")
            if recurring and by_name[recurring].get("nullable", True):
                set_value(recurring, None, f"{recurring} cleared because {auto} is false")
        else:
            if period_count and rec.get(period_count) in (None, 0):
                params = by_name[period_count].get("params") or {}
                lo = _to_finite_float(params.get("min", params.get("lo")), 1.0) or 1.0
                hi = _to_finite_float(params.get("max", params.get("hi")), max(lo, 12.0))
                candidate = int(round(lo if hi is None else _rng().uniform(max(1.0, lo), max(lo, hi))))
                set_value(period_count, max(1, candidate), f"{period_count} aligned to enabled recurring processing")
            if recurring and rec.get(recurring) is None:
                choices = list((by_name[recurring].get("params") or {}).get("choices") or [])
                if choices:
                    set_value(recurring, _rng().choice(choices), f"{recurring} populated for enabled recurring processing")

    # 2) Completed outcomes require a confirmation/completion instant when such an instant exists.
    status_fields = [
        n for n in names
        if any(token in norm(n).split("_") for token in ("status", "state", "outcome", "result"))
        and str(by_name[n].get("dtype") or "").lower() in {"categorical", "string"}
    ]
    for status_name in status_fields:
        value = norm(rec.get(status_name))
        if value not in {"completed", "complete", "success", "successful", "settled", "fulfilled", "done"}:
            continue
        base = norm(status_name)
        prefix = base.rsplit("_status", 1)[0] if "_status" in base else base.rsplit("_state", 1)[0] if "_state" in base else base
        confirmation = next((n for n in names if norm(n).startswith(prefix) and any(x in norm(n) for x in ("confirmation", "confirmed", "completion", "completed")) and str(by_name[n].get("dtype") or "").lower() == "datetime"), None)
        requested = next((n for n in names if norm(n).startswith(prefix) and any(x in norm(n) for x in ("requested", "request")) and str(by_name[n].get("dtype") or "").lower() == "datetime"), None)
        if confirmation and dt(confirmation) is None and requested and dt(requested) is not None:
            params = by_name[confirmation].get("params") or {}
            value_dt = dt(requested) + timedelta(seconds=_rng().randint(30, 900))
            set_value(confirmation, _format_datetime_for_variable(value_dt, by_name[confirmation]), f"{confirmation} generated from completed {status_name}")

    # 3) Accepted decisions imply a presented decision opportunity when both flags exist. Reuse the
    # generic offer lifecycle helper for the actual implication; here only ground conversion on status.
    rec, offer_issues = _offer_lifecycle_consistency(rec, variables)
    issues.extend(offer_issues)

    # 4) Numeric remaining/reserved relationship when the same resource exposes both quantities.
    remaining_fields = [n for n in names if "remaining" in norm(n) and str(by_name[n].get("dtype") or "").lower() in _NUMERIC_DTYPES]
    for remaining in remaining_fields:
        r_value = _to_finite_float(rec.get(remaining), None)
        if r_value is None:
            continue
        tokens = set(norm(remaining).split("_")) - {"remaining", "value", "amount", "quantity", "unit", "units"}
        reserved = next((n for n in names if "reserved" in norm(n) and tokens & set(norm(n).split("_")) and str(by_name[n].get("dtype") or "").lower() in _NUMERIC_DTYPES), None)
        if reserved:
            reserved_value = _to_finite_float(rec.get(reserved), None)
            if reserved_value is not None and r_value > reserved_value:
                set_value(remaining, round(reserved_value, int((by_name[remaining].get("params") or {}).get("precision", 2) or 2)), f"{remaining} reduced to remain within {reserved}")

    # 6) Validity windows must contain the transaction anchor when a same-resource anchor is obvious.
    validity_starts = [n for n in names if norm(n).endswith(("_start_date_time", "_start_datetime", "_start_date", "_valid_from", "_validity_start")) and str(by_name[n].get("dtype") or "").lower() == "datetime"]
    for start in validity_starts:
        start_dt = dt(start)
        end = None
        stem = norm(start)
        for suffix in ("_start_date_time", "_start_datetime", "_start_date", "_valid_from", "_validity_start"):
            if stem.endswith(suffix):
                prefix = stem[:-len(suffix)]
                end = next((n for n in names if norm(n) == prefix + "_end_date_time" or norm(n) == prefix + "_end_datetime" or norm(n) == prefix + "_end_date" or norm(n) == prefix + "_valid_to" or norm(n) == prefix + "_validity_end"), None)
                break
        if not end:
            continue
        end_dt = dt(end)
        anchors = [
            n for n in names
            if str(by_name[n].get("dtype") or "").lower() == "datetime"
            and any(x in norm(n) for x in ("requested", "transaction", "event", "record", "created", "presentation", "response", "confirmation"))
            and set(prefix.split("_")) & set(norm(n).split("_"))
        ]
        anchor = next((dt(n) for n in anchors if dt(n) is not None), None)
        if anchor is None:
            continue
        if start_dt is None or end_dt is None:
            continue
        if start_dt > anchor:
            duration = max(1, int((end_dt - start_dt).total_seconds())) if end_dt > start_dt else 86400
            new_start = anchor - timedelta(seconds=duration)
            set_value(start, _format_datetime_for_variable(new_start, by_name[start]), f"{start} moved before its transaction anchor")
            start_dt = new_start
        if end_dt < anchor:
            duration = max(1, int((end_dt - start_dt).total_seconds())) if start_dt is not None and end_dt > start_dt else 86400
            new_end = anchor + timedelta(seconds=duration)
            set_value(end, _format_datetime_for_variable(new_end, by_name[end]), f"{end} moved after its transaction anchor")

    return rec, issues

def _offer_lifecycle_consistency(rec: dict, variables: list[dict]) -> tuple[dict, list[str]]:
    """Enforce obvious offer lifecycle dependencies without assuming an industry-specific schema."""
    rec = dict(rec)
    issues: list[str] = []
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    names = list(by_name)

    def find_flag(*tokens: str) -> str | None:
        for name in names:
            low = name.casefold()
            if "flag" not in low:
                continue
            if "offer" not in low:
                continue
            if any(token in low for token in tokens):
                return name
        return None

    presented = find_flag("presented", "presentation")
    accepted = find_flag("accepted", "acceptance")
    converted = find_flag("converted", "conversion")

    def find_time(*tokens: str) -> str | None:
        candidates = []
        for name, var in by_name.items():
            if str(var.get("dtype") or "").casefold() != "datetime":
                continue
            low = name.casefold()
            if "offer" in low and all(token in low for token in tokens):
                candidates.append(name)
        return sorted(candidates)[0] if candidates else None

    impression_ts = find_time("impression")
    conversion_ts = find_time("conversion")

    def set_value(name: str | None, value: Any, reason: str) -> None:
        if not name or name not in rec:
            return
        if rec.get(name) != value:
            rec[name] = value
            issues.append(reason)

    # An offer cannot be accepted or converted when the same record says it was never presented.
    if presented and rec.get(presented) is False:
        set_value(accepted, False, f"{accepted} set false because {presented} is false")
        set_value(converted, False, f"{converted} set false because {presented} is false")
        if impression_ts and by_name.get(impression_ts, {}).get("nullable", True):
            set_value(impression_ts, None, f"{impression_ts} cleared because {presented} is false")

    if accepted and rec.get(accepted) is False:
        set_value(converted, False, f"{converted} set false because {accepted} is false")
        if conversion_ts and by_name.get(conversion_ts, {}).get("nullable", True):
            set_value(conversion_ts, None, f"{conversion_ts} cleared because {accepted} is false")

    if converted and rec.get(converted) is True:
        set_value(accepted, True, f"{accepted} set true because {converted} is true")
        set_value(presented, True, f"{presented} set true because {converted} is true")

    return rec, issues


def _enforce_obvious_semantic_consistency(rec: dict, variables: list[dict], rules: dict | None = None) -> tuple[dict, list[str]]:
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

    rec, offer_issues = _offer_lifecycle_consistency(rec, variables)
    issues.extend(offer_issues)

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


def _assert_temporal_consistency(
    rec: dict,
    variables: list[dict],
    *,
    relations: list[tuple[str, str, int | None, int]] | tuple[tuple[str, str, int | None, int], ...] | None = None,
) -> None:
    """Fail closed when any declared/inferred temporal relation remains contradictory."""
    rels = list(relations) if relations is not None else _infer_temporal_relationships(variables)
    for parent_name, child_name, max_gap, min_gap in rels:
        parent_dt = _qa_parse_dt(rec.get(parent_name))
        child_dt = _qa_parse_dt(rec.get(child_name))
        if parent_dt is None or child_dt is None:
            continue
        delta = (child_dt - parent_dt).total_seconds()
        if delta < min_gap:
            raise ValueError(
                f"temporal relationship violation: {child_name} occurs before {parent_name}"
            )
        if max_gap is not None and delta > max_gap:
            raise ValueError(
                f"temporal relationship violation: {child_name} exceeds allowed gap from {parent_name}"
            )


def _validate_record(
    rec: dict,
    variables: list[dict],
    field_order: list[str],
    transactional: bool,
    rules: dict | None = None,
    temporal_relations: list[tuple[str, str, int | None, int]] | tuple[tuple[str, str, int | None, int], ...] | None = None,
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
                raise ValueError(f"{name} has invalid datetime value")
            elif dtype == "date":
                try:
                    parsed_date = date.fromisoformat(str(value))
                    if str(value) != parsed_date.isoformat():
                        raise ValueError("non-canonical date")
                except Exception:
                    raise ValueError(f"{name} has invalid date value")
        except Exception as exc:
            # Never replace a generation/contract failure with a dtype default such as 0, False,
            # or an empty string. Those defaults turn bad records into plausible-looking data.
            raise ValueError(f"{name} failed confirmed schema validation: {exc}") from exc

        if dtype in {"float", "decimal", "number", "numeric"} and isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool):
            rec[name] = round(float(rec[name]), precision)

        declared_options = _declared_param_options(params)
        if declared_options and rec.get(name) is not None:
            if not any(_matches_declared_option(rec.get(name), opt) for opt in declared_options):
                preferred = _coerce_rule_values(_rule_constraint_for(name, rules).get("preferred_values"))
                selected = next(
                    (opt for opt in declared_options if any(_matches_declared_option(opt, pref) for pref in preferred)),
                    None,
                )
                if selected is None:
                    selected = _weighted_choice({"choices": declared_options, "weights": params.get("weights")}, rec)
                rec[name] = selected
                issues.append(f"{name} resampled from declared schema values")
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
                # another generated field (min=event_count_30d). Never call
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
        if _temporal_field_is_absent_by_scenario(field_def, rules) and rec.get(field) is None:
            continue
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
        if _temporal_field_is_absent_by_scenario(field_def, rules) and rec.get(field) is None:
            continue
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

    # Temporal relationships are enforced once after all scenario/domain semantic mutations.
    # This avoids repeatedly parsing the same timestamp pairs on the hot path.
    rec, temporal_issues = _enforce_temporal_consistency(
        rec, variables, rules=rules, relations=temporal_relations
    )
    issues.extend(temporal_issues)
    # Recalculate explicit formulas after temporal repair because a moved parent timestamp
    # can legitimately change a dependent duration/formula field.
    rec, temporal_formula_issues = _enforce_authoritative_formulas(rec, variables, rules=rules)
    issues.extend(temporal_formula_issues)
    rec, final_contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(final_contract_issues)

    rec, generic_issues = _enforce_generic_business_consistency(rec, variables, rules=rules)
    issues.extend(generic_issues)
    rec, semantic_issues = _enforce_obvious_semantic_consistency(rec, variables, rules=rules)
    issues.extend(semantic_issues)
    # Domain semantics can move timestamps/amounts/statuses, so authoritative formulas must
    # be recalculated once more before the final contract/strict validation boundary.
    rec, post_domain_formula_issues = _enforce_authoritative_formulas(rec, variables, rules=rules)
    issues.extend(post_domain_formula_issues)
    rec, final_contract_issues = _enforce_csv_contract(rec, variables, rules=rules)
    issues.extend(final_contract_issues)

    # A formula or scenario-semantic repair above may change a timestamp after the earlier
    # temporal pass. Enforce the temporal graph once more as the final pre-validation invariant.
    rec, terminal_temporal_issues = _enforce_temporal_consistency(
        rec, variables, rules=rules, relations=temporal_relations
    )
    issues.extend(terminal_temporal_issues)

    # Fail closed after every repair/constraint pass. This prevents a future change to
    # one repair stage from silently reintroducing a contradiction into final_records.
    _assert_temporal_consistency(rec, variables, relations=temporal_relations)

    bad_placeholders = []
    for var in variables:
        name = str(var.get("name") or "")
        if name in rec and _is_placeholder_value(name, rec.get(name), var):
            bad_placeholders.append(name)
    if bad_placeholders:
        raise ValueError(f"Unresolved placeholder values remain: {bad_placeholders}")

    # Presentation format is a deterministic CSV concern, not an LLM concern.
    # Formatting preserves the parsed instant and therefore does not require another
    # temporal-consistency scan after serialization.
    rec = _format_datetime_fields(rec, variables)

    rec, final_formula_issues = _enforce_authoritative_formulas(rec, variables, rules=rules)
    issues.extend(final_formula_issues)
    rec = _format_datetime_fields(rec, variables)

    # Serialization is part of the client-facing contract. Re-check temporal relations and formulas
    # after the final format conversion so presentation cannot hide an invalid causal or arithmetic gap.
    _assert_temporal_consistency(rec, variables, relations=temporal_relations)
    _assert_authoritative_formulas(rec, variables, rules=rules)
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
    seed: int | None = None,
    as_of: Any = None,
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

    # Behaviour-pack path: a scenario confirmed against a behaviour pack is generated by a causal journey
    # simulation and scored on its delivered rows. Scenarios without a pack keep the legacy generator below.
    from synth.service import generate as pack_generate, resolve_pack

    pack = resolve_pack(scenario_context)
    if pack is not None:
        if type_of_data != "transactional":
            raise ValueError(f"Behaviour pack '{pack.pack_id}' generates transactional data only.")
        from core.scenario_semantics import classify_outcome_mode

        mode = classify_outcome_mode(
            scenario_type=str(state.scenario_type or ""),
            expected_outcome=str(state.expected_outcome or ""),
            business_response=str(state.business_response or ""),
            business_scenario=str(state.business_scenario or ""),
        )
        result = pack_generate(
            pack, variables, count=state.count, per_entity=state.records_per_user,
            seed=seed, as_of=as_of, mode=mode,
        )
        state.raw_records = list(result.records)
        state.final_records = list(result.records)
        state.field_order = list(result.fields)
        state.validation_report = result.validation_report
        return state

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
        expected_transactional_records = state.count * state.records_per_user
        if AGENTIC_REQUIRE_EXACT_RECORD_COUNT and len(state.raw_records) != expected_transactional_records:
            raise ValueError(
                f"Agentic generation produced {len(state.raw_records)} clean transactional records; "
                f"expected exactly {expected_transactional_records}. Refusing to deliver a partial dataset."
            )
    else:
        transactional_fixes = []
        from config.runtime import GENERATION_MAX_ATTEMPTS_PER_RECORD
        aggregate_plan = _build_generation_plan(variables, rules=state.rules)
        for index in range(state.count):
            last_exc: Exception | None = None
            for _attempt in range(GENERATION_MAX_ATTEMPTS_PER_RECORD):
                try:
                    state.raw_records.append(_generate_record(variables, rules=state.rules, plan=aggregate_plan, apply_repairs=False))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                state.record_errors.append({"record_index": index, "error": str(last_exc), "record": {}})

    checked: list[dict] = []
    fixes = 0
    clean_records = 0
    if type_of_data == "transactional":
        # _transactional_records now runs the complete validator exactly once for every
        # accepted row, including its final strict validation boundary. Re-validating the
        # same rows here only duplicated the most expensive work.
        checked = list(state.raw_records)
        clean_records = sum(1 for fix_count in transactional_fixes if fix_count == 0)
        fixes = sum(transactional_fixes)
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
                if not issues:
                    clean_records += 1
            except Exception as exc:
                state.record_errors.append({"record_index": record_index, "error": str(exc), "record": dict(record)})

    state.final_records = checked
    expected_records = state.count * state.records_per_user if type_of_data == "transactional" else state.count
    valid_rate = (len(checked) / expected_records) if expected_records else 1.0
    clean_rate = (clean_records / len(checked)) if checked else 0.0
    repaired_record_count = max(0, len(checked) - clean_records)
    repaired_record_rate = (repaired_record_count / len(checked)) if checked else 0.0
    state.validation_report = {
        "requested_records": expected_records,
        "total_input": len(state.raw_records),
        "total_valid": len(checked),
        "total_dropped": len(state.record_errors),
        "record_errors": len(state.record_errors),
        "valid_record_rate": round(valid_rate * 100.0, 3),
        "clean_record_rate": round(clean_rate * 100.0, 3),
        "repaired_record_rate": round(repaired_record_rate * 100.0, 3),
        "target_valid_record_rate": 100.0,
        "target_clean_record_rate": 95.0,
        "quality_target_met": valid_rate >= 1.0 and clean_rate >= 0.95,
        "contract_pass_rate": 100.0 if checked else 0.0,
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
        qa_temporal_relations = _build_generation_plan(variables, rules=state.rules).temporal_relations

        checked: list[dict] = []
        algo_fixed = 0
        for record_index, record in enumerate(records):
            try:
                repaired, issues = _validate_record(
                    record, variables, state.field_order, transactional,
                    rules=state.rules, temporal_relations=qa_temporal_relations,
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
            ) + (
                "\n\nGENERATION POLICY (records should arrive pre-correct):\n"
                + "\n".join(f"- {r}" for r in state.rules.get("generation_policy", []))
            ) + f"\n\nFULL SCENARIO SCHEMA CONTRACT (authoritative):\n{schema_contract}\n"
            for i in range(0, len(checked), _CHUNK):
                chunk = checked[i:i + _CHUNK]
                try:
                    qa_chunk = []
                    original_by_id: dict[str, dict] = {}
                    for offset, original in enumerate(chunk):
                        qa_id = f"{i + offset}"
                        tagged = dict(original)
                        tagged["__qa_id"] = qa_id
                        qa_chunk.append(tagged)
                        original_by_id[qa_id] = dict(original)
                    # Re-run the request with internal IDs; the IDs are provenance metadata, not schema fields.
                    result = self._llm.generate_json(
                        system_prompt,
                        f"Scenario: {state.scenario}\nIndustry: {state.industry}\nCountry: {state.country or 'GLOBAL'}\n"
                        f"Domain: {state.domain}\nBusiness scenario: {state.business_scenario}\n"
                        f"Business response: {state.business_response or ''}\nExpected outcome: {state.expected_outcome or ''}\n"
                        f"Use case: {state.use_case or ''}\nScenario type: {state.scenario_type or ''}\nRecords to validate:\n{json.dumps(qa_chunk, default=str)}",
                        temperature=0.1,
                    )
                    validated = result.get("valid_records", qa_chunk)
                    dropped = result.get("dropped_records", []) or []
                    returned: dict[str, dict] = {}
                    invalid_qa = False
                    for item in validated if isinstance(validated, list) else []:
                        if not isinstance(item, dict) or "__qa_id" not in item:
                            invalid_qa = True
                            break
                        qa_id = str(item.get("__qa_id") or "")
                        if qa_id not in original_by_id or qa_id in returned:
                            invalid_qa = True
                            break
                        cleaned = {k: item[k] for k in state.field_order if k in item}
                        returned[qa_id] = cleaned
                    dropped_ids = {
                        str(item.get("__qa_id") if isinstance(item, dict) else item)
                        for item in dropped if isinstance(item, (dict, str, int))
                    }
                    if not dropped_ids.issubset(original_by_id.keys()) or (set(returned) & dropped_ids):
                        invalid_qa = True
                    if invalid_qa:
                        logger.warning("[QA] Chunk %d returned invalid record identities; deterministic chunk retained", i)
                        state.errors.append(f"QA chunk {i} returned invalid record identities; deterministic records retained")
                        valid_all.extend(chunk)
                    else:
                        # Missing IDs must be retained unless explicitly dropped. This prevents LLM QA
                        # from silently shrinking the dataset.
                        for qa_id, original in original_by_id.items():
                            if qa_id in returned:
                                valid_all.append(returned[qa_id])
                            elif qa_id not in dropped_ids:
                                valid_all.append(original)
                        dropped_all.extend(sorted(dropped_ids))
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
                        record, variables, state.field_order, transactional,
                        rules=state.rules, temporal_relations=qa_temporal_relations,
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

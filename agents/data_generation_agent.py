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

logger = logging.getLogger(__name__)

# Canonical numeric dtypes used by the CSV schema/generation QA layer.
# csv_scenario.py normalizes integer/decimal/uuid client types before they
# reach the generation agent, but keeping the aliases here makes the helper
# safe for both normalized and direct callers.
_NUMERIC_DTYPES = {"int", "integer", "float", "decimal", "number", "numeric"}

# Optional LLM QA configuration.  Deterministic QA remains the default; these
# constants are only used when QA_LLM_MODE=full is explicitly enabled.
_CHUNK = max(1, int(os.getenv("QA_LLM_CHUNK_SIZE", "10")))
_QA_SYSTEM = """You are the final QA validator for generated synthetic data.
Validate each supplied record against the CSV-defined schema and the supplied
business/cross-field rules. Preserve valid values, repair only clear deterministic
violations when possible, and do not invent fields that are not in the schema.
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

    Scenario definitions can come from an LLM or CSV, so numeric params may be
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
    return base.isoformat()


def _ts_offset(params: dict, rec: dict) -> str:
    base_str = rec.get(params["base_field"], datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    offset = timedelta(seconds=random.randint(params["min_sec"], params["max_sec"]))
    return (base + offset).isoformat()


def _ts_add_field(params: dict, rec: dict) -> str:
    base_str = rec.get(params["base_field"], datetime.now(timezone.utc).isoformat())
    base = _parse_dt(base_str)
    seconds = int(rec.get(params["add_seconds_field"], 60))
    return (base + timedelta(seconds=seconds)).isoformat()


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


def _parse_dt(s: str) -> datetime:
    """Parse ISO-8601 string to timezone-aware datetime."""
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


# ── Dispatch table ─────────────────────────────────────────────────────────────

_GENERATORS = {
    "prefixed_int":   lambda v, rec: _prefixed_int(v["params"], rec),
    "id_mirror":      lambda v, rec: _id_mirror(v["params"], rec),
    "e164_phone":     lambda v, rec: _e164_phone(v["params"], rec),
    "constant":       lambda v, rec: _constant(v["params"], rec),
    "weighted_choice":lambda v, rec: _weighted_choice(v["params"], rec),
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
}


def get_known_generator_types() -> set[str]:
    """Public accessor for the set of valid 'gen' type strings — used by
    core/csv_scenario.py to validate industry-supplied CSV variable definitions."""
    return set(_GENERATORS.keys())


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


def _apply_generation_constraint(var: dict, value, rec: dict, rules: dict | None):
    """Apply safe machine-readable SchemaAgent constraints to a generated value."""
    constraint = _rule_constraint_for(str(var.get("name", "")), rules)
    if value is None:
        return value

    params = var.get("params") if isinstance(var.get("params"), dict) else {}
    dtype = str(var.get("dtype", "")).strip().lower()
    precision = int(params.get("precision", 2) or 2)

    # Params are authoritative: if explicit values are provided, do not emit
    # anything outside that set. Preserve the original token/casing from params.
    declared_options = _declared_param_options(params)
    if declared_options:
        for opt in declared_options:
            if _matches_declared_option(value, opt):
                return opt
        # CSV choices are authoritative as the allowed set, while scenario-derived
        # preferred_values select the semantically appropriate member of that set.
        preferred = _coerce_rule_values(constraint.get("preferred_values")) if constraint else []
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

    CSV dependencies may point forward.  A topological pass makes those definitions
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
                # tokens that are not schema fields (common in human-readable CSVs).
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
    return _apply_scenario_semantics(rec, rules)


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
    return _apply_scenario_semantics(rec, rules)


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


def _transactional_records(compiled, user_count: int, records_per_user: int = 10,
                            rules: dict | None = None, record_errors_out: list[dict] | None = None) -> list[dict]:
    """Generate a fixed-length recent history for each user/entity.

    Stable user-context variables are generated once and copied into each row.
    History variables are regenerated for every historical row. The output remains flat so
    downstream QA operates on ordinary records; the API groups those records by
    entity_key after generation.
    """
    variables=list(compiled.variables); entity_key=compiled.entity_key
    records_per_user=max(1,min(50,int(records_per_user or 10)))
    timestamp_field=_pick_timestamp_field(variables)
    generated=[]; used_entity_keys=set()

    for user_index in range(user_count):
        try:
            user_context=_generate_selected_record(variables,set(compiled.user_fields),rules=rules)
            if entity_key and entity_key not in user_context:
                # Ensure the entity key is generated even if inferred user context omitted it.
                key_var=compiled.variable_by_name.get(entity_key)
                if key_var:
                    user_context=_generate_selected_record(variables,{entity_key},base=user_context,rules=rules)
            if entity_key and entity_key in user_context:
                attempts=0
                while str(user_context[entity_key]) in used_entity_keys and attempts < 10:
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
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fill_missing(rec: dict, field_order: list, dtype_map: dict) -> tuple[dict, int]:
    """Preserve the existing API behavior: fill missing/null fields with a safe default."""
    filled = 0
    for name in field_order:
        if rec.get(name) is None:
            rec[name] = _default_for_dtype(dtype_map.get(name, "string"))
            filled += 1
    return rec, filled


def _safe_formula(expr: str, rec: dict):
    """Evaluate the same small arithmetic expression language used by the generator."""
    allowed_funcs = {"round": round, "min": min, "max": max, "abs": abs}
    names = {}
    for k, v in rec.items():
        if k == "__current_field__" or v is None:
            continue
        if isinstance(v, str):
            parsed = _qa_parse_dt(v)
            names[k] = parsed if parsed is not None and ("T" in v or "+" in v or v.endswith("Z")) else v
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

    for var in variables:
        name = var["name"]
        if name not in rec or rec[name] is None:
            continue
        value = rec[name]
        dtype = var.get("dtype", "string")
        params = var.get("params") or {}
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
                rec[name] = datetime.now(timezone.utc).isoformat(); issues.append(f"{name} repaired as datetime")
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
        # A derived timestamp formula can be descriptive in the CSV. The executable
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

    # Re-validate authoritative formulas after semantic/conditional repairs.
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

    timestamp_field = next((name for name in ("record_timestamp", "transaction_timestamp", "timestamp", "created_at", "updated_at") if name in rec), None)
    base_ts = _qa_parse_dt(rec.get(timestamp_field)) if timestamp_field else None
    dispatch_ts = _qa_parse_dt(rec.get("notification_dispatch_ts"))
    response_ts = _qa_parse_dt(rec.get("customer_response_ts"))
    if base_ts and dispatch_ts and dispatch_ts < base_ts:
        rec["notification_dispatch_ts"] = base_ts.isoformat(); issues.append("notification_dispatch_ts corrected")
    if dispatch_ts and response_ts and response_ts < dispatch_ts:
        rec["customer_response_ts"] = dispatch_ts.isoformat(); issues.append("customer_response_ts corrected")

    allowed = set(field_order)
    return {k: v for k, v in rec.items() if k in allowed}, issues


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
            system_prompt = _QA_SYSTEM.format(
                rules="\n".join(f"- {r}" for r in state.rules.get("business_rules", [])) or "Apply the supplied CSV schema and deterministic checks.",
                cross_field_rules="\n".join(f"- {r}" for r in state.rules.get("cross_field_rules", [])) or "Validate declared mathematical, temporal, and business relationships.",
            )
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
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
import random
import uuid
import ast
from datetime import date, datetime, timedelta, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from core.dynamic_scenarios import resolve_variables, resolve_events, resolve_entity_key
from core.llm_client import GeminiClient
from core.state import WorkflowState
from config.industry_profiles import get_profile, country_allowed_value, payment_methods_for_profile

logger = logging.getLogger(__name__)

_MAX_EDGE_CASE_ATTEMPTS = 20


def _normalize_upi_in_generated_value(value: Any):
    """Normalize UPI references for Telecom output only.

    The caller must explicitly enable this for Telecom. Other industries may
    legitimately use UPI and must preserve it unchanged.
    """
    if isinstance(value, dict):
        return {k: _normalize_upi_in_generated_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_upi_in_generated_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_upi_in_generated_value(v) for v in value)
    if isinstance(value, str) and "upi" in value.lower():
        # Canonicalize the whole generated value rather than leaking phrases such
        # as "UPI Payment" or "UPI-linked Account" into final synthetic data.
        return "DIGITAL_WALLET"
    return value


def _normalize_upi_in_records(records: list[dict], industry: str | None = None) -> list[dict]:
    """Normalize UPI only for Telecom records; preserve it for other industries."""
    if str(industry or "").strip().lower() != "telecom":
        return records
    return [_normalize_upi_in_generated_value(dict(record)) for record in records]

def _edge_case_groups(edge_case_variables: list[dict] | None) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for item in edge_case_variables or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item.get("edge_case_name") or "Scenario Edge Case").strip()
        group = groups.setdefault(name, {
            "description": str(item.get("edge_case_description") or ""),
            "condition": str(item.get("condition") or "").strip(),
            "event_type": str(item.get("event_type") or "").strip().upper().replace(" ", "_"),
            "variables": [],
        })
        if item.get("condition") and not group.get("condition"):
            group["condition"] = str(item["condition"]).strip()
        if item.get("event_type") and not group.get("event_type"):
            group["event_type"] = str(item["event_type"]).strip().upper().replace(" ", "_")
        group["variables"].append(item)
    return groups

def _boolean_semantic(value):
    """Return True/False for common boolean encodings, otherwise None.

    This is intentionally data-driven and does not depend on any field name.
    """
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

def _safe_edge_condition(expression: str, rec: dict) -> bool:
    """Evaluate a configured condition against the actual record deterministically.

    The evaluator is schema/value aware: numeric-looking strings can participate in
    numeric comparisons and categorical equality/membership is case-insensitive.
    This keeps conditions stable across LLM/CSV casing and serialization differences
    without hard-coding any industry or field names.
    """
    if not expression:
        return True
    try:
        tree = ast.parse(str(expression), mode="eval")
        allowed_nodes = (
            ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.Compare,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
            ast.Is, ast.IsNot, ast.Name, ast.Constant, ast.List, ast.Tuple,
            ast.UnaryOp, ast.USub, ast.UAdd, ast.Load,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            if isinstance(node, ast.Name) and node.id not in rec:
                return False

        def _num(value):
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
            if isinstance(value, str):
                try:
                    x = float(value.strip())
                    return x if math.isfinite(x) else None
                except (TypeError, ValueError):
                    return None
            return None

        class _Transformer(ast.NodeTransformer):
            def visit_Compare(self, node):
                self.generic_visit(node)
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    return node
                op = node.ops[0]
                right = node.comparators[0]
                if isinstance(node.left, ast.Name):
                    name = node.left.id
                    actual = rec.get(name)
                    # Numeric comparison: make a numeric-looking actual value numeric.
                    if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)) and not isinstance(right.value, bool):
                        n = _num(actual)
                        if n is not None:
                            node.left = ast.Constant(value=n)
                    # Boolean-semantic values may be serialized as YES/NO, Y/N,
                    # enabled/disabled, 1/0, etc. Compare by semantic value.
                    elif isinstance(right, ast.Constant) and isinstance(right.value, bool):
                        actual_bool = _boolean_semantic(actual)
                        if actual_bool is not None:
                            node.left = ast.Constant(value=actual_bool)
                    # String equality/membership: canonicalize literal casing to the
                    # actual record value when they are equal ignoring case.
                    elif isinstance(actual, str) and isinstance(right, ast.Constant) and isinstance(right.value, str):
                        if actual.strip().lower() == right.value.strip().lower():
                            node.comparators[0] = ast.Constant(value=actual)
                    elif isinstance(actual, str) and isinstance(right, (ast.List, ast.Tuple)):
                        new_elts = []
                        for element in right.elts:
                            if isinstance(element, ast.Constant) and isinstance(element.value, str) and actual.strip().lower() == element.value.strip().lower():
                                new_elts.append(ast.Constant(value=actual))
                            else:
                                new_elts.append(element)
                        right.elts = new_elts
                return node

        tree = _Transformer().visit(tree)
        ast.fix_missing_locations(tree)
        return bool(eval(compile(tree, "<edge-condition>", "eval"), {"__builtins__": {}}, dict(rec)))
    except Exception:
        return False

def _apply_edge_case_overrides(rec: dict, edge_group: dict, variables: list[dict], profile: dict | None = None, rules: dict | None = None, active_fields: set[str] | None = None) -> dict:
    """Apply only relevant edge-case overrides, then recalculate available formulas.

    ``active_fields`` prevents an edge-case override belonging to another event from
    leaking into sparse transactional rows.  Entity-level generation calls this with
    the entity fields; event generation calls it with the event fields.
    """
    by_name = {v["name"]: v for v in variables}
    for edge_var in edge_group.get("variables", []):
        name = str(edge_var.get("name", ""))
        if name not in by_name:
            continue
        if active_fields is not None and name not in active_fields and name not in rec:
            continue
        effective = dict(by_name[name])
        for key in ("gen", "params", "formula", "dtype", "nullable"):
            if key in edge_var and edge_var[key] not in (None, ""):
                effective[key] = edge_var[key]
        generator = _GENERATORS.get(effective.get("gen"))
        if generator:
            helper = dict(rec)
            helper["__current_field__"] = name
            helper["__country_profile__"] = profile
            value = generator(effective, helper)
            rec[name] = _apply_generation_constraint(by_name[name], value, rec, rules)

    # Recalculate every formula whose dependencies are now available. Edge-case
    # overrides must not leave derived fields inconsistent.
    formula_vars = [v for v in variables if v.get("formula") and v.get("name") in rec]
    for _ in range(len(formula_vars) + 1):
        changed = False
        for var in formula_vars:
            expr = str(var["formula"])
            deps = _formula_dependencies(expr)
            if deps and not all(rec.get(d) is not None for d in deps):
                continue
            value = _formula(var, rec)
            if value is not None and rec.get(var["name"]) != value:
                rec[var["name"]] = value
                changed = True
        if not changed:
            break
    return rec

def _edge_case_count(total: int, percentage: float) -> int:
    """Return the deterministic number of edge-case units requested.

    Percentage is always applied to the scenario's declared grain: records for
    aggregational data and entities/journeys for transactional data.  We use
    floor so the generator never creates more edge-case units than requested.
    """
    if total <= 0 or percentage <= 0:
        return 0
    return max(0, min(total, math.floor(total * percentage + 1e-12)))


def _edge_case_indices(total: int, edge_count: int) -> set[int]:
    """Choose conceptual indexes evenly across the full dataset.

    Transactional generation intentionally materializes only the latest response
    window for performance.  Edge cases are selected against the *full* conceptual
    entity set, never against that truncated window.  Consequently the visible
    latest-10 window may contain zero, one, or several edge entities; we never
    inflate the requested percentage just to force edge cases into the response.
    """
    if total <= 0 or edge_count <= 0:
        return set()
    if edge_count >= total:
        return set(range(total))
    # Evenly distribute edge units over the conceptual dataset.  The +0.5 center
    # avoids clustering them at the beginning and is deterministic for a given count.
    return {
        min(total - 1, max(0, int(((i + 0.5) * total) / edge_count)))
        for i in range(edge_count)
    }

# ── Generator functions ────────────────────────────────────────────────────────

def _prefixed_int(params: dict, _rec: dict) -> str:
    return f"{params['prefix']}{random.randint(10**( params['digits']-1), 10**params['digits']-1)}"


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


def _constant(params: dict, rec: dict):
    field_name = str(rec.get("__current_field__", "")).lower()
    profile = rec.get("__country_profile__")
    if profile and "currency" in field_name:
        return profile.get("currency", params.get("value"))
    return params["value"]


def _weighted_choice(params: dict, rec: dict):
    choices = list(params.get("choices", []))
    weights = list(params.get("weights", []))
    field_name = str(rec.get("__current_field__", ""))
    profile = rec.get("__country_profile__")
    if profile and ("payment" in field_name.lower() or "pay_method" in field_name.lower()):
        allowed = {m.lower().replace(" ", "_").replace("-", "_") for m in payment_methods_for_profile(profile)}
        filtered = [(c, w) for c, w in zip(choices, weights) if str(c).strip().lower().replace(" ", "_").replace("-", "_") in allowed]
        if filtered:
            choices, weights = zip(*filtered)
            choices, weights = list(choices), list(weights)
        elif allowed:
            choices = payment_methods_for_profile(profile)
            weights = [1.0] * len(choices)
    if not choices:
        return None
    if len(weights) != len(choices) or sum(weights) <= 0:
        weights = [1.0] * len(choices)
    return random.choices(choices, weights=weights, k=1)[0]


def _uniform(params: dict, _rec: dict) -> float:
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    if lo is None:
        lo = 0.0
    if hi is None:
        hi = lo
    return round(random.uniform(lo, max(lo, hi)), 2)


def _lognormal(params: dict, _rec: dict) -> float:
    mu = _to_finite_float(params.get("mu"), 0.0)
    sigma = _to_finite_float(params.get("sigma"), 1.0)
    lo = _to_finite_float(params.get("min", params.get("lo")), 0.0)
    hi = _to_finite_float(params.get("max", params.get("hi")), lo)
    raw = math.exp(random.gauss(mu if mu is not None else 0.0, sigma if sigma is not None else 1.0))
    clipped = max(lo if lo is not None else 0.0, min(hi if hi is not None else raw, raw))
    return round(clipped, 2)


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
    seg = rec.get("subscriber_segment", "Occasional Rechargers")
    rng = params.get(seg, list(params.values())[0])
    return round(random.uniform(rng["min"], rng["max"]), 4)


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


def _uniform_bounded(params: dict, rec: dict) -> float:
    hi = _to_finite_float(rec.get(params.get("hi_field")), None)
    if hi is None:
        hi = _to_finite_float(params.get("hi", params.get("max")), 1.00)
    lo = _to_finite_float(params.get("lo", params.get("min")), 0.00)
    if lo is None:
        lo = 0.00
    return round(random.uniform(lo, max(lo, hi)), 2)


def _recent_datetime(params: dict, _rec: dict) -> str:
    base = datetime.now(timezone.utc) - timedelta(
        days=random.randint(0, params["days_back"]),
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


def _id_mirror(params: dict, rec: dict) -> str:
    """Copy the numeric suffix from source_field and attach a new prefix."""
    source = rec.get(params["source_field"], "")
    number = source.replace(params["source_prefix"], "")
    return f"{params['prefix']}{number}"


def _prefixed_uuid(params: dict, _rec: dict) -> str:
    raw = str(uuid.uuid4())           # 8-4-4-4-12
    suffix = raw[len(params["prefix"]):]
    return params["prefix"] + suffix


def _tx_id(params: dict, rec: dict) -> str:
    ts = rec.get("event_timestamp", datetime.now(timezone.utc).isoformat())
    date_part = ts[:10].replace("-", "")
    rand_part = random.randint(1_000_000, 9_999_999)
    return f"{params['prefix']}{date_part}-{rand_part}"


def _formula(var: dict, rec: dict):
    """Evaluate a simple arithmetic/field-reference formula safely."""
    expr = var["formula"]
    # Build a local namespace from the current record (numeric fields only)
    ns = {k: v for k, v in rec.items() if isinstance(v, (int, float))}
    ns["round"] = round
    try:
        return eval(expr, {"__builtins__": {}, "round": round, "min": min, "max": max}, ns)  # noqa: S307
    except Exception:
        return None


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
    "lognormal":      lambda v, rec: _lognormal(v["params"], rec),
    "lognormal_int":  lambda v, rec: _lognormal_int(v["params"], rec),
    "beta":           lambda v, rec: _beta(v["params"], rec),
    "segment_range":  lambda v, rec: _segment_range(v["params"], rec),
    "uniform_bounded":lambda v, rec: _uniform_bounded(v["params"], rec),
    "recent_datetime":lambda v, rec: _recent_datetime(v["params"], rec),
    "ts_offset":      lambda v, rec: _ts_offset(v["params"], rec),
    "ts_add_field":   lambda v, rec: _ts_add_field(v["params"], rec),
    "date_offset":    lambda v, rec: _date_offset(v["params"], rec),
    "prefixed_uuid":  lambda v, rec: _prefixed_uuid(v["params"], rec),
    "tx_id":          lambda v, rec: _tx_id(v["params"], rec),
    "formula":        lambda v, rec: _formula(v, rec),
}


def get_known_generator_types() -> set[str]:
    """Public accessor for the set of valid 'gen' type strings — used by
    core/csv_scenario.py to validate industry-supplied CSV variable catalogs."""
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


def _apply_generation_constraint(var: dict, value, rec: dict, rules: dict | None):
    """Apply safe machine-readable SchemaAgent constraints to a generated value."""
    constraint = _rule_constraint_for(str(var.get("name", "")), rules)
    if not constraint or value is None:
        return value

    allowed = _coerce_rule_values(constraint.get("preferred_values"))
    if not allowed:
        allowed = _coerce_rule_values(constraint.get("valid_values"))
    if allowed:
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
                and node.id not in {"round", "min", "max", "abs", "sum"}}
    except Exception:
        return set()


def _generate_record(variables: list[dict], profile: dict | None = None, rules: dict | None = None) -> dict:
    """Generate one record by resolving variables in dependency order."""
    rec: dict = {}
    for var in variables:
        gen_type = var["gen"]
        generator = _GENERATORS.get(gen_type)
        effective_var = var
        # Explicit variable formulas are authoritative; rules are only a fallback.
        rule_formula = var.get("formula") or _formula_from_rules(var["name"], rules)
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        generator = _GENERATORS.get(effective_var.get("gen"))
        if generator:
            helper_rec = dict(rec)
            helper_rec["__current_field__"] = var["name"]
            helper_rec["__country_profile__"] = profile
            value = generator(effective_var, helper_rec)
        else:
            value = None
        rec[var["name"]] = _apply_generation_constraint(var, value, rec, rules)
    return rec


def _generate_selected_record(
    variables: list[dict],
    selected_names: set[str],
    base: dict | None = None,
    profile: dict | None = None,
    rules: dict | None = None,
) -> dict:
    """Generate only selected variables (plus their declared dependencies).

    This is the hot-path primitive for transactional generation. Entity fields are
    generated once and reused; event fields are generated only for the event row.
    """
    rec = dict(base or {})
    required = set(selected_names)

    changed = True
    by_name = {v["name"]: v for v in variables}
    while changed:
        changed = False
        for name in tuple(required):
            var = by_name.get(name)
            if not var:
                continue
            for dep in var.get("depends_on", []) or []:
                if dep in by_name and dep not in required:
                    required.add(dep)
                    changed = True

    for var in variables:
        name = var["name"]
        if name not in required or name in rec:
            continue
        effective_var = var
        # Explicit variable formulas are authoritative; rules are only a fallback.
        rule_formula = var.get("formula") or _formula_from_rules(name, rules)
        if rule_formula:
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = str(rule_formula)
        generator = _GENERATORS.get(effective_var.get("gen"))
        if generator:
            helper_rec = dict(rec)
            helper_rec["__current_field__"] = name
            helper_rec["__country_profile__"] = profile
            value = generator(effective_var, helper_rec)
        else:
            value = None
        rec[name] = _apply_generation_constraint(var, value, rec, rules)
    return rec


# ── Transactional generation helpers ─────────────────────────────────────────

_TRANSACTIONAL_COMMON_FIELDS = {
    "journey_id", "transaction_id", "subscriber_id", "customer_id",
    "subscriber_msisdn", "phone_number", "account_id", "event_type",
    "event_sequence", "event_timestamp", "event_occurrence",
}

# Publicly documented response behavior: at most 10 entities and 10 records/event/entity.
MAX_RESPONSE_ENTITIES = 10
MAX_EVENT_RECORDS = 10


def _journey_id() -> str:
    return f"JRN-{uuid.uuid4().hex[:12].upper()}"


def _condition_refs(expression: str) -> set[str]:
    """Return field names referenced by an edge-case expression."""
    try:
        tree = ast.parse(str(expression or ""), mode="eval")
        return {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
    except Exception:
        return set()


def _condition_branches(expression: str, variables: list[dict] | None = None) -> list[dict[str, object]]:
    """Compile supported edge conditions into deterministic assignments.

    This is deliberately schema-driven rather than industry/field-name driven.
    Each returned branch is sufficient to make the corresponding boolean branch
    true.  Equality/range/membership and boolean combinations are supported.
    """
    try:
        tree = ast.parse(str(expression or ""), mode="eval")
    except Exception:
        return []

    def _literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            vals=[]
            for x in node.elts:
                if not isinstance(x, ast.Constant):
                    return None
                vals.append(x.value)
            return vals
        return None

    def compare(node: ast.Compare) -> list[dict[str, object]]:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return []
        op, left, right = node.ops[0], node.left, node.comparators[0]
        # Build a candidate assignment for both field-to-literal and
        # field-to-field comparisons. Field-to-field comparisons are common in
        # real edge definitions (e.g. balance_before < recharge_amount) and
        # cannot be solved by the old literal-only branch builder.
        if isinstance(left, ast.Name):
            name = left.id
            lit = _literal(right)
            if isinstance(right, ast.Name):
                other = right.id
                if isinstance(op, (ast.Eq, ast.Is)):
                    return [{name: {"__copy_from__": other}}]
                if isinstance(op, (ast.NotEq, ast.IsNot)):
                    return [{"__field_not_eq__": (name, other)}]
                if isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                    return [{"__field_compare__": (name, op.__class__.__name__, other)}]
                return []
            if isinstance(op, (ast.Eq, ast.Is)) and not isinstance(right, (ast.List, ast.Tuple)):
                return [{name: lit}]
            if isinstance(op, ast.Gt) and isinstance(lit,(int,float)) and not isinstance(lit,bool):
                return [{name: lit + (1 if isinstance(lit,int) else 0.01)}]
            if isinstance(op, ast.GtE) and isinstance(lit,(int,float)) and not isinstance(lit,bool):
                return [{name: lit}]
            if isinstance(op, ast.Lt) and isinstance(lit,(int,float)) and not isinstance(lit,bool):
                return [{name: lit - (1 if isinstance(lit,int) else 0.01)}]
            if isinstance(op, ast.LtE) and isinstance(lit,(int,float)) and not isinstance(lit,bool):
                return [{name: lit}]
            if isinstance(op, ast.In) and isinstance(lit,list):
                return [{name: x} for x in lit]
            if isinstance(op, ast.NotIn) and isinstance(lit,list):
                return [{"__not_in__": (name, tuple(lit))}]
            if isinstance(op, (ast.NotEq, ast.IsNot)):
                return [{"__not_eq__": (name, lit)}]
        if isinstance(right, ast.Name) and isinstance(left, ast.Constant):
            if isinstance(op, (ast.Eq, ast.Is)):
                return [{right.id: left.value}]
            if isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                # Reverse constant <op> field into field <reverse-op> constant.
                reverse = {ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Lt: ast.Gt, ast.LtE: ast.GtE}.get(type(op))
                if reverse:
                    fake = ast.Compare(left=ast.Name(id=right.id), ops=[reverse()], comparators=[left])
                    return compare(fake)
        return []

    def walk(node: ast.AST) -> list[dict[str, object]]:
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Compare): return compare(node)
        if isinstance(node, ast.Name): return [{node.id: True}]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Name):
            return [{node.operand.id: False}]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            branches=[{}]
            for child in node.values:
                cb=walk(child)
                if not cb: return []
                merged=[]
                for a in branches:
                    for b in cb:
                        special = {k:v for k,v in b.items() if k.startswith("__") }
                        normal = {k:v for k,v in b.items() if not k.startswith("__") }
                        if any(k in a and a[k] != v for k,v in normal.items()): continue
                        merged.append({**a, **normal, **special})
                branches=merged
            return branches
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            out=[]
            for child in node.values: out.extend(walk(child))
            return out
        return []
    return walk(tree)

def _apply_condition_assignments(
    candidate: dict,
    expression: str,
    variables: list[dict],
    assignments: dict[str, object],
) -> dict:
    """Apply deterministic condition constraints to an edge-case candidate.

    Missing fields are allowed here because an edge case may intentionally introduce
    a field that is sparse in the normal event.  Formula outputs are never assigned
    directly; their inputs are assigned and formulas are recalculated afterwards.
    """
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    formula_names = {str(v.get("name")) for v in variables if v.get("formula")}
    for key, raw_value in assignments.items():
        if key.startswith("__"):
            if key == "__not_eq__":
                name, forbidden = raw_value
                var = by_name.get(name, {})
                if name in formula_names: continue
                params = var.get("params") if isinstance(var.get("params"), dict) else {}
                choices = params.get("choices") if isinstance(params.get("choices"), list) else []
                alternatives = [c for c in choices if str(c).strip().lower() != str(forbidden).strip().lower()]
                if alternatives:
                    candidate[name] = alternatives[0]
                elif isinstance(candidate.get(name), (int,float)) and not isinstance(candidate.get(name), bool):
                    candidate[name] = float(forbidden) + (1 if isinstance(forbidden, int) else 0.01)
                continue
            if key == "__not_in__":
                name, forbidden = raw_value
                var = by_name.get(name, {})
                if name in formula_names: continue
                params = var.get("params") if isinstance(var.get("params"), dict) else {}
                choices = params.get("choices") if isinstance(params.get("choices"), list) else []
                alternatives = [c for c in choices if c not in forbidden]
                if alternatives:
                    candidate[name] = alternatives[0]
                continue
            if key == "__copy_from__":
                name, source = raw_value if isinstance(raw_value, tuple) else (None, None)
                if name and source in candidate and name not in formula_names:
                    candidate[name] = candidate[source]
                continue
            if key == "__field_not_eq__":
                name, other = raw_value
                if name in formula_names: continue
                if other in candidate and str(candidate.get(name)).strip().lower() == str(candidate.get(other)).strip().lower():
                    var = by_name.get(name, {})
                    params = var.get("params") if isinstance(var.get("params"), dict) else {}
                    choices = params.get("choices") if isinstance(params.get("choices"), list) else []
                    alt = next((c for c in choices if str(c).strip().lower() != str(candidate.get(other)).strip().lower()), None)
                    if alt is not None: candidate[name] = alt
                continue
            if key == "__field_compare__":
                name, op_name, other = raw_value
                if name in formula_names or other in formula_names:
                    continue
                def num(x):
                    try:
                        if isinstance(x, bool): return None
                        x = float(x)
                        return x if math.isfinite(x) else None
                    except (TypeError, ValueError):
                        return None
                def bounds(field):
                    var = by_name.get(field, {})
                    params = var.get("params") if isinstance(var.get("params"), dict) else {}
                    choices = params.get("choices") if isinstance(params.get("choices"), list) else []
                    nums = [num(x) for x in choices]
                    nums = [x for x in nums if x is not None]
                    lo = num(params.get("min")); hi = num(params.get("max"))
                    if nums:
                        lo = min(nums) if lo is None else lo
                        hi = max(nums) if hi is None else hi
                    cur = num(candidate.get(field))
                    if lo is None: lo = cur
                    if hi is None: hi = cur
                    return lo, hi, var
                llo, lhi, lvar = bounds(name)
                rlo, rhi, rvar = bounds(other)
                step_l = 1.0 if str(lvar.get("dtype", "")).lower() in {"int","integer"} else 0.01
                step_r = 1.0 if str(rvar.get("dtype", "")).lower() in {"int","integer"} else 0.01
                # Prefer values guaranteed to satisfy the relation using the
                # declared domains. If a normal domain is absent, synthesize a
                # minimal numeric pair. This is a generic constraint solver, not
                # a field-specific workaround.
                left = num(candidate.get(name)); right = num(candidate.get(other))
                if op_name in {"Lt", "LtE"}:
                    if llo is not None and rhi is not None and llo <= rhi - (0 if op_name == "LtE" else step_l):
                        left, right = llo, rhi
                    elif left is None or right is None or not (left < right if op_name == "Lt" else left <= right):
                        base = rhi if rhi is not None else (right if right is not None else 1.0)
                        right = base
                        left = base - (0 if op_name == "LtE" else max(step_l, step_r))
                        if llo is not None: left = max(llo, left)
                        if lhi is not None: left = min(lhi, left)
                else:
                    if lhi is not None and rlo is not None and lhi >= rlo + (0 if op_name == "GtE" else step_l):
                        left, right = lhi, rlo
                    elif left is None or right is None or not (left > right if op_name == "Gt" else left >= right):
                        base = rlo if rlo is not None else (right if right is not None else 0.0)
                        left = base + (0 if op_name == "GtE" else max(step_l, step_r))
                        right = base
                        if llo is not None: left = max(llo, left)
                        if lhi is not None: left = min(lhi, left)
                def cast(value, var):
                    dtype = str(var.get("dtype", "")).lower()
                    return int(round(value)) if dtype in {"int","integer"} else float(value)
                if left is not None: candidate[name] = cast(left, lvar)
                if right is not None: candidate[other] = cast(right, rvar)
                continue
            continue
        name, value = key, raw_value
        if name in formula_names: continue
        var = by_name.get(name, {})
        dtype = str(var.get("dtype", "string"))
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        choices = params.get("choices") if isinstance(params.get("choices"), list) else []
        if choices:
            # The condition is an explicit edge constraint. Prefer an existing
            # schema representation when one matches; otherwise keep the literal
            # condition value instead of silently dropping it. This is generic and
            # allows an edge state to sit outside the normal categorical distribution.
            match = next((c for c in choices if str(c).strip().lower() == str(value).strip().lower()), None)
            if match is not None:
                value = match
            elif isinstance(value, bool):
                # Preserve a schema's common YES/NO representation when possible.
                desired = value
                match = next((c for c in choices if _boolean_semantic(c) is desired), None)
                value = match if match is not None else value
        if dtype == "int" and isinstance(value, (int,float)) and not isinstance(value,bool):
            value = int(round(value))
        elif dtype == "float" and isinstance(value, (int,float)) and not isinstance(value,bool):
            value = float(value)
        elif dtype in {"boolean","bool"} and isinstance(value, bool):
            pass
        candidate[name] = value
    return candidate

def _recalculate_formulas(candidate: dict, variables: list[dict]) -> dict:
    formula_vars = [v for v in variables if v.get("formula") and v.get("name") in candidate]
    for _ in range(len(formula_vars) + 1):
        changed = False
        for var in formula_vars:
            deps = _formula_dependencies(str(var.get("formula", "")))
            if deps and not all(candidate.get(d) is not None for d in deps):
                continue
            value = _formula(var, candidate)
            if value is not None and candidate.get(var["name"]) != value:
                candidate[var["name"]] = value
                changed = True
        if not changed:
            break
    return candidate


def _condition_compatible_with_schema(
    expression: str,
    variables: list[dict],
    edge_override_names: set[str] | None = None,
) -> bool:
    """Validate an edge condition structurally and generically.

    This function answers only: "Can this condition be represented by the
    effective scenario schema?"  It deliberately does *not* try to prove the
    condition from normal-generation distributions.  Normal min/max values are
    distributions, not hard limits for an explicitly requested edge case.

    ``edge_override_names`` identifies fields for which the edge-case definition
    supplies an explicit value/domain.  Such overrides are authoritative over
    normal categorical choices.  This prevents /scenario/confirm from rejecting
    valid edge cases simply because an ordinary generator definition cannot
    produce the exceptional value.
    """
    by_name = {
        str(v.get("name")): v for v in variables
        if isinstance(v, dict) and v.get("name")
    }
    override_names = {str(x) for x in (edge_override_names or set())}

    try:
        tree = ast.parse(str(expression or ""), mode="eval")
    except Exception:
        return False

    allowed = (
        ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.Compare,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Name, ast.Constant,
        ast.List, ast.Tuple, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load,
    )
    if any(not isinstance(n, allowed) for n in ast.walk(tree)):
        return False

    refs = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not refs.issubset(by_name):
        return False

    def literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            values = []
            for child in node.elts:
                if not isinstance(child, ast.Constant):
                    return None
                values.append(child.value)
            return values
        return None

    def numeric(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        if isinstance(v, str):
            try:
                x = float(v.strip())
                return x if math.isfinite(x) else None
            except (TypeError, ValueError):
                return None
        return None

    def semantic_kind(var: dict):
        dtype = str(var.get("dtype", "string")).strip().lower()
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        choices = params.get("choices") if isinstance(params.get("choices"), list) else []
        if dtype in {"boolean", "bool"}:
            return "boolean"
        if dtype in {"int", "integer"}:
            return "number"
        if dtype in {"float", "number", "double", "decimal"}:
            return "number"
        boolean_values = [_boolean_semantic(x) for x in choices] if choices else []
        if choices and all(x is not None for x in boolean_values) and len(set(boolean_values)) <= 2:
            return "boolean"
        if choices and all(numeric(x) is not None for x in choices):
            return "number"
        if params.get("min") is not None and params.get("max") is not None:
            if numeric(params.get("min")) is not None and numeric(params.get("max")) is not None:
                return "number"
        if dtype in {"datetime", "date", "timestamp"}:
            return "datetime"
        return "string"

    def compatible_scalar(name: str, op, value) -> bool:
        var = by_name[name]
        kind = semantic_kind(var)
        if kind == "number":
            if numeric(value) is None:
                return False
        elif kind == "boolean":
            # Boolean-semantic categorical schemas commonly use YES/NO, Y/N,
            # enabled/disabled, or 1/0. Conditions may use either representation.
            if _boolean_semantic(value) is None:
                return False
        elif kind == "datetime":
            if not isinstance(value, str):
                return False
        else:
            # String/categorical variables accept string literals.  Do not reject
            # a value merely because its normal choices omit it when an explicit
            # edge override exists; the override is precisely how an exceptional
            # categorical state is declared.
            if not isinstance(value, str):
                return False

        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        choices = params.get("choices") if isinstance(params.get("choices"), list) else []
        # For an explicit edge condition, the condition literal itself is an edge
        # constraint. Normal categorical choices describe ordinary generation and must
        # not make confirmation fail. If an explicit edge override exists it is even
        # more authoritative; generation will materialize the condition value.
        return True

    def validate(node) -> bool:
        if isinstance(node, ast.Expression):
            return validate(node.body)
        if isinstance(node, ast.BoolOp):
            return all(validate(x) for x in node.values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return validate(node.operand)
        if isinstance(node, ast.Compare):
            if len(node.ops) != len(node.comparators):
                return False
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                if isinstance(left, ast.Name):
                    value = literal(right)
                    if isinstance(right, ast.Name):
                        # Field-to-field comparisons are supported only when both
                        # fields are declared; actual satisfiability is handled at
                        # generation time from the generated record.
                        if right.id not in by_name:
                            return False
                    elif value is None and not (isinstance(right, ast.Constant) and right.value is None):
                        return False
                    elif isinstance(value, list) and isinstance(op, (ast.In, ast.NotIn)):
                        if not all(compatible_scalar(left.id, op, x) for x in value):
                            return False
                    elif not isinstance(right, ast.Name) and not compatible_scalar(left.id, op, value):
                        return False
                elif isinstance(right, ast.Name) and isinstance(left, ast.Constant):
                    if right.id not in by_name:
                        return False
                    if not compatible_scalar(right.id, op, literal(left)):
                        return False
                else:
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            return node.id in by_name
        return False

    return validate(tree)

def _condition_literals(expression: str) -> dict[str, object]:
    """Backward-compatible first branch of the generic condition constraint solver."""
    branches = _condition_branches(expression)
    return branches[0] if branches else {}


def _apply_edge_generation_value(var: dict, value, rec: dict, rules: dict | None):
    """Apply hard categorical/rule domains to an edge value without clamping normal numeric ranges."""
    if value is None:
        return value
    params = var.get("params") if isinstance(var.get("params"), dict) else {}
    choices = params.get("choices") if isinstance(params.get("choices"), list) else []
    if choices:
        match = next((c for c in choices if str(c).strip().lower() == str(value).strip().lower()), None)
        if match is not None:
            return match
        # A configured edge condition may use a semantically equivalent numeric choice.
        try:
            nv = float(value)
            for c in choices:
                if not isinstance(c, bool) and math.isclose(float(c), nv, abs_tol=1e-12):
                    return c
        except (TypeError, ValueError):
            pass
        return value
    # RuleAgent preferred/valid values are hard domains; numeric min/max rules are not.
    constraint = _rule_constraint_for(str(var.get("name", "")), rules)
    if constraint:
        allowed = _coerce_rule_values(constraint.get("preferred_values")) or _coerce_rule_values(constraint.get("valid_values"))
        if allowed:
            match = next((x for x in allowed if str(x).strip().lower().replace("-", "_").replace(" ", "_") == str(value).strip().lower().replace("-", "_").replace(" ", "_")), None)
            return match if match is not None else value
    return value


def _solve_edge_condition(candidate: dict, expression: str, variables: list[dict]) -> dict:
    """Generic deterministic constraint repair for an edge condition.

    Repeatedly applies atomic constraints, recalculates formulas, and re-evaluates the
    complete expression. It never knows industry or field names. Formula fields are
    treated as derived; when a condition targets a derived field, its dependencies are
    adjusted by the existing formula model where possible.
    """
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    formula_names = {str(v.get("name")) for v in variables if v.get("formula")}

    def numeric(x):
        if isinstance(x, bool): return None
        try:
            y = float(x)
            return y if math.isfinite(y) else None
        except (TypeError, ValueError): return None

    def set_value(target, name, value):
        if name in formula_names:
            return
        var = by_name.get(name, {})
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        choices = params.get("choices") if isinstance(params.get("choices"), list) else []
        if choices:
            # Prefer an exact/semantic schema representation; if none exists, preserve
            # the explicit edge literal because normal categorical choices are not hard
            # limits for an explicitly requested edge state.
            match = next((c for c in choices if str(c).strip().lower() == str(value).strip().lower()), None)
            if match is None and isinstance(value, bool):
                match = next((c for c in choices if _boolean_semantic(c) is value), None)
            if match is not None: value = match
        dtype = str(var.get("dtype", "")).lower()
        if dtype in {"int", "integer"} and numeric(value) is not None:
            value = int(round(float(value)))
        elif dtype in {"float", "number", "double", "decimal"} and numeric(value) is not None:
            value = float(value)
        target[name] = value

    def atomic_nodes(node):
        if isinstance(node, ast.Expression): return atomic_nodes(node.body)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            out=[]
            for x in node.values: out.extend(atomic_nodes(x))
            return out
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            # Try each branch independently; caller will stop once expression is true.
            out=[]
            for x in node.values: out.extend(atomic_nodes(x))
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Compare):
            cmp_node = node.operand
            if len(cmp_node.ops) == 1:
                inverse = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}.get(type(cmp_node.ops[0]))
                if inverse:
                    return [ast.Compare(left=cmp_node.left, ops=[inverse()], comparators=cmp_node.comparators)]
        return [node] if isinstance(node, (ast.Compare, ast.Name, ast.UnaryOp)) else []

    def repair_compare(target, node):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            return
        op=node.ops[0]; right=node.comparators[0]; left=node.left
        if isinstance(left, ast.Name):
            name=left.id
            if name not in by_name or name in formula_names: return
            if isinstance(right, ast.Name):
                other=right.id
                if other not in by_name: return
                a=numeric(target.get(name)); b=numeric(target.get(other))
                step_l = 1.0 if str(by_name[name].get("dtype", "")).lower() in {"int", "integer"} else 0.01
                step_r = 1.0 if str(by_name[other].get("dtype", "")).lower() in {"int", "integer"} else 0.01
                if isinstance(op, ast.Eq):
                    if b is None and a is None: a = b = 0.0
                    elif b is None: b = a
                    else: a = b
                    set_value(target, name, a); set_value(target, other, b)
                elif isinstance(op, ast.NotEq):
                    if a is None and b is None: a, b = 0.0, 1.0
                    elif a is None: a = b + step_l
                    elif b is None: b = a + step_r
                    elif a == b: a = b + step_l
                    set_value(target, name, a); set_value(target, other, b)
                elif isinstance(op, ast.Lt):
                    if a is None and b is None: a, b = 0.0, 1.0
                    elif a is None: a = b - step_l
                    elif b is None: b = a + step_r
                    if not a < b: b = a + max(step_l, step_r)
                    set_value(target, name, a); set_value(target, other, b)
                elif isinstance(op, ast.LtE):
                    if a is None and b is None: a = b = 0.0
                    elif a is None: a = b
                    elif b is None: b = a
                    if a > b: b = a
                    set_value(target, name, a); set_value(target, other, b)
                elif isinstance(op, ast.Gt):
                    if a is None and b is None: a, b = 1.0, 0.0
                    elif a is None: a = b + step_l
                    elif b is None: b = a - step_r
                    if not a > b: a = b + max(step_l, step_r)
                    set_value(target, name, a); set_value(target, other, b)
                elif isinstance(op, ast.GtE):
                    if a is None and b is None: a = b = 0.0
                    elif a is None: a = b
                    elif b is None: b = a
                    if a < b: a = b
                    set_value(target, name, a); set_value(target, other, b)
                return
            if isinstance(right, ast.Constant):
                value=right.value
                if isinstance(op, (ast.Eq, ast.Is)): set_value(target, name, value)
                elif isinstance(op, ast.NotEq) or isinstance(op, ast.IsNot):
                    params=by_name[name].get("params") if isinstance(by_name[name].get("params"),dict) else {}
                    choices=params.get("choices") if isinstance(params.get("choices"),list) else []
                    alt=next((c for c in choices if str(c).strip().lower()!=str(value).strip().lower()), None)
                    if alt is not None: set_value(target, name,alt)
                elif isinstance(value,(int,float)) and not isinstance(value,bool):
                    n=float(value); cur=numeric(candidate.get(name)); step=1.0 if str(by_name[name].get("dtype","")).lower() in {"int","integer"} else 0.01
                    if isinstance(op,ast.Gt): set_value(target, name,n+step)
                    elif isinstance(op,ast.GtE): set_value(target, name,n)
                    elif isinstance(op,ast.Lt): set_value(target, name,n-step)
                    elif isinstance(op,ast.LtE): set_value(target, name,n)
                return
            if isinstance(right,(ast.List,ast.Tuple)):
                vals=[x.value for x in right.elts if isinstance(x,ast.Constant)]
                if isinstance(op,ast.In) and vals: set_value(target, name,vals[0])
                elif isinstance(op,ast.NotIn):
                    params=by_name[name].get("params") if isinstance(by_name[name].get("params"),dict) else {}
                    choices=params.get("choices") if isinstance(params.get("choices"),list) else []
                    alt=next((c for c in choices if c not in vals), None)
                    if alt is not None: set_value(target, name,alt)

    try:
        tree=ast.parse(str(expression or ""), mode="eval")
    except Exception:
        return candidate
    # Multiple passes matter when a formula depends on a constrained field, or when
    # one atomic constraint changes the value needed by another.
    for _ in range(max(4, len(by_name)*2)):
        if _safe_edge_condition(expression, candidate):
            return candidate
        # For OR, try each branch by building candidates and keep the first successful one.
        body=tree.body if isinstance(tree,ast.Expression) else tree
        branches=[body]
        if isinstance(body,ast.BoolOp) and isinstance(body.op,ast.Or): branches=list(body.values)
        best=None
        for branch in branches:
            trial=dict(candidate)
            for node in atomic_nodes(branch): repair_compare(trial, node)
            trial=_recalculate_formulas(trial,variables)
            if _safe_edge_condition(expression,trial): return trial
            best=trial
        if best is not None: candidate=best
    return candidate

def _edge_candidate(
    row: dict,
    edge_group: dict,
    variables: list[dict],
    profile: dict | None,
    rules: dict | None,
    event_fields: set[str],
    assignments: dict[str, object] | None = None,
) -> dict:
    """Build an edge-case candidate deterministically from its condition.

    The condition is the source of truth. Explicit edge overrides are applied first,
    then deterministic constraints from the condition are applied to non-formula
    fields, formulas are recalculated, and the final condition is checked by the
    caller. Edge-only fields are materialized only when the edge definition itself
    declares them; normal fields from another sparse event are never invented.
    """
    candidate = dict(row)
    variable_by_name = {str(v.get("name")): v for v in variables if isinstance(v, dict) and v.get("name")}
    edge_by_name = {str(v.get("name")): v for v in edge_group.get("variables", []) if v.get("name")}
    refs = _condition_refs(edge_group.get("condition", ""))
    formula_names = {str(v.get("name")) for v in variables if v.get("formula")}

    # A condition can only be represented by this record if every referenced field is
    # already present or explicitly declared by the edge definition. This prevents a
    # sparse event from accidentally borrowing fields from an unrelated event.
    for name in refs:
        if name in candidate:
            continue
        # A condition is an explicit request that this field participate in the
        # edge-case record. If the sparse event did not normally contain it,
        # materialize it from the edge definition when present, otherwise from the
        # canonical scenario variable. This is generic and avoids variable-name
        # specific bypasses. The field is only introduced on records selected as
        # edge cases; normal sparse records are unchanged.
        definition = edge_by_name.get(name) or variable_by_name.get(name)
        if definition is None:
            continue
        effective = dict(variable_by_name.get(name, definition))
        for key in ("gen", "params", "formula", "dtype", "nullable"):
            if key in definition and definition[key] not in (None, ""):
                effective[key] = definition[key]
        generator = _GENERATORS.get(effective.get("gen"))
        if generator and (effective.get("gen") != "formula"):
            helper = dict(candidate)
            helper["__current_field__"] = name
            helper["__country_profile__"] = profile
            value = generator(effective, helper)
            candidate[name] = _apply_edge_generation_value(
                variable_by_name.get(name, effective), value, candidate, rules
            )

    # Apply explicit edge overrides. They are allowed to introduce an edge-only field,
    # but only for this record when the field is part of the event or condition.
    for edge_var in edge_group.get("variables", []):
        name = str(edge_var.get("name", ""))
        if not name:
            continue
        base_var = variable_by_name.get(name, edge_var)
        effective = dict(base_var)
        for key in ("gen", "params", "formula", "dtype", "nullable"):
            if key in edge_var and edge_var[key] not in (None, ""):
                effective[key] = edge_var[key]
        generator = _GENERATORS.get(effective.get("gen"))
        if generator and (effective.get("gen") != "formula"):
            helper = dict(candidate)
            helper["__current_field__"] = name
            helper["__country_profile__"] = profile
            value = generator(effective, helper)
            candidate[name] = _apply_edge_generation_value(base_var, value, candidate, rules)

    # Apply condition constraints after generation. For OR conditions, each branch is
    # tried by the caller; here the first branch is sufficient for candidate creation.
    assignments = assignments if assignments is not None else _condition_literals(edge_group.get("condition", ""))
    candidate = _apply_condition_assignments(candidate, edge_group.get("condition", ""), variables, assignments)

    # Recalculate formulas in declaration/dependency order. This is intentionally done
    # after edge overrides so derived values stay mathematically consistent.
    candidate = _recalculate_formulas(candidate, variables)
    candidate = _solve_edge_condition(candidate, edge_group.get("condition", ""), variables)
    candidate = _recalculate_formulas(candidate, variables)
    return candidate


def _edge_case_candidate_for_aggregation(
    row: dict,
    edge_group: dict,
    variables: list[dict],
    profile: dict | None,
    rules: dict | None,
    assignments: dict[str, object] | None = None,
) -> dict:
    """Apply the same generic condition solver used by transactional records."""
    return _edge_candidate(
        row,
        edge_group,
        variables,
        profile,
        rules,
        event_fields=set(v.get("name") for v in variables if v.get("name")),
        assignments=assignments,
    )


def _matches_any_edge_condition(rec: dict, edge_groups: dict[str, dict]) -> bool:
    """Return whether the actual record is already an edge case by definition."""
    for group in edge_groups.values():
        condition = str(group.get("condition") or "").strip()
        if condition and _safe_edge_condition(condition, rec):
            return True
    return False


def _transactional_records(
    compiled,
    journey_count: int,
    event_counts_out: dict[str, dict[str, int]] | None = None,
    profile: dict | None = None,
    rules: dict | None = None,
    edge_case_variables: list[dict] | None = None,
    edge_case_percentage: float = 0.0,
    record_errors_out: list[dict] | None = None,
) -> list[dict]:
    """Generate transactional records with record-level edge-case semantics.

    ``journey_count`` remains the requested number of conceptual entities, while
    the API materializes only the latest response window for performance.  The
    edge-case percentage is applied to the requested transactional dataset count
    (the user's ``count``), while the resulting flag is placed on actual event
    records inside ``events[].records[]``. Every true flag is
    independently proven against that actual record; normal records are never
    promoted to edge cases merely because they happen to match a condition.
    """
    events = compiled.events
    variables = list(compiled.variables)
    entity_key = compiled.entity_key
    if not events:
        from types import SimpleNamespace
        events = (SimpleNamespace(event_type="BUSINESS_EVENT", sequence=1, fields=(), min_occurrences=1, max_occurrences=10),)

    edge_groups = _edge_case_groups(edge_case_variables)
    edge_enabled = bool(edge_groups) and edge_case_percentage > 0
    response_entity_count = min(journey_count, MAX_RESPONSE_ENTITIES)
    response_start_index = max(0, journey_count - response_entity_count)
    generated: list[dict] = []
    used_entity_keys: set[str] = set()

    for entity_index in range(response_entity_count):
        try:
            entity_context = _generate_selected_record(
                variables, set(compiled.entity_fields), profile=profile, rules=rules
            )
        except Exception as exc:
            error = {
                "record_index": len(generated),
                "error": str(exc),
                "record": {"entity_index": entity_index},
            }
            if record_errors_out is not None:
                record_errors_out.append(error)
            logger.warning("[DataGeneration] Skipping transactional entity %d: %s", entity_index, exc)
            continue
        if entity_key and entity_key in entity_context:
            attempts = 0
            while str(entity_context[entity_key]) in used_entity_keys and attempts < 10:
                key_var = compiled.variable_by_name.get(entity_key)
                if key_var:
                    generator = _GENERATORS.get(key_var.get("gen"))
                    if generator:
                        entity_context[entity_key] = generator(key_var, entity_context)
                attempts += 1
            used_entity_keys.add(str(entity_context[entity_key]))

        journey_id = _journey_id()
        entity_value = str(entity_context.get(entity_key, "")) if entity_key else journey_id
        if event_counts_out is not None:
            event_counts_out.setdefault(entity_value, {})
        base_ts = _parse_dt(entity_context.get("event_timestamp", datetime.now(timezone.utc).isoformat()))
        elapsed_seconds = 0

        for event in events:
            occurrence_count = random.randint(event.min_occurrences, event.max_occurrences)
            if event_counts_out is not None:
                event_counts_out[entity_value][event.event_type] = occurrence_count
            start_occurrence = max(0, occurrence_count - MAX_EVENT_RECORDS)
            for occurrence in range(start_occurrence, occurrence_count):
                elapsed_seconds += random.randint(5, 300)
                event_ts = base_ts + timedelta(seconds=elapsed_seconds)
                row = None
                generation_exception = None
                for _attempt in range(_MAX_EDGE_CASE_ATTEMPTS if edge_enabled else 1):
                    try:
                        candidate_row = _generate_selected_record(
                            variables,
                            set(event.fields),
                            base=entity_context,
                            profile=profile,
                            rules=rules,
                        )
                        candidate_row.update({
                            "journey_id": journey_id,
                            "transaction_id": f"TXN-{event_ts.strftime('%Y%m%d')}-{uuid.uuid4().hex[:10].upper()}",
                            "event_type": event.event_type,
                            "event_sequence": event.sequence,
                            "event_occurrence": occurrence + 1,
                            "event_timestamp": event_ts.isoformat(),
                            "isEdgeCaseData": False,
                        })
                    except Exception as exc:
                        generation_exception = exc
                        row = None
                        break
                    # Edge classification is deterministic: if a normal generated
                    # record already satisfies an edge condition, it cannot be labelled
                    # false. Regenerate it instead. This prevents two identical records
                    # from receiving contradictory edge labels.
                    if edge_enabled and _matches_any_edge_condition(candidate_row, edge_groups):
                        row = None
                        continue
                    row = candidate_row
                    break
                if row is None:
                    error = {
                        "record_index": len(generated),
                        "error": (
                            str(generation_exception) if generation_exception is not None else
                            "Unable to generate a normal transactional record that does not "
                            "satisfy any configured edge-case condition. The edge-case definitions "
                            "leave no valid normal-domain value for this event."
                        ),
                        "record": {
                            "journey_id": journey_id,
                            "event_type": event.event_type,
                            "event_sequence": event.sequence,
                            "event_occurrence": occurrence + 1,
                        },
                    }
                    if record_errors_out is not None:
                        record_errors_out.append(error)
                    logger.warning("[DataGeneration] Skipping transactional record: %s", error["error"])
                    continue
                generated.append(row)

    if not edge_groups or edge_case_percentage <= 0 or not generated:
        return generated

    # The API's ``count`` is the user's requested dataset count. For transactional
    # scenarios we keep the existing response optimization (latest 10 entities /
    # latest 10 records per event), but edgeCasePercentage is still interpreted
    # against the requested count so 100 × 0.02 always means 2 edge-case units.
    target = min(_edge_case_count(journey_count, edge_case_percentage), len(generated))
    if target <= 0:
        return generated

    # Find real candidates. A candidate is eligible only when the condition actually
    # evaluates to true after deterministic condition constraints and formula repair.
    # We never mark a record true merely because it was selected by percentage.
    group_items = list(edge_groups.items())
    eligible: list[tuple[int, str, dict]] = []
    for idx, row in enumerate(generated):
        event_def = compiled.event_by_type.get(str(row.get("event_type")))
        event_fields = set(event_def.fields) if event_def else set()
        for group_name, group in group_items:
            configured_event = str(group.get("event_type") or "").strip().upper().replace(" ", "_")
            if configured_event and configured_event != str(row.get("event_type") or "").strip().upper().replace(" ", "_"):
                continue
            refs = _condition_refs(group.get("condition", ""))
            if not _condition_compatible_with_schema(
                group.get("condition", ""), variables,
                edge_override_names={str(v.get("name")) for v in group.get("variables", []) if v.get("name")},
            ):
                continue
            # If a condition references fields that are neither present in the row nor
            # explicitly supplied by this edge definition, this event cannot represent it.
            edge_names = {str(v.get("name")) for v in group.get("variables", []) if v.get("name")}
            if any(name not in row and name not in edge_names for name in refs):
                continue
            branches = _condition_branches(group.get("condition", ""), variables) or [{}]
            for assignments in branches:
                candidate = _edge_candidate(
                    row, group, variables, profile, rules, event_fields, assignments=assignments
                )
                if _safe_edge_condition(group.get("condition", ""), candidate):
                    eligible.append((idx, group_name, candidate))
                    break
            if eligible and eligible[-1][0] == idx:
                break

    if len(eligible) < target:
        # Do not fail merely because the randomly materialized normal window contains
        # no natural matches. Build additional candidates from fresh valid base rows
        # and solve the configured condition deterministically. This is the generic
        # path for rare edge states; no variable or industry is special-cased.
        needed = target - len(eligible)
        for group_name, group in group_items:
            if needed <= 0:
                break
            configured_event = str(group.get("event_type") or "").strip().upper().replace(" ", "_")
            for _attempt in range(max(_MAX_EDGE_CASE_ATTEMPTS, needed * 2)):
                if needed <= 0:
                    break
                event_candidates = [e for e in events if not configured_event or str(e.event_type).strip().upper().replace(" ", "_") == configured_event]
                if not event_candidates:
                    continue
                event_def = random.choice(event_candidates)
                try:
                    base = _generate_selected_record(
                        variables, set(event_def.fields), base=entity_context,
                        profile=profile, rules=rules
                    )
                except Exception as exc:
                    if record_errors_out is not None:
                        record_errors_out.append({
                            "record_index": len(generated),
                            "error": str(exc),
                            "record": {"journey_id": journey_id, "event_type": event_def.event_type},
                        })
                    continue
                base.update({
                    "journey_id": journey_id,
                    "transaction_id": f"TXN-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:10].upper()}",
                    "event_type": event_def.event_type,
                    "event_sequence": event_def.sequence,
                    "event_occurrence": 1,
                    "event_timestamp": datetime.now(timezone.utc).isoformat(),
                    "isEdgeCaseData": False,
                })
                branches = _condition_branches(group.get("condition", ""), variables) or [{}]
                for assignments in branches:
                    candidate = _edge_candidate(base, group, variables, profile, rules, set(event_def.fields), assignments=assignments)
                    if _safe_edge_condition(group.get("condition", ""), candidate):
                        # Replace an existing non-edge response row so the requested
                        # dataset cardinality never changes merely because an edge
                        # condition is rare.
                        available_indices = [i for i, r in enumerate(generated) if r.get("isEdgeCaseData") is not True and i not in {x[0] for x in eligible}]
                        if not available_indices:
                            break
                        replacement_idx = available_indices[0]
                        candidate["isEdgeCaseData"] = True
                        generated[replacement_idx] = candidate
                        eligible.append((replacement_idx, group_name, candidate))
                        needed -= 1
                        break
        if len(eligible) < target:
            missing = target - len(eligible)
            descriptions = "; ".join(f"{name}: {group.get('condition', '')}" for name, group in group_items)
            # Edge-case construction is record-level work. A rare/impossible edge
            # definition must not abort the entire /scenario/generate request.
            # Keep all successfully generated records and report each unconstructed
            # edge-case unit through record_errors.
            for offset in range(missing):
                if record_errors_out is not None:
                    record_errors_out.append({
                        "record_index": len(generated) + offset,
                        "error": (
                            "Unable to construct transactional edge-case record after "
                            "deterministic constraint solving. "
                            f"Constructed {len(eligible)} of {target}. Definitions: {descriptions}"
                        ),
                        "record": {},
                    })
            target = len(eligible)
            if target <= 0:
                return generated

    # Spread selected edge records through the materialized response rather than
    # clustering all edge cases in one event/entity.
    chosen_positions = {
        eligible[min(len(eligible) - 1, int(i * len(eligible) / target))][0]
        for i in range(target)
    }
    # Guarantee exactly target unique positions even with integer rounding.
    if len(chosen_positions) < target:
        for idx, _group_name, _candidate in eligible:
            chosen_positions.add(idx)
            if len(chosen_positions) == target:
                break

    chosen_lookup = {idx: (group_name, candidate) for idx, group_name, candidate in eligible if idx in chosen_positions}
    for idx, (group_name, candidate) in chosen_lookup.items():
        group = edge_groups[group_name]
        if not _safe_edge_condition(group.get("condition", ""), candidate):
            if record_errors_out is not None:
                record_errors_out.append({
                    "record_index": idx,
                    "error": (
                        f"Edge-case invariant failed for condition '{group.get('condition', '')}' "
                        f"on event '{generated[idx].get('event_type')}'."
                    ),
                    "record": dict(generated[idx]),
                })
            continue
        candidate["isEdgeCaseData"] = True
        generated[idx] = candidate

    # The generator is best-effort: any mismatch is reported as a record-level
    # error rather than aborting the whole request.
    true_count = sum(1 for row in generated if row.get("isEdgeCaseData") is True)
    if true_count != target and record_errors_out is not None:
        record_errors_out.append({
            "record_index": -1,
            "error": (
                f"Transactional edge-case assignment produced {true_count} valid edge-case "
                f"record(s) out of {target} requested."
            ),
            "record": {},
        })
    return generated


# ── QA / validation-layer helpers ───────────────────────────────────────────────

_NUMERIC_DTYPES = {"int", "float"}
_CHUNK = 50

_QA_SYSTEM = """
You are the QA Validation layer for a synthetic data generation pipeline covering
multiple industries and countries.

Validate the supplied records against the scenario's business rules, field
constraints, country/industry conventions, cross-field rules, calculations,
and transactional event semantics.

Do not invent fields that are intentionally absent from sparse transactional
events. Do not rewrite valid records unnecessarily.

SEMANTIC RELEVANCE IS A HARD REQUIREMENT, NOT A STYLE PREFERENCE:
- Every value must make business sense for the target industry, domain, scenario, and country.
- A value can be syntactically valid and still be invalid. Reject/repair semantically wrong values.
- Never use placeholder organization/entity values such as Provider_A, Provider_B, Company_A, Product_A, Gateway_A, Synthetic_Provider, or similar.
- When the scenario schema identifies a real-world entity field (provider, operator, merchant, bank, retailer, carrier, manufacturer, etc.), use the country/industry vocabulary supplied by the scenario profile.
- Never borrow terminology from another industry merely because it fits the datatype.
- Do not create filler values solely to satisfy a variable count; every generated value must serve the scenario.
- Cross-field and event-state consistency is mandatory: validate relationships between fields, not only individual field types.
- INDUSTRY RELEVANCE IS A HARD GATE: every generated value must make sense for the exact
  target industry, country, domain, and scenario. A value that merely matches a datatype
  or a generic category is invalid if it belongs to another industry.
- REAL ENTITY FIELDS (provider/operator/bank/merchant/retailer/carrier/manufacturer/etc.)
  must use target-industry entities. Never use Provider_A, Company_A, telecom operators
  in non-telecom datasets, or another industry's brands. If the profile has no entity list,
  do not invent an unrelated cross-industry entity.
- VALUE-LEVEL AUDIT: inspect every categorical value individually and repair/drop values
  that fail the industry + country + scenario relevance test.
- STATE CONSISTENCY: after field-level validation, re-evaluate dependent fields together.
  Never allow a failure/decline/cancellation state to coexist with a success-only outcome
  unless the record explicitly represents a later recovery/completion state.
- TELECOM-SPECIFIC PAYMENT RULE: when Industry is Telecom, UPI must never appear
  in final generated data. Any Telecom generated value containing UPI (including
  "UPI payment", "UPI transaction", "UPI-linked", or similar) must be represented
  as the canonical value "DIGITAL_WALLET". This rule applies only to Telecom.
- For non-Telecom industries, do not remove, rename, or normalize UPI merely because
  it appears. Preserve UPI when it is valid for that industry's scenario and profile.

Return JSON:
  - "valid_records": [ ... ]
  - "dropped_records": [ ... ]
  - "fixes_applied": int
  - "issues_found": int
"""


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
    names = {k: v for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    try:
        tree = ast.parse(expr, mode="eval")
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
            ast.Name, ast.Call, ast.Load, ast.Tuple, ast.List,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Name) and node.id not in names and node.id not in allowed_funcs:
                return None
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs):
                return None
        return eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {**names, **allowed_funcs})
    except Exception:
        return None


def _normalize(v: Any) -> str:
    return str(v).strip().lower().replace(" ", "_").replace("-", "_")


def _country_repair(name: str, value: Any, profile: dict):
    if value is None:
        return value, None
    # Telecom-only output contract: UPI is represented as DIGITAL_WALLET.
    # Other industries are allowed to retain UPI when it is a valid value.
    if str(profile.get("industry", "")).strip().lower() == "telecom":
        normalized_value = _normalize_upi_in_generated_value(value)
        if normalized_value != value:
            return normalized_value, f"{name} normalized from UPI to DIGITAL_WALLET"
    lname = name.lower()
    if country_allowed_value(name, value, profile):
        return value, None
    if "service_provider" in lname or "provider" == lname or lname.endswith("_provider"):
        choices = profile.get("service_providers") or []
        if choices:
            normalized = {_normalize(c) for c in choices}
            if _normalize(value) not in normalized:
                return choices[0], f"{name} corrected to an industry/country-appropriate provider"
    if "payment" in lname or "pay_method" in lname:
        choices = payment_methods_for_profile(profile)
        if choices:
            return choices[0], f"{name} corrected for country {profile.get('country_code')}"
    if "currency" in lname:
        return profile.get("currency", value), f"{name} corrected for country {profile.get('country_code')}"
    return value, None


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
    profile: dict,
    transactional: bool,
    event_fields: set[str] | None = None,
    rules: dict | None = None,
) -> tuple[dict, list[str]]:
    """Validate and repair one record using the actual variable definitions."""
    rec = dict(rec)
    issues: list[str] = []
    by_name = {v["name"]: v for v in variables}
    active_fields = set(event_fields or field_order) if transactional else set(field_order)

    # 1. Country-sensitive values and declared categorical choices.
    for name, value in list(rec.items()):
        if name.startswith("__") or value is None:
            continue
        repaired, issue = _country_repair(name, value, profile)
        if issue:
            rec[name] = repaired
            issues.append(issue)

    for var in variables:
        name = var["name"]
        if name not in rec or rec[name] is None:
            # Sparse transactional rows are intentional. Missing event-local fields
            # are not errors and will be handled by the existing fill behavior later.
            continue
        value = rec[name]
        dtype = var.get("dtype", "string")
        params = var.get("params") or {}

        # 2. Type checks / safe coercion.
        try:
            if dtype == "int" and not isinstance(value, bool):
                if not isinstance(value, int):
                    rec[name] = int(float(value))
                    issues.append(f"{name} coerced to int")
            elif dtype == "float" and not isinstance(value, bool):
                if not isinstance(value, (int, float)):
                    rec[name] = float(value)
                    issues.append(f"{name} coerced to float")
            elif dtype == "boolean":
                if not isinstance(value, bool):
                    if str(value).strip().lower() in {"true", "1", "yes"}:
                        rec[name] = True
                    elif str(value).strip().lower() in {"false", "0", "no"}:
                        rec[name] = False
                    else:
                        raise ValueError("invalid boolean")
                    issues.append(f"{name} coerced to boolean")
            elif dtype == "datetime" and _qa_parse_dt(value) is None:
                rec[name] = datetime.now(timezone.utc).isoformat()
                issues.append(f"{name} repaired as datetime")
            elif dtype == "date":
                try:
                    date.fromisoformat(str(value)[:10])
                except Exception:
                    rec[name] = date.today().isoformat()
                    issues.append(f"{name} repaired as date")
        except Exception:
            # Do not drop the entire record for a single malformed value; regenerate
            # from the declared generator when possible, otherwise use a safe default.
            rec[name] = _default_for_dtype(dtype)
            issues.append(f"{name} repaired from invalid type")
            value = rec[name]

        # 3. Categorical values must stay within the declared scenario vocabulary.
        if dtype in {"categorical", "string"} and params.get("choices") and rec.get(name) is not None:
            choices = list(params.get("choices", []))
            # Real-world entity/payment vocabularies from the active country/industry
            # profile override stale choices that may exist in an already-confirmed
            # scenario or an older cached schema. This prevents a later categorical
            # check from undoing semantic country repair (e.g. Jio -> Provider_A).
            lname = str(name).strip().lower()
            if lname == "service_provider" and profile.get("service_providers"):
                choices = list(profile.get("service_providers") or [])
            elif lname == "payment_method" and str(profile.get("industry", "")).strip().lower() == "telecom":
                choices = ["DEBIT_CARD", "CREDIT_CARD", "DIGITAL_WALLET", "NETBANKING"]
            if choices and _normalize(rec[name]) not in {_normalize(c) for c in choices}:
                rec[name] = choices[0]
                issues.append(f"{name} corrected to authoritative industry/country choice")

        # 4. Generic numeric ranges from the variable definition.
        if isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool):
            val = float(rec[name])
            lo = params.get("min")
            hi = params.get("max")
            if lo is not None and val < float(lo):
                rec[name] = int(lo) if dtype == "int" else float(lo)
                issues.append(f"{name} raised to minimum")
            if hi is not None and val > float(hi):
                rec[name] = int(hi) if dtype == "int" else float(hi)
                issues.append(f"{name} lowered to maximum")

        # Percentage-like fields should remain 0..100 when explicitly named as percentages.
        if isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool) and (
            "pct" in name.lower() or "percent" in name.lower() or "percentage" in name.lower()
        ):
            old = rec[name]
            rec[name] = max(0, min(100, old))
            if old != rec[name]:
                issues.append(f"{name} clamped to percentage range")

    # 5. Formula invariants: recompute every active scenario formula.
    # Formulas are the source of truth across ALL industries; there are no
    # industry-specific arithmetic assumptions here. SchemaAgent formulas are
    # also considered when a business rule adds a formula not present on the
    # variable definition itself.
    formula_specs = _collect_formula_specs(variables, rules)
    for field, expr in formula_specs:
        if field not in active_fields:
            continue
        deps = _formula_dependencies(expr)
        missing = [dep for dep in deps if rec.get(dep) is None]
        if missing:
            # Do not invent a numeric default for a formula when its inputs are
            # absent. For sparse transactional events this is intentional; when
            # the formula field is explicitly part of the event, the record is
            # left for the generator/schema to repair rather than fabricating a value.
            issues.append(f"{field} formula could not be evaluated; missing dependencies: {missing}")
            continue
        expected = _safe_formula(expr, rec)
        if expected is None:
            issues.append(f"{field} formula could not be evaluated")
            continue
        actual = rec.get(field)
        mismatch = False
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            mismatch = not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=0.01)
        else:
            mismatch = str(actual) != str(expected)
        if mismatch:
            rec[field] = expected
            issues.append(f"{field} corrected from formula")

    # 6. Do not hard-code industry-specific arithmetic here.
    # All calculations must come from scenario-declared formulas so the same
    # validator works across every industry.

    # 7. Common timestamp relationships when fields exist.
    event_ts = _qa_parse_dt(rec.get("event_timestamp"))
    dispatch_ts = _qa_parse_dt(rec.get("notification_dispatch_ts"))
    response_ts = _qa_parse_dt(rec.get("customer_response_ts"))
    if event_ts and dispatch_ts and dispatch_ts < event_ts:
        rec["notification_dispatch_ts"] = event_ts.isoformat()
        dispatch_ts = event_ts
        issues.append("notification_dispatch_ts corrected to be >= event_timestamp")
    if dispatch_ts and response_ts and response_ts < dispatch_ts:
        rec["customer_response_ts"] = dispatch_ts.isoformat()
        issues.append("customer_response_ts corrected to be >= notification_dispatch_ts")

    # 8. Country phone-number convention. The profile supplies the expected prefix;
    # reject/repair only fields that clearly represent a phone number.
    phone_names = [n for n in rec if "msisdn" in n.lower() or "phone" in n.lower() or "mobile" in n.lower()]
    expected_cc = str(profile.get("phone_country_code", ""))
    for name in phone_names:
        value = str(rec.get(name, ""))
        if expected_cc and value and not value.startswith(expected_cc):
            # Preserve the subscriber's local digits where possible, but enforce the country code.
            digits = "".join(ch for ch in value if ch.isdigit())
            if expected_cc == "+91" and len(digits) >= 10:
                local = digits[-10:]
                if local[0] in "6789":
                    rec[name] = expected_cc + local
                else:
                    rec[name] = expected_cc + "9" + local[-9:]
            elif expected_cc == "+44" and len(digits) >= 10:
                local = digits[-10:]
                rec[name] = expected_cc + (local if local.startswith("7") else "7" + local[-9:])
            elif expected_cc == "+971":
                rec[name] = expected_cc + "50" + digits[-7:]
            elif expected_cc == "+1":
                rec[name] = expected_cc + digits[-10:].zfill(10)
            else:
                rec[name] = expected_cc + digits[-10:]
            issues.append(f"{name} corrected for country phone convention")

    # 9. Keep output limited to declared variables plus transactional metadata.
    metadata = {"journey_id", "transaction_id", "event_type", "event_sequence", "event_occurrence", "event_timestamp", "isEdgeCaseData"}
    if transactional:
        allowed = set(field_order) | metadata
    else:
        allowed = set(field_order)
    rec = {k: v for k, v in rec.items() if k in allowed}

    return rec, issues


def _repair_edge_record(
    rec: dict,
    edge_groups: dict,
    variables: list[dict],
    profile: dict | None = None,
    rules: dict | None = None,
    event_fields: set[str] | None = None,
) -> tuple[dict, bool]:
    """Make an edge-labelled record satisfy its configured condition.

    Edge constraints are authoritative for an edge-labelled record.  We use the same
    deterministic solver as the generator so QA cannot accidentally invalidate a valid
    edge case by clamping/replacing one of its condition fields.  Missing sparse-event
    fields may be materialized only for the edge record.
    """
    if rec.get("isEdgeCaseData") is not True:
        return rec, False
    for group in edge_groups.values():
        configured_event = str(group.get("event_type") or "").strip().upper().replace(" ", "_")
        record_event = str(rec.get("event_type") or "").strip().upper().replace(" ", "_")
        if configured_event and configured_event != record_event:
            continue
        condition = str(group.get("condition") or "").strip()
        if not condition:
            continue
        branches = _condition_branches(condition) or [{}]
        for assignments in branches:
            try:
                candidate = _edge_candidate(
                    dict(rec), group, variables, profile, rules, event_fields or set(), assignments=assignments
                )
            except Exception:
                continue
            if _safe_edge_condition(condition, candidate):
                candidate["isEdgeCaseData"] = True
                return candidate, True
    return rec, False



def _make_non_edge_record(
    rec: dict,
    edge_groups: dict[str, dict],
    variables: list[dict],
    profile: dict | None = None,
    rules: dict | None = None,
    transactional: bool = False,
) -> tuple[dict, bool]:
    """Dynamically repair a normal record that accidentally matches an edge condition.

    This is deliberately schema/condition driven.  It does not know any industry or
    field names.  We search for a minimally changed value assignment that makes every
    applicable edge condition false, while respecting declared categorical domains and
    recalculating formula fields.  This is used after ordinary QA repairs because those
    repairs can legitimately change a value and accidentally enter an edge state.
    """
    candidate = dict(rec)
    applicable = [g for g in _applicable_edge_groups(rec, edge_groups, transactional) if g.get("condition")]
    if not applicable:
        return candidate, True

    by_name = {str(v.get("name")): v for v in variables if isinstance(v, dict) and v.get("name")}
    formula_names = {str(v.get("name")) for v in variables if v.get("formula")}

    def alternatives(name: str, value: Any) -> list[Any]:
        var = by_name.get(name, {})
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        choices = params.get("choices") if isinstance(params.get("choices"), list) else []
        out: list[Any] = []
        def add(x):
            if x is None and value is not None:
                return
            if all(str(x) != str(y) for y in out):
                out.append(x)
        for c in choices:
            if str(c).strip().lower() != str(value).strip().lower():
                add(c)
        if isinstance(value, bool):
            add(not value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            n = float(value)
            add(int(n - 1) if isinstance(value, int) else n - 1.0)
            add(int(n + 1) if isinstance(value, int) else n + 1.0)
            params_min, params_max = params.get("min"), params.get("max")
            try:
                if params_min is not None and float(params_min) != n: add(int(float(params_min)) if isinstance(value, int) else float(params_min))
            except Exception: pass
            try:
                if params_max is not None and float(params_max) != n: add(int(float(params_max)) if isinstance(value, int) else float(params_max))
            except Exception: pass
            add(0 if isinstance(value, int) else 0.0)
        elif isinstance(value, str):
            add("")
            add("OTHER")
        return out[:8]

    def condition_refs(expr: str) -> list[str]:
        try:
            tree = ast.parse(str(expr), mode="eval")
            return list(dict.fromkeys(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in by_name))
        except Exception:
            return []

    # Build small domains only for fields actually used by edge conditions.  The
    # Cartesian search is bounded so QA remains fast even with many schema fields.
    domains: dict[str, list[Any]] = {}
    refs: list[str] = []
    for group in applicable:
        for name in condition_refs(group.get("condition", "")):
            if name not in refs and name not in formula_names and name in rec:
                refs.append(name)
    for name in refs:
        domains[name] = alternatives(name, rec.get(name))

    # Try one-field changes first, then small combinations.  Most accidental matches
    # are caused by a single QA repair, so this keeps the common path cheap.
    from itertools import product
    trials: list[dict[str, Any]] = [{}]
    for name in refs:
        vals = domains.get(name, [])
        if not vals:
            continue
        trials.extend({name: v} for v in vals)
    if len(refs) <= 5:
        value_lists = [domains.get(n, [])[:5] for n in refs]
        if all(value_lists):
            for combo in product(*value_lists):
                trials.append(dict(zip(refs, combo)))
                if len(trials) >= 250:
                    break

    for changes in trials[:250]:
        test = dict(candidate)
        test.update(changes)
        test = _recalculate_formulas(test, variables)
        if not any(
            _safe_edge_condition(g.get("condition", ""), test)
            for g in applicable
            if g.get("condition")
        ):
            test["isEdgeCaseData"] = False
            return test, True

    return candidate, False

def _applicable_edge_groups(rec: dict, edge_groups: dict[str, dict], transactional: bool) -> list[dict]:
    """Return edge definitions applicable to the actual record/event.

    Event-scoped definitions are never evaluated against unrelated sparse events.
    Unscoped definitions remain eligible for any record that contains their fields.
    """
    event_type = str(rec.get("event_type") or "").strip().upper().replace(" ", "_")
    out = []
    for group in edge_groups.values():
        configured = str(group.get("event_type") or "").strip().upper().replace(" ", "_")
        if transactional and configured and configured != event_type:
            continue
        out.append(group)
    return out


def _apply_telecom_client_contract(records: list[dict], scenario_id: str, industry: str) -> list[dict]:
    """Deterministically enforce audited Telecom LB-06/LB-07 business semantics.

    This is intentionally isolated from the generic generator/QA machinery. It only
    runs for the two audited Telecom scenarios and therefore cannot change generation
    behavior for other industries/scenarios. The schema remains the source of field
    definitions; this helper only reconciles values across the lifecycle events.
    """
    if str(industry or "").strip().lower() != "telecom":
        return records
    sid = str(scenario_id or "").strip().upper()
    if sid not in {"LB-06", "LB-07"}:
        return records

    by_journey: dict[str, list[dict]] = {}
    for row in records:
        by_journey.setdefault(str(row.get("journey_id", "")), []).append(row)

    def num(v, default=0.0):
        try:
            x = float(v)
            return x if math.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    for rows in by_journey.values():
        rows.sort(key=lambda r: (int(r.get("event_sequence", 0) or 0), int(r.get("event_occurrence", 0) or 0), str(r.get("event_timestamp", ""))))
        first = rows[0] if rows else {}
        balance = max(0.0, num(first.get("balance_before"), num(first.get("balance_after"), 0.0)))
        amount = max(10.0, num(first.get("recharge_amount"), 100.0))
        failed_txn = None

        if sid == "LB-06":
            failure_reasons = ["INSUFFICIENT_FUNDS", "GATEWAY_TIMEOUT", "NETWORK_FAILURE", "BANK_DECLINE", "AUTHENTICATION_FAILURE", "PAYMENT_INSTRUMENT_UNAVAILABLE"]
            reason = failure_reasons[sum(ord(c) for c in str(first.get("journey_id", ""))) % len(failure_reasons)]
            retry_total = sum(1 for r in rows if str(r.get("event_type", "")).upper() == "TOPUP_RETRY")
            retry_number = 0
            failure_rows = [r for r in rows if str(r.get("event_type", "")).upper() == "TOPUP_FAILURE"]
            for row in rows:
                et = str(row.get("event_type", "")).upper()
                row["balance_before"] = round(balance, 2)
                row["recharge_amount"] = round(amount, 2)
                if et == "LOW_BALANCE_DETECTED":
                    row["balance_after"] = round(balance, 2)
                elif et in {"TOPUP_ATTEMPT", "TOPUP_FAILURE"}:
                    row["recharge_status"] = "FAILED"
                    row["failure_reason"] = reason
                    row["balance_after"] = round(balance, 2)
                    if et == "TOPUP_FAILURE":
                        failed_txn = row.get("transaction_id")
                elif et == "RECOVERY_GUIDANCE":
                    row["recovery_action"] = "USE_ALTERNATE_PAYMENT_METHOD" if reason in {"GATEWAY_TIMEOUT", "BANK_DECLINE", "PAYMENT_INSTRUMENT_UNAVAILABLE"} else "RETRY_SAME_METHOD"
                    row["recovery_status"] = "RECOVERY_PENDING"
                    row["balance_after"] = round(balance, 2)
                elif et == "TOPUP_RETRY":
                    retry_number += 1
                    row["recharge_status"] = "SUCCESS" if retry_number == retry_total else "FAILED"
                    row["failure_reason"] = None if retry_number == retry_total else reason
                    row["parent_transaction_id"] = failed_txn or row.get("parent_transaction_id") or "PARENT_TXN"
                    row["retry_count"] = retry_number
                    row["balance_after"] = round(balance, 2)
                elif et == "TOPUP_SUCCESS":
                    row["recharge_status"] = "SUCCESS"
                    row["recovery_status"] = "RECOVERED_ALTERNATE_METHOD" if reason in {"GATEWAY_TIMEOUT", "BANK_DECLINE", "PAYMENT_INSTRUMENT_UNAVAILABLE"} else "RECOVERED_SAME_METHOD"
                    row["parent_transaction_id"] = failed_txn or row.get("parent_transaction_id") or "PARENT_TXN"
                    row["retry_count"] = max(1, retry_total)
                    success_balance = round(balance + amount, 2)
                    row["balance_after"] = success_balance
                    row["final_journey_status"] = "RECOVERED"
                    try:
                        row["recovery_timestamp"] = row.get("event_timestamp")
                    except Exception:
                        pass

        else:  # LB-07: preserve the exceptional state before reconciliation.
            expected = round(balance + amount, 2)
            observed = round(balance, 2)
            for row in rows:
                et = str(row.get("event_type", "")).upper()
                row["balance_before"] = round(balance, 2)
                row["recharge_amount"] = round(amount, 2)
                row["transaction_status"] = "SUCCESS"
                if et in {"TOPUP_INITIATED", "PAYMENT_AUTHORIZED", "PAYMENT_SETTLED"}:
                    row["balance_after"] = round(balance, 2)
                    if et == "PAYMENT_SETTLED":
                        row["settlement_status"] = "SETTLED"
                elif et == "BALANCE_UPDATE_FAILED":
                    row["balance_after"] = observed
                    row["expected_balance"] = expected
                    row["observed_balance"] = observed
                    row["balance_update_status"] = "FAILED"
                    row["balance_variance"] = round(observed - expected, 2)
                    row["exception_detected_flag"] = True
                    reasons = ["LEDGER_SYNCHRONIZATION_DELAY", "CORE_SYSTEM_UPDATE_DELAY", "CACHE_REFRESH_ISSUE", "EVENT_PROCESSING_LAG", "RECONCILIATION_MISMATCH", "BACKEND_TIMEOUT_AFTER_SUCCESSFUL_PROCESSING"]
                    row["exception_reason"] = reasons[sum(ord(c) for c in str(first.get("journey_id", ""))) % len(reasons)]
                elif et == "DISCREPANCY_DETECTED":
                    row["balance_after"] = observed
                    row["expected_balance"] = expected
                    row["observed_balance"] = observed
                    row["balance_variance"] = round(observed - expected, 2)
                    row["exception_detected_flag"] = True
                elif et == "VERIFICATION_REQUESTED":
                    row["balance_after"] = observed
                    row["verification_status"] = "VERIFIED"
                    row["verification_attempt_count"] = 1
                elif et == "TRANSACTION_LOOKUP":
                    row["balance_after"] = observed
                    row["settlement_status"] = "SETTLED"
                    row["verification_status"] = "VERIFIED"
                elif et == "RECONCILIATION_STARTED":
                    row["balance_after"] = observed
                    row["reconciliation_status"] = "RESOLVED"
                    row["resolution_type"] = "BALANCE_CORRECTION"
                elif et in {"STATUS_CONFIRMED", "CUSTOMER_NOTIFIED"}:
                    row["balance_after"] = expected
                    row["verification_status"] = "VERIFIED"
                    row["settlement_status"] = "SETTLED"
                    row["reconciliation_status"] = "RESOLVED"
                    row["final_status"] = "RESOLVED"
                elif et == "BALANCE_CORRECTED":
                    row["expected_balance"] = expected
                    row["final_balance"] = expected
                    row["balance_after"] = expected
                    row["resolution_type"] = "BALANCE_CORRECTION"
                elif et == "CASE_RESOLVED":
                    row["final_balance"] = expected
                    row["balance_after"] = expected
                    row["final_status"] = "RESOLVED"
                    row["resolution_type"] = "BALANCE_CORRECTION"

    return records


# ── Agent ──────────────────────────────────────────────────────────────────────

class DataGenerationAgent:
    """Agent 4 — generates records algorithmically (no LLM call per record), then
    validates them through an internal QA layer. 2-node LangGraph subgraph:
    generate -> qa_validate."""

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
        logger.info(
            "[DataGeneration] Context: industry=%s country=%s type=%s domain=%s business_scenario=%s use_case=%s scenario_type=%s entity_key=%s",
            state.industry, state.country or "GLOBAL", state.type_of_data, state.domain,
            state.business_scenario, state.use_case, state.scenario_type, state.entity_key,
        )
        # The confirmed scenario context is the source of truth. SchemaAgent has already
        # compiled its business/use-case constraints into state.rules, and EdgeCaseAgent
        # has already validated edge-case conditions are satisfiable; generation below
        # applies those constraints deterministically without an LLM call per record.
        dyn = resolve_variables(state.scenario)
        if dyn is None:
            raise ValueError(f"Unknown scenario '{state.scenario}'")
        variables, FIELD_ORDER = dyn

        if state.type_of_data == "transactional":
            from core.compiled_schema import compile_scenario
            compiled = compile_scenario(state.scenario)
            entity_key = compiled.entity_key
            state.transactional_event_counts = {}
            profile = get_profile(state.industry, state.country)
            records = _transactional_records(
                compiled, state.count, state.transactional_event_counts,
                profile=profile, rules=state.rules,
                edge_case_variables=state.edge_case_variables,
                edge_case_percentage=state.edge_case_percentage,
                record_errors_out=state.record_errors,
            )
            state.field_order = [
                "journey_id", "transaction_id", "event_type", "event_sequence",
                "event_occurrence", "event_timestamp",
            ] + [name for name in FIELD_ORDER if name != "event_timestamp"] + (["isEdgeCaseData"] if "isEdgeCaseData" not in FIELD_ORDER else [])
            logger.info(
                "[DataGeneration] Generated %d transactional records from %d journeys and %d events.",
                len(records), state.count, len(compiled.events),
            )
        else:
            state.field_order = FIELD_ORDER
            profile = get_profile(state.industry, state.country)
            edge_groups = _edge_case_groups(state.edge_case_variables)
            edge_count = _edge_case_count(state.count, state.edge_case_percentage)
            edge_names = list(edge_groups)
            records = []
            for index in range(state.count):
                rec = {}
                try:
                    rec = _generate_record(variables, profile=profile, rules=state.rules)
                    if index < edge_count and edge_names:
                        group = edge_groups[edge_names[index % len(edge_names)]]
                        matched = False
                        branches = _condition_branches(group.get("condition", ""), variables) or [{}]
                        for _ in range(_MAX_EDGE_CASE_ATTEMPTS):
                            for assignments in branches:
                                candidate = _edge_case_candidate_for_aggregation(
                                    dict(rec), group, variables, profile, state.rules, assignments
                                )
                                if _safe_edge_condition(group.get("condition", ""), candidate):
                                    rec = candidate
                                    rec["isEdgeCaseData"] = True
                                    matched = True
                                    break
                            if matched:
                                break
                            # Regenerate the normal base so random-dependent conditions
                            # get another opportunity; deterministic constraints still win.
                            rec = _generate_record(variables, profile=profile, rules=state.rules)
                        if not matched:
                            error = {
                                "record_index": index,
                                "error": (
                                    f"Unable to generate an aggregational edge-case record satisfying "
                                    f"'{group.get('condition', '')}' after {_MAX_EDGE_CASE_ATTEMPTS} attempts"
                                ),
                                "record": dict(rec) if isinstance(rec, dict) else {},
                            }
                            state.record_errors.append(error)
                            logger.warning("[DataGeneration] Skipping aggregational record %d: %s", index, error["error"])
                            continue
                    else:
                        # Normal records must be explicitly outside every configured
                        # edge condition. Use the same generic non-edge solver used by QA
                        # instead of relying only on random retries; this is important for
                        # categorical/low-cardinality fields where random retries can
                        # otherwise discard most or all normal records.
                        if edge_groups and state.edge_case_percentage > 0:
                            if _matches_any_edge_condition(rec, edge_groups):
                                rec, normal_ok = _make_non_edge_record(
                                    rec, edge_groups, variables, profile=profile,
                                    rules=state.rules, transactional=False
                                )
                            else:
                                normal_ok = True
                            if not normal_ok or _matches_any_edge_condition(rec, edge_groups):
                                error = {
                                    "record_index": index,
                                    "error": (
                                        "Unable to generate a normal aggregational record that does not "
                                        "satisfy any configured edge-case condition. The edge-case "
                                        "definitions leave no valid normal-domain value."
                                    ),
                                    "record": dict(rec) if isinstance(rec, dict) else {},
                                }
                                state.record_errors.append(error)
                                logger.warning("[DataGeneration] Skipping aggregational record %d: %s", index, error["error"])
                                continue
                        rec["isEdgeCaseData"] = False
                    records.append(rec)
                except Exception as exc:
                    error = {
                        "record_index": index,
                        "error": str(exc),
                        "record": dict(rec) if isinstance(locals().get("rec"), dict) else {},
                    }
                    state.record_errors.append(error)
                    logger.warning("[DataGeneration] Skipping aggregational record %d: %s", index, exc)
                    continue
            if "isEdgeCaseData" not in state.field_order:
                state.field_order = list(state.field_order) + ["isEdgeCaseData"]
            logger.info("[DataGeneration] Generated %d aggregational records with %d fields each; edge cases=%d.",
                        len(records), len(variables), edge_count)

        records = _apply_telecom_client_contract(records, state.scenario, state.industry)
        # Final pre-QA normalization: no UPI token/value is allowed to enter the
        # validation pipeline, even if it came from a legacy schema or generator.
        records = _normalize_upi_in_records(records, state.industry)
        state.raw_records = records
        return state

    def _qa_validate(self, state: WorkflowState) -> WorkflowState:
        records = state.raw_records
        logger.info("[QA] Validating %d records", len(records))
        edge_groups = {}
        for item in state.edge_case_variables:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            group = str(item.get("edge_case_name") or "Scenario Edge Case")
            edge_groups.setdefault(group, {"condition": str(item.get("condition") or ""), "variables": []})
            edge_groups[group]["variables"].append(item)

        dyn = resolve_variables(state.scenario)
        if dyn is None:
            raise ValueError(f"Unknown scenario '{state.scenario}'")
        variables, _ = dyn
        dtype_map = {v["name"]: v.get("dtype", "string") for v in variables}
        profile = get_profile(state.industry, state.country)
        transactional = state.type_of_data == "transactional"
        event_defs = {str(e.get("event_type")): e for e in resolve_events(state.scenario)} if transactional else {}
        formula_specs = _collect_formula_specs(variables, state.rules)

        algo_fixed = 0
        checked: list[dict] = []
        seen_transactions: set[str] = set()
        seen_entities: set[str] = set()
        journey_state: dict[str, tuple[int, datetime | None, str | None]] = {}
        journey_edge_context: dict[str, dict[str, Any]] = {}

        for record_index, rec in enumerate(records):
            try:
                pre_event_def = event_defs.get(str(rec.get("event_type"))) if transactional else None
                pre_event_fields = set(pre_event_def.get("fields", []) or []) if pre_event_def else set()
                rec, pre_repaired_edge = _repair_edge_record(dict(rec), edge_groups, variables, profile=profile, rules=state.rules, event_fields=pre_event_fields)
                rec, issues = _validate_record(
                    rec, variables, state.field_order, profile, transactional,
                    event_fields=pre_event_fields,
                    rules=state.rules,
                )

                if rec.get("isEdgeCaseData") is True:
                    rec, repaired_ok = _repair_edge_record(rec, edge_groups, variables, profile=profile, rules=state.rules, event_fields=pre_event_fields)
                    if not repaired_ok:
                        raise ValueError(
                            "Generated edge-case record failed deterministic validation: "
                            "the configured edge-case condition cannot be satisfied by the validated record"
                        )

                # Edge-case labels are trusted only after the configured condition is checked.
                # Aggregational rows are self-contained. Transactional rows are sparse, so their
                # final edge status is validated later against the complete journey context.
                if rec.get("isEdgeCaseData") is True:
                    matched = any(
                        _safe_edge_condition(group.get("condition", ""), rec)
                        for group in _applicable_edge_groups(rec, edge_groups, transactional)
                        if group.get("condition")
                    )
                    if not matched:
                        # Never silently turn a generated edge case into a normal record.
                        # If QA changed a value so the condition no longer holds, the
                        # dataset is internally inconsistent and must fail visibly.
                        raise ValueError(
                            "Generated edge-case record failed deterministic validation: "
                            "isEdgeCaseData=true but no configured edge-case condition is satisfied"
                        )

                # Classification is deterministic: a normal record must not satisfy an
                # applicable edge condition. Otherwise identical condition-relevant data
                # could legitimately receive both true and false labels. Generator-side
                # normal-record exclusion is the primary guard; QA repeats the invariant
                # after all repairs/fills so it cannot be broken here.
                if rec.get("isEdgeCaseData") is not True and any(
                    _safe_edge_condition(group.get("condition", ""), rec)
                    for group in _applicable_edge_groups(rec, edge_groups, transactional)
                    if group.get("condition")
                ):
                    rec, normal_ok = _make_non_edge_record(
                        rec, edge_groups, variables, profile=profile, rules=state.rules, transactional=transactional
                    )
                    if not normal_ok or any(
                        _safe_edge_condition(group.get("condition", ""), rec)
                        for group in _applicable_edge_groups(rec, edge_groups, transactional)
                        if group.get("condition")
                    ):
                        raise ValueError(
                            "Generated normal record satisfies a configured edge-case condition "
                            "and no schema-valid non-edge value could be derived dynamically"
                        )

                if transactional:
                    # Metadata is structural and should never be delegated to the LLM.
                    if not rec.get("journey_id"):
                        rec["journey_id"] = f"JRN-{uuid.uuid4().hex[:12].upper()}"
                        issues.append("journey_id generated")
                    if not rec.get("transaction_id") or str(rec.get("transaction_id")) in seen_transactions:
                        rec["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12].upper()}"
                        issues.append("transaction_id generated/renewed for uniqueness")
                    seen_transactions.add(str(rec["transaction_id"]))

                    event_type = str(rec.get("event_type") or "BUSINESS_EVENT")
                    rec["event_type"] = event_type
                    seq = int(rec.get("event_sequence") or 1)
                    if event_defs:
                        definition = event_defs.get(event_type)
                        if definition:
                            expected_seq = int(definition.get("sequence", seq))
                            if seq != expected_seq:
                                rec["event_sequence"] = expected_seq
                                seq = expected_seq
                                issues.append("event_sequence corrected from scenario definition")
                    rec["event_sequence"] = max(1, seq)

                    ts = _qa_parse_dt(rec.get("event_timestamp"))
                    if ts is None:
                        ts = datetime.now(timezone.utc)
                        rec["event_timestamp"] = ts.isoformat()
                        issues.append("event_timestamp generated")

                    journey = str(rec["journey_id"])
                    previous = journey_state.get(journey)
                    if previous:
                        prev_seq, prev_ts, prev_entity = previous
                        if rec["event_sequence"] < prev_seq:
                            rec["event_sequence"] = prev_seq
                            issues.append("event_sequence corrected to preserve journey ordering")
                        if prev_ts and ts < prev_ts:
                            rec["event_timestamp"] = prev_ts.isoformat()
                            ts = prev_ts
                            issues.append("event_timestamp corrected to preserve journey ordering")
                        entity_name = resolve_entity_key(state.scenario)
                        if entity_name and entity_name in rec and prev_entity and str(rec[entity_name]) != prev_entity:
                            rec[entity_name] = prev_entity
                            issues.append("entity key corrected for journey consistency")
                    entity_key = resolve_entity_key(state.scenario)
                    entity_value = str(rec.get(entity_key)) if entity_key and rec.get(entity_key) is not None else None
                    journey_state[journey] = (rec["event_sequence"], ts, entity_value)
                    if entity_value:
                        seen_entities.add(entity_value)

                # Fill missing values only after validation. Formula fields are
                # calculated first whenever their dependencies are available; never
                # replace a calculable formula with an arbitrary dtype default.
                active_fill_fields = set(event_defs.get(str(rec.get("event_type")), {}).get("fields", []) or []) if transactional else set(state.field_order)
                for formula_field, formula_expr in formula_specs:
                    if formula_field not in active_fill_fields or rec.get(formula_field) is not None:
                        continue
                    deps = _formula_dependencies(formula_expr)
                    if deps and all(rec.get(dep) is not None for dep in deps):
                        calculated = _safe_formula(formula_expr, rec)
                        if calculated is not None:
                            rec[formula_field] = calculated
                            issues.append(f"{formula_field} calculated from formula")

                # For transactional data, events are intentionally sparse: fill only
                # fields declared by this event, never every variable in the global schema.
                if transactional:
                    event_def = event_defs.get(str(rec.get("event_type")))
                    fill_order = list(event_def.get("fields", []) or []) if event_def else []
                else:
                    fill_order = state.field_order
                formula_field_names = {field for field, _ in formula_specs}
                safe_fill_order = [f for f in fill_order if f not in formula_field_names or rec.get(f) is not None]
                rec, filled = _fill_missing(rec, safe_fill_order, dtype_map)
                if filled:
                    issues.append(f"{filled} event field(s) defaulted")

                # FINAL EDGE-CASE RECONCILIATION: all deterministic QA repairs/fills are
                # complete now.  Re-apply the edge definition one last time so a normal
                # range/choice/default repair cannot invalidate an edge condition.  This
                # is the single source of truth for the edge flag and is intentionally
                # generic for every industry and field name.
                if rec.get("isEdgeCaseData") is True:
                    final_event_def = event_defs.get(str(rec.get("event_type"))) if transactional else None
                    final_event_fields = set(final_event_def.get("fields", []) or []) if final_event_def else set()
                    rec, repaired_ok = _repair_edge_record(
                        rec, edge_groups, variables, profile=profile, rules=state.rules,
                        event_fields=final_event_fields
                    )
                    if not repaired_ok:
                        raise ValueError(
                            "Generated edge-case record failed deterministic validation: "
                            "isEdgeCaseData=true but the configured edge-case condition could "
                            "not be satisfied after final QA reconciliation"
                        )
                    if not any(_safe_edge_condition(g.get("condition", ""), rec)
                               for g in _applicable_edge_groups(rec, edge_groups, transactional) if g.get("condition")):
                        raise ValueError(
                            "Generated edge-case record failed deterministic validation: "
                            "isEdgeCaseData=true but no configured edge-case condition is satisfied"
                        )

                # A later schema/default fill can also change a previously-normal record
                # into an edge state. Reconcile once more after ALL deterministic repairs.
                # Never allow a false label to coexist with a true edge condition.
                if rec.get("isEdgeCaseData") is not True and any(
                    _safe_edge_condition(group.get("condition", ""), rec)
                    for group in _applicable_edge_groups(rec, edge_groups, transactional)
                    if group.get("condition")
                ):
                    rec, normal_ok = _make_non_edge_record(
                        rec, edge_groups, variables, profile=profile, rules=state.rules, transactional=transactional
                    )
                    if not normal_ok or any(
                        _safe_edge_condition(group.get("condition", ""), rec)
                        for group in _applicable_edge_groups(rec, edge_groups, transactional)
                        if group.get("condition")
                    ):
                        raise ValueError(
                            "Generated normal record satisfies a configured edge-case condition "
                            "after final QA reconciliation and no schema-valid non-edge value could be derived dynamically"
                        )

                algo_fixed += len(issues)
                checked.append(rec)
                if transactional:
                    journey_id_for_edge = str(rec.get("journey_id") or "")
                    if journey_id_for_edge:
                        ctx = journey_edge_context.setdefault(journey_id_for_edge, {})
                        for key, value in rec.items():
                            if key not in {"journey_id", "transaction_id", "event_type", "event_sequence",
                                           "event_occurrence", "event_timestamp", "isEdgeCaseData"} and value is not None:
                                ctx[key] = value

            except Exception as exc:
                error = {
                    "record_index": record_index,
                    "error": str(exc),
                    "record": dict(rec) if isinstance(rec, dict) else {},
                }
                state.record_errors.append(error)
                logger.warning("[QA] Skipping invalid record %d: %s", record_index, exc)
                continue

        # Transactional edge-case labels are record-level. A sparse event must
        # satisfy an edge condition using the fields actually present on that
        # record; never promote a match on one event to every event in the journey.
        if transactional and checked:
            for rec in checked:
                if rec.get("isEdgeCaseData") is not True:
                    continue
                matched = any(
                    _safe_edge_condition(group.get("condition", ""), rec)
                    for group in _applicable_edge_groups(rec, edge_groups, transactional)
                    if group.get("condition")
                )
                if not matched:
                    state.record_errors.append({
                        "record_index": checked.index(rec),
                        "error": (
                            "Generated transactional edge-case record failed deterministic validation: "
                            "isEdgeCaseData=true but no configured edge-case condition is satisfied"
                        ),
                        "record": dict(rec),
                    })
                    checked.remove(rec)

        # Optional LLM semantic audit. Deterministic validation above always runs.
        rules_text = "\n".join(f"- {r}" for r in state.rules.get("business_rules", []))
        cross_text = "\n".join(f"- {r}" for r in state.rules.get("cross_field_rules", []))
        system_prompt = _QA_SYSTEM.format(
            rules=rules_text or "Apply the supplied variable definitions and deterministic checks.",
            cross_field_rules=cross_text or "Validate mathematical, temporal, event and business relationships.",
        )

        valid_all: list[dict] = checked
        dropped_all: list[dict] = []
        llm_fixes = 0
        llm_issues = 0
        qa_mode = os.getenv("QA_LLM_MODE", "off").strip().lower()

        if qa_mode == "full":
            valid_all = []
            for i in range(0, len(checked), _CHUNK):
                chunk = checked[i:i + _CHUNK]
                try:
                    result = self._llm.generate_json(
                        system_prompt,
                        f"Scenario: {state.scenario}\n"
                        f"Industry: {state.industry}\nCountry: {state.country or 'GLOBAL'}\n"
                        f"Domain: {state.domain}\nBusiness scenario: {state.business_scenario}\n"
                        f"Business response: {state.business_response or ''}\nExpected outcome: {state.expected_outcome or ''}\n"
                        f"Use case: {state.use_case or ''}\nScenario type: {state.scenario_type or ''}\n"
                        f"Records to validate:\n{json.dumps(chunk, default=str)}",
                        temperature=0.1,
                    )
                    validated = result.get("valid_records", chunk)
                    validated = [{k: r[k] for k in state.field_order if k in r} for r in validated]
                    valid_all.extend(validated)
                    dropped_records = result.get("dropped_records", []) or []
                    dropped_all.extend(dropped_records)
                    for dropped_offset, dropped in enumerate(dropped_records):
                        state.record_errors.append({
                            "record_index": i + dropped_offset,
                            "error": "Record dropped by LLM QA validation",
                            "record": dropped if isinstance(dropped, dict) else {},
                        })
                    llm_fixes += int(result.get("fixes_applied", 0))
                    llm_issues += int(result.get("issues_found", 0))
                except Exception as exc:
                    logger.warning("[QA] Chunk %d error: %s — deterministic validation retained", i, exc)
                    state.errors.append(f"QA chunk {i} error: {exc}")
                    valid_all.extend(chunk)
        elif qa_mode != "off" and checked:
            sample_size = max(1, int(os.getenv("QA_LLM_SAMPLE_SIZE", "20")))
            try:
                result = self._llm.generate_json(
                    system_prompt,
                    f"Scenario: {state.scenario}\n"
                    f"Industry: {state.industry}\nCountry: {state.country or 'GLOBAL'}\n"
                    f"Domain: {state.domain}\nBusiness scenario: {state.business_scenario}\n"
                    f"Use case: {state.use_case or ''}\n"
                    + ("For this Telecom dataset, UPI is forbidden; represent any UPI-related value as DIGITAL_WALLET.\n" if str(state.industry or "").strip().lower() == "telecom" else "For this non-Telecom dataset, preserve valid UPI values; do not apply the Telecom UPI rule.\n")
                    + f"This is a QA audit sample only. Do not rewrite the dataset.\nRecords:\n{json.dumps(checked[:sample_size], default=str)}",
                    temperature=0.1,
                )
                llm_fixes = int(result.get("fixes_applied", 0))
                llm_issues = int(result.get("issues_found", 0))
            except Exception as exc:
                logger.warning("[QA] Sample audit error: %s — deterministic validation retained", exc)
                state.errors.append(f"QA sample error: {exc}")

        # LLM QA is not authoritative for vocabulary. Re-apply the Telecom-only
        # output contract after LLM validation so it cannot reintroduce UPI into
        # Telecom final data. Other industries preserve UPI.
        valid_all = _normalize_upi_in_records(valid_all, state.industry)
        state.final_records = valid_all
        state.validation_report = {
            "total_input": len(records),
            "total_valid": len(valid_all),
            "total_dropped": len(dropped_all) + len(state.record_errors),
            "record_errors": len(state.record_errors),
            "recovered": 0,
            "algo_fixes": algo_fixed,
            "llm_fixes": llm_fixes,
            "llm_issues": llm_issues,
            "deterministic_checks": [
                "schema_and_type",
                "declared_ranges_and_choices",
                "formula_and_arithmetic",
                "timestamp_relationships",
                "country_conventions",
                "entity_and_transaction_consistency",
                "transactional_event_sequence",
            ],
        }
        logger.info(
            "[QA] Done. valid=%d dropped=%d algo_fixes=%d llm_fixes=%d",
            len(valid_all), len(dropped_all), algo_fixed, llm_fixes,
        )
        return state

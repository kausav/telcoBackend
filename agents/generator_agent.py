"""
Agent 3 — Data Generator Agent
Generates records purely algorithmically using the generation rules in
config/variables.py — no LLM call per record. Each generator type maps
directly to the constraints, distributions, and formulas from the Excel BRD.
"""
from __future__ import annotations
import logging
import math
import random
import uuid
import ast
from datetime import datetime, timedelta, timezone

from core.dynamic_scenarios import resolve_variables, resolve_events, resolve_entity_key
from core.llm_client import GeminiClient
from core.state import WorkflowState
from config.industry_profiles import get_profile, country_allowed_value, payment_methods_for_profile

logger = logging.getLogger(__name__)

_MAX_EDGE_CASE_ATTEMPTS = 20

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


# ── LB-02 generator helpers ───────────────────────────────────────────────────

# Deterministic circle-to-timezone mapping (DST-adjusted for summer)
_CIRCLE_TZ: dict[str, str] = {
    "US_EAST_NY":    "America/New_York",
    "US_WEST_CA":    "America/Los_Angeles",
    "US_CENTRAL_TX": "America/Chicago",
    "US_SOUTH_FL":   "America/New_York",
}
_TZ_UTC_OFFSET: dict[str, int] = {
    "America/New_York":    -4,
    "America/Los_Angeles": -7,
    "America/Chicago":     -5,
}
_QUIET_HOURS: set[int] = set(range(21, 24)) | set(range(0, 7))  # 21-23 and 0-6


def _circle_tz_lookup(params: dict, rec: dict) -> str:
    return _CIRCLE_TZ.get(rec.get("telecom_circle_code", "US_EAST_NY"), "America/New_York")


def _lb02_auto_recharge_enabled(params: dict, rec: dict) -> bool:
    mandate = rec.get("auto_recharge_mandate_status")
    if mandate in ("PROCESSING_IN_FLIGHT", "ACTIVE_IDLE"):
        return True
    return random.choices([False, True], weights=[0.85, 0.15], k=1)[0]


def _lb02_suppression_reason(params: dict, rec: dict) -> str:
    if rec.get("auto_recharge_mandate_status") == "PROCESSING_IN_FLIGHT":
        return "AUTO_RECHARGE_ACTIVE"
    choices = ["RECENT_RECHARGE_COOLDOWN", "PENDING_IN_FLIGHT_TXN",
               "FREQUENCY_CAP_EXCEEDED", "AUTO_RECHARGE_ACTIVE", "QUIET_HOURS_RESTRICTION"]
    return random.choices(choices, weights=[0.40, 0.20, 0.20, 0.10, 0.10], k=1)[0]


def _lb02_local_hour(params: dict, rec: dict) -> int:
    if rec.get("suppression_reason_code") == "QUIET_HOURS_RESTRICTION":
        return random.choice(list(_QUIET_HOURS))
    tz = rec.get("subscriber_local_tz", "America/New_York")
    offset = _TZ_UTC_OFFSET.get(tz, -4)
    try:
        hour = (_parse_dt(rec.get("event_timestamp", "")).hour + offset) % 24
        return hour if hour not in _QUIET_HOURS else random.randint(7, 20)
    except Exception:
        return random.randint(7, 20)


def _lb02_hours_since_recharge(params: dict, rec: dict) -> float:
    if rec.get("suppression_reason_code") == "RECENT_RECHARGE_COOLDOWN":
        return round(random.uniform(0.25, 23.99), 2)
    return round(random.uniform(24.01, 720.00), 2)


def _lb02_last_recharge_ts(params: dict, rec: dict) -> str:
    base = _parse_dt(rec.get("event_timestamp", datetime.now(timezone.utc).isoformat()))
    return (base - timedelta(hours=rec.get("hours_since_last_recharge", 24.0))).isoformat()


def _lb02_pending_txn_id(params: dict, rec: dict):
    if rec.get("suppression_reason_code") == "PENDING_IN_FLIGHT_TXN":
        return f"TXN-PG-{random.randint(100_000_000, 999_999_999)}"
    return None


def _lb02_alerts_sent(params: dict, rec: dict) -> int:
    return random.randint(1, 3) if rec.get("suppression_reason_code") == "FREQUENCY_CAP_EXCEEDED" else 0


def _lb02_last_outbound_ts(params: dict, rec: dict):
    if rec.get("alerts_sent_last_24h", 0) == 0:
        return None
    base = _parse_dt(rec.get("event_timestamp", datetime.now(timezone.utc).isoformat()))
    return (base - timedelta(hours=random.uniform(0.5, 23.5))).isoformat()


def _lb02_decision_action(params: dict, rec: dict) -> str:
    reason = rec.get("suppression_reason_code")
    if reason == "QUIET_HOURS_RESTRICTION":
        return "QUEUED_FOR_MORNING"
    if reason in ("RECENT_RECHARGE_COOLDOWN", "AUTO_RECHARGE_ACTIVE"):
        return "SUPPRESSED_SILENT"
    return "DROPPED"


def _lb02_journey_state(params: dict, rec: dict) -> str:
    return "DEFERRED_PENDING" if rec.get("decision_engine_action") == "QUEUED_FOR_MORNING" \
        else "CLOSED_SUPPRESSED"


def _lb02_subsequent_ts(params: dict, rec: dict):
    if rec.get("campaign_attribution_nature") == "UNCONVERTED":
        return None
    base = _parse_dt(rec.get("event_timestamp", datetime.now(timezone.utc).isoformat()))
    return (base + timedelta(hours=random.uniform(0.5, 48.0))).isoformat()


def _lb02_subsequent_channel(params: dict, rec: dict):
    if rec.get("subsequent_topup_timestamp") is None:
        return None
    if rec.get("campaign_attribution_nature") == "AUTO_RECURRING":
        return "AUTO_DEBIT_ACH"
    return random.choices(["MY_ACCOUNT_APP", "RETAILER_POS", "WEB_PORTAL"],
                          weights=[0.60, 0.25, 0.15], k=1)[0]


def _lb02_subsequent_payment(params: dict, rec: dict):
    if rec.get("subsequent_topup_timestamp") is None:
        return None
    if rec.get("campaign_attribution_nature") == "AUTO_RECURRING":
        return "DIRECT_DEBIT"
    return random.choices(["CREDIT_CARD", "DEBIT_CARD", "DIGITAL_WALLET"],
                          weights=[0.40, 0.35, 0.25], k=1)[0]


def _lb02_subsequent_amount(params: dict, rec: dict):
    if rec.get("subsequent_topup_timestamp") is None:
        return None
    return random.choices([10.00, 20.00, 50.00], weights=[0.30, 0.50, 0.20], k=1)[0]


def _lb02_balance_after(params: dict, rec: dict):
    amount = rec.get("subsequent_recharge_amount")
    return None if amount is None else round(rec.get("balance_before", 0.0) + amount, 2)


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
    # LB-02 specific
    "circle_tz_lookup":           lambda v, rec: _circle_tz_lookup(v.get("params", {}), rec),
    "lb02_auto_recharge_enabled": lambda v, rec: _lb02_auto_recharge_enabled(v.get("params", {}), rec),
    "lb02_suppression_reason":    lambda v, rec: _lb02_suppression_reason(v.get("params", {}), rec),
    "lb02_local_hour":            lambda v, rec: _lb02_local_hour(v.get("params", {}), rec),
    "lb02_hours_since_recharge":  lambda v, rec: _lb02_hours_since_recharge(v.get("params", {}), rec),
    "lb02_last_recharge_ts":      lambda v, rec: _lb02_last_recharge_ts(v.get("params", {}), rec),
    "lb02_pending_txn_id":        lambda v, rec: _lb02_pending_txn_id(v.get("params", {}), rec),
    "lb02_alerts_sent":           lambda v, rec: _lb02_alerts_sent(v.get("params", {}), rec),
    "lb02_last_outbound_ts":      lambda v, rec: _lb02_last_outbound_ts(v.get("params", {}), rec),
    "lb02_decision_action":       lambda v, rec: _lb02_decision_action(v.get("params", {}), rec),
    "lb02_journey_state":         lambda v, rec: _lb02_journey_state(v.get("params", {}), rec),
    "lb02_subsequent_ts":         lambda v, rec: _lb02_subsequent_ts(v.get("params", {}), rec),
    "lb02_subsequent_channel":    lambda v, rec: _lb02_subsequent_channel(v.get("params", {}), rec),
    "lb02_subsequent_payment":    lambda v, rec: _lb02_subsequent_payment(v.get("params", {}), rec),
    "lb02_subsequent_amount":     lambda v, rec: _lb02_subsequent_amount(v.get("params", {}), rec),
    "lb02_balance_after":         lambda v, rec: _lb02_balance_after(v.get("params", {}), rec),
}


def get_known_generator_types() -> set[str]:
    """Public accessor for the set of valid 'gen' type strings — used by
    core/csv_scenario.py to validate industry-supplied CSV variable catalogs."""
    return set(_GENERATORS.keys())


def _rule_constraint_for(field_name: str, rules: dict | None) -> dict:
    """Return machine-readable generation constraints for a field.

    RulesAgent produces these once per confirmed scenario. Keeping this lookup
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
        # RulesAgent normally returns a JSON list, but tolerate compact text.
        for sep in ("|", ";", ","):
            if sep in text:
                return [x.strip() for x in text.split(sep) if x.strip()]
        return [text]
    return []


def _apply_generation_constraint(var: dict, value, rec: dict, rules: dict | None):
    """Apply safe machine-readable RulesAgent constraints to a generated value."""
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
        rule_formula = _formula_from_rules(var["name"], rules)
        if rule_formula and var.get("gen") != "formula":
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = rule_formula
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
        rule_formula = _formula_from_rules(name, rules)
        if rule_formula and var.get("gen") != "formula":
            effective_var = dict(var)
            effective_var["gen"] = "formula"
            effective_var["formula"] = rule_formula
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
        entity_context = _generate_selected_record(
            variables, set(compiled.entity_fields), profile=profile, rules=rules
        )
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
                for _attempt in range(_MAX_EDGE_CASE_ATTEMPTS if edge_enabled else 1):
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
                    raise ValueError(
                        "Unable to generate a normal transactional record that does not "
                        "satisfy any configured edge-case condition. The edge-case definitions "
                        "leave no valid normal-domain value for this event."
                    )
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
                base = _generate_selected_record(variables, set(event_def.fields), base=entity_context, profile=profile, rules=rules)
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
            descriptions = "; ".join(f"{name}: {group.get('condition', '')}" for name, group in group_items)
            raise ValueError(
                f"Unable to construct {target} transactional edge-case record(s) after deterministic constraint solving. "
                f"Constructed {len(eligible)}. Definitions: {descriptions}"
            )

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
            raise ValueError(
                f"Edge-case invariant failed for condition '{group.get('condition', '')}' "
                f"on event '{generated[idx].get('event_type')}'."
            )
        candidate["isEdgeCaseData"] = True
        generated[idx] = candidate

    # Exactly the requested number of materialized records are true.
    true_count = sum(1 for row in generated if row.get("isEdgeCaseData") is True)
    if true_count != target:
        raise RuntimeError(
            f"Transactional edge-case assignment invariant failed: expected {target}, got {true_count}"
        )
    return generated


# ── Agent ──────────────────────────────────────────────────────────────────────

class DataGeneratorAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm  # available for future contextual enrichment

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info(
            "[DataGenerator] Context: industry=%s country=%s type=%s domain=%s business_scenario=%s use_case=%s scenario_type=%s entity_key=%s",
            state.industry, state.country or "GLOBAL", state.type_of_data, state.domain,
            state.business_scenario, state.use_case, state.scenario_type, state.entity_key,
        )
        # The confirmed scenario context is the source of truth. RulesAgent has already
        # compiled its business/use-case constraints into state.rules; generation below
        # applies those constraints deterministically without an LLM call per record.
        dyn = resolve_variables(state.scenario)
        if dyn is not None:
            variables, FIELD_ORDER = dyn
        else:
            from config.variables import VARIABLES as variables, FIELD_ORDER

        # Preflight edge-case definitions before any records are generated. A condition
        # that is syntactically valid but mathematically impossible under the declared
        # variable schema must never reach the hot generation path. This is generic and
        # prevents the recurring "unable to generate ... condition" failures caused by
        # incompatible LLM definitions.
        if state.edge_case_variables and state.edge_case_percentage > 0:
            groups = _edge_case_groups(state.edge_case_variables)
            for group_name, group in groups.items():
                condition = str(group.get("condition") or "").strip()
                if condition and not _condition_compatible_with_schema(
                    condition, variables,
                    edge_override_names={str(v.get("name")) for v in group.get("variables", []) if v.get("name")},
                ):
                    raise ValueError(
                        f"Edge case '{group_name}' has an unsatisfiable condition under the "
                        f"declared variable constraints: {condition}. Reconfirm the scenario "
                        "with a valid edge-case definition."
                    )

        if state.type_of_data == "transactional":
            from core.compiled_schema import compile_scenario
            compiled = compile_scenario(state.scenario)
            entity_key = compiled.entity_key
            state.transactional_event_counts = {}
            profile = get_profile(state.industry, state.country)
            records = _transactional_records(compiled, state.count, state.transactional_event_counts, profile=profile, rules=state.rules, edge_case_variables=state.edge_case_variables, edge_case_percentage=state.edge_case_percentage)
            state.field_order = [
                "journey_id", "transaction_id", "event_type", "event_sequence",
                "event_occurrence", "event_timestamp",
            ] + [name for name in FIELD_ORDER if name != "event_timestamp"] + (["isEdgeCaseData"] if "isEdgeCaseData" not in FIELD_ORDER else [])
            logger.info(
                "[DataGenerator] Generated %d transactional records from %d journeys and %d events.",
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
                        raise ValueError(
                            f"Unable to generate an aggregational edge-case record satisfying "
                            f"'{group.get('condition', '')}' after {_MAX_EDGE_CASE_ATTEMPTS} attempts"
                        )
                else:
                    # A record whose actual values satisfy a configured edge condition
                    # cannot be labelled false. Regenerate normal records until they are
                    # outside every edge condition. This makes the flag a deterministic
                    # property of the record, so identical condition-relevant data can
                    # never receive contradictory true/false labels.
                    if edge_groups and state.edge_case_percentage > 0:
                        normal_ok = False
                        for _attempt in range(_MAX_EDGE_CASE_ATTEMPTS):
                            if not _matches_any_edge_condition(rec, edge_groups):
                                normal_ok = True
                                break
                            rec = _generate_record(variables, profile=profile, rules=state.rules)
                        if not normal_ok:
                            raise ValueError(
                                "Unable to generate a normal aggregational record that does not "
                                "satisfy any configured edge-case condition. The edge-case "
                                "definitions leave no valid normal-domain value."
                            )
                    rec["isEdgeCaseData"] = False
                records.append(rec)
            if "isEdgeCaseData" not in state.field_order:
                state.field_order = list(state.field_order) + ["isEdgeCaseData"]
            logger.info("[DataGenerator] Generated %d aggregational records with %d fields each; edge cases=%d.",
                        len(records), len(variables), edge_count)

        state.raw_records = records
        return state

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
from datetime import datetime, timedelta, timezone

from core.dynamic_scenarios import resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState

logger = logging.getLogger(__name__)

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


def _constant(params: dict, _rec: dict):
    return params["value"]


def _weighted_choice(params: dict, _rec: dict):
    return random.choices(params["choices"], weights=params["weights"], k=1)[0]


def _uniform(params: dict, _rec: dict) -> float:
    return round(random.uniform(params["min"], params["max"]), 2)


def _lognormal(params: dict, _rec: dict) -> float:
    raw = math.exp(random.gauss(params["mu"], params["sigma"]))
    clipped = max(params["min"], min(params["max"], raw))
    return round(clipped, 2)


def _lognormal_int(params: dict, _rec: dict) -> int:
    raw = int(math.exp(random.gauss(params["mu"], params["sigma"])))
    return max(params["min"], min(params["max"], raw))


def _beta(params: dict, _rec: dict) -> float:
    return round(random.betavariate(params["alpha"], params["beta"]), 4)


def _segment_range(params: dict, rec: dict) -> float:
    seg = rec.get("subscriber_segment", "Occasional Rechargers")
    rng = params.get(seg, list(params.values())[0])
    return round(random.uniform(rng["min"], rng["max"]), 4)


def _uniform_bounded(params: dict, rec: dict) -> float:
    hi = rec.get(params["hi_field"], 1.00)
    lo = params.get("lo", 0.00)
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


def _generate_record(variables: list[dict]) -> dict:
    """Generate one record by resolving variables in dependency order."""
    rec: dict = {}
    for var in variables:
        gen_type = var["gen"]
        generator = _GENERATORS.get(gen_type)
        if generator:
            rec[var["name"]] = generator(var, rec)
        else:
            rec[var["name"]] = None
    return rec


# ── Agent ──────────────────────────────────────────────────────────────────────

class DataGeneratorAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm  # available for future contextual enrichment

    def run(self, state: WorkflowState) -> WorkflowState:
        dyn = resolve_variables(state.scenario)
        if dyn is not None:
            variables, FIELD_ORDER = dyn
        else:
            from config.variables import VARIABLES as variables, FIELD_ORDER

        state.field_order = FIELD_ORDER

        logger.info("[DataGenerator] Generating %d records for scenario=%s",
                    state.count, state.scenario)

        records = [_generate_record(variables) for _ in range(state.count)]
        state.raw_records = records
        logger.info("[DataGenerator] Done. Generated %d records with %d fields each.",
                    len(records), len(variables))
        return state
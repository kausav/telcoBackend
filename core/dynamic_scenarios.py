"""
Shared store for proposed drafts and confirmed scenario definitions.

Imported scenario drafts and confirmed scenarios are persisted in SQLite.
(not plain process-memory dicts) so that they are visible across all uvicorn
worker processes, not just the one that happened to handle a
/scenario/propose or /scenario/confirm call. They are still wiped if the DB
file is deleted / the volume is reset.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any

from config.runtime import DYNAMIC_SCENARIOS_DB, resolve_path

_DB_PATH = DYNAMIC_SCENARIOS_DB

_DRAFT_TTL_SECONDS = int(os.environ.get("DRAFT_TTL_SECONDS", 24 * 3600))


@contextmanager
def _connect():
    db_path = resolve_path(os.environ.get("DYNAMIC_SCENARIOS_DB"), _DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS drafts (
                draft_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS confirmed (
                scenario_id TEXT PRIMARY KEY,
                draft_id TEXT,
                meta TEXT NOT NULL,
                variables TEXT NOT NULL,
                field_order TEXT NOT NULL
            )"""
        )
        _migrate(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_confirmed_draft_id ON confirmed (draft_id)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                key TEXT NOT NULL,
                feedback TEXT NOT NULL
            )"""
        )


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns to tables that pre-date them, for DB files created before a schema change."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(confirmed)")}
    if "draft_id" not in cols:
        conn.execute("ALTER TABLE confirmed ADD COLUMN draft_id TEXT")


_init_db()


def _purge_expired_drafts(conn: sqlite3.Connection) -> None:
    cutoff = time.time() - _DRAFT_TTL_SECONDS
    conn.execute("DELETE FROM drafts WHERE created_at < ?", (cutoff,))


def _next_scenario_id_conn(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT scenario_id FROM confirmed").fetchall()
    nums = [int(match.group(1)) for value, in rows if (match := re.match(r"LB-(\d+)$", str(value)))]
    return f"LB-{(max(nums) + 1) if nums else 1:02d}"


def next_scenario_id() -> str:
    """Return the next display id. Reservation is performed atomically by confirm_scenario()."""
    with _connect() as conn:
        return _next_scenario_id_conn(conn)


def new_draft_id() -> str:
    return f"draft-{uuid.uuid4().hex}"


def save_draft(draft_id: str, data: dict[str, Any]) -> None:
    with _connect() as conn:
        _purge_expired_drafts(conn)
        conn.execute(
            "INSERT OR REPLACE INTO drafts (draft_id, data, created_at) VALUES (?, ?, ?)",
            (draft_id, json.dumps(data), time.time()),
        )


def get_draft(draft_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        _purge_expired_drafts(conn)
        row = conn.execute("SELECT data FROM drafts WHERE draft_id = ?", (draft_id,)).fetchone()
    return json.loads(row[0]) if row else None


def pop_draft(draft_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT data FROM drafts WHERE draft_id = ?", (draft_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM drafts WHERE draft_id = ?", (draft_id,))
    return json.loads(row[0])


def confirm_scenario(
    scenario_id: str | None,
    meta: dict[str, Any],
    variables: list[dict[str, Any]],
    field_order: list[str],
    draft_id: str | None = None,
) -> tuple[str, bool]:
    """Persist a confirmed scenario without allowing concurrent overwrites.

    If the requested id already exists, a fresh display id is allocated inside the
    same SQLite write transaction and the original id is recorded as requested metadata.
    """
    requested_id = str(scenario_id or "").strip() or None
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        final_id = requested_id or _next_scenario_id_conn(conn)
        reassigned = False
        try:
            conn.execute(
                "INSERT INTO confirmed (scenario_id, draft_id, meta, variables, field_order) VALUES (?, ?, ?, ?, ?)",
                (final_id, draft_id, json.dumps(meta), json.dumps(variables), json.dumps(field_order)),
            )
        except sqlite3.IntegrityError:
            final_id = _next_scenario_id_conn(conn)
            reassigned = True
            conn.execute(
                "INSERT INTO confirmed (scenario_id, draft_id, meta, variables, field_order) VALUES (?, ?, ?, ?, ?)",
                (final_id, draft_id, json.dumps(meta), json.dumps(variables), json.dumps(field_order)),
            )
    return final_id, reassigned


def resolve_scenario_id_from_draft(draft_id: str) -> str | None:
    """Look up the scenario_id a given draft was confirmed into, if any."""
    with _connect() as conn:
        row = conn.execute("SELECT scenario_id FROM confirmed WHERE draft_id = ?", (draft_id,)).fetchone()
    return row[0] if row else None


def get_confirmed(scenario_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT meta, variables, field_order FROM confirmed WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()
    if row is None:
        return None
    return {"meta": json.loads(row[0]), "variables": json.loads(row[1]), "field_order": json.loads(row[2])}


def scenario_exists(scenario_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM confirmed WHERE scenario_id = ?", (scenario_id,)).fetchone()
    return row is not None


def resolve_scenario_meta(scenario_id: str) -> dict[str, Any] | None:
    dyn = get_confirmed(scenario_id)
    return dyn["meta"] if dyn else None


def _repair_legacy_categorical(var: dict[str, Any], norm_name: str) -> None:
    """Correct known legacy enum mismatches where the confirmed field description is authoritative."""
    mapping = {
        "subscriber_segment": ["ULTRA_LOW", "MASS", "MID_TIER", "HIGH_VALUE"],
        "handset_network_capability": ["4G", "5G"],
        "low_balance_trigger_reason": ["LOW_BALANCE", "DATA_EXHAUSTED", "VALIDITY_EXPIRY"],
        "nudge_channel": ["SMS", "WHATSAPP", "APP_PUSH", "IVR"],
        "nudge_offer_type": ["EXTRA_DATA", "CASH_BACK", "VALIDITY_BOOSTER", "DISCOUNT_VOUCHER"],
        "recharge_channel": ["UPI_APP", "TELCO_APP", "RETAIL_POS", "ATM", "NETBANKING", "USSD"],
        "payment_instrument_type": ["UPI", "CREDIT_CARD", "DEBIT_CARD", "PREPAID_WALLET", "CASH", "AUTO_DEBIT"],
        "recharge_pack_category": ["UNLIMITED_COMBO", "DATA_ADDON", "TALKTIME_TOPUP", "INTERNATIONAL_ROAMING", "ISD"],
        "topup_fulfillment_status": ["COMPLETED", "FAILED", "PENDING", "REVERSED"],
    }
    choices = mapping.get(norm_name)
    if choices:
        params = dict(var.get("params") or {})
        params["choices"] = choices
        params["weights"] = [1.0] * len(choices)
        var["params"] = params
        var["gen"] = "weighted_choice"


def _repair_legacy_variables(variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair legacy confirmed variables so generation cannot replay placeholder artifacts."""
    repaired: list[dict[str, Any]] = []
    names = {
        str(raw.get("name") or "").strip().lower()
        for raw in variables or []
        if isinstance(raw, dict)
    }
    has_msisdn = "msisdn" in names

    for raw in variables or []:
        if not isinstance(raw, dict):
            continue
        var = dict(raw)
        name = str(var.get("name") or "").strip()
        if not name:
            continue
        name_key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        norm_name = name_key.replace("_", "")
        dtype = str(var.get("dtype") or "").strip().lower()
        gen = str(var.get("gen") or "").strip().lower()
        params = dict(var.get("params") or {})

        # One canonical subscriber phone identifier: msisdn.
        if has_msisdn and norm_name in {"phonenumber", "mobilenumber", "telephonenumber"}:
            continue

        # Flat generation cannot safely materialize arbitrary nested objects. Drop the
        # legacy generic/empty object contracts instead of returning {}.
        if (
            dtype in {"object", "array"}
            and gen in {"generic", ""}
            and name_key not in {"subscriber_id", "account_id", "msisdn"}
            and not (params.get("choices") or params.get("values") or params.get("value"))
        ):
            continue

        # Correct known semantic/categorical mismatches from earlier confirmed drafts.
        _repair_legacy_categorical(var, name_key)
        dtype = str(var.get("dtype") or dtype).strip().lower()
        gen = str(var.get("gen") or gen).strip().lower()
        params = dict(var.get("params") or {})

        # Mandatory telecom identity anchors are application-level contracts and must never
        # be downgraded to a generic generator during legacy migration.
        if name_key == "subscriber_id":
            var["dtype"] = "string"
            var["gen"] = "prefixed_int"
            var["params"] = {"prefix": "SUB-", "digits": 10}
        elif name_key == "account_id":
            var["dtype"] = "string"
            var["gen"] = "id_mirror"
            var["params"] = {
                "prefix": "ACC-", "source_field": "subscriber_id", "source_prefix": "SUB-"
            }
        elif name_key == "msisdn":
            var["dtype"] = "string"
            var["gen"] = "e164_phone"
            params.setdefault("country_codes", ["+91"])
            params.setdefault("country", "IN")
            var["params"] = params
        # Old schema versions used generic generators that intentionally produced placeholders.
        # Route other legacy generic strings through the safe semantic generator.
        elif dtype in {"string", "str", "text"} and gen in {"generic", ""}:
            var["gen"] = "semantic_string"

        # Preserve booleans as a concrete true/false generator, even when an old draft
        # accidentally attached a categorical lifecycle choice list to a boolean field.
        if dtype not in {"boolean", "bool"} and bool(re.match(r"^(?:is|has)[A-Z]", name)):
            dtype = "boolean"
            var["dtype"] = "boolean"
        if dtype in {"boolean", "bool"}:
            var["dtype"] = "boolean"
            var["gen"] = "weighted_choice"
            var["params"] = {"choices": [False, True], "weights": [0.5, 0.5]}

        # Normalize JSON/OpenAPI's "date-time" format so it can never be used as a literal strftime mask.
        if dtype == "datetime" and str(params.get("format") or "").strip().lower() in {"date-time", "datetime", "timestamp"}:
            params.pop("format", None)
            params.setdefault("timestamp_format", "dd/mm/yyyy hh:mm a")
            var["params"] = params

        repaired.append(var)
    return repaired

def resolve_variables(scenario_id: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return repaired variables and a field order consistent with the repaired contract."""
    dyn = get_confirmed(scenario_id)
    if not dyn:
        return None
    variables = _repair_legacy_variables(dyn["variables"])
    allowed = {str(v.get("name")) for v in variables if isinstance(v, dict) and v.get("name")}
    field_order = [name for name in list(dyn["field_order"]) if str(name) in allowed]
    return variables, field_order



def resolve_scenario_context(scenario_id: str) -> dict[str, Any]:
    """Return the complete confirmed scenario context used by generation agents."""
    meta = resolve_scenario_meta(scenario_id) or {}
    return {
        "scenario_id": scenario_id,
        "label": meta.get("label", scenario_id),
        "journey": meta.get("journey", ""),
        "description": meta.get("description", ""),
        "domain": meta.get("domain", ""),
        "business_scenario": meta.get("business_scenario", ""),
        "business_response": meta.get("business_response"),
        "expected_outcome": meta.get("expected_outcome"),
        "scenario_type": meta.get("scenario_type"),
        "use_case": meta.get("use_case"),
        "industry": meta.get("industry", "generic"),
        "country": meta.get("country"),
        "type_of_data": meta.get("type_of_data", "aggregational"),
        "entity_key": meta.get("entity_key"),
        "records_per_user": int(meta.get("records_per_user", 10) or 10),
        "agentic": bool(meta.get("agentic", False)),
    }


def resolve_data_type(scenario_id: str) -> str:
    """Return the persisted data type for a scenario.

    Old scenarios created before typeOfData was introduced are treated as
    aggregational so existing scenarios continue to work unchanged.
    """
    meta = resolve_scenario_meta(scenario_id) or {}
    value = str(meta.get("type_of_data", "aggregational")).strip().lower()
    return value if value in {"transactional", "aggregational"} else "aggregational"


def resolve_entity_key(scenario_id: str) -> str | None:
    meta = resolve_scenario_meta(scenario_id) or {}
    value = meta.get("entity_key")
    return str(value) if value else None



def list_scenarios() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT scenario_id, meta FROM confirmed").fetchall()
    return [{"id": r[0], **json.loads(r[1])} for r in rows]


def _feedback_key(domain: str, business_scenario: str) -> str:
    return f"{domain.strip().lower()}::{business_scenario.strip().lower()}"


def add_feedback(domain: str, business_scenario: str, feedback: str) -> None:
    if not feedback:
        return
    key = _feedback_key(domain, business_scenario)
    with _connect() as conn:
        conn.execute("INSERT INTO feedback (key, feedback) VALUES (?, ?)", (key, feedback))

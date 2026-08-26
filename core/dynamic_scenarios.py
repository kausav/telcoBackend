"""
Shared store for LLM-proposed scenarios (LB-04, LB-05, ...).

Static scenarios (LB-01/02/03) keep living in config/scenarios.py and
config/variables*.py. Dynamic scenarios proposed via the /scenario/* API
are persisted in a SQLite file (not plain process-memory dicts) so that
they are visible across all uvicorn worker processes, not just the one
that happened to handle the /scenario/propose or /scenario/confirm call.
They are still wiped if the DB file is deleted / the volume is reset.
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

from config.scenarios import SCENARIOS

_DB_PATH = os.environ.get(
    "DYNAMIC_SCENARIOS_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dynamic_scenarios.db"),
)

_DRAFT_TTL_SECONDS = int(os.environ.get("DRAFT_TTL_SECONDS", 24 * 3600))


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30)
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


def next_scenario_id() -> str:
    """Next free LB-0N id, considering both static and confirmed dynamic scenarios."""
    with _connect() as conn:
        rows = conn.execute("SELECT scenario_id FROM confirmed").fetchall()
    existing = list(SCENARIOS.keys()) + [r[0] for r in rows]
    nums = [int(m.group(1)) for k in existing if (m := re.match(r"LB-(\d+)$", k))]
    return f"LB-{(max(nums) + 1) if nums else 1:02d}"


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


def confirm_scenario(scenario_id: str, meta: dict[str, Any],
                      variables: list[dict[str, Any]], field_order: list[str],
                      draft_id: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO confirmed (scenario_id, draft_id, meta, variables, field_order) "
            "VALUES (?, ?, ?, ?, ?)",
            (scenario_id, draft_id, json.dumps(meta), json.dumps(variables), json.dumps(field_order)),
        )


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
    if scenario_id in SCENARIOS:
        return True
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM confirmed WHERE scenario_id = ?", (scenario_id,)).fetchone()
    return row is not None


def resolve_scenario_meta(scenario_id: str) -> dict[str, Any] | None:
    if scenario_id in SCENARIOS:
        return SCENARIOS[scenario_id]
    dyn = get_confirmed(scenario_id)
    return dyn["meta"] if dyn else None


def resolve_variables(scenario_id: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return (variables, field_order) for a dynamic scenario, or None if it's a static one."""
    dyn = get_confirmed(scenario_id)
    return (dyn["variables"], dyn["field_order"]) if dyn else None



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


def resolve_events(scenario_id: str) -> list[dict[str, Any]]:
    """Return transactional event definitions persisted with a scenario."""
    meta = resolve_scenario_meta(scenario_id) or {}
    events = meta.get("events", [])
    return events if isinstance(events, list) else []

def list_scenarios() -> list[dict[str, Any]]:
    out = [{"id": k, **v} for k, v in SCENARIOS.items()]
    with _connect() as conn:
        rows = conn.execute("SELECT scenario_id, meta FROM confirmed").fetchall()
    out += [{"id": r[0], **json.loads(r[1])} for r in rows]
    return out


def _feedback_key(domain: str, business_scenario: str) -> str:
    return f"{domain.strip().lower()}::{business_scenario.strip().lower()}"


def add_feedback(domain: str, business_scenario: str, feedback: str) -> None:
    if not feedback:
        return
    key = _feedback_key(domain, business_scenario)
    with _connect() as conn:
        conn.execute("INSERT INTO feedback (key, feedback) VALUES (?, ?)", (key, feedback))


def get_feedback_history(domain: str, business_scenario: str) -> list[str]:
    key = _feedback_key(domain, business_scenario)
    with _connect() as conn:
        rows = conn.execute("SELECT feedback FROM feedback WHERE key = ?", (key,)).fetchall()
    return [r[0] for r in rows]

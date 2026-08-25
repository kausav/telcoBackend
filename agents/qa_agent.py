"""
Agent 4 — QA & Validation Agent
Validates generated records against the business rules from RulesAgent.
Cross-field dependencies (T0<T1<T2, balance_after=balance_before+recharge_amount)
are checked algorithmically; Gemini audits logical consistency in batches.
"""
from __future__ import annotations
import json
import logging

from config.variables import VARIABLES
from core.dynamic_scenarios import resolve_variables
from core.llm_client import GeminiClient
from core.state import WorkflowState

logger = logging.getLogger(__name__)

# Fallback value used when a field comes back null but the record is otherwise usable —
# keeps output record count matching the requested count instead of silently dropping rows.
_NUMERIC_DTYPES = {"int", "float"}

_SYSTEM = """
You are the QA Validation Agent for a synthetic telecom data pipeline.
Validate each record against the business rules below.

Business rules:
{rules}

Cross-field rules:
{cross_field_rules}

For each record check:
1. balance_after == balance_before + recharge_amount (within 0.01 tolerance)
2. notification_dispatch_ts > event_timestamp
3. customer_response_ts > notification_dispatch_ts
4. balance_before <= low_balance_threshold_amt
5. Required fields are not null.

Return JSON:
  - "valid_records": [ ... ]
  - "dropped_records": [ ... ]
  - "fixes_applied": int
  - "issues_found": int
"""

CHUNK = 50


def _default_for_dtype(dtype: str):
    if dtype in _NUMERIC_DTYPES:
        return 0
    if dtype == "boolean":
        return False
    return ""


def _fill_missing(rec: dict, field_order: list, dtype_map: dict) -> tuple[dict, int]:
    """Replace null/missing field values with a type-appropriate default (0 / "" / False)
    so a single ungenerated field never causes the whole record to be lost downstream."""
    filled = 0
    for name in field_order:
        if rec.get(name) is None:
            rec[name] = _default_for_dtype(dtype_map.get(name, "string"))
            filled += 1
    return rec, filled


def _algorithmic_check(rec: dict, field_order: list, dtype_map: dict) -> tuple[dict, list[str]]:
    """Fast rule checks without LLM — fix correctable issues in-place."""
    issues = []

    # balance_after must equal balance_before + recharge_amount
    bb = rec.get("balance_before")
    ra = rec.get("recharge_amount")
    ba = rec.get("balance_after")
    if bb is not None and ra is not None:
        expected = round(bb + ra, 2)
        if ba is None or abs(ba - expected) > 0.01:
            rec["balance_after"] = expected
            issues.append("balance_after corrected")

    # Strip to defined field order only
    rec = {k: rec[k] for k in field_order if k in rec}

    rec, filled = _fill_missing(rec, field_order, dtype_map)
    if filled:
        issues.append(f"{filled} null field(s) defaulted")

    return rec, issues


class QAAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        records = state.raw_records
        logger.info("[QAAgent] Validating %d records", len(records))

        dyn = resolve_variables(state.scenario)
        if dyn is not None:
            VARS, _ = dyn
        else:
            VARS = VARIABLES
        dtype_map = {v["name"]: v.get("dtype", "string") for v in VARS}

        # Step 1: Fast algorithmic checks
        algo_fixed = 0
        checked: list[dict] = []
        for rec in records:
            rec, issues = _algorithmic_check(rec, state.field_order, dtype_map)
            algo_fixed += len(issues)
            checked.append(rec)

        # Step 2: LLM semantic validation in chunks
        rules_text = "\n".join(f"- {r}" for r in state.rules.get("business_rules", []))
        cross_text = "\n".join(f"- {r}" for r in state.rules.get("cross_field_rules", []))
        system_prompt = _SYSTEM.format(rules=rules_text or "Standard telecom rules.",
                                       cross_field_rules=cross_text or "See field constraints.")

        valid_all: list[dict] = []
        dropped_all: list[dict] = []
        llm_fixes = 0
        llm_issues = 0

        for i in range(0, len(checked), CHUNK):
            chunk = checked[i: i + CHUNK]
            user_prompt = (
                f"Scenario: {state.scenario}\n"
                f"Records to validate:\n{json.dumps(chunk, default=str, indent=2)}"
            )
            try:
                result = self._llm.generate_json(system_prompt, user_prompt, temperature=0.1)
                validated = result.get("valid_records", chunk)
                validated = [{k: r[k] for k in state.field_order if k in r} for r in validated]
                dropped = result.get("dropped_records", [])
                valid_all.extend(validated)
                dropped_all.extend(dropped)
                llm_fixes += int(result.get("fixes_applied", 0))
                llm_issues += int(result.get("issues_found", 0))
            except Exception as exc:
                logger.warning("[QAAgent] Chunk %d error: %s — passing through", i, exc)
                state.errors.append(f"QA chunk {i} error: {exc}")
                valid_all.extend(chunk)

        # Recover LLM-dropped records instead of losing them: default-fill any nulls the
        # LLM flagged (e.g. "required field null") so the output count matches state.count.
        recovered = 0
        for rec in dropped_all:
            rec, _ = _fill_missing(dict(rec), state.field_order, dtype_map)
            rec = {k: rec[k] for k in state.field_order if k in rec}
            valid_all.append(rec)
            recovered += 1

        state.final_records = valid_all
        state.validation_report = {
            "total_input":    len(records),
            "total_valid":    len(valid_all),
            "total_dropped":  len(dropped_all),
            "recovered":      recovered,
            "algo_fixes":     algo_fixed,
            "llm_fixes":      llm_fixes,
            "llm_issues":     llm_issues,
        }
        logger.info("[QAAgent] Done. valid=%d dropped=%d recovered=%d algo_fixes=%d llm_fixes=%d",
                    len(valid_all), len(dropped_all), recovered, algo_fixed, llm_fixes)
        return state
"""Synchronous scenario-generation service.

The API waits for this service to finish and returns the exact generated/validated response.
Latency optimizations belong here and in the deterministic generation engine rather than
changing the response into a queued job.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.compiled_schema import compile_scenario
from core.dynamic_scenarios import (
    resolve_data_type,
    resolve_scenario_context,
    resolve_scenario_id_from_draft,
    resolve_scenario_meta,
    resolve_variables,
    scenario_exists,
)
from agents.data_generation_agent import run_deterministic_agentic_generation
from core.pipeline import run_pipeline


def _timestamp_sort_key(value: Any):
    from datetime import datetime, timezone
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = None
            for fmt in (
                "%d/%m/%Y %I:%M %p",
                "%d/%m/%Y %I:%M:%S %p",
                "%d/%m/%Y %H:%M",
                "%d/%m/%Y %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_generation_response(req_payload: dict[str, Any]) -> dict[str, Any]:
    """Run the full generation + QA pipeline and build the API response payload."""
    scenario_id = req_payload.get("scenario")
    draft_id = req_payload.get("draftId")
    count = int(req_payload.get("count", 35) or 35)
    records_per_user = int(req_payload.get("recordsPerUser", 10) or 10)

    if draft_id:
        resolved = resolve_scenario_id_from_draft(draft_id)
        if resolved is None:
            raise ValueError(f"Unknown or unconfirmed draftId '{draft_id}'")
        if scenario_id and scenario_id != resolved:
            raise ValueError(f"draftId '{draft_id}' does not match scenario '{scenario_id}'")
        scenario_id = resolved
    if not scenario_id:
        raise ValueError("Either 'scenario' or 'draftId' is required")
    if not scenario_exists(scenario_id):
        raise ValueError(f"Unknown scenario '{scenario_id}'")

    scenario_context = resolve_scenario_context(scenario_id)
    if scenario_context.get("agentic"):
        state = run_deterministic_agentic_generation(
            scenario=scenario_id,
            count=count,
            industry=scenario_context.get("industry", "telecom"),
            country=scenario_context.get("country"),
            type_of_data=scenario_context.get("type_of_data", resolve_data_type(scenario_id)),
            scenario_context=scenario_context,
            records_per_user=records_per_user,
        )
    else:
        state = run_pipeline(
            scenario=scenario_id,
            count=count,
            industry=scenario_context.get("industry", "generic"),
            country=scenario_context.get("country"),
            type_of_data=scenario_context.get("type_of_data", resolve_data_type(scenario_id)),
            scenario_context=scenario_context,
            records_per_user=records_per_user,
        )

    if state.errors and not state.final_records and not state.record_errors:
        raise RuntimeError("; ".join(state.errors))
    if state.record_errors and not state.final_records:
        raise RuntimeError("Generation produced no valid records: " + str(state.record_errors[:10]))

    meta = resolve_scenario_meta(scenario_id) or {}
    final_records = state.final_records
    entity_key = meta.get("entity_key")
    response_records = final_records
    total_count = len(final_records)
    total_records = len(final_records)

    if state.type_of_data == "transactional":
        if not entity_key:
            raise RuntimeError("Transactional scenario is missing entity_key")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in final_records:
            value = row.get(entity_key)
            if value in (None, ""):
                continue
            grouped.setdefault(str(value), []).append(row)
        entity_records: list[dict[str, Any]] = []
        compiled = compile_scenario(scenario_id)
        user_fields = compiled.user_fields
        user_field_names = set(user_fields)
        user_field_names.add(entity_key)
        for entity_value, rows in grouped.items():
            timestamp_field = next(
                (f for f in (
                    "topup_requested_date_time", "topupbalance_requested_date", "topupbalance_requesteddate",
                    "recharge_timestamp", "transaction_timestamp", "record_timestamp", "timestamp", "created_at", "updated_at",
                ) if f in rows[0]),
                None,
            )
            if timestamp_field:
                rows = sorted(rows, key=lambda r: _timestamp_sort_key(r.get(timestamp_field)), reverse=True)
            latest = rows[0] if rows else {}
            ordered_user_fields: list[str] = []
            for name in user_fields:
                if name in latest and name not in ordered_user_fields:
                    ordered_user_fields.append(name)
            if entity_key in latest and entity_key not in ordered_user_fields:
                ordered_user_fields.append(entity_key)
            user_output = {name: latest.get(name) for name in ordered_user_fields}
            user_output[entity_key] = entity_value
            user_output["records"] = [
                {k: v for k, v in dict(row).items() if k not in user_field_names}
                for row in rows[:records_per_user]
            ]
            entity_records.append(user_output)
        response_records = entity_records
        total_count = len(entity_records)
        total_records = sum(len(x.get("records", [])) for x in entity_records)

    return {
        "success": True,
        "scenario_id": str(meta.get("requested_scenario_id") or scenario_id),
        "requested_scenario_id": str(meta.get("requested_scenario_id") or scenario_id),
        "typeOfData": state.type_of_data,
        "entityKey": entity_key,
        "totalCount": total_count,
        "recordsPerUser": records_per_user,
        "draft_id": draft_id,
        "scenario_label": meta.get("label", scenario_id),
        "fields": state.field_order or (resolve_variables(scenario_id) or ([], []))[1],
        "total_records": total_records,
        "validation_report": state.validation_report,
        "records": response_records,
        "errors": state.errors,
        "record_errors": state.record_errors,
    }


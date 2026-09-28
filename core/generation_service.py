"""Synchronous scenario-generation service.

The API waits for this service to finish and returns the exact generated/validated response.
Latency optimizations belong here and in the deterministic generation engine rather than
changing the response into a queued job.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.compiled_schema import compile_scenario
from core.dynamic_scenarios import (
    resolve_data_type,
    resolve_scenario_context,
    resolve_requested_scenario_id_from_draft,
    resolve_scenario_meta,
    resolve_variables,
    scenario_exists,
)
from agents.data_generation_agent import run_deterministic_agentic_generation
from core.pipeline import run_pipeline
from core.industry_source_store import list_source_documents
from core.scenario_variable_store import get_recommended

logger = logging.getLogger(__name__)


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



def _require_generation_source(
    requested_scenario_id: str,
    scenario_context: dict[str, Any],
) -> None:
    """Fail closed unless the exact industry/domain has JSON sources or scenario_variables.

    Industry/domain JSON documents are the preferred executable source. The only supported
    fallback is enabled scenario-level recommendations stored in MongoDB. This check is kept in
    the generation service itself so every caller, not just the HTTP route, observes the same
    source-of-truth boundary.
    """
    industry = str(scenario_context.get("industry") or "").strip()
    domain = str(scenario_context.get("domain") or "").strip()
    if not industry or not domain:
        raise ValueError(
            f"Generation is blocked for scenario '{requested_scenario_id}': the confirmed scenario is missing industryType or domain."
        )

    # Avoid reparsing every uploaded Swagger document on every generation call. The confirmed
    # schema already stores source IDs; generation only needs to verify that those source IDs are
    # still active. This keeps generate latency independent of source-document size.
    active_sources = list_source_documents(industry_type=industry, domain=domain, active_only=True)
    source_available = bool(active_sources)
    scenario_version = int(scenario_context.get("scenario_version", 1) or 1)
    scenario_variables = get_recommended(requested_scenario_id, scenario_version)
    if not source_available and not scenario_variables:
        raise ValueError(
            "Generation is blocked because no active industry-standard JSON source documents exist "
            f"in MongoDB for industryType='{industry}', domain='{domain}', and no enabled "
            f"scenario_variables exist for scenario='{requested_scenario_id}', scenarioVersion={scenario_version}. "
            "Upload at least one industry-standard JSON for this industry/domain or save scenario "
            "variables before calling /scenario/generate."
        )

    if source_available:
        active_source_ids = {str(row.get("source_id") or "").strip() for row in active_sources}
        scenario_variable_names = {
            str(row.get("name") or "").strip().casefold()
            for row in scenario_variables
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        }
        confirmed_variables = scenario_context.get("variables") or []
        variable_source_ids = {
            str(name).strip().casefold(): str(source_id).strip()
            for name, source_id in (scenario_context.get("variable_source_ids") or {}).items()
            if str(name).strip() and str(source_id).strip()
        }
        missing_sources: set[str] = set()
        unknown = []
        for row in confirmed_variables:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            source_id = str(provenance.get("source_json_id") or variable_source_ids.get(name.casefold()) or "").strip()
            normalized_source = str(row.get("source") or "").strip().upper()
            if normalized_source == "MONGODB_JSON" and source_id and source_id not in active_source_ids:
                missing_sources.add(source_id)
            elif normalized_source != "MONGODB_JSON" and name.casefold() not in scenario_variable_names and not source_id:
                unknown.append(name)
        if missing_sources:
            raise ValueError(
                "Generation is blocked because confirmed variables reference inactive or removed source documents: "
                + ", ".join(sorted(missing_sources)[:25])
            )
        if unknown:
            raise ValueError(
                "Generation is blocked because the confirmed scenario contains variables outside the current MongoDB source boundary: "
                + ", ".join(sorted(set(unknown))[:25])
            )
    return

def build_generation_response(req_payload: dict[str, Any]) -> dict[str, Any]:
    """Run the full generation + QA pipeline and build the API response payload."""
    requested_scenario_id = req_payload.get("requested_scenario_id")
    draft_id = req_payload.get("draftId")
    count = int(req_payload.get("count", 35) or 35)
    records_per_user = int(req_payload.get("recordsPerUser", 10) or 10)

    if draft_id:
        resolved = resolve_requested_scenario_id_from_draft(draft_id)
        if resolved is None:
            raise ValueError(f"Unknown or unconfirmed draftId '{draft_id}'")
        if requested_scenario_id and requested_scenario_id != resolved:
            raise ValueError(f"draftId '{draft_id}' does not match requested_scenario_id '{requested_scenario_id}'")
        requested_scenario_id = resolved
    if not requested_scenario_id:
        raise ValueError("Either 'requested_scenario_id' or 'draftId' is required")
    if not scenario_exists(requested_scenario_id):
        raise ValueError(f"Unknown requested_scenario_id '{requested_scenario_id}'")

    scenario_context = resolve_scenario_context(requested_scenario_id)
    _require_generation_source(requested_scenario_id, scenario_context)
    if scenario_context.get("agentic"):
        state = run_deterministic_agentic_generation(
            scenario=requested_scenario_id,
            count=count,
            industry=scenario_context.get("industry", "generic"),
            country=scenario_context.get("country"),
            type_of_data=scenario_context.get("type_of_data", resolve_data_type(requested_scenario_id)),
            scenario_context=scenario_context,
            records_per_user=records_per_user,
        )
    else:
        state = run_pipeline(
            scenario=requested_scenario_id,
            count=count,
            industry=scenario_context.get("industry", "generic"),
            country=scenario_context.get("country"),
            type_of_data=scenario_context.get("type_of_data", resolve_data_type(requested_scenario_id)),
            scenario_context=scenario_context,
            records_per_user=records_per_user,
        )

    if state.errors and not state.final_records and not state.record_errors:
        raise RuntimeError("; ".join(state.errors))
    if state.record_errors and not state.final_records:
        raise RuntimeError("Generation produced no valid records: " + str(state.record_errors[:10]))

    meta = resolve_scenario_meta(requested_scenario_id) or {}
    raw_final_records = getattr(state, "final_records", None)
    if isinstance(raw_final_records, dict):
        candidate_records = raw_final_records.get("records")
        if candidate_records is None:
            candidate_records = [raw_final_records]
    elif isinstance(raw_final_records, (list, tuple)):
        candidate_records = list(raw_final_records)
    else:
        candidate_records = []

    final_records = [row for row in candidate_records if isinstance(row, dict)]
    malformed_count = len(candidate_records) - len(final_records)
    if malformed_count:
        logger.error(
            "Generation produced %d malformed record(s) for scenario=%s; refusing to serialize a partial dataset",
            malformed_count,
            requested_scenario_id,
        )
        raise RuntimeError(
            f"Generation produced {malformed_count} malformed record object(s); refusing to return a partial dataset"
        )
    if candidate_records and not final_records:
        raise RuntimeError("Generation produced no valid record objects")

    entity_key = meta.get("entity_key")
    if state.type_of_data == "transactional":
        expected_records = max(1, count) * max(1, records_per_user)
        if len(final_records) != expected_records:
            raise RuntimeError(
                f"Generation produced {len(final_records)} of {expected_records} requested transactional records"
            )
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
        compiled = compile_scenario(requested_scenario_id)
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
        "scenario_id": str(meta.get("requested_scenario_id") or requested_scenario_id),
        "requested_scenario_id": str(meta.get("requested_scenario_id") or requested_scenario_id),
        "typeOfData": state.type_of_data,
        "entityKey": entity_key,
        "totalCount": total_count,
        "recordsPerUser": records_per_user,
        "draft_id": draft_id,
        "scenario_label": meta.get("label", requested_scenario_id),
        "fields": state.field_order or (resolve_variables(requested_scenario_id) or ([], []))[1],
        "total_records": total_records,
        "validation_report": state.validation_report,
        "records": response_records,
        "errors": state.errors,
        "record_errors": state.record_errors,
    }


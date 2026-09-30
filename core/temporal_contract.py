"""Evidence-based temporal relationship utilities for synthetic generation.

Temporal edges are executable constraints. They must be supported by either an explicit
confirmed dependency or an unambiguous same-resource lifecycle pairing. Generic words such as
"request", "confirmation", "start", and "end" are not sufficient to relate two different
business resources.
"""
from __future__ import annotations

import re
from typing import Any


_TEMPORAL_SUFFIXES = (
    "_confirmation_date_time", "_confirmation_datetime", "_confirmation_timestamp", "_confirmation_date",
    "_requested_date_time", "_requested_datetime", "_requested_timestamp", "_requested_date",
    "_created_date_time", "_created_datetime", "_created_timestamp", "_created_at", "_created_date",
    "_creation_date_time", "_creation_datetime", "_creation_timestamp", "_creation_at", "_creation_date",
    "_start_date_time", "_start_datetime", "_start_date", "_start_at",
    "_end_date_time", "_end_datetime", "_end_date", "_end_at",
    "_completion_date_time", "_completion_datetime", "_completion_timestamp", "_completion_at", "_completion_date",
    "_completed_date_time", "_completed_datetime", "_completed_timestamp", "_completed_at", "_completed_date",
    "_processed_date_time", "_processed_datetime", "_processed_timestamp", "_processed_at",
    "_settled_date_time", "_settled_datetime", "_settled_timestamp", "_settled_at",
    "_finished_date_time", "_finished_datetime", "_finished_timestamp", "_finished_at",
    "_decision_timestamp", "_decision_date", "_decision_at",
    "_impression_timestamp", "_impression_date", "_impression_at",
    "_acceptance_timestamp", "_acceptance_date", "_acceptance_at",
    "_conversion_timestamp", "_conversion_date", "_conversion_at",
    "_presentation_timestamp", "_presentation_date", "_presentation_at",
    "_presented_timestamp", "_presented_date", "_presented_at",
    "_response_timestamp", "_response_date", "_response_at",
    "_dispatch_timestamp", "_dispatch_date", "_dispatch_at",
)


def normalize_temporal_family(name: Any) -> str:
    """Return a canonical resource family for a temporal field name.

    Underscore/casing variants are normalized so e.g. ``topupbalance_*`` and
    ``topup_balance_*`` can match, while genuinely different resources such as
    ``adjust_balance_*`` and ``topup_balance_*`` remain separate.
    """
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").casefold()).strip("_")
    for suffix in _TEMPORAL_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip("_")
            break
    return text.replace("_", "")


def _source_model(var: dict[str, Any]) -> str:
    provenance = var.get("provenance") if isinstance(var.get("provenance"), dict) else {}
    return str(
        provenance.get("source_json_model")
        or provenance.get("source_model")
        or var.get("_source_model")
        or ""
    ).strip().casefold()


def _source_identity(var: dict[str, Any]) -> str:
    provenance = var.get("provenance") if isinstance(var.get("provenance"), dict) else {}
    source_id = str(provenance.get("source_json_id") or var.get("_source_id") or "").strip().casefold()
    model = str(
        provenance.get("source_json_model")
        or provenance.get("source_model")
        or var.get("_source_model")
        or ""
    ).strip().casefold()
    if source_id and model:
        return f"{source_id}::{model}"
    return model or source_id


def same_source_model(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    left = _source_model(parent)
    right = _source_model(child)
    return bool(left and right and left == right)


def same_source_resource(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    """Return whether two fields share the same preserved source resource identity."""
    left = _source_identity(parent)
    right = _source_identity(child)
    return bool(left and right and left == right)


def is_temporal_name(name: Any) -> bool:
    """Whether a field name explicitly carries a temporal lifecycle suffix."""
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").casefold()).strip("_")
    return any(text.endswith(suffix) for suffix in _TEMPORAL_SUFFIXES)


def explicit_datetime_dependency(parent_name: str, child: dict[str, Any]) -> bool:
    return any(str(dep).strip() == str(parent_name).strip() for dep in (child.get("depends_on") or []))


def is_supported_temporal_rule(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    """Whether a temporal edge has structural evidence in the confirmed schema.

    A cross-resource datetime dependency is not accepted merely because the model placed the
    parent in ``depends_on``. For source-grounded schemas, temporal dependencies must either
    belong to the same canonical resource family or be represented by a non-temporal business
    formula/dependency mechanism elsewhere in the contract.
    """
    if not parent or not child or parent is child:
        return False
    if str(parent.get("dtype", "")).strip().lower() != "datetime":
        return False
    if str(child.get("dtype", "")).strip().lower() != "datetime":
        return False

    parent_name = str(parent.get("name") or "")
    child_name = str(child.get("name") or "")
    if not parent_name or not child_name:
        return False

    parent_family = normalize_temporal_family(parent_name)
    child_family = normalize_temporal_family(child_name)
    same_family = bool(parent_family and parent_family == child_family)

    # A confirmed source dependency may legitimately use generic field names (for example
    # ``created_at`` -> ``completed_at``). In that case source provenance is stronger evidence
    # than the flattened field-name prefix.
    if explicit_datetime_dependency(parent_name, child):
        return same_family or same_source_resource(parent, child)

    # Inferred lifecycle pairs remain name-structural only. This keeps generic words such as
    # request/confirmation/start/end from linking unrelated sibling resources.
    return same_family


def source_declared_min_delay_seconds(parent: dict[str, Any], child: dict[str, Any]) -> int | None:
    """Extract only an explicit lower-bound delay from source descriptions."""
    text = f"{parent.get('description', '')} {child.get('description', '')}".casefold()
    patterns = (
        r"(?:at\s+least|minimum(?:\s+delay)?(?:\s+of)?)\s+(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)",
        r"not\s+before\s+(?:\+\s*)?(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            return amount * {
                "second": 1, "seconds": 1,
                "minute": 60, "minutes": 60,
                "hour": 3600, "hours": 3600,
                "day": 86400, "days": 86400,
            }[unit]
    return None


def source_declared_max_delay_seconds(parent: dict[str, Any], child: dict[str, Any]) -> int | None:
    """Extract only an explicit source/contract delay bound; never invent one.

    Examples supported by source descriptions are phrases such as ``within 30 minutes`` or
    ``within 2 days``. Generic lifecycle roles do not imply a maximum delay.
    """
    text = f"{parent.get('description', '')} {child.get('description', '')}".casefold()
    match = re.search(r"within\s+(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)", text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        return amount * {
            "second": 1, "seconds": 1,
            "minute": 60, "minutes": 60,
            "hour": 3600, "hours": 3600,
            "day": 86400, "days": 86400,
        }[unit]
    if "same day" in text or "same-day" in text:
        return 86400
    return None


def temporal_relationship_reason(parent: dict[str, Any], child: dict[str, Any]) -> str:
    if explicit_datetime_dependency(str(parent.get("name") or ""), child):
        return "Confirmed datetime dependency implies chronological causality."
    return "Same canonical resource family implies lifecycle ordering."

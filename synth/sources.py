"""What the industry source documents say about the columns of a scenario.

A column that came from an industry standard (TMF, FHIR, BIAN, ACORD, ...) carries more meaning than its name and
type: the resource it belongs to, that resource's own description, and its path inside the standard. The designer of a
scenario's behaviour reads this so that, for example, a "history" resource is understood as the history of the main
resource and not as an unrelated second record. The documents are the same ``industry_source_documents`` the proposal
used; nothing here is specific to an industry.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_RESOURCE_TEXT = 260


def source_notes(brief: dict[str, Any], variables: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """``(notes, resources)`` for the columns that exist in the source catalog of the scenario's industry and domain.

    ``notes``: column name -> ``{"resource": ..., "standard_path": ...}``; ``resources``: resource -> its description.
    Never raises: a missing catalog or an unreachable database simply yields no notes.
    """
    industry, domain = str(brief.get("industry") or "").strip(), str(brief.get("domain") or "").strip()
    if not industry or not domain or not variables:
        return {}, {}
    try:
        from core.industry_source_store import catalog_for_request

        catalog = catalog_for_request(industry, domain)
    except Exception as exc:
        logger.warning("source context unavailable for %s / %s: %s: %s", industry, domain, type(exc).__name__, exc)
        return {}, {}
    from core.variable_semantics import variable_semantic_aliases

    rows = [r for r in catalog.get("models") or [] if isinstance(r, dict) and r.get("name")]
    rows.sort(key=lambda r: (str(r.get("model_kind") or "resource") != "resource", len(str(r.get("path") or ""))))
    by_alias: dict[str, dict[str, Any]] = {}
    for row in rows:
        for alias in variable_semantic_aliases(row["name"]):
            by_alias.setdefault(alias, row)
    notes: dict[str, dict[str, Any]] = {}
    resources: dict[str, str] = {}
    for variable in variables:
        name = str(variable.get("name") or "") if isinstance(variable, dict) else ""
        row = next((by_alias[a] for a in sorted(variable_semantic_aliases(name)) if a in by_alias), None) if name else None
        if row is None:
            continue
        resource = str(row.get("business_model") or row.get("model") or "").strip()
        note = {k: v for k, v in (("resource", resource), ("standard_path", str(row.get("path") or "").strip())) if v}
        if note:
            notes[name] = note
        description = " ".join(str(row.get("model_description") or "").split())
        if resource and description and resource not in resources:
            resources[resource] = description[:MAX_RESOURCE_TEXT]
    return notes, resources

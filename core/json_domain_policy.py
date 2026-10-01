"""Machine-readable industry/domain source policy.

Source documents live in MongoDB, keyed by ``industryType`` and ``domain``, and may contain one or
more Swagger/OpenAPI/JSON-Schema documents.

There is no bundled industry/domain source fallback. Active MongoDB source documents are the only
source of truth for executable industry/domain JSON schemas.
"""
from __future__ import annotations

from typing import Any

from core.industry_source_store import (
    catalog_for_request as _catalog_for_request,
    has_sources,
    source_manifest as _source_manifest,
)


def is_json_grounded_domain(value: str | None, industry_type: str | None = None) -> bool:
    """Return True only when active MongoDB source documents exist for the exact pair.

    Fails closed when the source registry has no active documents for the requested pair.
    """
    if not industry_type or not str(value or "").strip():
        return False
    return has_sources(industry_type, value)


def catalog_for_request(industry_type: str, domain: str) -> dict[str, Any]:
    """Return the complete source-backed catalog and source provenance for a request."""
    return _catalog_for_request(industry_type, domain)


def source_manifest(industry_type: str, domain: str) -> list[dict[str, Any]]:
    return _source_manifest(industry_type, domain)

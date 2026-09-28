"""Machine-readable industry/domain source policy.

The original implementation hard-coded two TM Forum files for Low Balance & Top-up. The
operational source of truth is now MongoDB: source documents are keyed by ``industryType`` and
``domain`` and may contain one or more Swagger/OpenAPI/JSON-Schema documents.

There is no bundled industry/domain source fallback. Active MongoDB source documents are the only
source of truth for executable industry/domain JSON schemas.
"""
from __future__ import annotations

from typing import Any

from core.industry_source_store import (
    catalog_for_request as _catalog_for_request,
    has_sources,
    normalize_domain_key,
    normalize_industry_key,
    source_manifest as _source_manifest,
)

LOW_BALANCE_DOMAIN_ALIASES = {
    "low balance & top-up",
    "low balance and top up",
    "low balance and top-up",
    "prepay balance",
    "prepay balance management",
    "top up",
    "top-up",
    "recharge balance",
}

LOW_BALANCE_SOURCE_IDS = ("tmf654_v4", "tmf629_v4")
LOW_BALANCE_MAIN_MODEL_IDS = (
    "tmf654_v4__topup_balance",
    "tmf654_v4__bucket",
    "tmf629_v4__customer",
)
def normalize_domain(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def is_low_balance_domain(value: str | None, industry_type: str | None = None) -> bool:
    text = normalize_domain(value)
    if industry_type and normalize_industry_key(industry_type) != "telecom":
        return False
    return text in LOW_BALANCE_DOMAIN_ALIASES or (
        "low balance" in text and any(token in text for token in ("top", "recharge"))
    )


def is_json_grounded_domain(value: str | None, industry_type: str | None = None) -> bool:
    """Return True only when active MongoDB source documents exist for the exact pair.

    Historical domain aliases are used only to select business-policy code. They never bypass
    the MongoDB source check. This deliberately fails closed when the source registry has no
    active documents for the requested industry/domain pair.
    """
    if not industry_type or not str(value or "").strip():
        return False
    return has_sources(industry_type, value)


def expanded_scalar_catalog(
    industry_type: str = "Telecommunications",
    domain: str = "Low Balance & Top-up",
) -> list[dict[str, Any]]:
    """Return the active scalar catalog for an exact industry/domain pair."""
    catalog = _catalog_for_request(industry_type, domain)
    return [dict(row) for row in catalog.get("models") or []]


def catalog_for_request(industry_type: str, domain: str) -> dict[str, Any]:
    """Return the complete source-backed catalog and source provenance for a request."""
    return _catalog_for_request(industry_type, domain)


def source_manifest(
    industry_type: str = "Telecommunications",
    domain: str = "Low Balance & Top-up",
) -> list[dict[str, Any]]:
    return _source_manifest(industry_type, domain)


# Kept for backward compatibility with callers that inspect these helpers. They intentionally
# do not read the filesystem; an empty result simply means the requested source is not in Mongo.
def _material_scalar_spec(row: dict[str, Any]) -> bool:
    dtype = str(row.get("dtype") or "string").lower()
    return bool(str(row.get("name") or "").strip()) and dtype not in {"object", "array"}

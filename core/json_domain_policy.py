"""Machine-readable industry/domain source policy.

The original implementation hard-coded two TM Forum files for Low Balance & Top-up. The
operational source of truth is now MongoDB: source documents are keyed by ``industryType`` and
``domain`` and may contain one or more Swagger/OpenAPI/JSON-Schema documents.

The bundled Low Balance files remain only as a one-time bootstrap migration. No proposal or
runtime generation path reads their filesystem paths directly.
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
LOW_BALANCE_SOURCE_NAMES = {
    "tmf654_v4": "TMF654 Prepay Balance Management API v4.0.0 Swagger",
    "tmf629_v4": "TMF629 Customer Management API v4.0.0 Swagger",
}
LOW_BALANCE_SOURCE_FILES = {
    "tmf654_v4": "TMF654_Prepay_Balance_Management_API_v4.0.0_swagger.json",
    "tmf629_v4": "TMF629_Customer_Management_API_v4.0.0_swagger.json",
}


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
    """Return whether the exact request is backed by active MongoDB JSON source documents.

    Low Balance remains recognized by its historical aliases so legacy confirmed scenarios keep
    their special business rules. Other industries/domains are JSON-grounded only when MongoDB
    actually contains active documents for the supplied pair.
    """
    if is_low_balance_domain(value, industry_type):
        return True
    if industry_type and str(value or "").strip():
        return has_sources(industry_type, value)
    return False


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

"""Country-only metadata for HTTP/CSV convenience logic.

This module is never provided to an LLM and contains no industry/domain standards.
"""
from __future__ import annotations

COUNTRY_BASE: dict[str, dict[str, str]] = {
    "GLOBAL": {"country_name": "Global / not country-specific", "currency": "USD", "phone_country_code": "+1"},
    "US": {"country_name": "United States", "currency": "USD", "phone_country_code": "+1"},
    "IN": {"country_name": "India", "currency": "INR", "phone_country_code": "+91"},
    "GB": {"country_name": "United Kingdom", "currency": "GBP", "phone_country_code": "+44"},
    "AE": {"country_name": "United Arab Emirates", "currency": "AED", "phone_country_code": "+971"},
    "CA": {"country_name": "Canada", "currency": "CAD", "phone_country_code": "+1"},
    "AU": {"country_name": "Australia", "currency": "AUD", "phone_country_code": "+61"},
    "SG": {"country_name": "Singapore", "currency": "SGD", "phone_country_code": "+65"},
    "DE": {"country_name": "Germany", "currency": "EUR", "phone_country_code": "+49"},
    "FR": {"country_name": "France", "currency": "EUR", "phone_country_code": "+33"},
    "IT": {"country_name": "Italy", "currency": "EUR", "phone_country_code": "+39"},
}

"""
Country + industry conventions used to ground LLM-generated synthetic data in
real-world standards, instead of implicit US/telecom-only assumptions.

Two layers, merged by get_profile(industry, country):
  - COUNTRY_BASE: country-level facts that apply regardless of industry
    (currency, phone numbering format, country name).
  - INDUSTRY_NOTES: industry+country-specific facts (regulator, market
    character, typical product/plan types, identity/KYC conventions,
    typical transaction denominations). Telecom is fully populated as the
    original domain this project was built for; add more industries here
    (banking, retail, healthcare, insurance, ...) as they're needed —
    unknown industries fall back to a generic per-country entry so the
    pipeline still works, just with less domain-specific detail.

Every agent that needs industry/country grounding calls get_profile() rather
than hardcoding "telecom" anywhere.
"""
from __future__ import annotations

# Country is intentionally optional. Missing country means GLOBAL/non-country-specific data.
DEFAULT_INDUSTRY = "generic"
GLOBAL_COUNTRY = "GLOBAL"

COUNTRY_BASE: dict[str, dict] = {
    "GLOBAL": {
        "country_name": "Global / not country-specific",
        "currency": "USD",
        "phone_country_code": "+1",
        "phone_format": "Generic E.164-style mobile number (no specific national numbering plan enforced)",
        "payment_methods": ["credit_card", "debit_card", "bank_transfer", "digital_wallet", "cash"],
    },
    "US": {
        "country_name": "United States",
        "currency": "USD",
        "phone_country_code": "+1",
        "phone_format": "NANP: +1 followed by 3-digit area code (200-999) + 3-digit exchange (200-999) + 4-digit line number",
        "payment_methods": ["credit_card", "debit_card", "bank_account", "bank_transfer", "ach", "digital_wallet", "paypal", "cash"],
    },
    "IN": {
        "country_name": "India",
        "currency": "INR",
        "phone_country_code": "+91",
        "phone_format": "10-digit mobile number starting with 6, 7, 8, or 9 (no leading 0 in E.164 form)",
        "payment_methods": ["upi", "credit_card", "debit_card", "net_banking", "bank_transfer", "digital_wallet", "cash", "cod"],
    },
    "GB": {
        "country_name": "United Kingdom",
        "currency": "GBP",
        "phone_country_code": "+44",
        "phone_format": "UK mobile: +44 followed by 10 digits, first digit 7 (e.g. 7xxx xxxxxx)",
        "payment_methods": ["credit_card", "debit_card", "bank_transfer", "direct_debit", "digital_wallet", "paypal", "cash"],
    },
    "AE": {
        "country_name": "United Arab Emirates",
        "currency": "AED",
        "phone_country_code": "+971",
        "phone_format": "+971 5X XXX XXXX (mobile prefixes 50, 52, 54, 55, 56, 58)",
        "payment_methods": ["credit_card", "debit_card", "bank_transfer", "digital_wallet", "cash"],
    },
}

# Keys are lower-cased industry names; match loosely against user-supplied industryType.
INDUSTRY_NOTES: dict[str, dict[str, dict]] = {
    "telecom": {
        "US": {
            "regulator": "FCC (Federal Communications Commission)",
            "market_character": "Postpaid-dominant; prepaid mainly MVNOs and budget segments",
            "product_types": ["Postpaid Unlimited", "Postpaid Family Share", "Prepaid Pay-as-you-go", "Prepaid Monthly"],
            "identity_notes": "No national biometric SIM KYC requirement; account tied to billing address/credit check for postpaid",
            "typical_denominations": [10.00, 25.00, 50.00, 100.00],
        },
        "IN": {
            "regulator": "TRAI (Telecom Regulatory Authority of India)",
            "market_character": "Prepaid-dominant (~95% of subscribers); postpaid a small premium segment",
            "product_types": ["Prepaid Combo (data+voice+SMS)", "Prepaid Unlimited Daily Data", "Postpaid Family Plan", "Prepaid Top-up Voucher"],
            "identity_notes": "Mandatory Aadhaar/e-KYC or biometric verification for SIM issuance; circle-based numbering",
            "typical_denominations": [10.00, 19.00, 49.00, 99.00, 199.00, 299.00, 599.00],
        },
        "GB": {
            "regulator": "Ofcom",
            "market_character": "Mixed prepaid (PAYG)/postpaid (SIM-only and contract) market",
            "product_types": ["Pay Monthly SIM-only", "Pay Monthly Handset Contract", "Pay As You Go (PAYG)"],
            "identity_notes": "No mandatory biometric KYC; identity/credit check required for postpaid contracts",
            "typical_denominations": [10.00, 15.00, 20.00, 30.00],
        },
        "AE": {
            "regulator": "TDRA (Telecommunications and Digital Government Regulatory Authority)",
            "market_character": "Prepaid-dominant among expatriate population; postpaid common for residents/business",
            "product_types": ["Prepaid Visitor SIM", "Prepaid Data+Voice Bundle", "Postpaid Family Plan"],
            "identity_notes": "Mandatory Emirates ID verification for SIM registration",
            "typical_denominations": [10.00, 25.00, 55.00, 100.00],
        },
    },
    "banking": {
        "US": {
            "regulator": "OCC / FDIC / Federal Reserve",
            "market_character": "Mature retail + digital banking; checking/savings dominant, high card usage",
            "product_types": ["Checking Account", "Savings Account", "Credit Card", "Personal Loan", "Certificate of Deposit"],
            "identity_notes": "KYC via SSN + government ID under the Bank Secrecy Act / CIP rules",
            "typical_denominations": [20.00, 50.00, 100.00, 500.00, 1000.00],
        },
        "IN": {
            "regulator": "RBI (Reserve Bank of India)",
            "market_character": "Rapid UPI/digital-payments growth alongside traditional branch banking; large unbanked-to-banked transition",
            "product_types": ["Savings Account", "Current Account", "UPI-linked Account", "Fixed Deposit", "Personal Loan"],
            "identity_notes": "Mandatory Aadhaar/PAN-based KYC",
            "typical_denominations": [100.00, 500.00, 1000.00, 5000.00],
        },
    },
    "retail": {
        "US": {
            "regulator": "FTC (consumer protection); state sales tax authorities",
            "market_character": "Omnichannel (in-store + e-commerce), high card/digital-wallet payment share",
            "product_types": ["In-Store Purchase", "Online Order", "Buy-Online-Pickup-In-Store", "Loyalty Program Redemption"],
            "identity_notes": "No mandatory identity verification for purchases; loyalty programs use opt-in email/phone",
            "typical_denominations": [9.99, 19.99, 49.99, 99.99],
        },
        "IN": {
            "regulator": "CCPA (Central Consumer Protection Authority); GST authorities",
            "market_character": "Fast-growing e-commerce alongside dominant unorganized/local retail; COD still significant",
            "product_types": ["In-Store Purchase", "Online Order (COD)", "Online Order (Prepaid)", "Loyalty Program Redemption"],
            "identity_notes": "No mandatory identity verification for purchases",
            "typical_denominations": [99.00, 499.00, 999.00, 1999.00],
        },
    },
}

_GENERIC_NOTES = {
    "regulator": "Not country-specific — no single national regulator assumed",
    "market_character": "Generic/global market conventions (no single country's standards enforced)",
    "product_types": ["Standard Product", "Premium Product"],
    "identity_notes": "No specific identity/KYC convention modeled (country not specified)",
    "typical_denominations": [10.00, 25.00, 50.00, 100.00],
    "payment_methods": ["credit_card", "debit_card", "bank_transfer", "digital_wallet", "cash"],
}


def match_industry_key(industry: str | None) -> str:
    key = (industry or DEFAULT_INDUSTRY).strip().lower()
    if key in INDUSTRY_NOTES:
        return key
    for known in INDUSTRY_NOTES:
        if known in key or key in known:
            return known
    return DEFAULT_INDUSTRY


def payment_methods_for_profile(profile: dict) -> list[str]:
    """Return country-appropriate payment methods for generation/validation.

    Values are normalized to lowercase snake_case so they can be compared against
    generator choices such as ``UPI`` or ``CREDIT_CARD``.
    """
    return list(profile.get("payment_methods", _GENERIC_NOTES["payment_methods"]))


def country_allowed_value(field_name: str, value, profile: dict):
    """Return True when a value is compatible with a known country-sensitive field.

    Unknown fields are left alone. This deliberately avoids pretending that every
    categorical field is country-specific.
    """
    if value is None:
        return True
    name = str(field_name or "").strip().lower()
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if "currency" in name:
        currency = str(profile.get("currency", "")).strip().lower()
        return normalized in {currency, profile.get("currency", "").upper().lower()}
    if "payment" in name or "pay_method" in name or name in {"payment_method", "payment_method_type"}:
        return normalized in {m.lower().replace(" ", "_").replace("-", "_") for m in payment_methods_for_profile(profile)}
    return True


def get_profile(industry: str | None, country: str | None) -> dict:
    """Merge country-level facts with industry-specific notes for the given pair.
    If country is omitted, uses the GLOBAL pseudo-country (generic, not tied to any
    real country's regulator/market conventions) rather than silently assuming a
    specific country like US. An unrecognized country code also falls back to GLOBAL."""
    country_code = (country or GLOBAL_COUNTRY).strip().upper()
    base = dict(COUNTRY_BASE.get(country_code, COUNTRY_BASE[GLOBAL_COUNTRY]))

    industry_key = match_industry_key(industry)
    notes = INDUSTRY_NOTES.get(industry_key, {}).get(country_code)
    if notes is None:
        notes = _GENERIC_NOTES

    profile = {**base, **notes}
    profile["industry"] = industry or DEFAULT_INDUSTRY
    profile["country_code"] = country_code
    return profile
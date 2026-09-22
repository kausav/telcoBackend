"""PDF-only semantic policy for the supplied TMF654/TMF635 guides.

When the requested domain is ``Low Balance & Top-up``, proposal semantics are
restricted to the two user guides supplied with this application.  This module
contains the small flat/scalar catalog used by the agentic proposal path. Nested
reference/list objects are intentionally not emitted as flat CSV fields.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Any

PDF_SOURCE_IDS = {"tmf654_v4", "tmf635_v4"}
PDF_SOURCE_STANDARDS = {"TMF654", "TMF635"}
PDF_SOURCE_NAMES = {
    "tmf654_v4": "TMF654 Prepay Balance Management API User Guide v4.0.0",
    "tmf635_v4": "TMF635 Usage Management API User Guide v4.0.0",
}
PDF_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources" / "telecom" / "pdf_sources"

PDF_DOMAIN_ALIASES = {
    "low balance & top-up",
    "low balance and top up",
    "low balance and top-up",
    "prepay balance",
    "prepay balance management",
    "top up",
    "top-up",
    "recharge balance",
    "usage management",
}

# Only scalar fields from the supplied guides are exposed.  Quantity fields are
# represented by their documented Quantity.amount / Quantity.units members;
# arbitrary nested reference/list objects are excluded from the flat record contract.
PDF_CATALOG: dict[str, dict[str, Any]] = {
    "bucket": {
        "source_id": "tmf654_v4", "standard": "TMF654", "name": "Bucket",
        "description": "A bucket represents and tracks a quantity of usage remaining or consumed, including a monetary amount.",
        "fields": {
            "href": ("string", "Hyperlink/resource URI for the bucket."),
            "id": ("string", "Unique identifier within the server for the bucket."),
            "@baseType": ("string", "Super-class when sub-classing."),
            "@schemaLocation": ("string", "URI to a JSON Schema defining additional attributes and relationships."),
            "@type": ("string", "Extensible sub-class name."),
            "description": ("string", "Text describing the contents of the balance managed by the bucket."),
            "isShared": ("boolean", "Whether the bucket is shared between devices or users."),
            "name": ("string", "Friendly name identifying the bucket."),
            "remainingValue": ("float", "Numeric amount remaining on the bucket; Quantity.amount."),
            "remainingValue_units": ("string", "Unit for remainingValue; Quantity.units."),
            "remainingValueName": ("string", "Formatted remaining amount for display."),
            "reservedValue": ("float", "Numeric amount reserved on the bucket; Quantity.amount."),
            "reservedValue_units": ("string", "Unit for reservedValue; Quantity.units."),
            "status": ("categorical", "Bucket status; the guide gives active, expired and suspended as examples."),
            "usageType": ("string", "Type of the underlying balance, such as data, voice or currency."),
            "validFor": ("string", "Time period for which the balance in the bucket is valid."),
        },
        "choices": {"status": ["active", "expired", "suspended"]},
    },
    "topupbalance": {
        "source_id": "tmf654_v4", "standard": "TMF654", "name": "TopupBalance",
        "description": "A top-up operation on a prepay balance bucket.",
        "fields": {
            "confirmationDate": ("datetime", "Date when the deduction was confirmed in the server."),
            "description": ("string", "Description of the recharge operation."),
            "href": ("string", "Hyperlink reference."),
            "id": ("string", "Unique identifier."),
            "reason": ("string", "Text describing the reason for the action/task."),
            "requestedDate": ("datetime", "Date when the deduction request was received in the server."),
            "status": ("string", "Status of the operation."),
            "usageType": ("string", "Type of the underlying balance."),
            "@baseType": ("string", "Super-class when sub-classing."),
            "@schemaLocation": ("string", "URI to a JSON Schema defining additional attributes and relationships."),
            "@type": ("string", "Extensible sub-class name."),
            "isAutoTopup": ("boolean", "Whether the requested top-up is an auto-top-up processed periodically."),
            "numberOfPeriods": ("integer", "Number of occurrences for an auto-top-up; if absent, no limit is set."),
            "recurringPeriod": ("string", "Periodicity for an auto-top-up, such as monthly or weekly."),
            "voucher": ("string", "Identifier for a voucher when the top-up is performed by this means."),
            "amount": ("float", "Positive numeric amount from the Quantity sub-resource."),
            "amount_units": ("string", "Unit from the Quantity sub-resource."),
        },
        "choices": {},
    },
    "transferbalance": {
        "source_id": "tmf654_v4", "standard": "TMF654", "name": "TransferBalance",
        "description": "Transfer of a monetary amount from a source bucket to a colleague/target bucket.",
        "fields": {
            "confirmationDate": ("datetime", "Date when the deduction was confirmed in the server."),
            "description": ("string", "Description of the transfer operation."),
            "href": ("string", "Hyperlink reference."),
            "id": ("string", "Unique identifier."),
            "reason": ("string", "Text describing the reason for the action/task."),
            "requestedDate": ("datetime", "Date when the deduction request was received in the server."),
            "status": ("string", "Status of the operation."),
            "usageType": ("string", "Type of the underlying balance."),
            "@baseType": ("string", "Super-class when sub-classing."),
            "@schemaLocation": ("string", "URI to a JSON Schema defining additional attributes and relationships."),
            "@type": ("string", "Extensible sub-class name."),
            "costOwner": ("string", "Related party bearing transfer costs, for example originator or receiver."),
            "receiverBucketUsageType": ("string", "Type of the receiver prepay balance bucket."),
            "amount": ("float", "Positive numeric transfer amount from the Quantity sub-resource."),
            "amount_units": ("string", "Unit from the Quantity sub-resource."),
            "transferCost": ("float", "Numeric transfer cost from the Quantity sub-resource."),
            "transferCost_units": ("string", "Unit from the transferCost Quantity sub-resource."),
        },
        "choices": {},
    },
    "reservebalance": {
        "source_id": "tmf654_v4", "standard": "TMF654", "name": "ReserveBalance",
        "description": "Reservation of an amount on a bucket.",
        "fields": {
            "confirmationDate": ("datetime", "Date when the deduction was confirmed in the server."),
            "description": ("string", "Description of the reservation operation."),
            "href": ("string", "Hyperlink reference."),
            "id": ("string", "Unique identifier."),
            "reason": ("string", "Text describing the reason for the action/task."),
            "requestedDate": ("datetime", "Date when the deduction request was received in the server."),
            "status": ("string", "Status of the operation."),
            "usageType": ("string", "Type of the underlying balance."),
            "@baseType": ("string", "Super-class when sub-classing."),
            "@schemaLocation": ("string", "URI to a JSON Schema defining additional attributes and relationships."),
            "@type": ("string", "Extensible sub-class name."),
            "amount": ("float", "Positive numeric reservation amount from the Quantity sub-resource."),
            "amount_units": ("string", "Unit from the Quantity sub-resource."),
        },
        "choices": {},
    },
    "usage": {
        "source_id": "tmf635_v4", "standard": "TMF635", "name": "Usage",
        "description": "An occurrence of employing a Product, Service or Resource for its intended purpose that is of business interest.",
        "fields": {
            "href": ("string", "Hyperlink reference."),
            "id": ("string", "Unique identifier."),
            "@baseType": ("string", "Super-class when sub-classing."),
            "@schemaLocation": ("string", "URI to a JSON Schema defining additional attributes and relationships."),
            "@type": ("string", "Extensible sub-class name."),
            "description": ("string", "Description of usage."),
            "status": ("categorical", "Usage status type."),
            "usageDate": ("datetime", "Date and time of usage."),
            "usageType": ("string", "Type of usage."),
        },
        "choices": {"status": ["Received", "Rejected", "Guided", "Rated", "Rerated", "Billed", "Recycled"]},
    },
    "usagespecification": {
        "source_id": "tmf635_v4", "standard": "TMF635", "name": "UsageSpecification",
        "description": "A detailed description of a usage event, including characteristics for a particular type of usage.",
        "fields": {
            "description": ("string", "Description of the specification."),
            "isBundle": ("boolean", "Whether the specification represents a single specification or a bundle."),
            "lastUpdate": ("datetime", "Date and time of the last update."),
            "lifecycleStatus": ("string", "Current lifecycle status of the catalog item."),
            "name": ("string", "Name given to the specification."),
            "version": ("string", "Specification version."),
            "href": ("string", "Hyperlink reference."),
            "id": ("string", "Unique identifier."),
            "@baseType": ("string", "Super-class when sub-classing."),
            "@schemaLocation": ("string", "URI to a JSON Schema defining additional attributes and relationships."),
            "@type": ("string", "Extensible sub-class name."),
            "validFor": ("string", "Time period for which the REST resource is valid."),
        },
        "choices": {},
    },
}

USE_CASES = {
    "tmf654_uc1": "TMF654 Use Case 1 - One-time top-up",
    "tmf654_uc2": "TMF654 Use Case 2 - Recurring auto top-up",
    "tmf654_uc3": "TMF654 Use Case 3 - Cancel recurring top-up",
    "tmf654_uc4": "TMF654 Use Case 4 - Transfer a monetary amount to a colleague",
    "tmf654_uc5": "TMF654 Use Case 5 - Reserve an amount on a bucket",
    "tmf635_uc_usage": "TMF635 sample use case - Create/manage a Usage instance",
    "tmf635_uc_spec": "TMF635 sample use case - Create/manage a UsageSpecification",
}


def normalize_domain(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def is_pdf_grounded_domain(value: str | None) -> bool:
    text = normalize_domain(value)
    return text in PDF_DOMAIN_ALIASES or "low balance" in text and ("top" in text or "recharge" in text)


def catalog_for_request(business_scenario: str = "", use_case: str = "") -> dict[str, dict[str, Any]]:
    text = f"{business_scenario} {use_case}".lower()
    # Low Balance & Top-up is TMF654-first. Bring TMF635 into the semantic catalog only
    # when the actual request asks about usage/usage specification concepts.
    include_usage = any(token in text for token in ("usage", "voice call", "voicemail", "usage specification", "rated", "billed"))
    selected = {k: v for k, v in PDF_CATALOG.items() if v["source_id"] == "tmf654_v4"}
    if include_usage:
        selected.update({k: v for k, v in PDF_CATALOG.items() if v["source_id"] == "tmf635_v4"})
    return selected


def use_case_for(resource: str, field: str, business_scenario: str = "", use_case: str = "") -> str:
    text = f"{business_scenario} {use_case}".lower()
    if "cancel" in text or "cancellation" in text:
        if resource == "topupbalance":
            return USE_CASES["tmf654_uc3"]
        if resource == "bucket":
            return USE_CASES["tmf654_uc3"]
    if resource == "topupbalance":
        if field in {"isAutoTopup", "numberOfPeriods", "recurringPeriod"} or "recurr" in text or "auto top" in text:
            return USE_CASES["tmf654_uc2"]
        if field == "voucher" or "voucher" in text or "one time" in text or "one-time" in text:
            return USE_CASES["tmf654_uc1"]
        return f"{USE_CASES['tmf654_uc1']} / {USE_CASES['tmf654_uc2']} / {USE_CASES['tmf654_uc3']}"
    if resource == "transferbalance":
        return USE_CASES["tmf654_uc4"]
    if resource == "reservebalance":
        return USE_CASES["tmf654_uc5"]
    if resource == "bucket":
        if "transfer" in text or "colleague" in text:
            return USE_CASES["tmf654_uc4"]
        if "reserve" in text or "reservation" in text:
            return USE_CASES["tmf654_uc5"]
        if "recurr" in text or "auto top" in text:
            return USE_CASES["tmf654_uc2"]
        if "one time" in text or "one-time" in text:
            return USE_CASES["tmf654_uc1"]
        return " / ".join(USE_CASES[k] for k in ("tmf654_uc1", "tmf654_uc2", "tmf654_uc3", "tmf654_uc4", "tmf654_uc5"))
    if resource == "usage":
        return USE_CASES["tmf635_uc_usage"]
    if resource == "usagespecification":
        return USE_CASES["tmf635_uc_spec"]
    return "TMF654/TMF635 resource definition in supplied guide"


def pdf_provenance(resource: str, field: str, use_case: str) -> dict[str, Any]:
    item = PDF_CATALOG[resource]
    return {
        "source_policy": "supplied_pdf_only",
        "source_id": item["source_id"],
        "standard": item["standard"],
        "source_document": PDF_SOURCE_NAMES[item["source_id"]],
        "pdf_field": field,
        "useCase": use_case,
    }


def source_manifest() -> list[dict[str, str]]:
    out = []
    for source_id, filename in {
        "tmf654_v4": "TMF654_PrepayBalance_Management_API_User_Guide_v4.0.0.pdf",
        "tmf635_v4": "TMF635_Usage_Management_API_User_Guide_v4.0.0.pdf",
    }.items():
        path = PDF_RESOURCE_DIR / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        out.append({"source_id": source_id, "name": PDF_SOURCE_NAMES[source_id], "filename": filename, "sha256": digest})
    return out

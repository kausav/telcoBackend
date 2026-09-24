"""MongoDB-backed store for drafts, confirmed scenarios and feedback."""
from __future__ import annotations
import re
from typing import Any
from models.scenario import ScenarioModel
from models.scenario_draft import ScenarioDraftModel
from models.scenario_feedback import ScenarioFeedbackModel
from core.json_domain_policy import is_json_grounded_domain
from core.low_balance_variable_policy import validate_low_balance_variable_sources

def next_scenario_id() -> str:
    return ScenarioModel.next_id()

def new_draft_id() -> str:
    return ScenarioDraftModel.new_id()

def save_draft(draft_id: str, data: dict[str, Any]) -> None:
    ScenarioDraftModel.save(draft_id, data)

def get_draft(draft_id: str) -> dict[str, Any] | None:
    return ScenarioDraftModel.get(draft_id)

def pop_draft(draft_id: str) -> dict[str, Any] | None:
    return ScenarioDraftModel.pop(draft_id)

def confirm_scenario(requested_scenario_id: str | None, meta: dict[str, Any], variables: list[dict[str, Any]], field_order: list[str], draft_id: str | None = None) -> tuple[str, bool]:
    requested_id = str(requested_scenario_id or "").strip() or None
    return ScenarioModel.create_with_allocation(requested_id, draft_id, meta, variables, field_order)

def resolve_requested_scenario_id_from_draft(draft_id: str) -> str | None:
    return ScenarioModel.by_draft_id(draft_id)

def get_confirmed(requested_scenario_id: str) -> dict[str, Any] | None:
    return ScenarioModel.get(requested_scenario_id)

def scenario_exists(requested_scenario_id: str) -> bool:
    return ScenarioModel.exists(requested_scenario_id)

def resolve_scenario_meta(requested_scenario_id: str) -> dict[str, Any] | None:
    dyn = get_confirmed(requested_scenario_id)
    return dyn["meta"] if dyn else None

def _repair_legacy_categorical(var: dict[str, Any], norm_name: str) -> None:
    """Correct known legacy enum mismatches where the confirmed field description is authoritative."""
    mapping = {
        # These fields are known to have had semantically incorrect legacy values in the
        # supplied confirmed proposal; keep the repair narrowly scoped to those fields.
        "subscriber_segment": ["ULTRA_LOW", "MASS", "MID_TIER", "HIGH_VALUE"],
        "handset_network_capability": ["4G", "5G"],
        "subscriber_circle": [
            "Delhi", "Haryana", "Punjab", "Rajasthan", "Uttar Pradesh East", "Uttar Pradesh West",
            "Maharashtra", "Mumbai", "Gujarat", "Karnataka", "Tamil Nadu", "Kerala",
            "Andhra Pradesh", "Telangana", "West Bengal", "Bihar", "Odisha", "Assam",
            "North East", "Himachal Pradesh", "Jammu Kashmir", "Madhya Pradesh", "Kolkata",
        ],
        "low_balance_trigger_reason": ["LOW_BALANCE", "DATA_EXHAUSTED", "VALIDITY_EXPIRY"],
        "nudge_channel": ["SMS", "FLASH_SMS", "WHATSAPP", "APP_PUSH", "IVR"],
        "nudge_offer_type": ["EXTRA_DATA", "CASH_BACK", "VALIDITY_BOOSTER", "DISCOUNT_VOUCHER"],
        "recharge_channel": ["UPI_APP", "TELCO_APP", "RETAIL_POS", "ATM", "NETBANKING", "USSD"],
        "payment_instrument_type": ["UPI", "CREDIT_CARD", "DEBIT_CARD", "PREPAID_WALLET", "CASH", "AUTO_DEBIT"],
        "recharge_pack_category": ["UNLIMITED_COMBO", "DATA_ADDON", "TALKTIME_TOPUP", "INTERNATIONAL_ROAMING", "ISD"],
        "topup_fulfillment_status": ["COMPLETED", "FAILED", "PENDING", "REVERSED"],
        "accountType": ["INDIVIDUAL", "JOINT", "ORGANIZATION"],
        "paymentStatus": ["DUE", "PAID", "IN_ARREARS"],
        "lifecycleStatus": ["ACTIVE", "INACTIVE", "PENDING", "SUSPENDED", "TERMINATED"],
        "statusReason": ["CUSTOMER_REQUEST", "PAYMENT_FAILURE", "SYSTEM_ERROR", "POLICY_VIOLATION"],
        "usageType": ["DATA", "VOICE", "SMS", "CURRENCY"],
    }
    # Match both snake_case and compact/camelCase names. Older code compared only the
    # underscore-stripped form against snake_case mapping keys, which meant most repairs
    # silently never executed.
    normalized_mapping = {
        re.sub(r"[^a-z0-9]+", "", str(key).lower()): value
        for key, value in mapping.items()
    }
    lookup = re.sub(r"[^a-z0-9]+", "", str(norm_name).lower())
    choices = normalized_mapping.get(lookup)
    if choices:
        params = dict(var.get("params") or {})
        existing = params.get("choices", params.get("values"))
        existing_values = list(existing) if isinstance(existing, (list, tuple)) else []

        # Only replace an existing enum when it clearly conflicts with the authoritative
        # description/repair vocabulary. This prevents legacy migration from overwriting a
        # user-edited set that already contains values consistent with the field semantics.
        def norm(value):
            return re.sub(r"[^a-z0-9]+", "", str(value).lower())

        # These are explicit legacy repairs, so once the field is in the narrow repair map
        # its vocabulary is replaced as a whole rather than merged with the old values.
        params["choices"] = choices
        params.pop("values", None)
        params["weights"] = [1.0] * len(choices)
        var["params"] = params
        var["gen"] = "weighted_choice"
        if lookup == "subscribercircle":
            var["dtype"] = "categorical"


def _repair_legacy_variables(variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair legacy confirmed variables so generation cannot replay placeholder artifacts."""
    repaired: list[dict[str, Any]] = []
    names = {
        str(raw.get("name") or "").strip().lower()
        for raw in variables or []
        if isinstance(raw, dict)
    }
    has_msisdn = "msisdn" in names

    for raw in variables or []:
        if not isinstance(raw, dict):
            continue
        var = dict(raw)
        name = str(var.get("name") or "").strip()
        if not name:
            continue
        name_key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        norm_name = name_key.replace("_", "")
        dtype = str(var.get("dtype") or "").strip().lower()
        gen = str(var.get("gen") or "").strip().lower()
        params = dict(var.get("params") or {})

        # One canonical subscriber phone identifier: msisdn.
        if has_msisdn and norm_name in {"phonenumber", "mobilenumber", "telephonenumber"}:
            continue

        # Flat generation cannot safely materialize arbitrary nested objects. Drop the
        # legacy generic/empty object contracts instead of returning {}.
        if (
            dtype in {"object", "array"}
            and gen in {"generic", ""}
            and name_key not in {"subscriber_id", "account_id", "msisdn"}
            and not (params.get("choices") or params.get("values") or params.get("value"))
        ):
            continue

        # Correct known semantic/categorical mismatches from earlier confirmed drafts.
        _repair_legacy_categorical(var, name_key)
        dtype = str(var.get("dtype") or dtype).strip().lower()
        gen = str(var.get("gen") or gen).strip().lower()
        params = dict(var.get("params") or {})

        # Mandatory telecom identity anchors are application-level contracts and must never
        # be downgraded to a generic generator during legacy migration.
        if name_key == "subscriber_id":
            var["dtype"] = "string"
            var["gen"] = "prefixed_int"
            var["params"] = {"prefix": "SUB-", "digits": 10}
        elif name_key == "account_id":
            var["dtype"] = "string"
            var["gen"] = "id_mirror"
            var["params"] = {
                "prefix": "ACC-", "source_field": "subscriber_id", "source_prefix": "SUB-"
            }
        elif name_key == "msisdn":
            var["dtype"] = "string"
            var["gen"] = "e164_phone"
            params.setdefault("country_codes", ["+91"])
            params.setdefault("country", "IN")
            var["params"] = params
        # Old schema versions used generic generators that intentionally produced placeholders.
        # Route other legacy generic strings through the safe semantic generator.
        elif dtype in {"string", "str", "text"} and gen in {"generic", ""}:
            var["gen"] = "semantic_string"

        # Preserve booleans as a concrete true/false generator, even when an old draft
        # accidentally attached a categorical lifecycle choice list to a boolean field.
        if dtype not in {"boolean", "bool"} and bool(re.match(r"^(?:is|has)[A-Z]", name)):
            dtype = "boolean"
            var["dtype"] = "boolean"
        if dtype in {"boolean", "bool"}:
            var["dtype"] = "boolean"
            var["gen"] = "weighted_choice"
            var["params"] = {"choices": [False, True], "weights": [0.5, 0.5]}

        # Unit/denomination fields are always textual. Older confirmed drafts could
        # accidentally classify names such as ``topup_amount_currency_unit`` as numeric
        # because they also contain ``amount``; normalize them before generation.
        if any(token in norm_name for token in ("unit", "units", "currencyunit", "usageunit")):
            var["dtype"] = "string"
            if gen in {"uniform", "uniform_int", "range", "segment_range", "generic"}:
                var["gen"] = "semantic_string"
                var["params"] = {}
            dtype = "string"
            gen = str(var.get("gen") or "").strip().lower()

        # Normalize JSON/OpenAPI's "date-time" format so it can never be used as a literal strftime mask.
        if dtype == "datetime" and str(params.get("format") or "").strip().lower() in {"date-time", "datetime", "timestamp"}:
            params.pop("format", None)
            params.setdefault("timestamp_format", "dd/mm/yyyy hh:mm a")
            var["params"] = params

        repaired.append(var)
    return repaired

def resolve_variables(requested_scenario_id: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return the confirmed executable variables without rewriting Low Balance DB definitions."""
    dyn = get_confirmed(requested_scenario_id)
    if not dyn:
        return None

    raw_variables = [dict(v) for v in (dyn.get("variables") or []) if isinstance(v, dict)]
    meta = dyn.get("meta") or {}
    domain = meta.get("domain") or meta.get("journey") or ""

    if is_json_grounded_domain(domain):
        # Low Balance is source-locked. Never apply the generic legacy-repair layer here:
        # DB definitions must remain unchanged and official JSON variables must remain exactly
        # the source-backed contract selected during proposal/confirmation.
        validate_low_balance_variable_sources(
            raw_variables,
            meta.get("variable_sources") or {},
            db_variable_names=set(meta.get("db_variable_names") or []),
        )
        variables = raw_variables
    else:
        variables = _repair_legacy_variables(raw_variables)

    allowed = {str(v.get("name")) for v in variables if isinstance(v, dict) and v.get("name")}
    field_order = [name for name in list(dyn.get("field_order") or []) if str(name) in allowed]
    return variables, field_order



def resolve_scenario_context(requested_scenario_id: str) -> dict[str, Any]:
    """Return the complete confirmed scenario context used by generation agents."""
    meta = resolve_scenario_meta(requested_scenario_id) or {}
    return {
        "scenario_id": requested_scenario_id,
        "requested_scenario_id": meta.get("requested_scenario_id", requested_scenario_id),
        "label": meta.get("label", requested_scenario_id),
        "journey": meta.get("journey", ""),
        "description": meta.get("description", ""),
        "domain": meta.get("domain", ""),
        "business_scenario": meta.get("business_scenario", ""),
        "business_response": meta.get("business_response"),
        "expected_outcome": meta.get("expected_outcome"),
        "scenario_type": meta.get("scenario_type"),
        "use_case": meta.get("use_case"),
        "industry": meta.get("industry", "generic"),
        "country": meta.get("country"),
        "type_of_data": meta.get("type_of_data", "aggregational"),
        "entity_key": meta.get("entity_key"),
        "records_per_user": int(meta.get("records_per_user", 10) or 10),
        "agentic": bool(meta.get("agentic", False)),
        "variable_sources": dict(meta.get("variable_sources") or {}),
        "db_variable_names": sorted(set(meta.get("db_variable_names") or [])),
        "db_variable_definitions": dict(meta.get("db_variable_definitions") or {}),
    }


def resolve_data_type(requested_scenario_id: str) -> str:
    """Return the persisted data type for a scenario.

    Old scenarios created before typeOfData was introduced are treated as
    aggregational so existing scenarios continue to work unchanged.
    """
    meta = resolve_scenario_meta(requested_scenario_id) or {}
    value = str(meta.get("type_of_data", "aggregational")).strip().lower()
    return value if value in {"transactional", "aggregational"} else "aggregational"


def resolve_entity_key(requested_scenario_id: str) -> str | None:
    meta = resolve_scenario_meta(requested_scenario_id) or {}
    value = meta.get("entity_key")
    return str(value) if value else None



def list_scenarios() -> list[dict[str, Any]]:
    return ScenarioModel.list()

def _feedback_key(domain: str, business_scenario: str) -> str:
    return f"{domain.strip().lower()}::{business_scenario.strip().lower()}"

def add_feedback(requested_scenario_id: str, domain: str, business_scenario: str, feedback: str) -> None:
    if not feedback:
        return
    requested = str(requested_scenario_id or "").strip()
    if not requested:
        raise ValueError("requested_scenario_id is required")
    ScenarioFeedbackModel.add(requested, domain, business_scenario, feedback)


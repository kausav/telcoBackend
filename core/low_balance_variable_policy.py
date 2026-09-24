"""Strict variable-source policy for the Low Balance & Top-up journey.

The Low Balance journey has exactly two allowed variable sources:

1. the bundled TMF654 + TMF629 Swagger/OpenAPI scalar catalog; or
2. variables explicitly persisted in MongoDB as scenario/user variables.

Gemini is a selector/reviewer only. It can select official JSON variables for the
current scenario and identify semantic duplicates, but it can never introduce a
new variable into the executable schema.
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Iterable

from core.agentic_models import GeneratedSchemaField, ScenarioSchema
from core.json_domain_policy import expanded_scalar_catalog


_ALLOWED_CONTEXT_PREFIXES = {
    "topupbalance",
    "topup_balance",
    "topup",
    "recharge",
    "transaction",
}

# Low Balance rows should maximize analytical coverage without reproducing transport/display
# metadata or customer PII. These exclusions are based only on the supplied TMF654/TMF629
# scalar catalog; they do not introduce any new business vocabulary.
LOW_BALANCE_REQUIRED_FIELDS = ("customer_id", "account_id", "msisdn")
LOW_BALANCE_DB_REQUIRED_FIELDS = ("account_id", "msisdn")

# ``customer_id`` is present in the bundled TMF629 catalog. MongoDB may still contain a
# legacy/recommended variable with the same name, but it is only allowed to override the
# official JSON contract when its executable definition is fully compatible with the Low Balance
# entity-identity contract. Incompatible DB copies are quarantined from the executable schema;
# the stored Mongo document is never modified.
LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT = {
    "dtype": "string",
    "gen": "prefixed_int",
    "prefix": "cust-",
    "digits": 8,
    "scope": "entity",
    "required": True,
    "nullable": False,
}
_LOW_BALANCE_TECHNICAL_SUFFIXES = ("_href", "_description", "_referred_type")
_LOW_BALANCE_PII_NAMES = {
    "customer_name",
    "customer_engaged_party_name",
    "topupbalance_requestor_name",
}
_LOW_BALANCE_DISPLAY_NAMES = {"bucket_remaining_value_name"}


def is_compatible_low_balance_customer_db_override(variable: dict[str, Any] | None) -> bool:
    """Return whether a Mongo ``customer_id`` definition may override TMF629 Customer.id.

    This is intentionally a narrow, deterministic compatibility gate. It protects generation
    integrity without requiring the DB owner to clean up historical recommendation rows first.
    The function does not mutate the supplied Mongo definition.
    """
    if not isinstance(variable, dict):
        return False
    name = _normalize_name(variable.get("name"))
    if name != "customer_id":
        return True
    params = variable.get("params") if isinstance(variable.get("params"), dict) else {}
    try:
        digits = int(params.get("digits", 0) or 0)
    except (TypeError, ValueError):
        digits = 0
    return (
        str(variable.get("dtype") or "").strip().lower() == LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["dtype"]
        and str(variable.get("gen") or "").strip().lower() == LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["gen"]
        and str(params.get("prefix") or "") == LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["prefix"]
        and digits == LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["digits"]
        and str(variable.get("scope") or "").strip().lower() == LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["scope"]
        and bool(variable.get("required")) is LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["required"]
        and bool(variable.get("nullable")) is LOW_BALANCE_CUSTOMER_DB_OVERRIDE_CONTRACT["nullable"]
    )


def official_customer_id_definition() -> dict[str, Any]:
    """Build the canonical executable ``customer_id`` definition from the bundled TMF629 leaf.

    The Swagger source marks ``Customer.id`` optional at the API-schema level; the synthetic
    Low Balance contract deliberately promotes it to a required stable entity identifier.
    """
    spec = dict(official_catalog_by_name().get("customer_id") or {})
    if not spec:
        raise ValueError("The bundled TMF629 catalog is missing the required customer_id field")
    return {
        "name": "customer_id",
        "dtype": "string",
        "description": str(spec.get("description") or "Unique identifier for Customers")[:500],
        "gen": "prefixed_int",
        "params": {"prefix": "cust-", "digits": 8},
        "depends_on": [],
        "nullable": False,
        "required": True,
        "formula": None,
        "scope": "entity",
        "useCase": None,
        "provenance": {
            "generated_from": "official_json_source",
            "source_json_id": spec.get("source_id"),
            "source_json_model": spec.get("model"),
            "source_json_path": spec.get("path"),
            "grain": "entity",
        },
    }


def official_customer_id_field() -> GeneratedSchemaField:
    """Return the canonical executable Pydantic field for Low Balance ``customer_id``."""
    return GeneratedSchemaField.model_validate(official_customer_id_definition())


def filter_incompatible_low_balance_db_customer_overrides(
    recommended: list[dict[str, Any]],
    user_selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Remove only incompatible Mongo ``customer_id`` overrides from the executable input.

    All other DB variables, including variables that do not exist in the two Swagger sources,
    pass through unchanged. This is the key source-boundary rule that lets Mongo extend the
    official catalog without allowing an old ``customer_id`` definition to corrupt the stable
    entity contract.
    """
    suppressed: list[str] = []

    def clean(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for raw in items or []:
            if not isinstance(raw, dict):
                cleaned.append(raw)
                continue
            if _normalize_name(raw.get("name")) == "customer_id" and not is_compatible_low_balance_customer_db_override(raw):
                suppressed.append(str(raw.get("name") or "customer_id"))
                continue
            cleaned.append(dict(raw))
        return cleaned

    return clean(recommended), clean(user_selected), sorted(set(suppressed))


def reconcile_low_balance_executable_variables(
    variables: list[dict[str, Any]],
    source_by_name: dict[str, str] | None = None,
    db_variable_names: set[str] | None = None,
    field_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], set[str], list[str]]:
    """Repair legacy Low Balance executable state without changing MongoDB.

    The current proposal path should already produce this shape, but confirmed scenarios can
    outlive compiler/policy revisions. This function is therefore also the generation-time
    migration boundary: an incompatible DB ``customer_id`` is replaced locally by the official
    TMF629 contract, missing official ``customer_id`` is restored, and all unrelated DB variables
    remain untouched.
    """
    normalized_sources = {
        _normalize_name(k): str(v).strip().upper()
        for k, v in (source_by_name or {}).items()
        if str(k).strip() and str(v).strip()
    }
    normalized_db_names = {
        _normalize_name(name) for name in (db_variable_names or set()) if str(name).strip()
    }
    normalized_variables: list[dict[str, Any]] = []
    customer_seen = False

    for raw in variables or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        name = _normalize_name(item.get("name"))
        if not name:
            continue
        source = normalized_sources.get(name, "")
        is_db_source = source in {"DB_RECOMMENDED", "USER_SELECTED"} or name in normalized_db_names
        is_db_customer = name == "customer_id" and is_db_source

        if name == "customer_id":
            if customer_seen:
                # Confirmed drafts should already have unique names, but legacy rows can contain
                # duplicate customer_id entries. Keep the first deterministic winner only.
                continue
            customer_seen = True
            # An official-source customer_id is repaired from the bundled source definition if
            # a legacy confirmed draft contains an incomplete/old executable shape.
            if source == "OFFICIAL_JSON":
                item = official_customer_id_definition()
            elif is_db_customer and not is_compatible_low_balance_customer_db_override(item):
                item = official_customer_id_definition()
                normalized_sources[name] = "OFFICIAL_JSON"
                normalized_db_names.discard(name)
            elif not source:
                # Old drafts may lack provenance. The exact public field name is source-grounded
                # and therefore safely recoverable from the bundled TMF629 catalog.
                item = official_customer_id_definition()
                normalized_sources[name] = "OFFICIAL_JSON"
                normalized_db_names.discard(name)
        elif name in LOW_BALANCE_DB_REQUIRED_FIELDS and is_db_source:
            # account_id/msisdn are DB-only fields. Keep their generator contract exactly as
            # stored, but repair only the stable identity-grain flags required by this journey.
            item, _ = normalize_low_balance_db_identity_definition(item)

        normalized_variables.append(item)

    if not customer_seen:
        normalized_variables.append(official_customer_id_definition())
        normalized_sources["customer_id"] = "OFFICIAL_JSON"
        normalized_db_names.discard("customer_id")

    normalized_order = [str(name) for name in (field_order or []) if str(name).strip()]
    present = {_normalize_name(item.get("name")) for item in normalized_variables if isinstance(item, dict)}
    normalized_order = [name for name in normalized_order if _normalize_name(name) in present]
    if "customer_id" not in {_normalize_name(name) for name in normalized_order}:
        customer_position = next((i for i, item in enumerate(normalized_variables) if _normalize_name(item.get("name")) == "customer_id"), None)
        if customer_position is None:
            normalized_order.append("customer_id")
        else:
            normalized_order.insert(customer_position, "customer_id")

    return normalized_variables, normalized_sources, normalized_db_names, normalized_order


def reconcile_low_balance_schema(
    schema: ScenarioSchema,
    variables: list[dict[str, Any]],
    source_by_name: dict[str, str] | None = None,
    db_variable_names: set[str] | None = None,
    field_order: list[str] | None = None,
) -> tuple[ScenarioSchema, list[dict[str, Any]], dict[str, str], set[str], list[str]]:
    """Align a draft schema with the legacy-safe executable variable reconciliation."""
    normalized_variables, normalized_sources, normalized_db_names, normalized_order = reconcile_low_balance_executable_variables(
        variables, source_by_name, db_variable_names, field_order
    )
    existing_fields = {
        _normalize_name(field.name): field
        for field in schema.fields
        if isinstance(field, GeneratedSchemaField) and str(field.name).strip()
    }
    rebuilt_fields: list[GeneratedSchemaField] = []
    for item in normalized_variables:
        name = _normalize_name(item.get("name"))
        if name == "customer_id":
            rebuilt_fields.append(official_customer_id_field())
            continue
        if name in LOW_BALANCE_DB_REQUIRED_FIELDS:
            rebuilt_fields.append(GeneratedSchemaField.model_validate(item))
            continue
        existing = existing_fields.get(name)
        rebuilt_fields.append(existing if existing is not None else GeneratedSchemaField.model_validate(item))
    represented_names = {_normalize_name(field.name) for field in rebuilt_fields}
    cleaned_unresolved: list[str] = []
    for item in schema.unresolved_items:
        text = str(item).strip()
        lower = text.casefold()
        if "entity key '" in lower or "requested entity key '" in lower:
            candidate = lower.split("entity key '", 1)[-1].split("'", 1)[0].strip()
            if candidate and candidate in represented_names:
                continue
        cleaned_unresolved.append(text)

    return (
        schema.model_copy(update={"fields": rebuilt_fields, "unresolved_items": cleaned_unresolved}),
        normalized_variables,
        normalized_sources,
        normalized_db_names,
        normalized_order,
    )


def is_bucket_variable_name(value: Any) -> bool:
    """Return True for fields that expose balance-bucket context rather than the top-up/customer journey itself."""
    name = _normalize_name(value)
    if not name:
        return False
    tokens = name.split("_")
    return (
        name.startswith("bucket_")
        or name.startswith("balance_bucket_")
        or "_bucket_" in name
        or name.endswith("_bucket")
        or name == "bucket"
        or (tokens and tokens[0] == "bucket")
    )


def is_material_low_balance_spec(spec: dict[str, Any]) -> bool:
    """Return whether an official scalar leaf is analytically useful for Low Balance.

    The function only filters low-value metadata/PII from the supplied official JSON catalog.
    Material Bucket state is deliberately allowed through so scenario policy can decide whether it
    adds independent value.
    Every retained field remains traceable to TMF654/TMF629.
    """
    name = _normalize_name(spec.get("name"))
    if not name:
        return False
    if name in _LOW_BALANCE_PII_NAMES or name in _LOW_BALANCE_DISPLAY_NAMES:
        return False
    if name in {
        "bucket_name",
        "bucket_requested_date",
        "bucket_confirmation_date",
        "bucket_party_account_id",
        "bucket_party_account_name",
        "bucket_party_account_status",
        "bucket_remaining_value_name",
    }:
        return False
    # Bucket is a legitimate Low Balance business resource. Its material state fields are
    # eligible for selection; scenario-specific policy below removes low-value bucket metadata.
    if name.endswith(_LOW_BALANCE_TECHNICAL_SUFFIXES):
        return False
    dtype = str(spec.get("dtype") or "string").strip().lower()
    return dtype not in {"object", "array"}


def material_low_balance_catalog() -> tuple[dict[str, Any], ...]:
    """Return the broad, quality-safe official Low Balance variable catalog."""
    rows = [dict(row) for row in expanded_scalar_catalog() if is_material_low_balance_spec(row)]
    rows.sort(key=lambda item: (str(item.get("model") or ""), str(item.get("path") or ""), str(item.get("name") or "")))
    return tuple(rows)


def normalize_low_balance_db_identity_definition(variable: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return the executable Low Balance contract for a DB-backed identity field.

    MongoDB remains the source of truth for the variable's semantic generator definition,
    name, dtype and parameters. The application overlays only the Low Balance identity-grain
    contract at runtime because historical DB rows may predate the journey's stable-entity
    requirement. The stored Mongo document is never modified.
    """
    item = dict(variable)
    name = _normalize_name(item.get("name"))
    if name not in LOW_BALANCE_DB_REQUIRED_FIELDS:
        return item, False

    changed = (
        str(item.get("scope") or "").strip().lower() != "entity"
        or bool(item.get("required")) is not True
        or bool(item.get("nullable")) is not False
    )
    if changed:
        item["scope"] = "entity"
        item["required"] = True
        item["nullable"] = False
    return item, changed


def normalize_low_balance_db_identity_variables(
    db_variables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize only mandatory DB identity fields for executable Low Balance state.

    DB-only extension variables remain byte-for-byte equivalent dict copies. Their scope and
    nullable/required semantics are deliberately not changed.
    """
    normalized: list[dict[str, Any]] = []
    changed_names: list[str] = []
    for raw in db_variables or []:
        if not isinstance(raw, dict):
            normalized.append(raw)
            continue
        item, changed = normalize_low_balance_db_identity_definition(raw)
        normalized.append(item)
        if changed:
            changed_names.append(str(item.get("name") or "").strip())
    return normalized, sorted(set(name for name in changed_names if name))


def validate_low_balance_required_identity_sources(db_variables: list[dict[str, Any]]) -> None:
    """Require the exact DB-only telecom identity fields that are absent from the two Swagger files.

    ``customer_id`` exists in the official TMF629 catalog. ``account_id`` and ``msisdn`` do not, so
    Low Balance must receive those exact names from MongoDB. Their executable identity-grain contract
    is normalized before this validation; this function therefore checks presence, not historical
    DB metadata that may legitimately be stale.
    """
    normalized = {
        _normalize_name(item.get("name")): item
        for item in (db_variables or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    missing = [name for name in LOW_BALANCE_DB_REQUIRED_FIELDS if _normalize_name(name) not in normalized]
    if missing:
        raise ValueError(
            "Low Balance & Top-up requires the exact DB variables "
            + ", ".join(missing)
            + ". They are not present in the supplied TMF654/TMF629 Swagger scalar catalog and must not be invented by the LLM or another generator."
        )
# Synonyms are deliberately narrow. This is only the deterministic backstop for
# the LLM's duplicate review; it must not become a second semantic registry.
_TOKEN_ALIASES = {
    "automatic": "auto",
    "autotopup": "auto_topup",
    "top_up": "topup",
    "requested": "requested",
    "confirmation": "confirmation",
    "result": "outcome",
    "state": "status",
    "recharge_result": "outcome",
    "lifecycle_state": "status",
    "lifecycle_status": "status",
    "credited": "credit",
}


def _normalize_name(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)


def _tokenize(value: Any) -> list[str]:
    normalized = _normalize_name(value)
    return [token for token in normalized.split("_") if token]


def _canonical_tokens(value: Any) -> list[str]:
    """Normalize naming variants without collapsing materially different concepts."""
    tokens = _tokenize(value)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        triple = tuple(tokens[i:i + 3])
        pair = tuple(tokens[i:i + 2])

        if triple in {
            ("requested", "date", "time"),
            ("request", "date", "time"),
        }:
            out.append("requested_timestamp")
            i += 3
            continue
        if pair in {
            ("requested", "date"),
            ("request", "date"),
            ("requested", "datetime"),
            ("request", "datetime"),
            ("request", "timestamp"),
        }:
            out.append("requested_timestamp")
            i += 2
            continue
        if triple in {
            ("confirmation", "date", "time"),
            ("confirm", "date", "time"),
        }:
            out.append("confirmation_timestamp")
            i += 3
            continue
        if pair in {
            ("confirmation", "date"),
            ("confirm", "date"),
            ("confirmation", "datetime"),
            ("confirm", "datetime"),
            ("confirmation", "timestamp"),
            ("confirm", "timestamp"),
        }:
            out.append("confirmation_timestamp")
            i += 2
            continue

        token = _TOKEN_ALIASES.get(tokens[i], tokens[i])
        out.append(token)
        i += 1

    # Remove only the well-known source/context prefix. Keep meaningful qualifiers such as
    # party_account_status and amount_units so distinct fields never collapse accidentally.
    if out[:2] == ["topup", "balance"]:
        out = out[2:]
    elif out and out[0] in _ALLOWED_CONTEXT_PREFIXES:
        out = out[1:]

    while out and out[0] in {"is", "has"}:
        out = out[1:]

    # Collapse only consecutive duplicate flattening tokens such as amount_amount.
    if out:
        collapsed: list[str] = []
        for token in out:
            if collapsed and token == collapsed[-1] and token in {"amount", "date", "time"}:
                continue
            collapsed.append(token)
        out = collapsed
    return out


def _semantic_context(raw_tokens: list[str]) -> str:
    """Determine the broad context without discarding qualifying business terms."""
    if not raw_tokens:
        return ""
    if raw_tokens[0] in {"customer", "subscriber", "bucket"}:
        return raw_tokens[0]
    if any(token in {"topupbalance", "topup", "recharge", "transaction"} for token in raw_tokens):
        return "topup"
    return ""


def semantic_signature(variable: dict[str, Any] | GeneratedSchemaField) -> tuple[str, str]:
    """Return a conservative business-use signature for duplicate detection.

    The function intentionally catches only well-understood aliases. It must never turn
    related-but-distinct fields such as ``amount`` vs ``amount_units`` or
    ``status`` vs ``party_account_status`` into duplicates.
    """
    if isinstance(variable, GeneratedSchemaField):
        data = variable.model_dump()
    else:
        data = variable

    name = str(data.get("name") or "")
    raw_tokens = _tokenize(name)
    tokens = _canonical_tokens(name)
    if not tokens:
        tokens = _canonical_tokens(str(data.get("description") or ""))

    context = _semantic_context(raw_tokens)
    token_set = set(tokens)

    # Normalize the complete business concept only for exact, recognized TopupBalance/recharge
    # aliases. Qualifying terms remain intact for all other concepts.
    concept_group: str | None = None
    if context == "topup":
        if tokens == ["auto_topup"]:
            concept_group = "auto_topup"
        elif tokens in (["amount"], ["credit", "amount"], ["recharge", "amount"], ["topup", "amount"]):
            concept_group = "amount"
        elif tokens == ["status"] or tokens == ["outcome"] or tokens == ["result"]:
            concept_group = "status_outcome"
        elif tokens == ["requested_timestamp"]:
            concept_group = "requested_timestamp"
        elif tokens == ["confirmation_timestamp"]:
            concept_group = "confirmation_timestamp"

    dtype = str(data.get("dtype") or "string").strip().lower()
    if concept_group == "auto_topup":
        dtype_key = "boolean"
    elif concept_group == "amount":
        dtype_key = "numeric"
    elif concept_group == "status_outcome":
        dtype_key = "status"
    elif concept_group in {"requested_timestamp", "confirmation_timestamp"}:
        dtype_key = "datetime"
    elif dtype in {"boolean", "bool"}:
        dtype_key = "boolean"
    elif dtype in {"integer", "int", "float", "decimal", "number", "numeric"}:
        dtype_key = "numeric"
    elif dtype in {"datetime", "date", "timestamp"}:
        dtype_key = "datetime"
    else:
        dtype_key = "categorical" if str(data.get("role") or "").lower() in {"status", "decision"} else "string"

    concept = concept_group or "_".join(tokens or ["field"])
    return context, f"{concept}::{dtype_key}"


LOW_BALANCE_OFFICIAL_MAX_FIELDS = 45


def _lb_norm_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _lb_scenario_tokens(value: object) -> set[str]:
    """Return exact normalized scenario tokens; avoids substring false positives such as data/dataset."""
    return {token for token in _lb_norm_text(value).split("_") if token}


def _lb_has_scenario_terms(value: object, terms: set[str]) -> bool:
    tokens = _lb_scenario_tokens(value)
    normalized = _lb_norm_text(value)
    return any(term in tokens or ("_" in term and term in normalized) for term in terms)


def _low_balance_relevance_profile(outcome_mode: str) -> dict[str, dict[str, float]]:
    """Return scenario-specific concept weights for official Low Balance fields.

    The selector never creates a field name. It only ranks exact leaves already present in the
    supplied TMF654/TMF629 catalog. Scenario type changes the ranking so distinct scenarios do not
    collapse to the same variable set merely because they share the same business description.
    """
    common = {
        "amount": 4.0,
        "requested": 3.0,
        "confirmation": 3.0,
        "status": 4.0,
        "reason": 4.0,
        "channel": 3.0,
        "payment_method": 3.0,
        "voucher": 2.0,
        "auto_topup": 3.0,
        "recurring_period": 2.0,
        "number_of_periods": 2.0,
        "usage_type": 2.0,
        "valid_for": 1.5,
        "customer_status": 2.0,
        "customer_status_reason": 2.0,
        "party_account_status": 2.0,
        "requestor": 1.5,
        "id": 1.0,
        # Bucket state is a legitimate Low Balance signal, but only its independent
        # analytical measures should rank highly. Transport/reference/display fields
        # are filtered elsewhere.
        "remaining": 6.0,
        "reserved": 3.0,
        "is_shared": 2.5,
    }
    profiles = {
        "positive": {
            **common,
            "amount": 7.0,
            "requested": 6.0,
            "confirmation": 7.0,
            "status": 6.0,
            "usage_type": 5.0,
            "channel": 4.5,
            "payment_method": 4.5,
            "voucher": 4.0,
            "auto_topup": 5.5,
            "recurring_period": 5.0,
            "number_of_periods": 4.0,
            "valid_for": 3.0,
            "reason": 3.0,
            "remaining": 10.0,
            "reserved": 4.0,
            "is_shared": 3.0,
        },
        "suppression": {
            **common,
            "status": 10.0,
            "reason": 10.0,
            "customer_status": 9.0,
            "customer_status_reason": 9.5,
            "party_account_status": 8.0,
            "channel": 7.5,
            "payment_method": 5.5,
            "requestor": 5.5,
            "auto_topup": 6.0,
            "requested": 4.0,
            "confirmation": 4.0,
            "valid_for": 2.5,
            "amount": 2.0,
            "usage_type": 1.5,
            "voucher": 3.0,
            "amount_units": 0.5,
            "remaining": 12.0,
            "reserved": 5.0,
            "is_shared": 4.0,
        },
        "negative": {
            **common,
            "status": 9.0,
            "reason": 8.5,
            "customer_status": 6.0,
            "customer_status_reason": 6.5,
            "party_account_status": 6.0,
            "requested": 5.0,
            "confirmation": 6.0,
            "amount": 5.0,
            "channel": 5.0,
            "payment_method": 5.0,
            "requestor": 4.0,
            "voucher": 3.0,
            "auto_topup": 3.0,
            "remaining": 11.0,
            "reserved": 5.0,
            "is_shared": 3.0,
        },
        "decline_or_no_response": {
            **common,
            "status": 8.0,
            "reason": 7.5,
            "channel": 6.5,
            "requested": 6.0,
            "confirmation": 4.0,
            "customer_status": 6.0,
            "customer_status_reason": 6.0,
            "party_account_status": 5.0,
            "payment_method": 5.0,
            "requestor": 4.0,
            "amount": 3.0,
            "auto_topup": 4.0,
            "voucher": 3.0,
            "remaining": 11.0,
            "reserved": 4.0,
            "is_shared": 3.0,
        },
        "concurrent": {
            **common,
            "status": 7.0,
            "reason": 5.0,
            "channel": 6.0,
            "payment_method": 5.0,
            "requestor": 5.0,
            "customer_status": 5.0,
            "party_account_status": 5.0,
            "amount": 4.0,
            "requested": 4.0,
            "confirmation": 4.0,
            "auto_topup": 4.0,
            "remaining": 10.0,
            "reserved": 4.0,
            "is_shared": 3.0,
        },
    }
    return profiles.get(outcome_mode, common)



def _low_balance_profile_exclusions(outcome_mode: str, business_scenario: str = "") -> set[str]:
    """Exclude source fields that add little independent signal for the requested scenario.

    These are quality filters over the approved TMF654/TMF629 catalog, not source restrictions.
    MongoDB variables are handled separately and are never removed here.
    """
    mode = str(outcome_mode or "").strip().lower()
    scenario_text = _lb_norm_text(business_scenario)
    exclusions = {
        "customer_engaged_party_id",
        "customer_engaged_party_role",
        "topupbalance_balance_topup_id",
        "topupbalance_balance_topup_name",
        "topupbalance_balance_topup_role",
        "topupbalance_party_account_name",
        # The TopupBalance bucket reference is a transport-level pointer. The direct Bucket
        # resource below provides the useful balance-state facts, so reference labels are not
        # duplicated in the flat row.
        "topupbalance_bucket_id",
        "topupbalance_bucket_name",
        # Low-value bucket metadata/display fields.
        "bucket_name",
        "bucket_requested_date",
        "bucket_confirmation_date",
        "bucket_party_account_id",
        "bucket_party_account_name",
        "bucket_party_account_status",
        "bucket_remaining_value_name",
    }

    has_reservation = _lb_has_scenario_terms(scenario_text, {
        "reserved", "reservation", "reserve", "hold"
    })
    has_shared = _lb_has_scenario_terms(scenario_text, {
        "shared", "family", "multi_device", "multidevice"
    })
    has_validity = _lb_has_scenario_terms(scenario_text, {
        "expiry", "expire", "expiration", "validity", "valid_for", "validity_period"
    })
    has_usage = _lb_has_scenario_terms(scenario_text, {
        "usage", "data", "voice", "sms", "monetary", "currency"
    })

    if not has_reservation:
        exclusions.update({"bucket_reserved_value_amount", "bucket_reserved_value_units"})
    if not has_shared:
        exclusions.add("bucket_is_shared")
    if not has_validity:
        exclusions.update({"bucket_valid_for_start_date_time", "bucket_valid_for_end_date_time"})
    if mode in {"suppression", "negative", "decline_or_no_response", "concurrent"} and not has_usage:
        exclusions.add("bucket_usage_type")

    if mode == "suppression":
        # Recurrence configuration is only useful when the suppression scenario explicitly
        # discusses recurring/automatic top-up behavior.
        scenario_tokens = _lb_scenario_tokens(scenario_text)
        if not scenario_tokens.intersection({"recurring", "periodic", "automatic", "autotopup"}) and not "auto_topup" in scenario_text:
            exclusions.update({"topupbalance_recurring_period", "topupbalance_number_of_periods"})
        if not scenario_tokens.intersection({"usage", "data", "voice", "sms"}):
            exclusions.add("topupbalance_usage_type")

    return exclusions


def low_balance_official_relevance_score(
    row: dict[str, Any],
    *,
    outcome_mode: str,
    business_scenario: str = "",
    scenario_type: str = "",
) -> float:
    """Score an exact official scalar by scenario relevance without inventing any field."""
    name = _lb_norm_text(row.get("name"))
    description = _lb_norm_text(row.get("description"))
    text = f"{name} {description}"
    profile = _low_balance_relevance_profile(outcome_mode)
    score = 45.0

    if name == "customer_id":
        score += 18.0
    if str(row.get("model") or "").lower() == "topupbalance":
        score += 7.0
    elif str(row.get("model") or "").lower() == "customer":
        score += 4.0

    for token, weight in profile.items():
        if token in text:
            score += weight

    # Scenario language is an additional weak signal, never a source of new field names.
    scenario_text = f"{_lb_norm_text(scenario_type)} {_lb_norm_text(business_scenario)}"
    for token in ("retention", "intervention", "trigger", "recharge", "topup", "balance", "customer"):
        if token in scenario_text and token in text:
            score += 1.5
    if outcome_mode == "suppression":
        if any(token in text for token in ("name", "role")):
            score -= 3.0
        if "balance_topup" in text:
            score -= 5.0
    elif outcome_mode == "positive":
        if "balance_topup" in text:
            score -= 3.0
        if "customer_engaged_party" in text:
            score -= 2.0

    # Names/roles are relation context rather than the primary behavioral signal. Keep them only
    # when they clear the scenario-specific quality threshold or are required by dependencies.
    if name.endswith("_name"):
        score -= 4.0
    if name.endswith("_role"):
        score -= 3.0
    if name.endswith("_units"):
        score -= 1.0
    return score


def select_low_balance_official_catalog(
    *,
    outcome_mode: str,
    business_scenario: str = "",
    scenario_type: str = "",
    excluded_names: set[str] | None = None,
    preferred_names: set[str] | None = None,
    max_fields: int = LOW_BALANCE_OFFICIAL_MAX_FIELDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the widest high-quality scenario-specific official field set.

    Only exact fields from the supplied TMF654/TMF629 material catalog are returned. The LLM can
    influence priority through ``preferred_names`` but cannot force unrelated/low-quality fields.
    """
    excluded = {_lb_norm_text(name) for name in (excluded_names or set()) if _lb_norm_text(name)}
    excluded.update(_low_balance_profile_exclusions(outcome_mode, business_scenario))
    preferred = {_lb_norm_text(name) for name in (preferred_names or set()) if _lb_norm_text(name)}
    rows = [
        dict(row) for row in material_low_balance_catalog()
        if _lb_norm_text(row.get("name")) not in excluded
    ]

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        name = _lb_norm_text(row.get("name"))
        score = low_balance_official_relevance_score(
            row,
            outcome_mode=outcome_mode,
            business_scenario=business_scenario,
            scenario_type=scenario_type,
        )
        if name in preferred:
            score += 3.0
        scored.append((score, index, row))

    # Core journey facts are always useful when present. Everything else must earn inclusion through
    # the scenario-specific score, which is what creates meaningful differences between Normal,
    # Suppression, Failure, etc. instead of returning the entire catalog for every scenario.
    core_names = {
        "topupbalance_id",
        "topupbalance_requested_date",
        "topupbalance_confirmation_date",
        "topupbalance_status",
        "topupbalance_amount_amount",
        "customer_id",
        # Current bucket state is central to a Low Balance journey. Optional bucket dimensions
        # are added only when the scenario context supports them.
        "bucket_id",
        "bucket_remaining_value_amount",
        "bucket_remaining_value_units",
        "bucket_status",
    }
    scenario_text = _lb_norm_text(business_scenario)
    if outcome_mode == "positive" or _lb_has_scenario_terms(
        scenario_text, {"usage", "data", "voice", "sms", "monetary", "currency"}
    ):
        core_names.add("bucket_usage_type")
    core: list[tuple[float, int, dict[str, Any]]] = [item for item in scored if _lb_norm_text(item[2].get("name")) in core_names]
    core_keys = {_lb_norm_text(item[2].get("name")) for item in core}
    ranked = sorted(scored, key=lambda item: (-item[0], item[1], _lb_norm_text(item[2].get("name"))))

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for item in core + ranked:
        score, _index, row = item
        name = _lb_norm_text(row.get("name"))
        if name in selected_keys:
            continue
        if name not in core_keys and score < 47.0:
            continue
        if len(selected) >= max(1, int(max_fields)):
            break
        selected.append(row)
        selected_keys.add(name)

    report = {
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "max_fields": int(max_fields),
        "outcome_mode": outcome_mode,
        "preferred_names_used": sorted(preferred & selected_keys),
        "selected_names": [str(row.get("name") or "") for row in selected],
    }
    return selected, report


def official_catalog() -> tuple[dict[str, Any], ...]:
    """Return the immutable official Low Balance scalar catalog from the two bundled Swagger files.

    The catalog contains material journey fields from TMF654 TopupBalance, TMF654 Bucket, and
    TMF629 Customer. Bucket transport/display/reference noise is filtered, while materially useful
    balance-state fields remain available for scenario-aware selection.
    """
    rows = []
    for row in material_low_balance_catalog():
        item = dict(row)
        item["name"] = _normalize_name(item.get("name"))
        rows.append(item)
    # Exact normalized field names are unique in expanded_scalar_catalog(); keep a
    # deterministic order for prompt/caching reproducibility.
    rows.sort(key=lambda item: str(item.get("name") or ""))
    return tuple(rows)


@lru_cache(maxsize=1)
def official_catalog_by_name() -> dict[str, dict[str, Any]]:
    return {str(row["name"]): dict(row) for row in official_catalog()}


def validate_llm_official_selection(variables: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only exact official catalog variables returned by Gemini.

    Any LLM-created name that is not an exact normalized official source leaf is
    rejected from the executable path. We return the rejected names for logging/
    diagnostics, never as schema fields.
    """
    catalog = official_catalog_by_name()
    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for raw in variables or []:
        name = _normalize_name(raw.get("name") if isinstance(raw, dict) else "")
        if not name:
            continue
        if name not in catalog:
            rejected.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        canonical = dict(raw)
        canonical["name"] = name
        # Replace model-provided semantic metadata with the source catalog's identity
        # metadata. Gemini may choose fields; it does not define what the field means.
        spec = catalog[name]
        canonical["description"] = str(spec.get("description") or canonical.get("description") or "")[:500]
        source_dtype = str(spec.get("dtype") or "string").lower()
        if source_dtype in {"boolean", "bool"}:
            canonical["dtype"] = "boolean"
        elif source_dtype in {"number", "float", "double", "decimal"}:
            canonical["dtype"] = "float"
        elif source_dtype in {"integer", "int", "bigint", "smallint"}:
            canonical["dtype"] = "integer"
        elif source_dtype in {"date-time", "datetime", "timestamp"}:
            canonical["dtype"] = "datetime"
        elif source_dtype == "date":
            canonical["dtype"] = "date"
        elif spec.get("enum_values"):
            canonical["dtype"] = "categorical"
        else:
            canonical["dtype"] = "string"
        selected.append(canonical)
    return selected, sorted(set(rejected))


def source_spec_for_name(name: str) -> dict[str, Any] | None:
    """Return the official JSON spec for an exact catalog field name."""
    return dict(official_catalog_by_name().get(_normalize_name(name), {})) or None


def _best_json_winner(fields: list[GeneratedSchemaField], indices: list[int]) -> int:
    """Choose one deterministic representative for a JSON-only semantic duplicate group."""
    def rank(index: int) -> tuple[int, int, float, int]:
        field = fields[index]
        provenance = field.provenance or {}
        depth_raw = provenance.get("source_json_depth", provenance.get("depth", 0))
        try:
            depth = int(depth_raw or 0)
        except (TypeError, ValueError):
            depth = 0
        try:
            quality = float(provenance.get("quality_score", 0) or 0)
        except (TypeError, ValueError):
            quality = 0.0
        return (
            1 if field.required else 0,
            -depth,
            quality,
            -index,
        )
    return max(indices, key=rank)


def _normalized_dependency_names(variable: dict[str, Any]) -> set[str]:
    return {
        _normalize_name(dep)
        for dep in (variable.get("depends_on") or [])
        if str(dep).strip()
    }


def dedupe_db_variable_sources(
    recommended: list[dict[str, Any]],
    user_selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Deduplicate MongoDB variables before they enter the Low Balance pipeline.

    Rules are intentionally strict and deterministic:
      * DB definitions are never rewritten.
      * USER_SELECTED wins over DB_RECOMMENDED for the same exact name or semantic use.
      * Within one DB source, the first deterministic definition wins.
      * If removing a semantic duplicate would break another surviving DB variable's explicit
        dependency, fail closed rather than rewriting the dependency.

    The returned definitions are copies of the winning MongoDB records; callers must continue
    to use those records unchanged. The third return value contains names suppressed from the
    DB source set for diagnostics.
    """
    entries: list[tuple[str, int, dict[str, Any]]] = []
    for source_name, items in (
        ("DB_RECOMMENDED", recommended or []), ("USER_SELECTED", user_selected or [])
    ):
        for position, raw in enumerate(items):
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid MongoDB scenario variable: {raw!r}")
            validate_db_definition(raw)
            name = str(raw.get("name") or "").strip()
            if not name:
                raise ValueError("MongoDB scenario variable is missing its name")
            # Validate execution shape, but retain the original DB mapping as the winning
            # definition. Pydantic normalization is intentionally not written back.
            entries.append((source_name, position, dict(raw)))

    # Exact names: USER_SELECTED replaces the recommendation; otherwise first occurrence wins.
    exact_winners: dict[str, tuple[str, int, dict[str, Any]]] = {}
    suppressed: set[str] = set()
    for source_name, position, item in entries:
        key = _normalize_name(item.get("name"))
        current = exact_winners.get(key)
        if current is None or (source_name == "USER_SELECTED" and current[0] == "DB_RECOMMENDED"):
            if current is not None:
                suppressed.add(str(current[2].get("name") or ""))
            exact_winners[key] = (source_name, position, item)
        else:
            suppressed.add(str(item.get("name") or ""))

    winners = list(exact_winners.values())
    # Preserve DB source ordering after exact-name overlay.
    winners.sort(key=lambda item: (0 if item[0] == "DB_RECOMMENDED" else 1, item[1], _normalize_name(item[2].get("name"))))

    # Semantic duplicate groups use the same conservative signature as JSON-vs-DB matching.
    semantic_groups: dict[tuple[str, str], list[tuple[str, int, dict[str, Any]]]] = {}
    for entry in winners:
        semantic_groups.setdefault(semantic_signature(entry[2]), []).append(entry)

    surviving: list[tuple[str, int, dict[str, Any]]] = []
    removed_names: set[str] = set(suppressed)
    for signature, group in semantic_groups.items():
        if len(group) == 1:
            surviving.append(group[0])
            continue

        # USER_SELECTED wins over recommendation. Within the same source the deterministic
        # first definition wins.
        group_sorted = sorted(
            group,
            key=lambda item: (0 if item[0] == "USER_SELECTED" else 1, item[1], _normalize_name(item[2].get("name"))),
        )
        winner = group_sorted[0]
        loser_names = [str(item[2].get("name") or "") for item in group_sorted[1:]]
        loser_keys = {_normalize_name(name) for name in loser_names if name}

        # Never mutate a DB dependency merely to perform deduplication. Such a database state
        # is ambiguous and must be repaired by the DB owner instead of silently changing meaning.
        for source_name, _, item in group_sorted:
            if source_name == winner[0] and _normalize_name(item.get("name")) == _normalize_name(winner[2].get("name")):
                continue
            name_key = _normalize_name(item.get("name"))
            for other_source, _, other_item in winners:
                other_key = _normalize_name(other_item.get("name"))
                if other_key == name_key:
                    continue
                deps = _normalized_dependency_names(other_item)
                if name_key in deps and other_key not in loser_keys:
                    raise ValueError(
                        f"MongoDB Low Balance variables '{item.get('name')}' and '{other_item.get('name')}' "
                        "are semantically duplicate, but the surviving variable depends on the duplicate. "
                        "Resolve the DB definitions before using this scenario."
                    )

        surviving.append(winner)
        removed_names.update(loser_names)

    surviving.sort(key=lambda item: (0 if item[0] == "DB_RECOMMENDED" else 1, item[1], _normalize_name(item[2].get("name"))))

    cleaned_recommended = [dict(item[2]) for item in surviving if item[0] == "DB_RECOMMENDED"]
    cleaned_user_selected = [dict(item[2]) for item in surviving if item[0] == "USER_SELECTED"]

    # A user-selected semantic winner may have suppressed its recommended counterpart. Make
    # the returned source lists mutually exclusive by exact normalized name as well.
    user_keys = {_normalize_name(item.get("name")) for item in cleaned_user_selected}
    cleaned_recommended = [
        item for item in cleaned_recommended
        if _normalize_name(item.get("name")) not in user_keys
    ]
    return cleaned_recommended, cleaned_user_selected, sorted(name for name in removed_names if name)


def dedupe_against_db(
    variables: list[dict[str, Any]],
    db_variables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove duplicate LLM-selected official JSON variables, with DB precedence.

    This function is applied before executable compilation. It performs two deterministic
    operations:
      1. DB-vs-JSON duplicate suppression: the DB variable wins unchanged.
      2. JSON-vs-JSON duplicate suppression: only one official variable survives.

    Dependency references inside surviving JSON candidates are redirected to the winning
    variable name where an alias was removed. DB definitions themselves are never changed.
    """
    db_sigs: dict[tuple[str, str], str] = {}
    exact_db_names = {_normalize_name(v.get("name")) for v in db_variables if isinstance(v, dict)}
    for raw in db_variables or []:
        if not isinstance(raw, dict):
            continue
        key = semantic_signature(raw)
        db_sigs.setdefault(key, str(raw.get("name") or ""))

    survivors: list[dict[str, Any]] = []
    seen_sigs: dict[tuple[str, str], str] = {}
    alias_to_survivor: dict[str, str] = {}
    removed: set[str] = set()

    for raw in variables or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        name = _normalize_name(item.get("name"))
        if not name:
            continue
        sig = semantic_signature(item)
        if name in exact_db_names or sig in db_sigs:
            replacement = db_sigs.get(sig) or next(
                (db_name for db_name in exact_db_names if db_name == name),
                "",
            )
            alias_to_survivor[name] = replacement
            removed.add(name)
            continue
        if sig in seen_sigs:
            alias_to_survivor[name] = seen_sigs[sig]
            removed.add(name)
            continue
        seen_sigs[sig] = name
        survivors.append(item)

    # Remap dependencies only for surviving JSON candidates. This never mutates DB data.
    for item in survivors:
        deps = item.get("depends_on") or []
        item["depends_on"] = [
            alias_to_survivor.get(_normalize_name(dep), str(dep))
            for dep in deps
        ]

    return survivors, sorted(removed)


def dedupe_schema_fields_against_db(
    fields: list[GeneratedSchemaField],
    db_variables: list[dict[str, Any]],
) -> tuple[list[GeneratedSchemaField], list[str]]:
    """Deduplicate the final Low Balance schema across JSON and MongoDB sources.

    Precedence is explicit:
        DB variable > official JSON variable.

    JSON-only aliases are collapsed to one deterministic representative. Dependencies of
    surviving JSON fields are redirected to the surviving/DB name. MongoDB definitions are
    never modified or deduplicated by this function.
    """
    db_sigs: dict[tuple[str, str], str] = {}
    db_names = {_normalize_name(raw.get("name")) for raw in (db_variables or []) if isinstance(raw, dict)}
    for raw in db_variables or []:
        if not isinstance(raw, dict):
            continue
        signature = semantic_signature(raw)
        name = str(raw.get("name") or "")
        existing = db_sigs.get(signature)
        if existing and _normalize_name(existing) != _normalize_name(name):
            raise ValueError(
                f"MongoDB Low Balance variables '{existing}' and '{name}' have the same business-use signature. "
                "Resolve the DB duplicate before compiling the scenario."
            )
        db_sigs[signature] = name

    groups: dict[tuple[str, str], list[int]] = {}
    for index, field in enumerate(fields or []):
        groups.setdefault(semantic_signature(field), []).append(index)

    keep: set[int] = set()
    replacements: dict[str, str] = {}
    removed: set[str] = set()

    for signature, indices in groups.items():
        db_winner = db_sigs.get(signature)
        if db_winner:
            for index in indices:
                name = _normalize_name(fields[index].name)
                if name != _normalize_name(db_winner):
                    replacements[name] = db_winner
                    removed.add(fields[index].name)
            continue

        winner = _best_json_winner(fields, indices)
        keep.add(winner)
        winner_name = fields[winner].name
        for index in indices:
            if index == winner:
                continue
            replacements[_normalize_name(fields[index].name)] = winner_name
            removed.add(fields[index].name)

    result: list[GeneratedSchemaField] = []
    for index, field in enumerate(fields or []):
        if index not in keep:
            continue
        data = field.model_copy(deep=True)
        data.depends_on = [
            replacements.get(_normalize_name(dep), dep)
            for dep in (data.depends_on or [])
        ]
        result.append(data)

    return result, sorted(removed)

def validate_low_balance_variable_sources(
    variables: list[dict[str, Any]],
    source_by_name: dict[str, str] | None = None,
    db_variable_names: set[str] | None = None,
) -> None:
    """Enforce the Low Balance executable-source boundary at every lifecycle stage.

    Every executable field must have explicit provenance:
      * OFFICIAL_JSON and exact name in the supplied TMF654/TMF629 scalar catalog, or
      * DB_RECOMMENDED / USER_SELECTED and exact name in the persisted DB set.

    No unlabelled, LLM_GENERATED, application-derived, CSV, or generic-registry field can enter
    the Low Balance executable contract. This is intentionally fail-closed.
    """
    catalog = official_catalog_by_name()
    sources = {
        _normalize_name(k): str(v).strip().upper()
        for k, v in (source_by_name or {}).items()
        if str(k).strip()
    }
    normalized_db_names = {
        _normalize_name(name) for name in (db_variable_names or set()) if str(name).strip()
    }
    allowed_db = {"DB_RECOMMENDED", "USER_SELECTED"}
    allowed_official = "OFFICIAL_JSON"
    invalid: list[str] = []
    seen_names: set[str] = set()
    seen_signatures: dict[tuple[str, str], str] = {}

    for raw in variables or []:
        if not isinstance(raw, dict):
            invalid.append("<invalid-variable>")
            continue
        name = _normalize_name(raw.get("name"))
        if not name:
            invalid.append("<missing-name>")
            continue
        if name in seen_names:
            invalid.append(f"{name} (duplicate exact variable name)")
            continue
        seen_names.add(name)

        source = sources.get(name, "")
        if not source:
            invalid.append(f"{name} (missing source provenance)")
            continue
        if source in allowed_db:
            if name not in normalized_db_names:
                invalid.append(f"{name} (DB source is not present in the persisted DB variable set)")
                continue
        elif source == allowed_official:
            if name not in catalog:
                invalid.append(f"{name} (not present in the supplied TMF654/TMF629 catalog)")
                continue
        else:
            invalid.append(f"{name} (unsupported source {source})")
            continue

        signature = semantic_signature(raw)
        prior = seen_signatures.get(signature)
        if prior and _normalize_name(prior) != name:
            invalid.append(f"{name} (duplicate business use of {prior})")
        else:
            seen_signatures[signature] = name

    if invalid:
        raise ValueError(
            "Low Balance & Top-up executable variables must come only from the supplied TMF654/TMF629 Swagger scalar catalog or MongoDB variables. "
            "Each field must also carry explicit source provenance. Invalid variables: "
            + ", ".join(sorted(set(invalid)))
        )

    by_name = {
        _normalize_name(raw.get("name")): raw
        for raw in (variables or [])
        if isinstance(raw, dict) and str(raw.get("name") or "").strip()
    }
    required = {_normalize_name(name) for name in LOW_BALANCE_REQUIRED_FIELDS}
    missing = sorted(name for name in required if name not in by_name)
    if missing:
        raise ValueError(
            "Low Balance & Top-up requires customer_id, account_id, and msisdn in every executable schema. "
            "Missing: " + ", ".join(missing)
        )
    for name in required:
        var = by_name[name]
        if str(var.get("scope") or "").strip().lower() != "entity":
            raise ValueError(f"Low Balance identity variable '{name}' must have scope='entity'.")
        if not bool(var.get("required")) or bool(var.get("nullable")):
            raise ValueError(f"Low Balance identity variable '{name}' must be required=true and nullable=false.")

    if sources.get("account_id") not in allowed_db:
        raise ValueError("Low Balance account_id must come from MongoDB; it is not present in the supplied TMF654/TMF629 catalog.")
    if sources.get("msisdn") not in allowed_db:
        raise ValueError("Low Balance msisdn must come from MongoDB; it is not present in the supplied TMF654/TMF629 catalog.")
    if sources.get("customer_id") not in {allowed_official, *allowed_db}:
        raise ValueError("Low Balance customer_id must come from TMF629 JSON or MongoDB.")

    customer = by_name["customer_id"]
    if sources.get("customer_id") in allowed_db:
        params = customer.get("params") if isinstance(customer.get("params"), dict) else {}
        if (
            str(customer.get("gen") or "").strip().lower() != "prefixed_int"
            or str(params.get("prefix") or "") != "cust-"
            or int(params.get("digits", 0) or 0) != 8
            or str(customer.get("dtype") or "").strip().lower() != "string"
        ):
            raise ValueError(
                "MongoDB customer_id is authoritative but its definition does not satisfy the required Low Balance contract: "
                "gen='prefixed_int', prefix='cust-', digits=8, dtype='string'. Update the DB definition; the application will not rewrite it."
            )


def validate_db_definition(variable: dict[str, Any]) -> dict[str, Any]:
    """Validate a DB variable without changing its semantics or name."""
    model = GeneratedSchemaField.model_validate(variable)
    return model.model_dump()

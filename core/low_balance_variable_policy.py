"""Strict variable-source policy for the Low Balance & Top-up journey.

The Low Balance journey has exactly two allowed variable sources:

1. active TMF654/TMF629 Swagger/OpenAPI source documents stored in the MongoDB industry/domain registry; or
2. variables explicitly persisted in MongoDB as scenario variables.

Gemini is a selector/reviewer only. It can select official JSON variables for the
current scenario and identify semantic duplicates, but it can never introduce a
new variable into the executable schema.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from core.agentic_models import GeneratedSchemaField
from core.json_domain_policy import expanded_scalar_catalog
from config.runtime import SCHEMA_MAX_VARIABLES


_ALLOWED_CONTEXT_PREFIXES = {
    "topupbalance",
    "topup_balance",
    "topup",
    "recharge",
    "transaction",
}

# Low Balance rows should maximize analytical coverage without reproducing transport/display
# metadata or customer PII. These exclusions operate only on the active MongoDB TMF654/TMF629
# scalar catalog; they do not introduce any new business vocabulary.
LOW_BALANCE_REQUIRED_FIELDS = ("customer_id",)
# The Low Balance journey must not require DB-recommended identity extensions. Every executable
# industry/domain field comes from the active MongoDB JSON source catalog, while existing user-selected
# scenario variables remain independently supported by the general DB-variable path.
LOW_BALANCE_DB_REQUIRED_FIELDS: tuple[str, ...] = ()
_LOW_BALANCE_TECHNICAL_SUFFIXES = ("_href", "_description", "_referred_type")
_LOW_BALANCE_PII_NAMES = {
    "customer_name",
    "customer_engaged_party_name",
    "topupbalance_requestor_name",
}
_LOW_BALANCE_DISPLAY_NAMES = {"bucket_remaining_value_name"}


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
    Every retained field remains traceable to an active MongoDB TMF654/TMF629 source document.
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


_LOW_BALANCE_CORE_RESOURCES = {"customer", "bucket", "topup"}
_LOW_BALANCE_OPERATION_ROOTS = {
    "adjust_balance",
    "transfer_balance",
    "reserve_balance",
    "accumulated_balance",
    "balance_action_history",
}
_LOW_BALANCE_EVENT_MARKERS = {
    "event",
    "payload",
    "create_event",
    "update_event",
    "delete_event",
    "cancel_event",
    "failure_event",
    "state_change_event",
    "attribute_value_change_event",
}
_LOW_BALANCE_REF_FIELDS = {
    "partyaccount": "party_account",
    "party_account": "party_account",
    "channel": "channel",
    "paymentmethod": "payment_method",
    "payment_method": "payment_method",
    "requestor": "requestor",
    "bucket": "bucket",
    "product": "product",
    "logicalresource": "logical_resource",
    "logical_resource": "logical_resource",
    "engagedparty": "engaged_party",
    "engaged_party": "engaged_party",
}
_LOW_BALANCE_EXCLUDED_REFERENCE_LEAVES = {"href", "description", "name", "referred_type", "@referredtype"}


def _lb_path_segments(path: object) -> list[str]:
    return [_normalize_name(part) for part in str(path or "").split(".") if _normalize_name(part)]


def _lb_root_family(segment: str) -> str | None:
    """Map only genuine business-resource roots to a Low Balance resource family."""
    token = _normalize_name(segment)
    if not token:
        return None
    allowed_roots = {
        "customer",
        "customer_create", "customer_update", "customer_delete",
        "customer_create_event", "customer_update_event", "customer_delete_event",
        "customer_state_change_event", "customer_attribute_value_change_event",
        "bucket",
        "bucket_create", "bucket_update", "bucket_delete",
        "bucket_create_event", "bucket_update_event", "bucket_delete_event",
        "topup_balance", "topup_balance_create", "topup_balance_update", "topup_balance_delete",
        "topup_balance_create_event", "topup_balance_update_event", "topup_balance_delete_event",
        "topup_balance_cancel_event", "topup_balance_failure_event",
    }
    if token in allowed_roots:
        if token.startswith("customer"):
            return "customer"
        if token.startswith("bucket"):
            return "bucket"
        return "topup"
    return None


def _lb_target_resource_and_tail(path: object) -> tuple[str | None, list[str], bool]:
    """Resolve event/create/update wrappers to the underlying business resource.

    The raw Mongo catalog is intentionally flat and contains the same business property many times
    through CRUD/event payload schemas. We canonicalize those wrappers before variable selection.
    Event-only payload fields remain excluded; the underlying business property is represented once.
    """
    segments = _lb_path_segments(path)
    if not segments:
        return None, [], False

    first_family = _lb_root_family(segments[0])
    first_has_event = "event" in segments[0] or segments[0].endswith("_event") or any(
        marker in segments[0] for marker in ("create_event", "update_event", "delete_event", "cancel_event", "failure_event", "state_change_event")
    )
    operation_root = normalize_lookup_operation_root = segments[0]
    if operation_root in _LOW_BALANCE_OPERATION_ROOTS or operation_root.startswith(tuple(f"{item}_" for item in _LOW_BALANCE_OPERATION_ROOTS)):
        return None, [], False

    # A direct base resource (Customer/Bucket/TopupBalance or their Create/Update variants).
    if first_family:
        if first_has_event:
            # Event schemas are transport wrappers around the same business resource. Locate the
            # embedded Customer/Bucket/TopupBalance object and reuse its canonical leaf so event,
            # payload, create/update and base-resource copies collapse into one analytical field.
            for idx, segment in enumerate(segments[1:], start=1):
                family = _lb_root_family(segment)
                if family == first_family:
                    return family, segments[idx + 1:], True
            return first_family, [], True
        return first_family, segments[1:], False

    # Other definitions such as Action.bucket, TransferBalance.bucket, etc. are deliberately not
    # mined for embedded references. Low Balance & Top-up is scoped to the Customer/Bucket/TopupBalance
    # business resources, otherwise unrelated operation schemas would leak duplicate reference leaves.
    return None, [], False


def _canonical_low_balance_name(row: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Convert a raw standards path into one canonical analytical variable name.

    Canonical names intentionally describe the business concept, not the Swagger event/transport
    path. The original source paths are retained in ``source_paths`` for traceability.
    """
    path = str(row.get("path") or "")
    resource, tail, from_event = _lb_target_resource_and_tail(path)
    if resource not in _LOW_BALANCE_CORE_RESOURCES:
        return None, {}
    if from_event:
        # Event wrappers are represented by their underlying base resource fields. A path that has
        # no business-resource tail is an event-only field, so it is not a scenario variable.
        if not tail:
            return None, {}

    # Convert property segments to their normalized forms.
    tail = [_normalize_name(part) for part in tail if _normalize_name(part)]
    if not tail:
        return None, {}

    leaf = tail[-1]
    ref_parent = _LOW_BALANCE_REF_FIELDS.get(tail[-2]) if len(tail) >= 2 else None

    canonical: str | None = None
    if resource == "customer":
        if tail == ["id"]:
            canonical = "customer_id"
        elif tail == ["status"]:
            canonical = "customer_status"
        elif tail == ["status_reason"]:
            canonical = "customer_status_reason"
        elif tail == ["valid_for", "start_date_time"]:
            canonical = "customer_valid_from"
        elif tail == ["valid_for", "end_date_time"]:
            canonical = "customer_valid_to"
        else:
            # engagedParty/account/agreement/contact/creditProfile nested metadata is either
            # relationship transport or unrelated to this flat retention/recharge scenario.
            return None, {}

    elif resource == "bucket":
        if tail == ["id"]:
            canonical = "bucket_id"
        elif tail == ["remaining_value", "amount"]:
            canonical = "bucket_remaining_amount"
        elif tail == ["remaining_value", "units"]:
            canonical = "bucket_remaining_unit"
        elif tail == ["reserved_value", "amount"]:
            canonical = "bucket_reserved_amount"
        elif tail == ["reserved_value", "units"]:
            canonical = "bucket_reserved_unit"
        elif tail == ["status"]:
            canonical = "bucket_status"
        elif tail == ["usage_type"]:
            canonical = "usage_type"
        elif tail == ["is_shared"]:
            canonical = "bucket_is_shared"
        elif tail == ["valid_for", "start_date_time"]:
            canonical = "bucket_valid_from"
        elif tail == ["valid_for", "end_date_time"]:
            canonical = "bucket_valid_to"
        elif len(tail) == 2 and ref_parent == "party_account" and leaf in {"id", "status"}:
            canonical = f"party_account_{leaf}"
        else:
            return None, {}

    elif resource == "topup":
        if tail == ["id"]:
            canonical = "topup_id"
        elif tail == ["requested_date"]:
            canonical = "topup_requested_at"
        elif tail == ["confirmation_date"]:
            canonical = "topup_confirmed_at"
        elif tail == ["status"]:
            canonical = "topup_status"
        elif tail == ["is_auto_topup"]:
            canonical = "topup_is_auto"
        elif tail == ["number_of_periods"]:
            canonical = "topup_number_of_periods"
        elif tail == ["recurring_period"]:
            canonical = "topup_recurring_period"
        elif tail == ["reason"]:
            canonical = "topup_reason"
        elif tail == ["amount", "amount"]:
            canonical = "topup_amount"
        elif tail == ["amount", "units"]:
            canonical = "topup_amount_unit"
        elif tail == ["usage_type"]:
            canonical = "usage_type"
        elif len(tail) == 2 and ref_parent == "party_account" and leaf in {"id", "status"}:
            canonical = f"party_account_{leaf}"
        elif len(tail) == 2 and ref_parent == "bucket" and leaf == "id":
            canonical = "bucket_id"
        elif len(tail) == 2 and ref_parent == "channel" and leaf == "name":
            canonical = "topup_channel_name"
        elif len(tail) == 2 and ref_parent == "payment_method" and leaf == "name":
            canonical = "topup_payment_method_name"
        elif len(tail) == 2 and ref_parent == "requestor" and leaf == "role":
            canonical = "topup_requestor_role"
        elif tail == ["valid_for", "start_date_time"]:
            canonical = "topup_valid_from"
        elif tail == ["valid_for", "end_date_time"]:
            canonical = "topup_valid_to"
        else:
            # Opaque voucher/reference IDs, names, hrefs and nested RelatedTopupBalance metadata are
            # intentionally excluded from the analytical variable set.
            return None, {}

    if not canonical:
        return None, {}
    meta = dict(row)
    meta["name"] = canonical
    meta["canonical_name"] = canonical
    meta["source_paths"] = [path] if path else []
    meta["source_models"] = [str(row.get("model") or "")] if row.get("model") else []
    raw_aliases = []
    for alias in (row.get("name"), path):
        normalized_alias = _normalize_name(alias)
        if normalized_alias and normalized_alias not in raw_aliases:
            raw_aliases.append(normalized_alias)
    meta["source_aliases"] = raw_aliases
    return canonical, meta


def _merge_low_balance_source_rows(group: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge source provenance and constraints for one canonical field."""
    for key, singular in (("source_paths", row.get("path")), ("source_models", row.get("model"))):
        values = group.setdefault(key, [])
        if singular and str(singular) not in values:
            values.append(str(singular))
    for sid in row.get("source_ids") or ([row.get("source_id")] if row.get("source_id") else []):
        sid = str(sid or "")
        if sid and sid not in group.setdefault("source_ids", []):
            group["source_ids"].append(sid)
    for alias in (row.get("name"), _normalize_name(row.get("path"))):
        alias = _normalize_name(alias)
        if alias and alias not in group.setdefault("source_aliases", []):
            group["source_aliases"].append(alias)
    if row.get("required"):
        group["required"] = True
    descriptions = [str(group.get("description") or "").strip(), str(row.get("description") or "").strip()]
    descriptions = [value for value in descriptions if value]
    if descriptions:
        group["description"] = max(descriptions, key=len)
    enum_values = list(group.get("enum_values") or [])
    for value in row.get("enum_values") or []:
        if value not in enum_values:
            enum_values.append(value)
    if enum_values:
        group["enum_values"] = enum_values


def material_low_balance_catalog() -> tuple[dict[str, Any], ...]:
    """Return canonical, deduplicated, source-backed Low Balance business variables.

    The raw MongoDB catalog deliberately contains Swagger path fan-out. This function is the semantic
    boundary between the standards documents and the scenario selector: one business concept becomes
    one variable, while all contributing source paths remain available as provenance.
    """
    groups: dict[str, dict[str, Any]] = {}
    for raw in expanded_scalar_catalog():
        if not isinstance(raw, dict) or not is_material_low_balance_spec(raw):
            continue
        canonical, item = _canonical_low_balance_name(raw)
        if not canonical:
            continue
        existing = groups.get(canonical)
        if existing is None:
            groups[canonical] = item
        else:
            _merge_low_balance_source_rows(existing, raw)

    rows = []
    for name, row in groups.items():
        item = dict(row)
        item["name"] = name
        item["canonical_name"] = name
        item["source_paths"] = sorted(set(str(value) for value in item.get("source_paths") or []))
        item["source_models"] = sorted(set(str(value) for value in item.get("source_models") or [] if value))
        item["source_ids"] = sorted(set(str(value) for value in item.get("source_ids") or [] if value))
        item["source_aliases"] = sorted(set(str(value) for value in item.get("source_aliases") or [] if value))
        rows.append(item)
    rows.sort(key=lambda item: str(item.get("name") or ""))
    return tuple(rows)


def validate_low_balance_required_identity_sources(db_variables: list[dict[str, Any]]) -> None:
    """Legacy compatibility hook; Low Balance identity fields are no longer DB-recommended.

    The active TMF654/TMF629 source catalog supplies the official customer identity (Customer.id).
    This function intentionally performs no DB-extension requirement so ``DB_RECOMMENDED`` variables
    cannot be forced into new Low Balance proposals.
    """
    return

def reconcile_low_balance_variables(
    variables: list[dict[str, Any]],
    source_by_name: dict[str, str] | None = None,
    db_variable_names: set[str] | None = None,
    field_order: list[str] | None = None,
    db_variable_definitions: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], set[str], list[str]]:
    """Reconcile Low Balance source metadata once, without rewriting any variable definition.

    The lifecycle has two authoritative source families only: active MongoDB industry/domain source
    documents and explicitly persisted MongoDB variables. DB provenance wins over stale official/LLM metadata
    for the same exact field name. The function also repairs legacy drafts whose source metadata
    was incomplete, while preserving every DB-owned definition exactly as supplied.
    """
    catalog = official_catalog_by_name()
    aliases = official_catalog_aliases()
    db_names = {
        _normalize_name(name)
        for name in (db_variable_names or set())
        if str(name).strip()
    }
    db_definitions = {
        _normalize_name(name): dict(value)
        for name, value in (db_variable_definitions or {}).items()
        if str(name).strip() and isinstance(value, dict)
    }
    db_names.update(db_definitions)
    sources = {
        _normalize_name(name): str(source).strip().upper()
        for name, source in (source_by_name or {}).items()
        if str(name).strip() and str(source).strip()
    }

    # Normalize exact names once. If an old draft omitted db_variable_names, existing DB
    # provenance is still sufficient to classify the field as DB-owned.
    for name, source in list(sources.items()):
        if source in {"DB_RECOMMENDED", "USER_SELECTED"}:
            db_names.add(name)

    def source_for(name: str) -> str:
        canonical_name = aliases.get(name, name)
        existing = sources.get(name, "") or sources.get(canonical_name, "")
        if name in db_names:
            return existing if existing in {"DB_RECOMMENDED", "USER_SELECTED"} else "DB_RECOMMENDED"
        if existing in {"DB_RECOMMENDED", "USER_SELECTED"}:
            return existing
        if canonical_name in catalog:
            return "MONGODB_JSON"
        return existing

    # Deduplicate exact names deterministically. A DB definition always replaces any earlier
    # schema/LLM copy, but the DB definition itself is never mutated. Definitions saved with a
    # draft are also rehydrated when a legacy draft's variables array is incomplete.
    normalized: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    inputs = [dict(raw) for raw in (variables or []) if isinstance(raw, dict)]
    inputs.extend(db_definitions.values())
    for raw in inputs:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        name = _normalize_name(item.get("name"))
        if not name:
            continue
        source = source_for(name)
        if name in positions:
            if source in {"DB_RECOMMENDED", "USER_SELECTED"}:
                normalized[positions[name]] = item
            continue
        positions[name] = len(normalized)
        normalized.append(item)
        if source:
            sources[name] = source

    # Recover provenance for every retained field from the immutable source boundary. This makes
    # legacy drafts robust even when their persisted source map is stale or incomplete.
    for item in normalized:
        name = _normalize_name(item.get("name"))
        source = source_for(name)
        if source:
            sources[name] = source

    normalized_order: list[str] = []
    seen_order: set[str] = set()
    for name in field_order or []:
        key = _normalize_name(name)
        if key in positions and key not in seen_order:
            normalized_order.append(str(name))
            seen_order.add(key)
    for item in normalized:
        name = str(item.get("name") or "")
        key = _normalize_name(name)
        if key and key not in seen_order:
            normalized_order.append(name)
            seen_order.add(key)

    return normalized, sources, db_names, normalized_order


def validate_low_balance_variable_sources(
    variables: list[dict[str, Any]],
    source_by_name: dict[str, str] | None = None,
    db_variable_names: set[str] | None = None,
) -> None:
    """Validate the Low Balance source boundary, independent of DB-owned field semantics.

    Official fields must exist in the immutable Swagger catalog. DB fields must exist in the
    persisted DB-name set. Scope/required/nullable/generator values belong to the source that
    owns the field and are intentionally not revalidated against a journey-local template.
    """
    catalog = official_catalog_by_name()
    aliases = official_catalog_aliases()
    sources = {
        _normalize_name(name): str(source).strip().upper()
        for name, source in (source_by_name or {}).items()
        if str(name).strip() and str(source).strip()
    }
    db_names = {
        _normalize_name(name)
        for name in (db_variable_names or set())
        if str(name).strip()
    }
    allowed_db = {"DB_RECOMMENDED", "USER_SELECTED"}
    invalid: list[str] = []
    seen_names: set[str] = set()

    for raw in variables or []:
        if not isinstance(raw, dict):
            invalid.append("<invalid-variable>")
            continue
        name = _normalize_name(raw.get("name"))
        if not name:
            invalid.append("<missing-name>")
            continue
        canonical_name = aliases.get(name, name)
        if canonical_name in seen_names:
            invalid.append(f"{name} (duplicate exact variable name)")
            continue
        seen_names.add(canonical_name)

        source = sources.get(name, "")
        if not source:
            source = sources.get(canonical_name, "")
        # DB membership is the authoritative provenance signal. This also repairs legacy drafts
        # where the source map was missing or still labeled the DB field as official JSON.
        if name in db_names:
            source = source if source in allowed_db else "DB_RECOMMENDED"
        elif not source and canonical_name in catalog:
            source = "MONGODB_JSON"

        if source in allowed_db:
            if name not in db_names:
                invalid.append(f"{name} (DB source is not present in the persisted DB variable set)")
                continue
        elif source in {"MONGODB_JSON", "OFFICIAL_JSON"}:
            if canonical_name not in catalog:
                invalid.append(f"{name} (not present in the active MongoDB TMF654/TMF629 catalog)")
                continue
        else:
            invalid.append(f"{name} (unsupported or missing source provenance)")
            continue

        if source in allowed_db:
            try:
                validate_db_definition(raw)
            except Exception as exc:
                invalid.append(f"{name} (invalid MongoDB definition: {exc})")

    required = {canonical_low_balance_name(name) for name in LOW_BALANCE_REQUIRED_FIELDS}
    missing = sorted(required - seen_names)
    invalid.extend(f"{name} (required Low Balance identity variable is missing)" for name in missing)

    if invalid:
        raise ValueError(
            "Low Balance & Top-up executable variables must come only from the active MongoDB TMF654/TMF629 source scalar catalog or MongoDB variables. "
            "Invalid variables: " + ", ".join(sorted(set(invalid)))
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

    name = canonical_low_balance_name(str(data.get("name") or ""))
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


LOW_BALANCE_OFFICIAL_MAX_FIELDS = SCHEMA_MAX_VARIABLES


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
    active MongoDB TMF654/TMF629 catalog. Scenario type changes the ranking so distinct scenarios do not
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
    """Exclude canonical source fields that add little independent value for this scenario."""
    mode = str(outcome_mode or "").strip().lower()
    scenario_text = _lb_norm_text(business_scenario)
    exclusions: set[str] = set()

    has_reservation = _lb_has_scenario_terms(scenario_text, {"reserved", "reservation", "reserve", "hold"})
    has_shared = _lb_has_scenario_terms(scenario_text, {"shared", "family", "multi_device", "multidevice"})
    has_validity = _lb_has_scenario_terms(
        scenario_text, {"expiry", "expire", "expiration", "validity", "valid_for", "validity_period"}
    )
    has_automatic = _lb_has_scenario_terms(
        scenario_text, {"automatic", "autotopup", "auto_topup", "recurring", "periodic"}
    )

    if not has_reservation:
        exclusions.update({"bucket_reserved_amount", "bucket_reserved_unit"})
    if not has_shared:
        exclusions.add("bucket_is_shared")
    if not has_validity:
        exclusions.update({
            "customer_valid_from", "customer_valid_to",
            "bucket_valid_from", "bucket_valid_to",
            "topup_valid_from", "topup_valid_to",
        })
    if mode == "suppression" and not has_automatic:
        exclusions.update({"topup_number_of_periods", "topup_recurring_period"})

    return exclusions

def low_balance_official_relevance_score(
    row: dict[str, Any],
    *,
    outcome_mode: str,
    business_scenario: str = "",
    scenario_type: str = "",
) -> float:
    """Score a canonical source variable by scenario relevance without creating vocabulary."""
    name = _lb_norm_text(row.get("name"))
    description = _lb_norm_text(row.get("description"))
    text = f"{name} {description}"
    mode = str(outcome_mode or "").strip().lower()
    score = 50.0

    resource = name.split("_", 1)[0] if name else ""
    if name == "customer_id":
        score += 20.0
    if name in {"topup_id", "bucket_id"}:
        score += 7.0
    if name in {"bucket_remaining_amount", "bucket_remaining_unit"}:
        score += 15.0
    if name in {"topup_requested_at", "topup_confirmed_at"}:
        score += 10.0
    if name in {"topup_status", "bucket_status", "customer_status"}:
        score += 9.0
    if name == "customer_status_reason":
        score += 8.0
    if name == "party_account_status":
        score += 8.0
    if name == "topup_is_auto":
        score += 8.0
    if name in {"topup_amount", "topup_amount_unit"}:
        score += 6.0
    if name in {"usage_type"}:
        score += 6.0
    if name in {"topup_channel_name", "topup_payment_method_name", "topup_requestor_role"}:
        score += 5.0

    # Scenario text is a weak relevance signal layered over the source-defined concept.
    scenario_text = f"{_lb_norm_text(scenario_type)} {_lb_norm_text(business_scenario)}"
    concept_terms = {
        "suppression": {"suppression", "cooldown", "contact", "intervention", "retention", "trigger"},
        "positive": {"recharge", "topup", "balance", "successful", "completed", "retention"},
        "negative": {"failure", "failed", "rejected", "error"},
        "decline_or_no_response": {"decline", "declined", "no_response", "no_response", "suppression"},
        "concurrent": {"concurrent", "simultaneous", "pending", "duplicate"},
    }
    for token in concept_terms.get(mode, set()):
        if token in scenario_text and token in text:
            score += 2.0

    # Independent analytical value: identifiers for related reference resources are lower value than
    # behavioral/category fields, but a role/status still carries useful segmentation information.
    if name.endswith("_role"):
        score -= 1.0
    if name.endswith("_unit"):
        score -= 0.5
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
    """Select the widest *canonical* source-backed set that remains relevant to the scenario."""
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

    ranked = sorted(scored, key=lambda item: (-item[0], item[1], _lb_norm_text(item[2].get("name"))))
    selected = [row for _score, _index, row in ranked[: max(1, int(max_fields))]]
    selected_keys = {_lb_norm_text(row.get("name")) for row in selected}

    report = {
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "max_fields": int(max_fields),
        "outcome_mode": outcome_mode,
        "preferred_names_used": sorted(preferred & selected_keys),
        "selected_names": [str(row.get("name") or "") for row in selected],
        "excluded_names": sorted(excluded),
    }
    return selected, report

def official_catalog() -> tuple[dict[str, Any], ...]:
    """Return the active MongoDB-backed Low Balance scalar catalog.

    The catalog contains material journey fields from the active MongoDB TMF654 TopupBalance, TMF654 Bucket, and
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


def official_catalog_by_name() -> dict[str, dict[str, Any]]:
    """Return the current MongoDB-backed Low Balance catalog without process-local staleness."""
    return {str(row["name"]): dict(row) for row in official_catalog()}


def official_catalog_aliases() -> dict[str, str]:
    """Map legacy raw Swagger field names to the new canonical analytical variable names."""
    aliases: dict[str, str] = {}
    for row in material_low_balance_catalog():
        canonical = _normalize_name(row.get("name"))
        if not canonical:
            continue
        aliases[canonical] = canonical
        for alias in row.get("source_aliases") or []:
            normalized = _normalize_name(alias)
            if normalized:
                aliases[normalized] = canonical
        for path in row.get("source_paths") or []:
            normalized = _normalize_name(path)
            if normalized:
                aliases[normalized] = canonical
    return aliases


def canonical_low_balance_name(value: Any) -> str:
    """Resolve a canonical or legacy source field name to the canonical analytical name."""
    name = _normalize_name(value)
    if not name:
        return ""
    return official_catalog_aliases().get(name, name)


def validate_llm_official_selection(variables: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only exact official catalog variables returned by Gemini.

    Any LLM-created name that is not an exact normalized source leaf from the active MongoDB catalog is
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

def validate_db_definition(variable: dict[str, Any]) -> dict[str, Any]:
    """Validate a DB variable without changing its semantics or name."""
    model = GeneratedSchemaField.model_validate(variable)
    return model.model_dump()

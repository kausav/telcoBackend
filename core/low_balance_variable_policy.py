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

from core.agentic_models import GeneratedSchemaField
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
_LOW_BALANCE_TECHNICAL_SUFFIXES = ("_href", "_description", "_referred_type")
_LOW_BALANCE_PII_NAMES = {
    "customer_name",
    "customer_engaged_party_name",
    "topupbalance_requestor_name",
}
_LOW_BALANCE_DISPLAY_NAMES = {"bucket_remaining_value_name"}


def is_material_low_balance_spec(spec: dict[str, Any]) -> bool:
    """Return whether an official scalar leaf is analytically useful for Low Balance.

    The function only filters low-value metadata/PII from the supplied official JSON catalog.
    Every retained field remains traceable to TMF654/TMF629.
    """
    name = _normalize_name(spec.get("name"))
    if not name:
        return False
    if name in _LOW_BALANCE_PII_NAMES or name in _LOW_BALANCE_DISPLAY_NAMES:
        return False
    if name.endswith(_LOW_BALANCE_TECHNICAL_SUFFIXES):
        return False
    dtype = str(spec.get("dtype") or "string").strip().lower()
    return dtype not in {"object", "array"}


def material_low_balance_catalog() -> tuple[dict[str, Any], ...]:
    """Return the broad, quality-safe official Low Balance variable catalog."""
    rows = [dict(row) for row in expanded_scalar_catalog() if is_material_low_balance_spec(row)]
    rows.sort(key=lambda item: (str(item.get("model") or ""), str(item.get("path") or ""), str(item.get("name") or "")))
    return tuple(rows)


def validate_low_balance_required_identity_sources(db_variables: list[dict[str, Any]]) -> None:
    """Require the exact DB-only telecom identity fields that are absent from the two Swagger files.

    ``customer_id`` exists in the official TMF629 catalog. ``account_id`` and ``msisdn`` do not, so
    Low Balance must receive those exact names from MongoDB. The policy intentionally refuses to
    synthesize aliases or application-level replacements.
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

    customer = normalized.get("customer_id")
    if customer is not None:
        validate_db_definition(customer)
        if str(customer.get("scope") or "").strip().lower() != "entity":
            raise ValueError("MongoDB customer_id must have scope='entity' for Low Balance & Top-up.")
        gen = str(customer.get("gen") or "").strip().lower()
        params = customer.get("params") if isinstance(customer.get("params"), dict) else {}
        try:
            customer_digits = int(params.get("digits", 0) or 0)
        except (TypeError, ValueError):
            customer_digits = 0
        if gen != "prefixed_int" or str(params.get("prefix") or "") != "cust-" or customer_digits != 8:
            raise ValueError(
                "MongoDB customer_id is authoritative but its definition does not satisfy the required "
                "Low Balance contract: gen='prefixed_int', prefix='cust-', digits=8. Update the DB definition; the application will not rewrite it."
            )

    for name in LOW_BALANCE_DB_REQUIRED_FIELDS:
        item = normalized[_normalize_name(name)]
        if str(item.get("scope") or "").strip().lower() != "entity":
            raise ValueError(f"MongoDB {name} must have scope='entity' for Low Balance & Top-up.")
        if not bool(item.get("required")) or bool(item.get("nullable")):
            raise ValueError(f"MongoDB {name} must be required=true and nullable=false for Low Balance & Top-up.")
    if customer is not None and (not bool(customer.get("required")) or bool(customer.get("nullable"))):
        raise ValueError("MongoDB customer_id must be required=true and nullable=false for Low Balance & Top-up.")

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

def official_catalog() -> tuple[dict[str, Any], ...]:
    """Return the immutable official scalar catalog from the two bundled Swagger files."""
    rows = []
    for row in expanded_scalar_catalog():
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
    """Enforce the Low Balance executable-source boundary at proposal, confirmation, and generation time."""
    catalog = official_catalog_by_name()
    sources = {str(k).strip().casefold(): str(v).strip().upper() for k, v in (source_by_name or {}).items()}
    normalized_db_names = {_normalize_name(name) for name in (db_variable_names or set()) if str(name).strip()}
    allowed_db = {"DB_RECOMMENDED", "USER_SELECTED"}
    allowed_official = "OFFICIAL_JSON"
    invalid: list[str] = []
    seen_signatures: dict[tuple[str, str], str] = {}
    for raw in variables or []:
        if not isinstance(raw, dict):
            invalid.append("<invalid-variable>")
            continue
        name = _normalize_name(raw.get("name"))
        if not name:
            invalid.append("<missing-name>")
            continue
        source = sources.get(name, "")
        if source in allowed_db:
            if normalized_db_names and name not in normalized_db_names:
                invalid.append(name)
                continue
        elif source and source != allowed_official:
            invalid.append(name)
            continue
        elif name not in catalog:
            invalid.append(name)
            continue

        signature = semantic_signature(raw)
        prior = seen_signatures.get(signature)
        if prior and _normalize_name(prior) != name:
            invalid.append(
                f"{name} (duplicate business use of {prior})"
            )
        else:
            seen_signatures[signature] = name
    if invalid:
        raise ValueError(
            "Low Balance & Top-up executable variables must come only from the supplied TMF654/TMF629 Swagger scalar catalog or MongoDB variables. "
            + "Invalid variables: " + ", ".join(sorted(set(invalid)))
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


def validate_db_definition(variable: dict[str, Any]) -> dict[str, Any]:
    """Validate a DB variable without changing its semantics or name."""
    model = GeneratedSchemaField.model_validate(variable)
    return model.model_dump()

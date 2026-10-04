"""Scenario-aware planning helpers.

The source catalog and persisted DB variables are catalogs, not output schemas.  This module
selects only concepts supported by the request, preserves explicit user selections, closes
reference dependencies, and keeps the result domain-neutral.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from core.industry_source_store import (
    _canonical_catalog_dtype,
    _catalog_role,
    _normalize_model_name,
    _source_redundancy_signature,
    canonical_variable_semantic_key,
    normalize_lookup_key,
)

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "case", "country", "data",
    "dataset", "do", "for", "from", "generate", "high", "in", "industry", "is", "it", "of",
    "on", "or", "pii", "scenario", "sensitive", "the", "this", "to", "type", "use", "used",
    "using", "with", "without", "normal", "transactional", "aggregational", "fidelity", "synthetic",
}
_GENERIC_ROLE_WORDS = {
    "id", "key", "name", "type", "status", "state", "date", "time", "timestamp", "amount", "value",
    "unit", "units", "reason", "code", "description", "reference", "identifier", "field", "record",
    "records", "resource", "entity", "object", "data", "model", "detail", "details", "profile", "balance",
}
_ENTITY_WORDS = {
    "customer", "user", "member", "patient", "account", "policyholder", "client",
    "tenant", "person", "organization", "company", "party", "phone", "mobile",
}


def _tokens(value: Any) -> set[str]:
    text = str(value or "").casefold()
    raw = re.findall(r"[a-z0-9]+", text)
    out: set[str] = set()
    for token in raw:
        if token in _STOP:
            continue
        out.add(token)
        if token.endswith("ies") and len(token) > 4:
            out.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us")):
            out.add(token[:-1])
    return out


def _context_tokens(request: Any) -> set[str]:
    values = (
        getattr(request, "scenario_type", ""), getattr(request, "industry_type", ""), getattr(request, "domain", ""),
        getattr(request, "business_scenario", ""), getattr(request, "use_case", ""), getattr(request, "country", ""),
        getattr(request, "type_of_data", ""),
    )
    return _tokens(" ".join(str(x or "") for x in values))


def _field_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(k) or "") for k in (
        "name", "semantic_key", "path", "description", "model", "business_model",
        "source_owner_model", "source_owner_relation", "model_description",
    ))


def _source_direct_signal(row: dict[str, Any], context: set[str]) -> set[str]:
    """Return business-term evidence without counting the owning resource name as field evidence."""
    business_context = context - _GENERIC_ROLE_WORDS - _ENTITY_WORDS
    model_tokens = _tokens(row.get("business_model") or row.get("model"))
    field_tokens = _tokens(row.get("field") or row.get("name"))
    description_tokens = _tokens(row.get("description"))
    semantic_tokens = _tokens(row.get("semantic_key") or row.get("path"))
    # Remove owning model tokens from full flattened names/paths so every ``topupbalance_*`` field
    # does not become a direct hit merely because the scenario is about top-ups.
    signal = (field_tokens | description_tokens | semantic_tokens) - model_tokens
    return signal & business_context


def _relevance_score(row: dict[str, Any], context: set[str]) -> float:
    name = _tokens(row.get("name"))
    semantic = _tokens(row.get("semantic_key"))
    path = _tokens(row.get("path"))
    description = _tokens(row.get("description"))
    model = _tokens(row.get("business_model") or row.get("model"))
    strong_context = context - _GENERIC_ROLE_WORDS
    score = 0.0
    score += 14.0 * len(name & strong_context)
    score += 10.0 * len(semantic & strong_context)
    score += 7.0 * len(path & strong_context)
    score += 5.0 * len(description & strong_context)
    score += 12.0 * len(model & strong_context)
    if _catalog_role(row) == "identity":
        score += 8.0
    elif _catalog_role(row) in {"status", "timing", "measurement", "metric", "decision", "transaction", "event"}:
        score += 4.0
    if row.get("required"):
        score += 3.0
    return score


def _model_evidence(rows: list[dict[str, Any]], context: set[str], preferred_models: Iterable[str] | None = None) -> set[str]:
    requested = {_normalize_model_name(v) for v in (preferred_models or ()) if _normalize_model_name(v)}
    result: set[str] = set()
    for row in rows:
        model = _normalize_model_name(row.get("business_model") or row.get("model"))
        if not model:
            continue
        model_tokens = _tokens(model) - _GENERIC_ROLE_WORDS - _ENTITY_WORDS
        strong_context = context - _GENERIC_ROLE_WORDS - _ENTITY_WORDS
        strong = model_tokens & strong_context
        direct = _source_direct_signal(row, context)
        if strong or direct:
            result.add(model)
            continue
        # An exact LLM model request is accepted only when it has an independent strong field signal.
        if model in requested and _relevance_score(row, context) >= 32:
            result.add(model)
    return result


def _definition_key(row: dict[str, Any]) -> tuple:
    """Identity of what a source leaf *is*, independent of the resource that carries it.

    Sibling resources of a standard repeat the same leaf (``product[].name``, ``usageType``, ``status`` ...) with
    the same type, allowed values and meaning. Those copies are one business concept, not several variables.
    """
    path = str(row.get("path") or "")
    leaf = path.split(".", 1)[1] if "." in path else path
    return (leaf, str(row.get("dtype") or ""), tuple(row.get("enum_values") or ()), " ".join(str(row.get("description") or "").split()))


_REFERENCE_NOISE = {"related", "ref", "refs", "reference", "or", "value", "link", "linked"}
_HISTORY_WORDS = {"history", "log", "audit", "journal", "ledger", "trail", "archive"}
_HISTORY_ASKED = _HISTORY_WORDS | {"timeline", "event", "past", "previous", "changes"}
_HIERARCHY_WORDS = {"parent", "child", "hierarchy", "hierarchical", "linked", "chain", "predecessor", "successor", "previous"}


def _reference_target(row: dict[str, Any]) -> str:
    """The resource a reference leaf points at (``related_topup_balance`` -> ``topup_balance``), else ``""``.

    The source marks scalar leaves that sit inside a reference wrapper (``X.relatedY.id``) with the wrapper's
    kind; the wrapper's own name says which resource it refers to.
    """
    kinds = {str(row.get("source_owner_kind") or "")} | {str(k) for k in row.get("source_owner_kinds") or ()}
    if "support_reference" not in kinds:
        return ""
    parts = [p for p in str(row.get("source_owner_model") or "").split("_") if p and p not in _REFERENCE_NOISE]
    return "_".join(parts)


_REFERENCE_SEGMENT = re.compile(r"^(?P<name>.+?)(RefOrValue|Ref|Reference)$")


def _reference_path(row: dict[str, Any]) -> tuple[str, str]:
    """``(referenced resource, property path inside it)`` for a leaf reached through a reference-or-value segment, else ``("", "")``.

    ``UsageConsumption.bucketRefOrValue[].remainingValue.amount`` is the ``Bucket`` resource's own ``remainingValue.amount``
    seen through the reference.
    """
    segments = [seg.replace("[]", "") for seg in str(row.get("path") or "").split(".")]
    for index, segment in enumerate(segments[1:], 1):
        match = _REFERENCE_SEGMENT.match(segment)
        if match and index + 1 < len(segments):
            return _normalize_model_name(match.group("name")), ".".join(segments[index + 1:])
    return "", ""


def _description_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _same_meaning(a: Any, b: Any) -> bool:
    """Equal descriptions, or one is a sentence of the other (at least three words, so ``Unit`` never matches)."""
    a, b = _description_text(a), _description_text(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short.split()) >= 3 and short in long_


def _leaf_path(row: dict[str, Any]) -> str:
    path = str(row.get("path") or "")
    return path.split(".", 1)[1] if "." in path else path


def _row_dtype(row: dict[str, Any]) -> str:
    fmt = str(row.get("format") or "").strip().lower()
    if fmt in {"date-time", "datetime"}:
        return "datetime"
    if fmt == "date":
        return "date"
    return _canonical_catalog_dtype(row.get("dtype"), bool(row.get("enum_values")))


def _restating_rows(
    selected: list[dict[str, Any]],
    *,
    protected: set[str],
    candidates: Iterable[dict[str, Any]],
    db_definitions: Iterable[dict[str, Any]] | None,
    owner_models: set[str] | None,
) -> set[str]:
    """Names of selected rows that only restate a fact another selected variable already carries.

    * A resource that points at another resource (a history or log row that references the balance it records)
      repeats that resource's own leaves: same property, type, allowed values and meaning. When the referenced
      resource's leaf is present - as a source row or as a persisted variable - the referencing copy is redundant.
      The resource's own identifier is never a copy (each resource has one).
    * ``<entity>_<x>`` beside an existing ``<x>`` variable is that variable seen through the entity.
    """
    def compact(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    def model_of(row: dict[str, Any]) -> str:
        return _normalize_model_name(row.get("business_model") or row.get("model"))

    by_model: dict[str, list[dict[str, Any]]] = {}
    links: dict[str, set[str]] = {}
    for row in selected:
        by_model.setdefault(model_of(row), []).append(row)
    for row in candidates:                                    # every candidate, also the reference leaves that were not selected
        target = _reference_target(row)
        if target and target != model_of(row):
            links.setdefault(model_of(row), set()).add(target)

    held: list[tuple[str, str, str, str]] = []              # (model, leaf-agnostic dtype, description, name)
    db_names: set[str] = set()
    for item in db_definitions or ():
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        name = normalize_lookup_key(item.get("name"))
        db_names.add(name)
        for model in owner_models or ():
            if compact(name).startswith(compact(model)):
                held.append((model, str(item.get("dtype") or "").strip().lower(), str(item.get("description") or ""), name))

    removed: set[str] = set()
    for row in selected:
        name = normalize_lookup_key(row.get("name"))
        if not name or name in protected:
            continue
        model = model_of(row)
        if _catalog_role(row) != "identity":
            for target in links.get(model, ()):
                same_source = any(
                    other is not row
                    and normalize_lookup_key(other.get("name")) not in removed
                    and _leaf_path(other) == _leaf_path(row)
                    and _row_dtype(other) == _row_dtype(row)
                    and tuple(other.get("enum_values") or ()) == tuple(row.get("enum_values") or ())
                    and _same_meaning(other.get("description"), row.get("description"))
                    for other in by_model.get(target, ())
                )
                same_db = any(
                    owner == target and dtype == _row_dtype(row) and _same_meaning(description, row.get("description"))
                    for owner, dtype, description, _ in held
                )
                if same_source or same_db:
                    removed.add(name)
                    break
        if name in removed:
            continue
        # The same property of a resource reached through a reference-or-value segment of another resource.
        ref_model, ref_rest = _reference_path(row)
        if ref_model and any(
            _leaf_path(other) == ref_rest for other in by_model.get(ref_model, ()) if other is not row
        ):
            removed.add(name)
            continue
        # <entity>_<x> next to a persisted <x>: the same fact reached through the entity.
        model_tokens = _tokens(row.get("business_model") or row.get("model"))
        head, _, rest = name.partition("_")
        if head in _ENTITY_WORDS and head in model_tokens and rest in db_names and rest != name:
            removed.add(name)
    # One resource that reaches the same shared object through two paths (a summary under the resource and again under each of
    # its nested items) carries that object's leaves twice: the owning object, its role and the property are the same.
    # The shallowest path is the resource's own; the deeper ones are repeats.
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in selected:
        name = normalize_lookup_key(row.get("name"))
        signature = _source_redundancy_signature(row)
        if signature is None or name in removed or name in protected:
            continue
        groups.setdefault((model_of(row), signature), []).append(row)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        keep = min(rows, key=lambda r: str(r.get("path") or "").count("."))
        for row in rows:
            if row is not keep:
                removed.add(normalize_lookup_key(row.get("name")))
    return removed


def owning_models(variables: Iterable[dict[str, Any]] | None, rows: list[dict[str, Any]]) -> set[str]:
    """Source models that own at least one of the given (curated) variables, matched on the flattened name prefix."""
    def compact(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    names = [compact(v.get("name")) for v in variables or [] if isinstance(v, dict) and v.get("name")]
    models = {_normalize_model_name(r.get("business_model") or r.get("model")) for r in rows}
    return {m for m in models if m and any(n.startswith(compact(m)) for n in names)}


def select_source_rows(
    rows: list[dict[str, Any]],
    *,
    context: set[str],
    preferred_names: set[str] | None = None,
    preferred_models: set[str] | None = None,
    excluded_names: set[str] | None = None,
    excluded_semantic_keys: set[str] | None = None,
    forced_names: set[str] | None = None,
    owner_models: set[str] | None = None,
    db_definitions: Iterable[dict[str, Any]] | None = None,
    max_fields: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select scenario-relevant source leaves without whole-model expansion."""
    limit = max(0, int(max_fields))
    preferred = {normalize_lookup_key(x) for x in preferred_names or set() if normalize_lookup_key(x)}
    excluded = {normalize_lookup_key(x) for x in excluded_names or set() if normalize_lookup_key(x)}
    excluded_sem = {normalize_lookup_key(x) for x in excluded_semantic_keys or set() if normalize_lookup_key(x)}
    # Fields the caller requires (for example what a DB variable depends on): selected unconditionally.
    forced = {normalize_lookup_key(x) for x in forced_names or set() if normalize_lookup_key(x)}

    canonical: list[dict[str, Any]] = []
    seen_semantics: set[str] = set()
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        name = normalize_lookup_key(row.get("name"))
        semantic = canonical_variable_semantic_key(row)
        if not name or name in excluded:
            continue
        if semantic and (semantic in seen_semantics or semantic in excluded_sem):
            continue
        if semantic:
            seen_semantics.add(semantic)
        kind = str(row.get("model_kind") or "resource")
        if kind in {"support", "support_reference", "abstract", "auxiliary", "crud_wrapper", "event_wrapper"}:
            continue
        canonical.append(row)

    models = _model_evidence(canonical, context, preferred_models)
    # LLM-selected source field names are useful evidence, but they must still clear a deterministic
    # relevance floor before their owning model is opened. This lets a valid field selection activate
    # its model without allowing a single noisy profile/reference field to resurrect an entire model.
    for row in canonical:
        name = normalize_lookup_key(row.get("name"))
        if name not in preferred:
            continue
        model = _normalize_model_name(row.get("business_model") or row.get("model"))
        if not model or _catalog_role(row) in {"profile", "configuration", "support", "support_reference", "abstract", "auxiliary", "other"}:
            continue
        direct = bool(_source_direct_signal(row, context))
        if direct or _relevance_score(row, context) >= 28:
            models.add(model)
    for row in canonical:
        if normalize_lookup_key(row.get("name")) in forced:
            model = _normalize_model_name(row.get("business_model") or row.get("model"))
            if model:
                models.add(model)
    anchor_models: set[str] = set()

    # Entity/resource anchors are selected without allowing generic "id" or "account" words to
    # open an unrelated source model.
    entity_candidates = [
        row for row in canonical
        if _catalog_role(row) == "identity"
        and (_tokens(row.get("name")) & _ENTITY_WORDS)
    ]
    for row in sorted(entity_candidates, key=lambda r: (-_relevance_score(r, context), str(r.get("name") or ""))):
        model = _normalize_model_name(row.get("business_model") or row.get("model"))
        if model:
            models.add(model)
            anchor_models.add(model)
            break

    # The resources the curated DB variables belong to are the scenario's primary resources. Sibling
    # resources that merely share vocabulary with the request (other operations on the same balance,
    # say) would only add look-alike reference columns, so they are not opened.
    # That only holds when the curated variables describe resources beyond the entity itself. A scenario whose
    # curated variables are just the entity's identifiers has no resource of its own yet, so the scenario
    # context alone decides which source models are relevant.
    requested_models = {_normalize_model_name(v) for v in (preferred_models or ()) if _normalize_model_name(v)}
    resource_owners = {m for m in (owner_models or set()) if not (_tokens(m) & _ENTITY_WORDS)}
    if resource_owners or requested_models:
        models = {m for m in models if m in resource_owners or m in requested_models or m in anchor_models}

    # Add source-defined related models only one hop from an already relevant model.  This is a
    # structural relation, not a broad resource expansion.
    related: set[str] = set()
    all_models = {
        _normalize_model_name(r.get("business_model") or r.get("model"))
        for r in canonical
        if _normalize_model_name(r.get("business_model") or r.get("model"))
    }
    graph: dict[str, set[str]] = {}
    for row in canonical:
        source_model = _normalize_model_name(row.get("business_model") or row.get("model"))
        if not source_model:
            continue
        for target in row.get("linked_business_models") or []:
            target_model = _normalize_model_name(target)
            if target_model in all_models and target_model != source_model:
                graph.setdefault(source_model, set()).add(target_model)
    for source in sorted(models):
        for target in sorted(graph.get(source, set())):
            if target not in models:
                related.add(target)

    def reference_leaf(row: dict[str, Any], model: str) -> bool:
        """An identifier that points at another resource, rather than naming this one or the entity."""
        if _catalog_role(row) != "identity":
            return False
        rest = _tokens(row.get("name")) - _tokens(model)
        rest -= {"id", "identifier", "key"}
        if not rest:
            return False                                        # the model's own identifier
        return not (model in anchor_models and rest <= _ENTITY_WORDS)

    def admissible(row: dict[str, Any], model: str, *, direct: bool, related_model: bool = False) -> bool:
        role = _catalog_role(row)
        if normalize_lookup_key(row.get("name")) not in forced and reference_leaf(row, model):
            return False
        if direct or row.get("required"):
            return True
        # Entity anchor models are opened only to provide identity/relationship anchors unless the
        # request explicitly mentions one of their other concepts. This prevents customer-profile
        # fields such as credit score/risk rating from leaking into unrelated transactional scenarios.
        if model in anchor_models and role != "identity":
            return False
        if role in {"profile", "configuration", "other"}:
            return False
        if related_model and role not in {"identity", "status", "timing", "measurement", "metric", "decision", "transaction", "event"}:
            return False
        if role == "categorical":
            # Generic descriptive categorical leaves (channel_name, method_name, etc.) are not
            # structurally required. Keep categorical state/type fields only when they have direct
            # scenario evidence or belong to an explicitly selected source model.
            leaf_tokens = _tokens(row.get("field") or row.get("name"))
            if not (leaf_tokens & {"status", "state", "usage", "category", "mode", "type", "decision", "outcome", "auto", "automatic", "recurring", "period"}):
                return False
        return True

    # A leaf inside a reference to a resource that is already part of the dataset (a top-up that points at a related
    # top-up, a history row that points at the top-up it records) restates that resource: its id is the id already
    # present, its name and role describe the resource's own columns. Only a request that is about the link itself
    # (parent/child, hierarchy, chains) keeps such leaves.
    hierarchy_asked = bool(context & _HIERARCHY_WORDS)
    entity_only = _ENTITY_WORDS | {"id", "identifier", "key"}

    def points_at_present_resource(row: dict[str, Any], model: str) -> bool:
        target = _reference_target(row)
        if not target or hierarchy_asked:
            return False
        name = normalize_lookup_key(row.get("name"))
        if name in forced or (name in preferred and _tokens(name) <= entity_only):
            return False
        return target == model or target in models

    # A history/log/audit resource that records actions on a resource already in the dataset describes those actions: its fields
    # either restate the resource (status, dates, amounts) or are specific to an operation the scenario does not run (the cost
    # of a transfer on a top-up). It is only wanted when the request asks for history, audit or a timeline.
    history_asked = bool(context & _HISTORY_ASKED)
    references: dict[str, set[str]] = {}
    for row in canonical:
        target = _reference_target(row)
        owner = _normalize_model_name(row.get("business_model") or row.get("model"))
        if target and owner and target != owner:
            references.setdefault(owner, set()).add(target)

    def history_of_present_resource(model: str) -> bool:
        if history_asked or not (_tokens(model.replace("_", " ")) & _HISTORY_WORDS):
            return False
        return bool(references.get(model, set()) & models)

    scored: list[tuple[float, dict[str, Any], str]] = []
    for row in canonical:
        model = _normalize_model_name(row.get("business_model") or row.get("model"))
        if model not in models and model not in related:
            continue
        if points_at_present_resource(row, model) or (history_of_present_resource(model) and normalize_lookup_key(row.get("name")) not in forced):
            continue
        score = _relevance_score(row, context)
        name = normalize_lookup_key(row.get("name"))
        direct = bool(_source_direct_signal(row, context))
        if name in preferred:
            score += 100.0
            direct = True
        if name in forced:
            score += 1000.0
            direct = True
        role = _catalog_role(row)
        if model in related:
            score -= 10.0
            if direct:
                score += 12.0
            elif role not in {"identity", "status", "timing", "measurement", "metric"}:
                score -= 14.0
        if model not in models:
            continue
        if not admissible(row, model, direct=direct, related_model=(model in related)):
            continue
        scored.append((score, row, model))

    # Keep a small amount of structural coverage per relevant model, then the strongest direct fields.
    selected: list[dict[str, Any]] = []
    selected_sem: set[str] = set()
    selected_defs: set[tuple] = set()

    def repeated(row: dict[str, Any]) -> bool:
        name = normalize_lookup_key(row.get("name"))
        return name not in preferred and name not in forced and _definition_key(row) in selected_defs

    # Models whose own name matches the scenario come first (then by their best field), and every opened model
    # gets a fair share of the budget, so the order of names in the catalog never decides what is selected.
    name_evidence = context - _GENERIC_ROLE_WORDS - _ENTITY_WORDS
    best_score: dict[str, float] = {}
    for score, _row, model in scored:
        best_score[model] = max(best_score.get(model, float("-inf")), score)
    ranked_models = sorted(models, key=lambda m: (-len((_tokens(m) - _GENERIC_ROLE_WORDS - _ENTITY_WORDS) & name_evidence), -best_score.get(m, 0.0), m))
    share = max(3, min(12, limit // max(1, len(ranked_models))))
    for model in ranked_models:
        candidates = sorted(
            [item for item in scored if item[2] == model],
            key=lambda item: (-item[0], item[1].get("depth", 0), normalize_lookup_key(item[1].get("name"))),
        )
        for score, row, _ in candidates[:share]:
            sem = canonical_variable_semantic_key(row) or normalize_lookup_key(row.get("name"))
            if sem in selected_sem or repeated(row):
                continue
            direct = bool(_source_direct_signal(row, context))
            if not admissible(row, model, direct=direct, related_model=False):
                continue
            selected.append(dict(row)); selected_sem.add(sem); selected_defs.add(_definition_key(row))

    for score, row, model in sorted(scored, key=lambda x: (-x[0], x[2], normalize_lookup_key(x[1].get("name")))):
        if len(selected) >= limit:
            break
        sem = canonical_variable_semantic_key(row) or normalize_lookup_key(row.get("name"))
        if sem in selected_sem or repeated(row):
            continue
        direct = bool(_source_direct_signal(row, context))
        if not admissible(row, model, direct=direct, related_model=(model in related)):
            continue
        if model in related and score < 18 and normalize_lookup_key(row.get("name")) not in preferred:
            continue
        selected.append(dict(row)); selected_sem.add(sem); selected_defs.add(_definition_key(row))

    # Resolve explicit preferred fields last only when their owning model is scenario-relevant. This
    # protects against one noisy LLM field opening a sibling operation model.
    for row in canonical:
        if len(selected) >= limit:
            break
        name = normalize_lookup_key(row.get("name"))
        model = _normalize_model_name(row.get("business_model") or row.get("model"))
        sem = canonical_variable_semantic_key(row) or name
        if name not in preferred or sem in selected_sem or model not in models or points_at_present_resource(row, model) or history_of_present_resource(model):
            continue
        selected.append(dict(row)); selected_sem.add(sem)

    # Copies of a fact another selected variable already carries (see _restating_rows) are dropped last, so which
    # of two look-alike resources keeps the fact never depends on the order candidates were visited in.
    restating = _restating_rows(selected, protected=forced, candidates=canonical, db_definitions=db_definitions, owner_models=owner_models)
    if restating:
        selected = [r for r in selected if normalize_lookup_key(r.get("name")) not in restating]

    # Forced fields come first so a budget can never squeeze them out.
    forced_rows = [dict(r) for r in canonical if normalize_lookup_key(r.get("name")) in forced]
    forced_keys = {canonical_variable_semantic_key(r) or normalize_lookup_key(r.get("name")) for r in forced_rows}
    selected = forced_rows + [r for r in selected if (canonical_variable_semantic_key(r) or normalize_lookup_key(r.get("name"))) not in forced_keys]
    selected = selected[:max(limit, len(forced_rows))]
    report = {
        "candidate_count": len(canonical),
        "canonical_candidate_count": len(canonical),
        "eligible_count_before_cap": len(selected),
        "selected_count": len(selected),
        "max_fields": limit,
        "capped": len(selected) >= limit and len(scored) > limit,
        "truncated_count": max(0, len(scored) - len(selected)),
        "selection_mode": "scenario_relevance_no_whole_model_expansion",
        "relevant_models": sorted(models),
        "related_models": sorted(related),
        "selected_business_models": sorted({_normalize_model_name(r.get("business_model") or r.get("model")) for r in selected}),
        "selected_names": [str(r.get("name") or "") for r in selected],
        "preferred_names_used": sorted(set(normalize_lookup_key(r.get("name")) for r in selected) & preferred),
        "excluded_names": sorted(excluded),
        "excluded_semantic_keys": sorted(excluded_sem),
        "restating_removed": sorted(restating),
    }
    return selected, report


def select_db_variables_for_scenario(
    variables: list[dict[str, Any]],
    *,
    context: Iterable[str],
    selected_source_models: Iterable[str] = (),
    user_selected_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select relevant persisted DB concepts; USER_SELECTED is always retained."""
    context_set = set(context or set())
    strong_context = context_set - _GENERIC_ROLE_WORDS - _ENTITY_WORDS
    model_tokens = _tokens(" ".join(selected_source_models)) - _GENERIC_ROLE_WORDS
    explicit = {normalize_lookup_key(x) for x in user_selected_names if normalize_lookup_key(x)}
    scored: list[tuple[float, dict[str, Any]]] = []
    for raw in variables or []:
        item = dict(raw)
        name = normalize_lookup_key(item.get("name"))
        if not name:
            continue
        if name in explicit:
            scored.append((1000.0, item)); continue
        text_tokens = _tokens(_field_text(item))
        field_name_tokens = _tokens(item.get("name"))
        direct = (_tokens(item.get("description")) | _tokens(item.get("field"))) & strong_context
        model_overlap = text_tokens & model_tokens
        role = _catalog_role(item)
        relationship_markers = {
            "party", "requestor", "owner", "receiver", "related", "payment", "method",
            "channel", "product", "logical", "resource", "voucher", "reference", "href", "role",
        }
        explicit_entity_name = bool(field_name_tokens & {"phone", "mobile"}) or normalize_lookup_key(item.get("name")) in {
            "customer_id", "user_id", "member_id", "patient_id", "account_id"
        }
        entity_identity = explicit_entity_name or (
            role == "identity"
            and bool(field_name_tokens & {"customer", "user", "member", "patient"})
            and not (field_name_tokens & relationship_markers)
        )
        resource_identity = bool(role == "identity" and model_overlap and not (field_name_tokens & relationship_markers))
        relationship_identity = role == "identity" and not entity_identity and not resource_identity
        if field_name_tokens & relationship_markers and not direct:
            # Party/channel/payment/reference metadata is auxiliary unless the scenario explicitly
            # asks for that relationship. Do not widen an otherwise focused contract with these leaves.
            continue
        matched_model = ""
        field_norm = normalize_lookup_key(item.get("name"))
        for model_name in sorted(set(_normalize_model_name(v) for v in selected_source_models if _normalize_model_name(v)), key=len, reverse=True):
            model_norm = normalize_lookup_key(model_name)
            if model_norm and (field_norm == model_norm or field_norm.startswith(model_norm + "_")):
                matched_model = model_norm
                break
        entity_model = bool(matched_model and _tokens(matched_model) & _ENTITY_WORDS)
        decision_context = context_set & {
            "offer", "accepted", "accept", "acceptance", "response", "decision",
            "intervention", "communication", "contact",
        }
        score = 14.0 * len(direct) + 8.0 * len(model_overlap)
        # An intervention/offer-style scenario has an implicit decision point when the persisted
        # catalog exposes a concrete offer/acceptance concept. This is semantic evidence from the
        # request terms, not a field-name allowlist for one industry.
        if (field_name_tokens & {"offer", "accepted", "accept", "acceptance", "recommend", "recommended"}) and (context_set & {"intervention", "offer"}):
            score += 18
            direct = True
        # Entity identity is structurally useful even if no model is explicitly named in the request.
        if entity_identity:
            score += 22
        if role in {"status", "timing", "measurement", "metric", "decision", "transaction", "event"}:
            score += 6
        # A relationship/reference identity (party/account/requestor/payment-method/etc.) is not
        # automatically required just because it shares the selected resource model. Keep only
        # identities that are themselves the scenario/entity anchor or are directly requested.
        relationship_identity = role == "identity" and not entity_identity
        if model_overlap and role == "identity" and not relationship_identity:
            score += 10
        elif relationship_identity and not direct:
            score -= 18
        # Keep state/type categorical fields that materially define the selected resource, but not
        # descriptive names such as channel_name/payment_method_name.
        categorical_leaf = field_name_tokens & {"status", "state", "usage", "category", "mode", "type", "decision", "outcome"}
        if model_overlap and role == "categorical" and categorical_leaf:
            score += 6
        # Quantity/unit pairs are useful support fields when a selected resource exposes both sides.
        if model_overlap and (field_name_tokens & {"unit", "units", "amount", "value", "quantity", "remaining", "reserved"}):
            score += 6
        # Auto/recurring configuration belongs with the transaction resource, not with the generic DB catalog.
        if model_overlap and field_name_tokens & {"auto", "automatic", "recurring", "period", "occurrence", "occurrences"}:
            score += 6
        # Only source/DB concepts that materially contribute to the request are retained. Generic entity
        # profile/status/validity metadata is not enough on its own.
        if role == "other" and not direct and not model_overlap and not entity_identity:
            continue
        if not direct and role == "categorical":
            leaf_tokens = _tokens(item.get("name"))
            if not (leaf_tokens & {"status", "state", "usage", "category", "mode", "type", "decision", "outcome", "auto", "automatic", "recurring", "period"}):
                continue
        # Generic descriptive leaves need stronger evidence than their resource prefix. For example,
        # a ``reason`` field is useful for exception/suppression/recovery journeys, but should not be
        # emitted merely because the scenario is about top-ups.
        leaf_tokens = set(_tokens(item.get("name")))
        exception_context = context_set & {"failure", "failed", "exception", "suppression", "decline", "declined", "rejected", "rejection", "recovery", "retry", "reason", "cause"}
        if not exception_context and leaf_tokens & {"reason", "voucher"}:
            continue
        # A free-form semantic string without authoritative choices/examples/pattern is not useful
        # merely because its resource prefix matches. Omit it unless the request directly asks for
        # that concept.
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        if (
            not direct
            and str(item.get("dtype") or "string").casefold() == "string"
            and not params.get("choices") and not params.get("values") and params.get("pattern") is None
            and not params.get("source_examples")
            and role in {"categorical", "other"}
            and not entity_identity
            and not (field_name_tokens & {"unit", "units"})
        ):
            continue
        if relationship_identity and not direct:
            continue
        # Customer/subscriber profile/status metadata is an entity catalog, not a reason to widen a
        # transaction scenario. Retain identity anchors and response/decision attributes when the
        # request explicitly contains a decision/response concept.
        entity_decision_signal = bool(decision_context and role in {"timing", "decision", "event"} and any(
            token in field_name_tokens for token in {"response", "event", "interaction", "engaged", "engagement", "decision"}
        ))
        if entity_model and not entity_identity and not direct and not entity_decision_signal:
            continue
        if score >= 14:
            scored.append((score, item))

    selected = [item for _score, item in sorted(scored, key=lambda x: (-x[0], normalize_lookup_key(x[1].get("name"))))]

    # Close persisted dependencies so a selected DB variable can execute without resurrecting an
    # otherwise irrelevant variable later in the generator.
    by_name = {normalize_lookup_key(v.get("name")): v for v in variables or [] if normalize_lookup_key(v.get("name"))}
    selected_names = {normalize_lookup_key(v.get("name")) for v in selected}
    changed = True
    while changed:
        changed = False
        for item in list(selected):
            for dep in item.get("depends_on", []) or []:
                key = normalize_lookup_key(dep)
                dep_item = by_name.get(key)
                if dep_item and key not in selected_names:
                    selected.append(dict(dep_item)); selected_names.add(key); changed = True

    return selected

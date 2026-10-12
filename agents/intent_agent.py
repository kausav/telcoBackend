"""Gemini intent agent: natural-language scenario in, validated semantic intent out.

The proposal path deliberately does not use provider-native structured-output schemas.
Gemini's structured-schema surface accepts only a subset of JSON Schema, and complex
Pydantic schemas can be rejected with 400 INVALID_ARGUMENT before the model returns an
answer. We therefore request plain JSON and validate it locally with Pydantic.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.agentic_models import GeneratedSchemaField, ScenarioIntent
from core.errors import LLMUpstreamError
from core.llm_client import GeminiClient
from core.json_domain_policy import is_json_grounded_domain
from core.scenario_variable_store import get_selection_lock, lock_selection
from synth.definitions import is_placeholder_only
from core.industry_source_store import (
    catalog_for_request,
    normalize_lookup_key,
    _catalog_business_metadata_penalty,
    _catalog_display_only,
    _catalog_role,
    _catalog_text_tokens,
    _canonical_context_tokens,
    _source_owner_family,
    _normalize_model_name,
    validate_catalog_selection,
    dedupe_catalog_against_db,
)
from config.runtime import JSON_SOURCE_LLM_CATALOG_LIMIT, JSON_SOURCE_LLM_FIELDS_PER_MODEL

logger = logging.getLogger(__name__)


INSTRUCTIONS = """
You are the intent-understanding and variable-ideation agent for a multi-industry synthetic-data platform.
Interpret the COMPLETE business request and propose a FRESH semantic variable set.

IMPORTANT BOUNDARIES:
- Return JSON only. Do not wrap the JSON in markdown fences.
- Do not output executable generators, generator parameters, formulas, SQL, or schema implementation details.
- Candidate variables are semantic ideas only. The deterministic compiler assigns executable generator contracts.
- scenarioId is an identifier only and MUST NOT influence what variables are proposed.
- Use scenarioType, industryType, domain, businessScenario, useCase, country, and typeOfData together. Entity key is backend schema metadata; do not use it to ideate variables.
- VARIABLE BREADTH IS A FIRST-CLASS REQUIREMENT. Do NOT return a short representative list. Review the
  supplied source catalog and maximize recall of materially relevant variables while staying strictly
  inside the authoritative source boundary. When two source-backed fields both add legitimate business,
  causal, temporal, relational, lifecycle, measurement, segmentation, configuration, or analytical value,
  include BOTH. Do not omit a relevant field merely to keep the list concise. The deterministic compiler
  performs an additional recall-first expansion over the complete MongoDB catalog, so this LLM list is a
  relevance/preference signal and is never the final breadth gate.
- There is NO small variable-count target. The only hard upper bound is the backend safety budget. Aim for
  the largest high-integrity related set the supplied catalog can support; do not optimize for brevity.
- Use a domain-neutral coverage pass: identify relevant entity/profile fields, identifiers and relationships,
  lifecycle/event/transaction fields, states/outcomes/reasons, timestamps/dates/durations, monetary/quantity/usage
  measures, channels/methods, configuration/eligibility, geography/segment attributes, and other fields that
  materially explain the scenario. Skip only fields that are truly redundant or are transport/display metadata.
  The deterministic compiler assigns executable contracts and rejects anything without a safe source-backed contract.
- Do not add application-specific identity anchors unless the supplied MongoDB source catalog or persisted
  MongoDB variables explicitly contain them. All executable variables must remain inside those authoritative
  sources regardless of industry or domain.
- Review every supplied source model card before selecting variables. ``requested_entities`` must contain only exact
  source business-model names from those cards and should identify the source models whose fields materially support the
  scenario. When a query/task model merely wraps or retrieves a concrete business resource, prefer the concrete resource
  model for data variables and use query/task fields only when the query itself is the subject of the scenario.
- When MongoDB source grounding is active, candidate variable names MUST exactly match a supplied MongoDB source field.
  When scenario_variables grounding is active, names MUST exactly match persisted MongoDB variable names.
- Avoid true semantic duplicates. Keep distinct fields when they represent different business concepts,
  entities, lifecycle steps, measures, relationships, or time points—even when their leaf names or descriptions
  are identical. A generic leaf such as ``status``, ``amount``, ``id``, ``percentage`` or ``date`` is NOT a global
  duplicate key. Treat two fields as duplicates only when the supplied source structure identifies the same owning
  business concept, relationship role, relative property path, and executable contract. Do not collapse
  ``Payment.status`` with ``Order.status`` merely because both are named ``status``. Conversely, repeated
  API wrappers of the same reusable schema concept should be represented by one canonical field. Never add copies
  merely to increase variable count.
- For transactional data, distinguish stable entity/profile fields from repeated transaction/event/decision fields using grain.
- Prefer variables that explain triggers, states, transitions, outcomes, timing, monetary/usage measures,
  decisions, contention, suppression, recovery, or retention when those concepts fit the scenario.
- Treat scenarioType as a behavioral mode and make the variable set materially reflect it. For positive/normal
  journeys, prioritize the source-backed fields that describe the intended successful lifecycle; for negative,
  exception, suppression, decline, failure, recovery, or mixed journeys, prioritize the corresponding state,
  reason, decision, timing, retry, recovery, and outcome fields. Never invent an outcome value; use only source-defined
  values. Scenario type changes relevance, not the authoritative vocabulary.
- Cover only concepts justified by the current business scenario and domain.
- The MongoDB source catalog is authoritative grounding, not a variable template. Select only source-backed fields that materially fit the request.
- For any JSON-grounded domain, the active MongoDB source catalog is the ONLY source from which the LLM may select variables. Review the complete catalog and return materially useful variables for the scenario. Every returned candidate variable name MUST exactly match one variable name from that catalog. Never invent, rename, alias, paraphrase, or synthesize a variable name.
- MongoDB scenario/user variables may contain additional business variables that are not present in source documents. Those DB variables are immutable inputs: never rename, rewrite, replace, or generate them. Review the full DB variable list and omit a source variable when it represents the same business use as a DB variable. DB variables always win semantic duplicates.
- Scenario-specific executable variables are NOT allowed to be invented by the LLM. If a concept is absent from the supplied MongoDB source catalog and is not already present as a persisted MongoDB variable, do not create it.
- If the business request asks for a behavioral concept that the supplied source models do not directly represent, do NOT substitute a nearby but different concept. Record that gap in ``notes`` using the source terminology (for example, qualification is not the same thing as acceptance, and a qualification result is not itself a conversion event).
- Do not propose unsupported nested/object fields when a flat synthetic dataset cannot deterministically
  populate their nested structure.
- Also return a small set of high-confidence behavioral_rules when the supplied fields support them.
  These rules are semantic relationships, not generators or formulas. Every field referenced by a rule
  MUST be an exact name from candidate_variables or a persisted MongoDB variable supplied to you.
  Use only simple machine-checkable conditions and assignments so the backend can validate and execute them deterministically.
  Equality example:
  {"when":{"offer_presented_flag":false},"then":{"offer_accepted_flag":false}}
  Threshold example:
  {"when":{"days_to_depletion":{"op":"<=","value":2}},"then":{"offer_presented_flag":true}}
  Conditions may use only equality/inequality (==, !=), numeric comparisons (<, <=, >, >=), or membership (in, not_in).
  The operator object must be {"op":"<=","value":2}; assignments remain direct scalar/list values.
  Only include relationships that are obvious from the field names/descriptions and the business scenario.
  Do not force a positive outcome merely because scenarioType is "Normal"; normal means a realistic mix
  of outcomes unless the business request explicitly specifies otherwise. Limit behavioral_rules to 30 items.

Return this JSON shape. `subdomain` is optional domain context; use "unknown" when the supplied request/source catalog does not define a meaningful subdomain taxonomy.
{
  "industry_type": "...",
  "domain": "...",
  "subdomain": "unknown",
  "scenario_type": "...",
  "type_of_data": "transactional|aggregational",
  "entity_key": "...",
  "use_case": "...",
  "requested_entities": ["..."],
  "requested_relationships": ["..."],
  "candidate_variables": [
    {
      "name": "exact_mongodb_source_name",
      "description": "...",
      "role": "identity|profile|event|transaction|status|measurement|metric|timing|decision|configuration|derived|other",
      "grain": "entity|transaction|event|derived",
      "dtype": "string|integer|float|decimal|boolean|categorical|datetime|date",
      "depends_on": ["existing_candidate_variable_name"],
      "useCase": "Relevant use-case label when applicable"
    }
  ],
  "behavioral_rules": [
    {
      "when": {"exact_field_name": "exact_value"},
      "then": {"exact_field_name": "exact_value"}
    },
    {
      "when": {"exact_numeric_field": {"op": "<=", "value": 2}},
      "then": {"exact_boolean_field": true}
    }
  ],
  "country": "...",
  "currency": "...",
  "record_count": null,
  "time_window_days": null,
  "notes": ["..."],
  "ambiguities": ["..."]
}
""".strip()


def _safe_exception_text(exc: Exception) -> str:
    text = str(exc)[:2000]
    return re.sub(
        r"(?i)(api[_-]?key|token|authorization|bearer|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )


def _extract_json_payload(payload: dict | list) -> dict[str, Any]:
    """Normalize Gemini JSON-mode output to one object suitable for validation."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    raise ValueError("Gemini returned JSON, but the intent payload was not a single object")


def _parse_text_json(text: str) -> dict[str, Any]:
    """Parse JSON from a plain-text Gemini response, tolerating markdown fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    return _extract_json_payload(parsed)


def _as_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Make common model-output deviations harmless before strict app validation."""
    normalized = dict(payload)

    variables: list[dict[str, Any]] = []
    allowed_roles = {
        "identity", "profile", "event", "transaction", "status", "measurement",
        "metric", "timing", "decision", "configuration", "derived", "other",
    }
    allowed_grains = {"entity", "transaction", "event", "derived"}
    allowed_dtypes = {"string", "integer", "float", "decimal", "boolean", "categorical", "datetime", "date"}

    # Candidate variables are intentionally unbounded. We preserve every model-proposed
    # variable and only normalize/deduplicate it; scenario semantics determine the final count.
    for raw in _as_list(normalized.get("candidate_variables"), len(normalized.get("candidate_variables") or [])):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if len(name) < 2:
            continue
        item = {
            "name": name[:120],
            "description": str(raw.get("description") or "")[:500],
            "role": str(raw.get("role") or "other").strip().lower(),
            "grain": str(raw.get("grain") or "transaction").strip().lower(),
            "dtype": str(raw.get("dtype") or "string").strip().lower(),
            "depends_on": [
                str(dep).strip()
                for dep in _as_list(raw.get("depends_on"), 8)
                if str(dep).strip()
            ],
            "useCase": (str(raw.get("useCase") or "").strip()[:300] or None),
        }
        if item["role"] not in allowed_roles:
            item["role"] = "other"
        if item["grain"] not in allowed_grains:
            item["grain"] = "transaction"
        if item["dtype"] not in allowed_dtypes:
            item["dtype"] = "string"
        variables.append(item)

    # Preserve first occurrence. There is intentionally no hard application limit.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in variables:
        key = item["name"].casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    normalized["candidate_variables"] = deduped

    # Keep only simple, machine-checkable behavioral relationships. Field names are validated
    # against the candidate set + persisted DB variables later, after source naming is canonicalized.
    raw_rules = normalized.get("behavioral_rules")
    normalized_rules: list[dict[str, Any]] = []
    if isinstance(raw_rules, list):
        for raw_rule in raw_rules[:30]:
            if not isinstance(raw_rule, dict):
                continue
            when = raw_rule.get("when") or raw_rule.get("conditions")
            then = raw_rule.get("then") or raw_rule.get("set")
            if not isinstance(when, dict) or not isinstance(then, dict) or not when or not then:
                continue
            safe_when = {}
            safe_then = {}
            for key, value in list(when.items())[:8]:
                name = str(key or "").strip()
                if not name:
                    continue
                if isinstance(value, dict):
                    op = str(value.get("op") or "").strip().lower()
                    operand = value.get("value")
                    if op in {"=", "==", "eq", "!=", "ne", "<", "lt", "<=", "lte", ">", "gt", ">=", "gte", "in", "not_in", "notin"} and (isinstance(operand, (str, int, float, bool)) or isinstance(operand, list)) and not isinstance(operand, (dict, tuple, set)):
                        if isinstance(operand, list) and any(isinstance(item, (dict, tuple, set)) for item in operand):
                            continue
                        safe_when[name[:120]] = {"op": op, "value": operand}
                elif isinstance(value, (str, int, float, bool)):
                    safe_when[name[:120]] = value
            for key, value in list(then.items())[:8]:
                name = str(key or "").strip()
                if name and isinstance(value, (str, int, float, bool)):
                    safe_then[name[:120]] = value
            if safe_when and safe_then:
                normalized_rules.append({"when": safe_when, "then": safe_then})
    normalized["behavioral_rules"] = normalized_rules

    # These are backend-owned and are overwritten after validation, so invalid model
    # guesses here should never make the provider request or application response fail.
    if not isinstance(normalized.get("industry_type"), str):
        normalized["industry_type"] = ""
    if not isinstance(normalized.get("domain"), str):
        normalized["domain"] = ""
    if not isinstance(normalized.get("scenario_type"), str):
        normalized["scenario_type"] = ""
    if not isinstance(normalized.get("type_of_data"), str):
        normalized["type_of_data"] = "transactional"
    if not isinstance(normalized.get("subdomain"), str) or not normalized.get("subdomain", "").strip():
        normalized["subdomain"] = "unknown"
    else:
        normalized["subdomain"] = str(normalized["subdomain"]).strip()[:100]
    if not isinstance(normalized.get("entity_key"), str):
        normalized["entity_key"] = ""
    if not isinstance(normalized.get("use_case"), str):
        normalized["use_case"] = ""
    for key in ("country", "currency"):
        value = normalized.get(key)
        if value is not None and not isinstance(value, str):
            normalized[key] = str(value)
    for key in ("record_count", "time_window_days"):
        value = normalized.get(key)
        if isinstance(value, bool):
            normalized[key] = None
        elif value is not None:
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                normalized[key] = None

    return normalized


def _compact_source_catalog_for_llm(
    models: list[dict[str, Any]],
    max_fields_per_model: int,
    max_total_fields: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a model-card projection from the COMPLETE active source catalog.

    The projection is prompt-size-only. It must never make a business model disappear before Gemini
    can reason about it, so every source model is represented by its name, description, total field
    count and a role-diverse sample of scalar leaves.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in models or []:
        if not isinstance(row, dict):
            continue
        model = str(row.get("business_model") or row.get("model") or "").strip()
        if not model:
            continue
        key = (model, str(row.get("source_id") or ""))
        groups.setdefault(key, []).append(row)

    role_order = ["timing", "measurement", "status", "categorical", "identity", "other"]

    def field_model_description_overlap(row: dict[str, Any]) -> int:
        model_tokens = _catalog_text_tokens(row.get("model_description"))
        field_tokens = _catalog_text_tokens(
            f"{row.get('name', '')} {row.get('description', '')}"
        )
        overlap = _canonical_context_tokens(model_tokens & field_tokens)
        return len(overlap - {"model", "resource", "entity", "task", "field"})

    def field_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        role = str(_catalog_role(row) or "other")
        metadata_penalty = float(_catalog_business_metadata_penalty(row))
        owner = _source_owner_family(row.get("source_owner_model") or row.get("model"))
        root_owner = _source_owner_family(row.get("business_model") or row.get("model"))
        role_rank = {
            "status": 0,
            "timing": 1,
            "measurement": 2,
            "categorical": 3,
            "identity": 4,
            "other": 5,
        }.get(role, 5)
        numeric_measurement = 0 if role == "measurement" and str(row.get("dtype") or "").lower() in {"integer", "float", "number"} else 1
        # Prefer analytically specific leaf properties over transport/display-style leaves. This is
        # structural and vocabulary-independent: ``remainingValue.amount`` beats ``remainingValue.name``
        # without ever naming a domain field explicitly.
        leaf_name = str(row.get("field") or row.get("name") or "").strip()
        leaf_tokens = _catalog_text_tokens(leaf_name)
        specificity_tokens = leaf_tokens - {
            "id", "name", "href", "url", "uri", "type", "role", "description",
            "value", "key", "code", "format", "schema",
        }
        specificity = len(specificity_tokens)
        display_penalty = 1 if _catalog_display_only(row, context_tokens=set(), preferred=set()) else 0
        return (
            1 if metadata_penalty >= 30.0 else 0,
            display_penalty,
            role_rank,
            numeric_measurement,
            -specificity,
            0 if owner == root_owner else 1,
            0 if row.get("required") else 1,
            -field_model_description_overlap(row),
            int(row.get("depth", 0) or 0),
            0 if row.get("enum_values") else 1,
            str(row.get("name") or ""),
        )

    cards: list[dict[str, Any]] = []
    per_model_limit = max(1, int(max_fields_per_model))
    for (model, source_id), model_rows in sorted(groups.items()):
        ordered = sorted(model_rows, key=field_sort_key)
        by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in role_order}
        for row in ordered:
            role = str(_catalog_role(row) or "other")
            by_role.setdefault(role, []).append(row)

        selected: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        def cluster_key(row: dict[str, Any]) -> tuple[str, str]:
            # Group by the structural source concept that owns a scalar leaf. This is schema-derived,
            # so nested analytical branches such as usage counters, remaining-value quantities and
            # consumption periods compete fairly with root IDs/names instead of being hidden by them.
            owner = _source_owner_family(row.get("source_owner_model") or row.get("model"))
            relation = normalize_lookup_key(row.get("source_owner_relation") or owner)
            return owner, relation

        def quality_key(row: dict[str, Any]) -> tuple[Any, ...]:
            role = str(_catalog_role(row) or "other")
            metadata_penalty = float(_catalog_business_metadata_penalty(row))
            owner = _source_owner_family(row.get("source_owner_model") or row.get("model"))
            root_owner = _source_owner_family(row.get("business_model") or row.get("model"))
            leaf_name = str(row.get("field") or row.get("name") or "").strip()
            specificity = len(_catalog_text_tokens(leaf_name) - {
                "id", "name", "href", "url", "uri", "type", "role", "description",
                "value", "key", "code", "format", "schema",
            })
            branch_tokens = _catalog_text_tokens(row.get("source_owner_relation") or row.get("source_owner_model"))
            model_tokens = _catalog_text_tokens(row.get("model_description"))
            branch_overlap = len(_canonical_context_tokens(branch_tokens & model_tokens) - {
                "model", "resource", "entity", "task", "field",
            })
            display_penalty = 1 if _catalog_display_only(row, context_tokens=set(), preferred=set()) else 0
            role_rank = {
                "status": 0,
                "timing": 1,
                "measurement": 2,
                "categorical": 3,
                "identity": 4,
                "other": 5,
            }.get(role, 5)
            numeric_measurement = 0 if role == "measurement" and str(row.get("dtype") or "").lower() in {"integer", "float", "number"} else 1
            return (
                1 if metadata_penalty >= 30.0 else 0,
                display_penalty,
                role_rank,
                numeric_measurement,
                -specificity,
                -branch_overlap,
                0 if owner == root_owner else 1,
                0 if row.get("required") else 1,
                -field_model_description_overlap(row),
                int(row.get("depth", 0) or 0),
                0 if row.get("enum_values") else 1,
                str(row.get("name") or ""),
            )

        # Allocate model-card space by semantic role. The quotas are generic coverage weights, not
        # business-field allowlists; unused quota is returned to the remainder pass below.
        quota_weights = {
            "status": 2,
            "timing": 2,
            "measurement": 5,
            "categorical": 2,
            "identity": 1,
            "other": 1,
        }
        total_weight = sum(quota_weights.values())
        quotas: dict[str, int] = {}
        assigned = 0
        for role in role_order:
            if role == role_order[-1]:
                quota = max(0, per_model_limit - assigned)
            else:
                quota = min(
                    max(1 if per_model_limit >= len(role_order) else 0,
                        round(per_model_limit * quota_weights[role] / total_weight)),
                    per_model_limit - assigned,
                )
            quotas[role] = quota
            assigned += quota
            if assigned >= per_model_limit:
                break
        for role in role_order:
            quotas.setdefault(role, 0)

        by_role_and_cluster: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
            role: {} for role in role_order
        }
        for row in ordered:
            role = str(_catalog_role(row) or "other")
            by_role_and_cluster.setdefault(role, {}).setdefault(cluster_key(row), []).append(row)

        def choose_role(role: str, limit: int) -> None:
            if limit <= 0:
                return
            clusters = by_role_and_cluster.get(role) or {}
            ranked_clusters = sorted(
                clusters.items(),
                key=lambda item: (
                    0 if any(float(_catalog_business_metadata_penalty(v)) < 30.0 for v in item[1]) else 1,
                    min(quality_key(v) for v in item[1]),
                    item[0],
                ),
            )
            for _cluster, candidates in ranked_clusters:
                if len(selected) >= per_model_limit or limit <= 0:
                    break
                pool = [r for r in candidates if normalize_lookup_key(r.get("name")) not in seen_names]
                if not pool:
                    continue
                row = min(pool, key=quality_key)
                name = normalize_lookup_key(row.get("name"))
                if not name:
                    continue
                selected.append(row)
                seen_names.add(name)
                limit -= 1

            # If this role still has room, allow a second field from an already-represented structural
            # branch before moving to unrelated transport/display fields.
            if limit > 0 and len(selected) < per_model_limit:
                pool = [
                    r for r in (by_role.get(role) or [])
                    if normalize_lookup_key(r.get("name")) not in seen_names
                ]
                for row in sorted(pool, key=quality_key):
                    if len(selected) >= per_model_limit or limit <= 0:
                        break
                    name = normalize_lookup_key(row.get("name"))
                    if not name:
                        continue
                    selected.append(row)
                    seen_names.add(name)
                    limit -= 1

        for role in role_order:
            choose_role(role, quotas.get(role, 0))

        # Redistribute unused quota to the strongest remaining fields, preferring a new structural
        # branch whenever one exists. This makes small models graceful and keeps large TMF models
        # broad without needing hand-maintained lists.
        while len(selected) < per_model_limit:
            remaining = [
                r for r in ordered
                if normalize_lookup_key(r.get("name")) not in seen_names
            ]
            if not remaining:
                break
            represented_clusters = {cluster_key(r) for r in selected}
            new_cluster = [r for r in remaining if cluster_key(r) not in represented_clusters]
            pool = new_cluster or remaining
            row = min(pool, key=quality_key)
            name = normalize_lookup_key(row.get("name"))
            if not name:
                break
            selected.append(row)
            seen_names.add(name)

        model_description = str((ordered[0] if ordered else {}).get("model_description") or "")[:500]
        cards.append({
            "model": model,
            "source_id": source_id,
            "model_description": model_description,
            "field_count": len(model_rows),
            "representative_fields": [
                {
                    "name": str(row.get("name") or ""),
                    "dtype": str(row.get("dtype") or ""),
                    "required": bool(row.get("required")),
                    "role": str(_catalog_role(row) or "other"),
                    "description": str(row.get("description") or "")[:300],
                    "source_owner_model": str(row.get("source_owner_model") or ""),
                    "source_owner_relation": str(row.get("source_owner_relation") or ""),
                    "source_owner_relative_path": str(row.get("source_owner_relative_path") or ""),
                }
                for row in selected
            ],
        })

    total_limit = max(1, int(max_total_fields)) if max_total_fields is not None else None
    total_representative = sum(len(card["representative_fields"]) for card in cards)
    if total_limit is not None and total_representative > total_limit and cards:
        # Keep every source model visible. Reduce only the number of representatives per card using
        # a fair-share quota; model descriptions and field counts remain intact.
        quota = max(1, total_limit // len(cards))
        remainder = max(0, total_limit - quota * len(cards))
        for index, card in enumerate(cards):
            card["representative_fields"] = card["representative_fields"][: quota + (1 if index < remainder else 0)]

    return cards, {
        "projection_mode": "complete_catalog_model_cards",
        "source_scalar_count": len(models or []),
        "model_card_count": len(cards),
        "fields_per_model": per_model_limit,
        "representative_field_count": sum(len(card["representative_fields"]) for card in cards),
        "total_field_limit": total_limit,
        "relevance_preselection": False,
        "role_diversity": True,
    }


# ---- value profiles ---------------------------------------------------------------------------------------------
# Standards (BIAN, FIBO, ISO 20022, TM Forum, ...) list few example values, so a free-text field could only be generated as a
# neutral SYN_<FIELD>_1234 token (stripped or rejected by the quality gates) and a number without declared bounds got a
# stand-in range (0-100, 0-1000). profile_values asks the model, once per scenario, what a real system of the industry and
# country holds in each such field, checks the answer against the field's own contract and locks it, so the same scenario
# always proposes the same values and generation needs no model call for these fields.
PROFILE_VERSION = "value-profile-v3"
CHUNK = 30                       # fields per model call
WORKERS = 3                      # chunks asked at the same time
MIN_VALUES = 4                   # fewer distinct usable values than this is no profile
MAX_VALUES = 20
_STALE_YEAR = re.compile(r"(?<![0-9])(?:19|20)\d{2}(?:(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))?(?![0-9])")          # a year inside a value goes stale
_GENERIC = re.compile(
    r"^(?:syn[_\-\s]?\w*|sample\w*|test\w*|example\w*|dummy\w*|lorem.*|foo\w*|bar|baz|xxx+|n/?a|na|none|null|nil|unknown|"
    r"string|value\s*\d*|text\s*\d*|type\s*[a-z0-9]|item\s*\d+|option\s*\d+|field\s*\d*|placeholder\w*|tbd|todo)$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You supply realistic values for fields of a synthetic-data schema, for any industry.

Each field is either a text field ("kind": "text") or a number ("kind": "number"). For a text field, list the values a real,
production system of the stated industry and country holds in it. For a number, give the range in which most real values lie
("min" and "max", in the field's own unit: the country's currency for money, whole units for counts), not the extremes. Work from the field's name, its description, its source path (abbreviated tags such as Dbtr, PstlAdr, Nm are the
standard's own shorthand: read them as the words they abbreviate) and the standard it comes from.

Rules:
1. Real, specific values only. Never a placeholder or generic token: no "Sample", "Test", "Example", "string", "N/A", "Unknown",
   "Type A", "Value 1", "Lorem ipsum", "SYN_...", "XXX", and never the field's own name.
2. The value must fit the field's meaning exactly. A name field holds realistic names for the country, an address line a real-looking
   street address, a code field the codes of the code list the standard uses (say which list in your head, then give its codes), an
   identifier field identifiers in the format real systems use, a free-text field short realistic text of the kind users enter.
3. Honour every constraint given for the field: minLength / maxLength, format, and the country's conventions (language, currency,
   phone, postal and ID formats).
4. Give 8 to 20 distinct values. Mix common and less common ones, as real data would. Keep every value on one line.
5. Describe the whole population of the field, not the scenario's special case. The business scenario may colour a narrative or
   a reason field (a fraud-alert narrative reads differently from a salary-credit narrative), but the values of a code, type,
   status or channel field are the ones the system holds for every kind of record, and a number's range covers the routine small
   values as well as the large ones. How often each value or size occurs for a scenario is decided later by the generator.
6. If you cannot tell what a field holds, return an empty list (text) or null min and max (number). That is better than a guess.
7. For a number, min must be below max. Give the range in which most real values of the field lie across all records, from the
   smallest routine value to the large ones (a retail payment ranges from tens to lakhs of rupees, not 45,000 to 950,000; a
   percentage is 0-100; a count of failed logins is small). Never narrow it to what only the scenario's suspicious or rare records show.
8. Identifiers, references and codes that systems generate: give the format with made-up digits (a prefix, a separator, digits). Never
   put a calendar year, a date or the name of a real company, bank or brand inside them. A value that is itself a name (a bank, a
   card network, a payment scheme, a city) may be a real one, only in a field that holds exactly that kind of name.
9. Text values never contain a calendar year or date that will go stale (no "2023", no "Jan-2024"), unless the field is a date.
10. One letter-case convention across the whole list: values of code, type, status, channel and category fields are UPPER_SNAKE_CASE
   (CASH_WITHDRAWAL, UPI_PAYMENT), unless the field's own description or allowed values show another style; names, places and
   narratives keep natural case. Never mix the two styles for fields of the same kind.

Return only JSON: {"fields": [{"name": "<exact field name>", "values": ["...", "..."]}, {"name": "<number field>", "min": 0, "max": 0}]}.
Include every field you were given, once."""


def _is_number_target(field: GeneratedSchemaField) -> bool:
    return str((field.params or {}).get("range_source") or "") == "default" and str(field.gen or "") in {"uniform", "uniform_int"}


def _signature(field: GeneratedSchemaField) -> dict[str, Any]:
    params = field.params or {}
    prov = field.provenance or {}
    return {
        "name": field.name,
        "kind": "number" if _is_number_target(field) else "text",
        "description": (field.description or "")[:300],
        "source_path": str(prov.get("source_json_path") or "")[:200],
        "source_model": str(prov.get("source_json_model") or "")[:120],
        "constraints": {k: params[k] for k in ("minLength", "maxLength", "min_length", "max_length", "format") if params.get(k) not in (None, "")},
    }


def _limits(field: GeneratedSchemaField) -> tuple[int, int]:
    params = field.params or {}

    def number(*keys: str) -> int | None:
        for key in keys:
            try:
                value = int(float(params[key]))
            except (KeyError, TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return None

    return number("minLength", "min_length") or 1, number("maxLength", "max_length") or 200


def _usable(field: GeneratedSchemaField, raw: Any) -> list[str]:
    """The values of ``raw`` that are real, distinct, one-line and inside the field's length contract."""
    if not isinstance(raw, list):
        return []
    low, high = _limits(field)
    key = re.sub(r"[^a-z0-9]+", "", field.name.lower())
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            continue
        text = re.sub(r"\s+", " ", str(item)).strip()
        folded = text.casefold()
        if not text or folded in seen or not (low <= len(text) <= high):
            continue
        if _GENERIC.match(text) or re.sub(r"[^a-z0-9]+", "", folded) == key or _STALE_YEAR.search(text):
            continue
        seen.add(folded)
        out.append(text)
        if len(out) == MAX_VALUES:
            break
    return out if len(out) >= MIN_VALUES else []


def _usable_range(field: GeneratedSchemaField, low: Any, high: Any) -> dict[str, float | int] | None:
    """``{"min", "max"}`` when the model gave a finite range with min below max (whole numbers for an integer field)."""
    try:
        lo, hi = float(low), float(high)
    except (TypeError, ValueError):
        return None
    if isinstance(low, bool) or isinstance(high, bool) or not (abs(lo) < 1e15 and abs(hi) < 1e15) or lo >= hi:
        return None
    if str(field.dtype).lower() in {"integer", "int"}:
        lo, hi = int(round(lo)), int(round(hi))
        return {"min": lo, "max": hi} if lo < hi else None
    return {"min": round(lo, 6), "max": round(hi, 6)}


# ---- readable names ---------------------------------------------------------------------------------------------
# A field compiled from a standard is named after its path inside that standard, which is exact but unreadable
# (bank_to_customer_debit_credit_notification_v14_grp_hdr_msg_rcpt_ctct_dtls_othr_chanl_tp). The fields are compiled from the data
# models first, with the source name kept in their provenance; humanize_names then gives each a short business name, once per
# set of fields, locks it, and the proposal carries the readable names from then on. Nothing downstream looks a field up by its
# technical name: provenance (source_json_original_name, source_json_semantic_key, source_json_path) still leads back to the model.
NAMES_VERSION = "readable-names-v2"
NAME_MAX_LEN = 48
NAME_MIN_LEN = 3
_MAX_WORD = 15                      # no ordinary word of a field name is longer (identification is 14)
_NAME_OK = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_NAME_FILLER = {"field", "value", "data", "column", "attribute", "item", "thing", "info", "misc", "syn"}
# Shorthand that standards and APIs use inside their tags. The model reads these too; this table is the fallback when it cannot be reached.
_ABBREVIATIONS = {
    "acct": "account", "addr": "address", "amt": "amount", "appl": "application", "auth": "authorization", "bal": "balance",
    "bic": "bic", "cd": "code", "chanl": "channel", "chnl": "channel", "ctct": "contact", "cntry": "country", "cre": "created",
    "cr": "credit", "cur": "currency", "ccy": "currency", "cust": "customer", "dbtr": "debtor", "cdtr": "creditor", "db": "debit",
    "dt": "date", "tm": "time", "dts": "date", "dtls": "details", "dtl": "detail", "desc": "description", "dest": "destination",
    "doc": "document", "dur": "duration", "end": "end", "exp": "expiry", "fee": "fee", "freq": "frequency", "grp": "group",
    "hdr": "header", "id": "id", "idx": "index", "info": "information", "inst": "instruction", "instd": "instructed",
    "instg": "instructing", "msg": "message", "nm": "name", "no": "number", "nb": "number", "num": "number", "ntfctn": "notification",
    "ntry": "entry", "ownr": "owner", "othr": "other", "pmt": "payment", "pstl": "postal", "prd": "period", "ref": "reference",
    "rcpt": "recipient", "rltd": "related", "rsn": "reason", "sts": "status", "stat": "status", "svc": "service", "tp": "type",
    "txn": "transaction", "trx": "transaction", "tx": "transaction", "ts": "timestamp", "val": "value", "ctry": "country",
    "ultmt": "ultimate", "orgnl": "original", "grpg": "grouping", "tot": "total", "qty": "quantity", "prty": "priority",
    "mmb": "member", "agt": "agent", "ctrl": "control", "sum": "summary", "seq": "sequence", "nbr": "number", "ind": "indicator",
}
_NOISE_TOKENS = {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}

NAMES_PROMPT = """You rename the columns of a synthetic dataset so that a business user, not a programmer, understands them at a glance.

You get a list of fields. Each has its current technical "name" (built from the path of the field inside an industry data model:
BIAN, FIBO, ISO 20022, TM Forum, FHIR, ACORD and so on), its "description", its "source_path", the "model" it belongs to and its
data type, and sometimes example values. Give every field a "readable" name.

Rules:
1. snake_case, lowercase letters, digits and underscores only, starting with a letter. 2 to 5 words, at most 40 characters.
2. Say what the value is, in the words a business person uses: transaction_amount, payment_channel, account_open_date,
   notification_sent_at, cash_withdrawal_type. Expand every abbreviation (Dbtr is debtor, Ntry is entry, ctct dtls is contact
   details, tp is type, cre dt tm is created date time, othr is other). Keep only universally known short forms (id, kyc, aml, sms).
3. Drop wrapper words that only describe the message or API envelope (the message name, version numbers such as v14, "request",
   "response", "grp hdr", "details", "record") unless two fields would otherwise be indistinguishable.
4. When the same leaf word appears for different owners (an amount of a deposit and an amount of a withdrawal, a date of the
   account opening and of the account closing), keep the owner word that tells them apart: deposit_amount, withdrawal_amount,
   account_open_date, account_close_date.
5. End with a word that names the kind of value when natural: _id, _date, _timestamp (date and time), _amount, _status, _type, _code,
   _flag (yes/no), _count, _name, _channel, _reason, _score, _rate.
6. Use the same word for the same concept across the whole list ("transaction" everywhere, never txn in one and trx in another;
   "customer" or "party" consistently).
7. Every readable name must be unique in the list, and must not be one of the names in "reserved".
8. Separate every word with an underscore (never glue words together such as currentaccountpayments) and correct obvious spelling
   mistakes of the source (wihdrawal is withdrawal, depositsand is deposit and). Drop action and envelope words of an API
   operation (update, create, initiate, execute, evaluate, request, response) when the field is the same fact without them.
9. Do not change what the field means and do not invent meaning that the description and path do not support. If a name is
   already clear, return it unchanged. Never use filler such as field, value, data, column, attribute, item, syn, and never
   end a name in a number.

Return only JSON: {"fields": [{"name": "<current technical name, exactly as given>", "readable": "<new name>"}]}.
Include every field you were given, once."""


def _name_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t]


_WRAPPER_TOKENS = {"details", "detail", "header", "group", "record", "message", "request", "response"}


def _plain_name(name: str) -> list[str]:
    """Deterministic readable tokens of a technical name: shorthand expanded, versions, envelope words and repeats dropped."""
    out: list[str] = []
    for token in _name_tokens(name):
        if token in _NOISE_TOKENS or re.fullmatch(r"v\d+", token):
            continue
        out.append(_ABBREVIATIONS.get(token, token))
    core = [t for t in out if t not in _WRAPPER_TOKENS]
    out = core if len(core) >= 2 else out
    merged: list[str] = []
    for token in out:                                   # date + time is a timestamp
        if token == "time" and merged and merged[-1] == "date":
            merged[-1] = "timestamp"
        else:
            merged.append(token)
    seen: set[str] = set()                              # a word said twice is said once, where it matters most (the later place)
    unique: list[str] = []
    for token in reversed(merged):
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique[::-1]


def _shorten(tokens: list[str], limit: int = NAME_MAX_LEN) -> str:
    """Whole name when it fits; otherwise the owner word at the front plus as many closing words as fit (the most specific part)."""
    if len("_".join(tokens)) <= limit:
        return "_".join(tokens)
    kept: list[str] = []
    room = limit - len(tokens[0]) - 1 if len(tokens) > 1 else limit
    for token in reversed(tokens[1:]):
        if len("_".join([token] + kept)) > room:
            break
        kept.insert(0, token)
    return "_".join(([tokens[0]] if tokens and len(tokens) > 1 else []) + kept) if kept else tokens[-1][:limit]


def _readable_ok(candidate: Any, taken: set[str]) -> str | None:
    text = re.sub(r"[^a-z0-9]+", "_", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(candidate or "").strip()).lower()).strip("_")
    if not (NAME_MIN_LEN <= len(text) <= NAME_MAX_LEN) or not _NAME_OK.match(text):
        return None
    parts = text.split("_")
    if any(len(p) > _MAX_WORD for p in parts):                      # words run together: currentaccountpayments
        return None
    # filler is a defect only when it is all the name says or the word it ends in ("bucket_remaining_value_amount" is fine)
    if all(p in _NAME_FILLER for p in parts) or parts[-1] in _NAME_FILLER or re.search(r"\d$", text) or _GENERIC.match(text):
        return None
    return None if text in taken else text


def _fallback_name(field: GeneratedSchemaField, taken: set[str]) -> str:
    """A readable name without the model: expanded tokens, shortened, made unique with the owner words of the source path."""
    tokens = _plain_name(field.name)
    context = _plain_name(str((field.provenance or {}).get("source_json_model") or ""))
    attempts = [tokens] + [context[-n:] + tokens for n in (1, 2, 3)] if context else [tokens]
    for attempt in attempts:
        candidate = _readable_ok(_shorten(attempt), taken)
        if candidate:
            return candidate
    base = _shorten(tokens, NAME_MAX_LEN - 3) or "field_value"
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}" if _NAME_OK.match(f"{base}_{n}") else f"attribute_{n}"


def _name_signature(field: GeneratedSchemaField) -> dict[str, Any]:
    params = field.params or {}
    prov = field.provenance or {}
    examples = params.get("source_examples") or params.get("choices") or []
    return {
        "name": field.name,
        "description": (field.description or "")[:220],
        "source_path": str(prov.get("source_json_path") or "")[:200],
        "model": str(prov.get("source_json_model") or "")[:100],
        "dtype": field.dtype,
        "examples": [str(x)[:40] for x in list(examples)[:3]] if isinstance(examples, list) else [],
    }


_OPERATION_WORDS = {"request", "response", "update", "create", "retrieve", "initiate", "execute", "evaluate", "control", "capture",
                    "notify", "exchange", "grp", "hdr"}


def _fact_key(name: str) -> tuple[str, ...]:
    """The words of a technical name that say WHAT the value is: operation/envelope words, versions, plurals and repeats removed."""
    words = [_ABBREVIATIONS.get(t, t) for t in _name_tokens(name) if t not in _OPERATION_WORDS and t not in _NOISE_TOKENS and not re.fullmatch(r"v\d+", t)]
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    out: list[str] = []
    for w in reversed(words):
        if w not in out:
            out.append(w)
    return tuple(reversed(out))


def drop_wrapper_duplicates(fields: list[GeneratedSchemaField], keep: set[str] | None = None) -> tuple[list[GeneratedSchemaField], list[str]]:
    """Remove a source field that is the same fact as another, differing only by the operation wrapper of its path.

    ``current_account_update_payments_request_payments_payment_transaction_amount`` and ``current_account_payments_payment_transaction_amount``
    are one amount seen through two API operations: both would be delivered as columns that must always agree. The one without
    operation words stays. Never removed: the ``keep`` names, a field another field depends on, anything that is not source-backed.
    """
    keep_keys = {str(k).casefold() for k in (keep or set())}
    referenced = {d for f in fields for d in (f.depends_on or [])}
    for f in fields:
        referenced |= {v for v in (f.params or {}).values() if isinstance(v, str)}
    groups: dict[tuple, list[GeneratedSchemaField]] = {}
    for f in fields:
        if (f.provenance or {}).get("source_json_id") and f.name.casefold() not in keep_keys and f.name not in referenced:
            key = _fact_key(f.name)
            if len(key) >= 2:
                groups.setdefault((key, f.dtype), []).append(f)
    drop: set[str] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        # keep the plainest path: fewest operation words, then the shortest name
        members.sort(key=lambda f: (sum(1 for t in _name_tokens(f.name) if t in _OPERATION_WORDS), len(f.name), f.name))
        drop |= {f.name for f in members[1:]}
    if not drop:
        return fields, []
    logger.info("[IntentAgent/Names] %d duplicate wrapper field(s) removed: %s", len(drop), ", ".join(sorted(drop)))
    return [f for f in fields if f.name not in drop], sorted(drop)


def rename_in_text(text: Any, mapping: dict[str, str]) -> Any:
    """``text`` with every whole-word technical name replaced (a formula, a rule text)."""
    if not isinstance(text, str) or not text or not mapping:
        return text
    for old in sorted(mapping, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", mapping[old], text)
    return text


def apply_renames(fields: list[GeneratedSchemaField], mapping: dict[str, str]) -> list[GeneratedSchemaField]:
    """Rename fields and every reference to them (depends_on, formula, reference parameters); the old name stays in provenance."""
    if not mapping:
        return fields
    reference_keys = ("depends_on_field", "field", "segment_field", "hi_field", "lo_field", "base_field", "source_field", "add_seconds_field")
    out: list[GeneratedSchemaField] = []
    for field in fields:
        params = dict(field.params or {})
        for key in reference_keys:
            if isinstance(params.get(key), str) and params[key] in mapping:
                params[key] = mapping[params[key]]
        update: dict[str, Any] = {
            "depends_on": [mapping.get(d, d) for d in (field.depends_on or [])],
            "formula": rename_in_text(field.formula, mapping),
            "params": params,
        }
        if field.name in mapping:
            prov = dict(field.provenance or {})
            prov.setdefault("source_json_original_name", field.name)
            prov.setdefault("source_json_name", field.name)
            prov["technical_name"] = field.name
            prov["readable_names"] = NAMES_VERSION
            update.update(name=mapping[field.name], provenance=prov)
        out.append(field.model_copy(update=update))
    return out


class GeminiIntentAgent:
    """Compatibility-named intent agent using Gemini JSON mode + local Pydantic validation.

    The intent path intentionally uses the official Google GenAI SDK in JSON mode.
    PydanticAI is intentionally not used here because provider-native structured
    output was the source of the observed Gemini 400 INVALID_ARGUMENT failures.
    """

    def __init__(self, api_key: str | None = None):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self._api_key = api_key
        self._client: GeminiClient | None = None

    def _get_client(self) -> GeminiClient:
        if self._client is None:
            self._client = GeminiClient(api_key=self._api_key)
        return self._client

    def run(
        self,
        request_context: str,
        country: str | None = None,
        industry_type: str = "",
        domain_query: str | None = None,
        excluded_variable_names: list[str] | None = None,
        persisted_variables: list[dict[str, Any]] | None = None,
        source_catalog: dict[str, Any] | None = None,
    ) -> ScenarioIntent:
        json_grounded = is_json_grounded_domain(domain_query, industry_type)
        excluded_names = {
            str(name).strip().casefold()
            for name in (excluded_variable_names or [])
            if str(name).strip()
        }
        if json_grounded:
            catalog = source_catalog or catalog_for_request(industry_type, domain_query or "")
            # Gemini must see the complete active source-model index. The old path first applied
            # deterministic lexical relevance selection and then compacted the survivors, which could
            # hide Usage/Consumption or Qualification models before the LLM could select them.
            # Compaction is now prompt-size-only; it does not decide source relevance.
            all_source_rows = [
                dict(row) for row in (catalog.get("models") or [])
                if isinstance(row, dict)
            ]
            compact_models, compact_report = _compact_source_catalog_for_llm(
                all_source_rows,
                JSON_SOURCE_LLM_FIELDS_PER_MODEL,
                max_total_fields=JSON_SOURCE_LLM_CATALOG_LIMIT,
            )
            projection_report = {
                "candidate_count": len(all_source_rows),
                "selection_mode": "prompt_only_complete_source_model_index",
                "relevance_preselection": False,
            }
            prompt_catalog = dict(catalog)
            prompt_catalog["models"] = compact_models
            prompt_catalog["llm_projection"] = {**projection_report, **compact_report}
            catalog_text = json.dumps(prompt_catalog, separators=(",", ":"), sort_keys=True)
            grounding_header = (
                "MONGODB INDUSTRY-SOURCE GROUNDING (authoritative):\n"
                "Use only active JSON source documents stored in MongoDB for this exact industryType/domain pair. "
                "Their scalar fields, descriptions, types, declared constraints, and enum values are the only standards vocabulary available to you. "
                "Do not use external standards, URLs, static application registries, profiles, templates, examples, memory, or generic industry knowledge.\n"
                + (
                    "The source projection is a breadth aid, not a contract limit. Omitted source fields are not evidence of irrelevance; "
                    "the deterministic backend will re-evaluate the complete active catalog before compilation.\n\n"
                )
            )
            mandatory_line = "Do not add variables outside the supplied MongoDB source catalog or separately persisted MongoDB scenario variables. "
        elif persisted_variables:
            prompt_catalog = {
                "source_policy": "scenario_variables",
                "industryType": industry_type,
                "domain": domain_query or "",
                "models": [
                    {
                        "name": str(item.get("name") or ""),
                        "description": str(item.get("description") or ""),
                        "dtype": str(item.get("dtype") or ""),
                        "scope": str(item.get("scope") or ""),
                        "role": str(item.get("role") or ""),
                        "params": item.get("params") or {},
                        "depends_on": item.get("depends_on") or [],
                        "formula": item.get("formula"),
                    }
                    for item in persisted_variables
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                ],
            }
            catalog_text = json.dumps(prompt_catalog, separators=(",", ":"), sort_keys=True)
            grounding_header = (
                "PERSISTED MONGODB SCENARIO VARIABLES (authoritative):\n"
                "No industry-standard JSON source is registered for this exact industryType/domain. "
                "Use only the persisted MongoDB scenario variables supplied below. Do not use external standards, static registries, profiles, templates, examples, memory, or generic industry knowledge.\n\n"
            )
            mandatory_line = "Every candidate variable name must exactly match a supplied persisted MongoDB scenario variable. "
        else:
            raise ValueError(
                f"No active JSON source documents and no persisted MongoDB scenario variables exist for industryType='{industry_type}', domain='{domain_query or ''}'."
            )
        prompt = (
            grounding_header +
            f"{catalog_text}\n\n"
            "Authoritative request context:\n"
            f"{request_context}\n\n"
            f"Selected industry: {industry_type}\n"
            f"Business domain: {domain_query or '<none>'}\n"
            f"Country: {country or '<none>'}\n\n"
            "Create a RECALL-FIRST official-variable relevance selection for this scenario. The numeric maximum is a CEILING, NOT a target: "
            "never pad a schema with irrelevant, marginal, technical, or semantically duplicated variables just to reach that number. "
            "Do not attempt to enumerate every field in the catalog. Instead, identify the strongest source-backed business anchors and relevant source models/entities "
            "that prove what should be expanded. The deterministic compiler will then expand a bounded structural neighborhood around those source models/fields up to the configured ceiling; it must not blindly import every nested technical branch. "
            "Cover the scenario's relevant business concepts: identity and relationship anchors, profile/context, transactions and events when explicitly relevant, "
            "states/statuses/reasons, timing/timestamps/dates, monetary/quantity/usage measures, decisions/outcomes, configuration/eligibility, "
            "channels/methods, geography/segments, lifecycle, and other source-backed analytical signals. "
            "When a source model is clearly relevant, prefer its distinct business fields rather than wrapper copies or repeated CRUD/event representations. "
            "A field is a true duplicate when it represents the same underlying source concept after wrapper/structural normalization; fields with similar types, descriptions, "
            "or enum values but different business-resource meaning are not automatically duplicates. "
            "Never pad with API href/referredType/reference metadata, display-only fields, transport plumbing, or privacy-sensitive customer contact fields when the request forbids PII. "
            "Do not let one generic or ambiguous field (for example status, amount, balance, id, or a single LLM-selected field) make an otherwise unrelated resource relevant. "
            "Use the scenario's domain, use case, business scenario, scenario type, data type, entity key, and country as relevance evidence. "
            "Review ALL supplied model cards before deciding. requested_entities must contain only exact source business model identifiers from the model cards that are materially relevant; use it to identify source models whose local structural neighborhoods should be inspected, not as permission to import every nested field. "
            "candidate_variables must name exact source scalar fields that are especially important. A candidate field from a relevant model is valid even when the field name itself does not lexically match the request, provided the model description/field semantics support it. "
            "For behavior-oriented scenarios, explicitly check source models that describe the triggering event, state/measurement, eligibility/qualification, outcomes and customer context before returning the selection. "
            "The final deterministic compiler evaluates the complete canonical MongoDB source catalog, so this LLM output is a relevance signal only, never the final variable-count gate. " + mandatory_line + "\n"
            + (
                "PERSISTED MONGODB VARIABLES (IMMUTABLE; NEVER RENAME OR MODIFY):\n"
                + json.dumps([
                    {
                        "name": str(item.get("name") or ""),
                        "description": str(item.get("description") or ""),
                        "dtype": str(item.get("dtype") or ""),
                        "scope": str(item.get("scope") or ""),
                        "role": str(item.get("role") or ""),
                    }
                    for item in (persisted_variables or [])
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                ], separators=(",", ":"), sort_keys=True)
                + "\n"
                + "Duplicate-review rule: compare every candidate official variable against every persisted DB variable. If two variables have the same business use, omit the official JSON variable and keep the DB variable unchanged.\n"
                if persisted_variables
                else ""
            )
            + (
                "PERSISTED SCENARIO VARIABLES THAT ARE ALREADY COVERED AND MUST NOT BE RE-PROPOSED:\n"
                + "- " + "\n- ".join(sorted(excluded_names)) + "\n"
                + "Return only complementary candidate variables; do not recreate these fields or semantic equivalents.\n"
                if excluded_names
                else ""
            )
            + "Every candidate_variables.name must exactly match a name in the supplied MongoDB source catalog when JSON grounding is active, or a supplied persisted MongoDB scenario variable when scenario_variables grounding is active. "
            "Return JSON only."
        )

        try:
            client = self._get_client()
            try:
                raw = client.generate_json(
                    system_instruction=INSTRUCTIONS,
                    user_prompt=prompt,
                    temperature=0.0,
                )
                payload = _extract_json_payload(raw)
            except LLMUpstreamError as exc:
                # Google can reject provider-native JSON configuration with a 400 even
                # though the model and credentials are valid. Retry once with plain text
                # JSON instructions, which avoids response_mime_type/response-schema validation.
                if exc.status_code != 400:
                    raise
                logger.warning("Gemini JSON mode rejected with 400; retrying intent request in plain-text mode")
                text = client.generate_text(
                    system_instruction=INSTRUCTIONS,
                    user_prompt=(
                        prompt
                        + "\n\nJSON MODE FALLBACK: output exactly one JSON object matching the requested shape. "
                        + "Do not add markdown, commentary, or code fences."
                    ),
                    temperature=0.0,
                )
                payload = _parse_text_json(text)

            payload = _normalize_payload(payload)
            if json_grounded:
                notes = list(payload.get("notes") or [])
                catalog_rows = [
                    dict(row) for row in (catalog.get("models") or [])
                    if isinstance(row, dict)
                ]
                available_models: dict[str, str] = {}
                for row in catalog_rows:
                    raw_model = str(row.get("business_model") or row.get("model") or "").strip()
                    if raw_model:
                        available_models.setdefault(_normalize_model_name(raw_model), raw_model)
                requested_models: list[str] = []
                invalid_requested_models: list[str] = []
                for raw_model in payload.get("requested_entities") or []:
                    token = _normalize_model_name(raw_model)
                    canonical_model = available_models.get(token) or available_models.get(_source_owner_family(token))
                    if canonical_model and canonical_model not in requested_models:
                        requested_models.append(canonical_model)
                    elif str(raw_model or "").strip():
                        invalid_requested_models.append(str(raw_model).strip())
                if invalid_requested_models:
                    notes.append(
                        "LLM-requested source models outside the active MongoDB model catalog were ignored: "
                        + ", ".join(sorted(set(invalid_requested_models)))
                    )
                payload["requested_entities"] = requested_models
                catalog_by_name = {
                    normalize_lookup_key(row.get("name")): dict(row)
                    for row in (catalog.get("models") or [])
                    if normalize_lookup_key(row.get("name"))
                }
                filtered, rejected = validate_catalog_selection(
                    payload.get("candidate_variables") or [],
                    catalog_by_name,
                    business_context=request_context,
                    preferred_names={str(item.get("name") or "") for item in (payload.get("candidate_variables") or []) if isinstance(item, dict)},
                    industry_type=industry_type,
                )
                filtered, duplicates = dedupe_catalog_against_db(filtered, persisted_variables or [])
                if rejected:
                    notes.append(
                        "LLM-proposed variable names outside the active MongoDB source catalog were rejected: "
                        + ", ".join(rejected)
                    )
                if duplicates:
                    notes.append(
                        "Source-backed variables semantically duplicated by MongoDB variables were suppressed; DB definitions remain authoritative: "
                        + ", ".join(duplicates)
                    )
                # ``_json_source_spec`` is backend-only compiler metadata added during
                # source validation. It must never cross the public ScenarioIntent/VariableIdea
                # boundary because those contracts intentionally reject unknown fields.
                # The deterministic compiler re-resolves the canonical source contract from the
                # complete MongoDB catalog, so only the semantic candidate fields belong here.
                sanitized_candidates: list[dict[str, Any]] = []
                for candidate in filtered:
                    if not isinstance(candidate, dict):
                        continue
                    sanitized_candidates.append({
                        key: value
                        for key, value in candidate.items()
                        if not str(key).startswith("_")
                    })
                payload["candidate_variables"] = sanitized_candidates
                allowed_rule_names = {
                    str(item.get("name") or "").strip().casefold()
                    for item in sanitized_candidates
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                }
                allowed_rule_names.update(
                    str(item.get("name") or "").strip().casefold()
                    for item in (persisted_variables or [])
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                )
                cleaned_rules: list[dict[str, Any]] = []
                for rule in payload.get("behavioral_rules") or []:
                    if not isinstance(rule, dict):
                        continue
                    when = rule.get("when") if isinstance(rule.get("when"), dict) else {}
                    then = rule.get("then") if isinstance(rule.get("then"), dict) else {}
                    if not when or not then:
                        continue
                    if not all(str(name).strip().casefold() in allowed_rule_names for name in when):
                        continue
                    if not all(str(name).strip().casefold() in allowed_rule_names for name in then):
                        continue
                    cleaned_rules.append({
                        "when": dict(when),
                        "then": dict(then),
                    })
                payload["behavioral_rules"] = cleaned_rules[:30]
                payload["notes"] = notes[:50]
            intent = ScenarioIntent.model_validate(payload)
        except LLMUpstreamError:
            raise
        except Exception as exc:
            logger.exception("Gemini intent request returned unusable JSON")
            raise LLMUpstreamError(
                f"Gemini intent response validation failed: {type(exc).__name__}: {_safe_exception_text(exc)}",
                provider="Google Gemini",
                model=self.model_name,
            ) from exc

        # Backend-owned values always win over model guesses.
        intent.industry_type = industry_type
        if country:
            intent.country = country
        if domain_query:
            intent.domain = domain_query
        return intent


    @staticmethod
    def profile_targets(fields: list[GeneratedSchemaField]) -> list[GeneratedSchemaField]:
        return [f for f in fields if is_placeholder_only(f.model_dump()) or _is_number_target(f)]

    def _ask_profiles(self, context: dict[str, Any], chunk: list[GeneratedSchemaField]) -> dict[str, Any]:
        prompt = json.dumps({"context": context, "fields": [_signature(f) for f in chunk]}, ensure_ascii=False, indent=1)
        reply = self._get_client().generate_json(SYSTEM_PROMPT, prompt, temperature=0.4)
        rows = reply.get("fields") if isinstance(reply, dict) else reply
        by_name = {f.name: f for f in chunk}
        found: dict[str, Any] = {}
        for row in rows or []:
            if not isinstance(row, dict) or row.get("name") not in by_name:
                continue
            field = by_name[row["name"]]
            if _is_number_target(field):
                bounds = _usable_range(field, row.get("min"), row.get("max"))
                if bounds:
                    found[field.name] = bounds
            else:
                values = _usable(field, row.get("values"))
                if values:
                    found[field.name] = values
        return found

    def profile_values(self, fields: list[GeneratedSchemaField], *, context: dict[str, Any], scenario_id: str) -> tuple[list[GeneratedSchemaField], bool]:
        """Give placeholder-only source fields model-authored, contract-checked, locked values: ``(fields, complete)``. ``complete`` is False when a model call failed, so the caller does not freeze the result."""
        targets = self.profile_targets(fields)
        if not targets:
            return fields, True
        digest = hashlib.sha256(json.dumps(
            [PROFILE_VERSION, context, sorted((_signature(f) for f in targets), key=lambda s: s["name"])], sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        lock_key = f"{PROFILE_VERSION}:{digest}"
        stored = None
        try:
            stored = get_selection_lock(lock_key)
        except Exception as exc:                                  # an unreachable store must not stop a proposal
            logger.warning("[IntentAgent/ValueProfile] lock store unavailable: %s: %s", type(exc).__name__, exc)
        complete = True
        if stored is None:
            chunks = [targets[i:i + CHUNK] for i in range(0, len(targets), CHUNK)]
            profiles: dict[str, Any] = {}

            def run(chunk: list[GeneratedSchemaField]) -> dict[str, Any] | None:
                try:
                    return self._ask_profiles(context, chunk)
                except Exception as exc:
                    logger.warning("[IntentAgent/ValueProfile] model call for %d field(s) failed: %s: %s", len(chunk), type(exc).__name__, exc)
                    return None

            with ThreadPoolExecutor(max_workers=min(WORKERS, len(chunks))) as pool:
                for answer in pool.map(run, chunks):
                    if answer is None:
                        complete = False
                    else:
                        profiles.update(answer)
            if complete and profiles:
                try:
                    stored = lock_selection(lock_key, scenario_id, {"profiles": profiles})
                except Exception as exc:
                    logger.warning("[IntentAgent/ValueProfile] could not lock profiles: %s: %s", type(exc).__name__, exc)
                    stored = {"profiles": profiles}
            else:
                stored = {"profiles": profiles}
        profiles = (stored or {}).get("profiles") or {}
        target_names = {f.name for f in targets}
        enriched: list[GeneratedSchemaField] = []
        filled = 0
        for field in fields:
            values = profiles.get(field.name)
            if values and field.name in target_names:
                params = dict(field.params or {})
                if isinstance(values, dict):                          # a number: the range where real values lie
                    params.update(min=values["min"], max=values["max"], range_source="value_profile")
                else:
                    params["source_examples"] = list(values)
                provenance = dict(field.provenance or {})
                provenance["value_profile"] = PROFILE_VERSION
                field = field.model_copy(update={"params": params, "provenance": provenance})
                filled += 1
            enriched.append(field)
        logger.info("[IntentAgent/ValueProfile] %d of %d placeholder-only field(s) given values", filled, len(targets))
        return enriched, complete


    def humanize_names(self, fields: list[GeneratedSchemaField], *, context: dict[str, Any], scenario_id: str,
                       keep: set[str] | None = None, reserved: set[str] | None = None) -> tuple[list[GeneratedSchemaField], dict[str, str], bool]:
        """Give source-backed fields short business names: ``(fields, {technical name: readable name}, complete)``.

        ``keep`` names are never renamed (the entity key, columns a DB variable refers to). ``reserved`` names (the DB variables)
        are never taken. The model proposes, every proposal is checked (shape, length, no filler, unique), and anything it did
        not give or got wrong is named by the deterministic fallback, so the result is always complete; ``complete`` is False only
        when the model could not be reached, so the caller does not freeze a fallback-only answer.
        """
        keep_keys = {str(k).casefold() for k in (keep or set())}
        targets = [f for f in fields if (f.provenance or {}).get("source_json_id") and f.name.casefold() not in keep_keys]
        if not targets:
            return fields, {}, True
        fixed = {f.name.casefold() for f in fields if f not in targets} | {str(r).casefold() for r in (reserved or set())}
        digest = hashlib.sha256(json.dumps(
            [NAMES_VERSION, {k: context.get(k) for k in ("industry", "domain", "standards")}, sorted(fixed),
             sorted((_name_signature(f) for f in targets), key=lambda s: s["name"])], sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")).hexdigest()
        lock_key = f"{NAMES_VERSION}:{digest}"
        stored = None
        try:
            stored = get_selection_lock(lock_key)
        except Exception as exc:
            logger.warning("[IntentAgent/Names] lock store unavailable: %s: %s", type(exc).__name__, exc)
        complete = True
        proposed: dict[str, str] = {}
        if stored is not None:
            proposed = {str(k): str(v) for k, v in ((stored or {}).get("names") or {}).items()}
        else:
            prompt = json.dumps({"context": {k: context.get(k) for k in ("industry", "domain", "use_case", "country", "standards")},
                                 "reserved": sorted(fixed)[:80], "fields": [_name_signature(f) for f in targets]}, ensure_ascii=False, indent=1)
            try:
                reply = self._get_client().generate_json(NAMES_PROMPT, prompt, temperature=0.2)
                rows = reply.get("fields") if isinstance(reply, dict) else reply
                for row in rows or []:
                    if isinstance(row, dict) and isinstance(row.get("name"), str):
                        proposed[row["name"]] = str(row.get("readable") or "")
            except Exception as exc:
                complete = False
                logger.warning("[IntentAgent/Names] model call for %d field(s) failed: %s: %s", len(targets), type(exc).__name__, exc)
        taken = set(fixed)
        mapping: dict[str, str] = {}
        from_model = 0
        for field in targets:                                               # pass 1: accepted model names, first come first served
            candidate = _readable_ok(proposed.get(field.name), taken)
            if candidate:
                mapping[field.name] = candidate
                taken.add(candidate)
                from_model += 1
        for field in targets:                                               # pass 2: the fallback names the rest
            if field.name not in mapping:
                candidate = _fallback_name(field, taken)
                mapping[field.name] = candidate
                taken.add(candidate)
        if stored is None and complete:
            try:
                lock_selection(lock_key, scenario_id, {"names": mapping})
            except Exception as exc:
                logger.warning("[IntentAgent/Names] could not lock names: %s: %s", type(exc).__name__, exc)
        mapping = {old: new for old, new in mapping.items() if old != new}
        logger.info("[IntentAgent/Names] %d of %d field(s) renamed (%d by the model)", len(mapping), len(targets), from_model)
        return apply_renames(fields, mapping), mapping, complete


# Backward-compatible internal alias for older imports.
PydanticAIIntentAgent = GeminiIntentAgent

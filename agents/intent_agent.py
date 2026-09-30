"""Gemini intent agent: natural-language scenario in, validated semantic intent out.

The proposal path deliberately does not use provider-native structured-output schemas.
Gemini's structured-schema surface accepts only a subset of JSON Schema, and complex
Pydantic schemas can be rejected with 400 INVALID_ARGUMENT before the model returns an
answer. We therefore request plain JSON and validate it locally with Pydantic.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from core.agentic_models import ScenarioIntent
from core.errors import LLMUpstreamError
from core.llm_client import GeminiClient
from core.json_domain_policy import is_json_grounded_domain
from core.industry_source_store import (
    catalog_for_request,
    normalize_lookup_key,
    select_json_source_catalog,
    _catalog_business_metadata_penalty,
    _catalog_display_only,
    _catalog_role,
    _catalog_text_tokens,
    _canonical_context_tokens,
    _source_owner_family,
    _normalize_model_name,
    semantic_exclusion_aliases,
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
  ``Bucket.status`` with ``TopupBalance.status`` merely because both are named ``status``. Conversely, repeated
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
        # without ever naming a telecom field explicitly.
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
        meaningful_by_role = {
            role: [row for row in values if float(_catalog_business_metadata_penalty(row)) < 30.0]
            for role, values in by_role.items()
        }

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
        industry_type: str = "telecom",
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
            "For behavior-oriented scenarios, explicitly check source models that describe usage/consumption, balance/depletion, recharge/top-up, eligibility/qualification, outcomes and customer context before returning the selection. "
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
                    temperature=0.2,
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
                    temperature=0.2,
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


# Backward-compatible internal alias for older imports.
PydanticAIIntentAgent = GeminiIntentAgent

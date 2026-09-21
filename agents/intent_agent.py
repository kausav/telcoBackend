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
from core.telecom_registry import TelecomRegistry

logger = logging.getLogger(__name__)


INSTRUCTIONS = """
You are the intent-understanding and variable-ideation agent for a telecom synthetic-data platform.
Interpret the COMPLETE business request and propose a FRESH semantic variable set.

IMPORTANT BOUNDARIES:
- Return JSON only. Do not wrap the JSON in markdown fences.
- Do not output executable generators, generator parameters, formulas, SQL, or schema implementation details.
- Candidate variables are semantic ideas only. The deterministic compiler assigns executable generator contracts.
- scenarioId is an identifier only and MUST NOT influence what variables are proposed.
- Use scenarioType, domain, businessScenario, useCase, country, and typeOfData together. Entity key is backend schema metadata; do not use it to ideate variables.
- There is NO variable-count target and NO variable-count maximum. Be COMPREHENSIVE in semantic coverage,
  but include ONLY variables that materially represent this business scenario. Never add a variable merely
  because an attribute exists somewhere in the telecom registry or in a related entity. The compiler will
  ground relevant ideas to the approved registry and will reject any idea that lacks a safe executable
  generation contract.
- For telecom transactional data, subscriber_id, account_id and msisdn are mandatory and must always be
  present as stable entity-level fields.
- Candidate variable names must be FRESH and should not simply copy a template or reference list.
- Avoid redundant identity/contact fields. In a telecom transactional scenario, msisdn is the canonical
  subscriber mobile identifier; do NOT also propose phoneNumber, mobileNumber, telephoneNumber, or equivalent
  duplicates unless the business scenario explicitly requires a separate contact-medium concept.
- For transactional data, distinguish stable entity/profile fields from repeated transaction/event/decision fields using grain.
- Prefer variables that explain triggers, states, transitions, outcomes, timing, monetary/usage measures,
  decisions, contention, suppression, recovery, or retention when those concepts fit the scenario.
- Treat scenarioType as a behavioral mode and make the variable set materially reflect it.
- Cover only concepts justified by the current business scenario and domain.
- The telecom catalog is grounding information, not a variable template. Do not dump catalog attributes.
- Do not propose unsupported nested/object fields when a flat synthetic dataset cannot deterministically
  populate their nested structure.

Return this JSON shape:
{
  "industry_type": "telecom",
  "domain": "...",
  "subdomain": "prepaid|postpaid|charging|usage|customer|network|unknown",
  "scenario_type": "...",
  "type_of_data": "transactional|aggregational",
  "entity_key": "...",
  "use_case": "...",
  "requested_entities": ["..."],
  "requested_relationships": ["..."],
  "candidate_variables": [
    {
      "name": "fresh_variable_name",
      "description": "...",
      "role": "identity|profile|event|transaction|status|measurement|metric|timing|decision|configuration|derived|other",
      "grain": "entity|transaction|event|derived",
      "dtype": "string|integer|float|decimal|boolean|categorical|datetime|date",
      "depends_on": ["existing_candidate_variable_name"]
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
        normalized["industry_type"] = "telecom"
    if not isinstance(normalized.get("domain"), str):
        normalized["domain"] = ""
    if not isinstance(normalized.get("scenario_type"), str):
        normalized["scenario_type"] = ""
    if not isinstance(normalized.get("type_of_data"), str):
        normalized["type_of_data"] = "transactional"
    if normalized.get("subdomain") not in {"prepaid", "postpaid", "charging", "usage", "customer", "network", "unknown"}:
        normalized["subdomain"] = "unknown"
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


class GeminiIntentAgent:
    """Compatibility-named intent agent using Gemini JSON mode + local Pydantic validation.

    The intent path intentionally uses the official Google GenAI SDK in JSON mode.
    PydanticAI is intentionally not used here because provider-native structured
    output was the source of the observed Gemini 400 INVALID_ARGUMENT failures.
    """

    def __init__(self, api_key: str | None = None, registry: TelecomRegistry | None = None):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.registry = registry or TelecomRegistry()
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
    ) -> ScenarioIntent:
        catalog = self.registry.llm_catalog_context(
            query=" ".join(part for part in (domain_query, request_context) if part)
        )
        catalog_text = json.dumps(catalog, separators=(",", ":"), sort_keys=True)
        prompt = (
            "APPROVED TELECOM STANDARDS REGISTRY (authoritative grounding only):\n"
            "The context below contains ALL official standard/model source URLs currently registered by the application, "
            "plus the complete relevant entity, attribute, relationship, and provenance records derived from those official models. "
            "Use the complete context to select the concepts needed by the business scenario; do not treat one source "
            "family as automatically sufficient when the scenario spans multiple telecom standards.\n\n"
            f"{catalog_text}\n\n"
            "Authoritative request context:\n"
            f"{request_context}\n\n"
            f"Selected industry: {industry_type}\n"
            f"Business domain: {domain_query or '<none>'}\n"
            f"Country: {country or '<none>'}\n\n"
            "Create a comprehensive fresh scenario intent. There is NO artificial variable-count target or maximum. "
            "Prefer the maximum set of relevant variables supported by the complete approved registry and the scenario semantics. "
            "Never truncate for count. For transactional telecom scenarios, ALWAYS include subscriber_id, account_id, and msisdn. "
            "Do not invent unsupported telecom entities or fields; the compiler will ground only relevant selected concepts from the approved registry. "
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

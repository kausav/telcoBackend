"""PydanticAI intent agent: natural language in, no schema authority out."""
from __future__ import annotations

import json
import os

from core.agentic_models import ScenarioIntent
from core.telecom_registry import TelecomRegistry
from core.errors import LLMUpstreamError
import logging
import re


logger = logging.getLogger(__name__)


def _safe_exception_text(exc: Exception) -> str:
    text = str(exc)[:1600]
    return re.sub(r"(?i)(api[_-]?key|token|authorization|bearer|password|secret)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)


INSTRUCTIONS = """
You are the intent-understanding and variable-ideation agent for a telecom synthetic-data platform.
You must interpret the COMPLETE business request and propose a FRESH semantic variable set.

IMPORTANT BOUNDARIES:
- Do not output executable generators, params, formulas, enum values, SQL, or schema implementation details.
- Candidate variables are semantic ideas only. The deterministic compiler assigns the executable generator contract.
- `scenarioId` is an identifier only and MUST NOT influence what variables are proposed.
- Use `scenarioType`, `domain`, `businessScenario`, `useCase`, `country`, `typeOfData`, and `entityKey` together.
- Aim for 30-35 candidate variables when the scenario naturally supports that breadth. If the scenario genuinely needs fewer variables, return fewer; never pad with irrelevant variables.
- Candidate variable names should be scenario-specific and avoid generic template replay.
- Always include the requested entity key when it is meaningful for the requested grain.
- For transactional data, distinguish stable entity/profile fields from repeated transaction/event/decision fields using `grain`.
- Prefer variables that explain triggers, state transitions, outcomes, timing, monetary/usage measures, decisions, and cross-journey behavior when those concepts fit the scenario.
- Treat scenarioType as a behavioral mode and make the variable set materially different across modes; do not reuse a generic prepaid template.
- Cover only concepts justified by the current businessScenario/domain/useCase. Prefer scenario-specific trigger, decision, outcome, timing, recovery, suppression, contention, or retention concepts over generic profile fields when relevant.
- Use the requested entityKey as the one allowed canonical identity field; otherwise prefer descriptive, scenario-specific names.
- The registry/catalog provided by the backend is authoritative for telecom entities and standards. Use it to stay grounded, but do not simply dump registry attributes.

Return a ScenarioIntent containing:
1. intent metadata;
2. a fresh `candidate_variables` list of semantic variable ideas with `name`, `description`, `role`, `grain`, and `dtype`;
3. requested entities/relationships that genuinely help the scenario.
"""


class PydanticAIIntentAgent:
    def __init__(self, api_key: str | None = None, registry: TelecomRegistry | None = None):
        try:
            from google.genai.types import HttpRetryOptions
            from pydantic_ai import Agent
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider
        except ImportError as exc:
            raise RuntimeError(
                "PydanticAI Google integration is not installed. Install 'pydantic-ai-slim[google]'."
            ) from exc

        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for the PydanticAI agent.")

        self.model_name = os.getenv("PYDANTIC_AI_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
        retry_options = HttpRetryOptions(
            attempts=max(1, min(6, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "1")))),
            initial_delay=1.0,
            max_delay=20.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        )
        provider = GoogleProvider(api_key=key, retry_options=retry_options)
        model = GoogleModel(self.model_name, provider=provider)
        self.registry = registry or TelecomRegistry()
        self.agent = Agent(model, output_type=ScenarioIntent, instructions=INSTRUCTIONS, retries=max(0, min(1, int(os.getenv("PYDANTIC_AI_RETRIES", "0")))))

    def run(
        self,
        message: str,
        country: str | None = None,
        industry_type: str = "telecom",
        domain_query: str | None = None,
    ) -> ScenarioIntent:
        # The backend chooses the industry first. The catalog passed to the LLM is
        # then constrained to the requested business-domain query, so the model is
        # never asked to search the full telecom ontology from scratch.
        catalog = self.registry.search(domain_query or "", limit=30) if domain_query else self.registry.catalog_summary(limit=30)
        catalog_text = json.dumps(catalog, separators=(",", ":"), sort_keys=True)
        prompt = (
            "Approved telecom domain catalog:\n" + catalog_text + "\n\n"
            f"Current user request:\n{message}\n\n"
            f"Selected industry (authoritative backend value): {industry_type}\n"
            f"Business domain query (authoritative backend value): {domain_query or '<none>'}\n"
            f"External country override: {country or '<none>'}\n"
            "Create a fresh candidate variable set. Aim for 30-35 only when justified by the business scenario; fewer is valid when semantically sufficient. "
            "Make the set materially reflect scenarioType, businessScenario, domain, useCase and typeOfData. "
            "Return only the ScenarioIntent structure."
        )
        try:
            result = self.agent.run_sync(prompt)
        except Exception as exc:
            logger.exception("PydanticAI/Gemini intent request failed", exc_info=exc)
            raise LLMUpstreamError(
                f"Gemini intent request failed: {type(exc).__name__}: {_safe_exception_text(exc)}",
                provider="Google Gemini",
                model=self.model_name,
            ) from exc
        return result.output

"""PydanticAI intent agent: natural language in, no schema authority out."""
from __future__ import annotations

import os

from core.agentic_models import ScenarioIntent
from core.telecom_registry import TelecomRegistry


INSTRUCTIONS = """
You are the intent-understanding agent for a telecom synthetic-data platform.
You NEVER design a database schema and NEVER invent telecom entities, attributes,
relationships, enum values, formulas, generators, or standards claims.

Your only job is to interpret the user's natural-language request and return:
- telecom subdomain intent
- requested concepts using names/aliases from the supplied catalog where possible
- country/currency hints
- requested record count/time window
- explicit ambiguities that require human clarification

If a concept is not in the catalog, keep it as an ambiguity; do not manufacture a synonym
that appears to be a real entity. Standards applicability is derived from registry provenance
later and must never be asserted by you.
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

        model_name = os.getenv("PYDANTIC_AI_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
        retry_options = HttpRetryOptions(
            attempts=max(1, min(6, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "4")))),
            initial_delay=1.0,
            max_delay=20.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        )
        provider = GoogleProvider(api_key=key, retry_options=retry_options)
        model = GoogleModel(model_name, provider=provider)
        self.registry = registry or TelecomRegistry()
        self.agent = Agent(model, output_type=ScenarioIntent, instructions=INSTRUCTIONS, retries=2)

    def run(
        self,
        message: str,
        history: list[dict[str, str]],
        country: str | None = None,
        record_count: int | None = None,
    ) -> ScenarioIntent:
        catalog = self.registry.catalog_summary(limit=500)
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-20:])
        prompt = (
            "Approved telecom catalog:\n" + str(catalog) + "\n\n"
            "Conversation history:\n" + (history_text or "<none>") + "\n\n"
            f"Current user request:\n{message}\n\n"
            f"External country override: {country or '<none>'}\n"
            f"External record-count override: {record_count if record_count is not None else '<none>'}\n"
            "Return only the ScenarioIntent structure."
        )
        result = self.agent.run_sync(prompt)
        return result.output

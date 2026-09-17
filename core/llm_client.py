"""Gemini client wrapper used by the legacy generation pipeline.

The agentic schema proposal path uses PydanticAI directly; this client remains for
legacy generation/QA stages. Provider failures are normalized into a stable application error contract.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

from config.runtime import ROOT
from core.errors import LLMUpstreamError

load_dotenv(ROOT / ".env", override=False)

logger = logging.getLogger(__name__)


def _safe_provider_error(exc: Exception) -> str:
    raw = str(exc)[:2000]
    return re.sub(
        r"(?i)(api[_-]?key|token|authorization|bearer|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        raw,
    )


class GeminiClient:
    """Thin, synchronous wrapper around the supported ``google-genai`` SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is required. Install requirements.txt.") from exc

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY before starting the service.")

        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        timeout_ms = max(1000, int(os.getenv("GEMINI_TIMEOUT_MS", "60000")))
        retry_attempts = max(1, min(6, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "1"))))
        retry_options = types.HttpRetryOptions(
            attempts=retry_attempts,
            initial_delay=1.0,
            max_delay=20.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        )
        http_options = types.HttpOptions(timeout=timeout_ms, retry_options=retry_options)
        self._types = types
        self._client = genai.Client(api_key=key, http_options=http_options)

    def generate_json(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> dict | list:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
        }
        if not self.model.startswith("gemini-3"):
            config_kwargs["temperature"] = temperature

        try:
            response = self._client.models.generate_content(
                model=self.model,
                config=self._types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except Exception as exc:
            logger.exception("Gemini JSON request failed")
            raise LLMUpstreamError(
                f"Gemini JSON request failed: {type(exc).__name__}: {_safe_provider_error(exc)}",
                provider="Google Gemini",
                model=self.model,
            ) from exc

        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

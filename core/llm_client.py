from __future__ import annotations
import json
import os
import logging
import re
from urllib.request import Request as UrlRequest, urlopen

from fastapi import HTTPException
from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _public_egress_ip() -> str | None:
    """Best-effort public egress address used by the deployed service."""
    try:
        req = UrlRequest("https://api.ipify.org", headers={"User-Agent": "telco-backend/1.0"})
        with urlopen(req, timeout=2.0) as response:
            value = response.read(64).decode("ascii", errors="ignore").strip()
        if re.fullmatch(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}|[0-9a-fA-F:]+", value):
            return value
    except Exception:
        logger.warning("Unable to determine public egress IP", exc_info=True)
    return None


def _raise_llm_upstream_error(exc: Exception) -> None:
    """Turn Gemini/network failures into a diagnosable HTTP 502."""
    public_ip = _public_egress_ip()
    raw = str(exc)[:3000]
    safe = re.sub(
        r"(?i)(api[_-]?key|token|authorization|bearer|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        raw,
    )
    details = {
        "provider": "Google Gemini",
        "provider_error": safe,
        "action": "If this is an IP allowlist/network restriction, whitelist the reported public_egress_ip for the deployed service.",
        "public_egress_ip": public_ip or "Unable to determine automatically; check the deployment's outbound/NAT public IP.",
    }
    raise HTTPException(502, detail={"error": "LLM upstream request failed", "details": details}) from exc

load_dotenv()  # loads .env from the project root


class GeminiClient:
    """Thin wrapper around google-genai for structured JSON generation."""

    MODEL = "gemini-3.6-flash"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "Set the GEMINI_API_KEY environment variable before running."
            )
        self._client = genai.Client(api_key=key)

    def generate_json(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> dict | list:
        """Call Gemini and parse the response as JSON."""
        # Gemini 3.x is optimized around its default sampling configuration.
        # Keep the public method signature unchanged for all agents, but do not send
        # legacy sampling knobs to Gemini 3.x. This is especially important on the
        # generation path, where the orchestrator/schema agents can otherwise fail
        # before any records are produced.
        config_kwargs = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
        }
        if not self.MODEL.startswith("gemini-3"):
            config_kwargs["temperature"] = temperature
        try:
            response = self._client.models.generate_content(
                model=self.MODEL,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except Exception as exc:
            _raise_llm_upstream_error(exc)
        text = (response.text or "").strip()
        # Gemini occasionally wraps JSON in ```json ... ``` fences despite response_mime_type.
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.startswith("json") else text
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

    def generate_text(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.4,
    ) -> str:
        """Call Gemini and return plain text."""
        config_kwargs = {"system_instruction": system_instruction}
        if not self.MODEL.startswith("gemini-3"):
            config_kwargs["temperature"] = temperature
        try:
            response = self._client.models.generate_content(
                model=self.MODEL,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except Exception as exc:
            _raise_llm_upstream_error(exc)
        return response.text.strip()
    
    
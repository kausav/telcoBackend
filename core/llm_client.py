"""Gemini client wrapper used by the legacy generation pipeline.

The agentic schema proposal path uses the Google GenAI SDK with local Pydantic validation; this
client remains the shared Gemini wrapper for proposal and legacy generation/QA stages. Provider failures are
normalized into a stable application error contract.
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


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    text = str(exc)
    match = re.search(r"\b([45]\d{2})\b", text)
    return int(match.group(1)) if match else None


def _close_open(text: str) -> str:
    """``text`` with its open string, objects and arrays closed (a reply that was cut off)."""
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    out = text + ('"' if in_string else "")
    out = re.sub(r"[,:\s]+$", "", out) if not in_string else out
    if out.rstrip().endswith('"') and stack and stack[-1] == "}" and re.search(r'[{,]\s*"[^"]*"$', out):
        out += ": null"                       # a key that was cut off before its value
    return out + "".join(reversed(stack))


def _mend(text: str, exc: json.JSONDecodeError) -> str | None:
    """One repair for the commonest ways a model breaks JSON (a stray backslash, a missing or trailing comma, an unescaped quote
    inside a string, trailing text, a reply cut off), or None when the error is not one of them."""
    pos, msg = exc.pos, exc.msg
    here = text[pos] if pos < len(text) else ""
    before = pos - 1
    while before >= 0 and text[before].isspace():
        before -= 1
    if msg.startswith("Invalid \\escape") or msg.startswith("Invalid control character"):
        if msg.startswith("Invalid control character"):
            return text[:pos] + {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(here, " ") + text[pos + 1:]
        return text[:pos] + "\\" + text[pos:]
    if msg.startswith("Extra data"):
        return text[:pos]
    if not here or msg.startswith("Unterminated string"):
        return _close_open(text)
    if here in "}]" and before >= 0 and text[before] == ",":
        return text[:before] + text[before + 1:]
    if msg.startswith("Expecting property name") and before >= 0 and text[before] == ",":
        return text[:before] + text[before + 1:]
    if msg.startswith("Expecting ',' delimiter"):
        if here in '"{[' or here.isdigit() or text.startswith(("true", "false", "null"), pos):
            return text[:pos] + "," + text[pos:]
        if before >= 0 and text[before] == '"':
            return text[:before] + "\\" + text[before:]            # that quote was part of the string
    return None


def loads_lenient(text: str) -> Any:
    """``json.loads``, after mending the small breakages a model's JSON reply is prone to (the error is raised when it cannot be mended)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as first:
        fixed, exc = text, first
        for _ in range(80):
            mended = _mend(fixed, exc)
            if mended is None or mended == fixed:
                raise first
            fixed = mended
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as again:
                exc = again
        raise first


class GeminiClient:
    """Thin, synchronous wrapper around the supported ``google-genai`` SDK."""

    def __init__(self, api_key: str | None = None, timeout_ms: int | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is required. Install requirements.txt.") from exc

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY before starting the service.")

        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        timeout_ms = max(1000, int(timeout_ms if timeout_ms else os.getenv("GEMINI_TIMEOUT_MS", "60000")))
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

    def generate_text(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Generate plain text without response-schema/JSON-mode constraints."""
        config_kwargs: dict[str, Any] = {"system_instruction": system_instruction}
        if not self.model.startswith("gemini-3"):
            config_kwargs["temperature"] = temperature
        try:
            response = self._client.models.generate_content(
                model=self.model,
                config=self._types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except Exception as exc:
            logger.exception("Gemini text request failed")
            raise LLMUpstreamError(
                f"Gemini text request failed: {type(exc).__name__}: {_safe_provider_error(exc)}",
                provider="Google Gemini",
                model=self.model,
                status_code=_extract_status_code(exc),
            ) from exc
        return (response.text or "").strip()

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
                status_code=_extract_status_code(exc),
            ) from exc

        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            return loads_lenient(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

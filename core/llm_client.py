from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
        response = self._client.models.generate_content(
            model=self.MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=temperature,
            ),
            contents=user_prompt,
        )
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
        response = self._client.models.generate_content(
            model=self.MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
            contents=user_prompt,
        )
        return response.text.strip()
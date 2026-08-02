"""
ai_service.py

Centralized wrapper around the Google Gemini API (google-genai SDK).
Every other AI module (ai_insights, ai_chat, recommendations, report
wording) goes through this single service so:
  - the client / API key handling only lives in one place
  - errors are normalized into a single AIResponse shape
  - the rest of the app never crashes because of an AI failure

If no API key is configured, or the SDK/package is missing, or a request
fails for any reason, `is_available` is False and the caller should fall
back to a "AI features are currently unavailable. Core analytics are
still available." message — the rest of the dashboard keeps working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from utils.constants import (
    GEMINI_MODEL_PRIMARY,
    GEMINI_MODEL_FALLBACK,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TEMPERATURE,
    AI_SYSTEM_GUARDRAILS,
    AI_UNAVAILABLE_MESSAGE,
)

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False


@dataclass
class AIResponse:
    success: bool
    text: str = ""
    error_message: Optional[str] = None


class AIService:
    """Thin, defensive wrapper around the Gemini API."""

    def __init__(self):
        self._api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None
        self._init_error: Optional[str] = None

        if not GENAI_SDK_AVAILABLE:
            self._init_error = "The google-genai package is not installed."
            return

        if not self._api_key:
            self._init_error = "No GEMINI_API_KEY was found in the environment."
            return

        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Failed to initialize the Gemini client: {exc}"
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str:
        return self._init_error or AI_UNAVAILABLE_MESSAGE

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = GEMINI_TEMPERATURE,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
    ) -> AIResponse:
        """Send a prompt to Gemini and return a normalized response."""
        if not self.is_available:
            return AIResponse(success=False, error_message=self.unavailable_reason)

        full_system_instruction = system_instruction or AI_SYSTEM_GUARDRAILS

        for model_name in (GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK):
            try:
                config = genai_types.GenerateContentConfig(
                    system_instruction=full_system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                if not text:
                    # Try the fallback model before giving up
                    continue
                return AIResponse(success=True, text=text.strip())
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                # If this looks like a quota/rate-limit/auth error, try fallback model once;
                # otherwise still try the fallback since we don't want a transient issue
                # to fully disable AI features for the session.
                continue

        return AIResponse(
            success=False,
            error_message=(
                "The AI service did not return a usable response. This can happen due to "
                "rate limits, quota exhaustion, or a temporary API issue. Core analytics "
                "remain available."
            ),
        )


# Module-level singleton so Streamlit doesn't reinitialize the client on every rerun.
_service_instance: Optional[AIService] = None


def get_ai_service() -> AIService:
    global _service_instance
    if _service_instance is None:
        _service_instance = AIService()
    return _service_instance


def reset_ai_service() -> None:
    """Force re-initialization (e.g. after the user edits their .env and restarts)."""
    global _service_instance
    _service_instance = None

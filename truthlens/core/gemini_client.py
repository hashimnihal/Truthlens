"""
core/gemini_client.py
---------------------
Centralized Gemini 2.5 Flash API client for TruthLens.

PRD reference: §7.1, §7.3, §18
Rules:
  - Temperature: Low (0.0 to 0.2) for fact checking and judgement calls
  - Structured JSON outputs validated by Pydantic (retry once on schema mismatch)
  - Exponential backoff on rate limits / transient network errors
  - Clear user-facing error message on quota exhaustion (429)
  - MOCK=1 mode: returns deterministic fixture responses without API key or network calls
  - API keys read via python-dotenv; never hardcoded or logged
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from core.models import (
    BatchedEvidenceJudgement,
    ClaimsAnalysis,
    ExtractedClaim,
    SourceStanceJudgement,
)

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = "gemini-3.5-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.5


# ── Exceptions ────────────────────────────────────────────────────────────────

class GeminiError(Exception):
    """Base exception for Gemini client errors."""


class GeminiConfigError(GeminiError):
    """Raised when configuration is invalid (e.g., missing API key in live mode)."""


class GeminiQuotaError(GeminiError):
    """Raised when Gemini API quota or rate limit is reached (HTTP 429)."""


class GeminiValidationError(GeminiError):
    """Raised when the model response fails Pydantic schema validation after retries."""


# ── Mode & Client Helpers ─────────────────────────────────────────────────────

def is_mock_mode() -> bool:
    """Return True if MOCK=1 or MOCK=true in environment."""
    return os.environ.get("MOCK", "0").strip().lower() in ("1", "true", "yes")


def get_gemini_api_key() -> str:
    """Retrieve GEMINI_API_KEY from environment or raise GeminiConfigError."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        raise GeminiConfigError(
            "GEMINI_API_KEY is not configured. "
            "Please set your API key in .env or run with MOCK=1 for development."
        )
    return key


def _get_mock_fixture(response_model: Type[T], prompt: str) -> T:
    """Generate a deterministic, valid fixture based on the requested model and prompt."""
    if response_model is ClaimsAnalysis:
        # Determine clickbait / claim indicators from prompt text
        prompt_lower = prompt.lower()
        is_shocking = any(w in prompt_lower for w in ["shocking", "won't believe", "secret", "exposed", "miracle"])
        clickbait_score = 0.85 if is_shocking else 0.15
        incongruity_score = 0.65 if "contradicts" in prompt_lower else 0.05

        headline_match = re.search(r'Headline:\s*"([^"]+)"', prompt)
        headline = headline_match.group(1) if headline_match else "Sample News Headline"

        fixture = ClaimsAnalysis(
            claims=[
                ExtractedClaim(
                    claim_text=f"Primary reported event: {headline}",
                    claim_type="event",
                    entities=["Primary Subject", "Location/Entity"],
                ),
                ExtractedClaim(
                    claim_text="Statistical metric or associated outcome cited in report",
                    claim_type="statistic",
                    entities=["Outcome", "Metric"],
                ),
            ],
            clickbait_rating=clickbait_score,
            incongruity=incongruity_score,
            rationale="Deterministic mock analysis: claims identified with baseline clickbait and consistency ratings.",
        )
        return fixture  # type: ignore[return-value]

    if response_model is BatchedEvidenceJudgement:
        # Extract all URLs listed in the prompt
        found_urls = re.findall(r"URL:\s*(https?://[^\s\)]+)", prompt)
        judgements: list[SourceStanceJudgement] = []
        is_contradictory = any(w in prompt.lower() for w in ["hoax", "debunked", "false claim", "fake"])

        for i, url in enumerate(found_urls):
            clean_url = url.strip()
            if is_contradictory and i == 0:
                stance = "contradicts"
                reason = "Source directly refutes the claim with official statements."
            elif i % 4 == 3:
                stance = "unrelated"
                reason = "Article discusses related background but does not substantiate the specific claim."
            else:
                stance = "supports"
                reason = "Article confirms the key details, parties involved, and timeline of the event."

            judgements.append(
                SourceStanceJudgement(
                    url=clean_url,
                    stance=stance,
                    reason=reason,
                )
            )

        summary = (
            "Evidence indicates solid consensus supporting the core factual claims."
            if not is_contradictory
            else "Evidence directly refutes or contradicts the reported claims."
        )

        return BatchedEvidenceJudgement(  # type: ignore[return-value]
            judgements=judgements,
            evidence_summary=summary,
        )

    # Fallback generic default instantiation if model allows empty
    try:
        return response_model.model_validate({})
    except Exception as exc:
        raise GeminiError(
            f"No mock fixture available for {response_model.__name__}: {exc}"
        ) from exc


def _clean_json_text(text: str) -> str:
    """Strip markdown code fence formatting from raw LLM output if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_structured(
    prompt: str,
    response_model: Type[T],
    temperature: float = 0.1,
    system_instruction: str | None = None,
    model_name: str | None = None,
) -> T:
    """Generate structured JSON output from Gemini validated against a Pydantic model.

    Args:
        prompt: User prompt content.
        response_model: Pydantic model class to validate and return.
        temperature: Low temperature (0.0 to 0.2) for deterministic judgements.
        system_instruction: Optional system instruction for the model.
        model_name: Gemini model name (defaults to GEMINI_MODEL or gemini-2.5-flash).

    Returns:
        Instance of response_model populated and validated.

    Raises:
        GeminiConfigError: If API key is missing and MOCK=0.
        GeminiQuotaError: If quota / rate limit is exceeded after retries.
        GeminiValidationError: If response cannot be parsed into response_model.
        GeminiError: On unhandled client or server exceptions.
    """
    if is_mock_mode():
        logger.info(f"[MOCK MODE] Generating fixture for {response_model.__name__}")
        return _get_mock_fixture(response_model, prompt)

    api_key = get_gemini_api_key()
    selected_model = model_name or DEFAULT_MODEL

    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_model,
        temperature=temperature,
        system_instruction=system_instruction,
    )

    last_error: Exception | None = None
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=config,
            )

            # Check if parsed attribute directly holds the validated object
            if hasattr(response, "parsed") and isinstance(response.parsed, response_model):
                return response.parsed

            # Parse from raw text
            raw_text = getattr(response, "text", "")
            if not raw_text:
                raise GeminiValidationError("Gemini returned an empty response")

            cleaned = _clean_json_text(raw_text)
            return response_model.model_validate_json(cleaned)

        except errors.APIError as exc:
            msg = str(exc)
            # Detect model deprecated/not found -> fallback to FALLBACK_MODEL
            if "404" in msg and selected_model != FALLBACK_MODEL:
                logger.warning(
                    f"Model {selected_model} returned 404; automatically switching to {FALLBACK_MODEL}"
                )
                selected_model = FALLBACK_MODEL
                last_error = exc
                continue

            # Detect quota / rate limiting / high demand (503 UNAVAILABLE)
            if "429" in msg or "503" in msg or "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg or "quota" in msg.lower():
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Gemini rate/load threshold hit (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    last_error = exc
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    raise GeminiQuotaError(
                        "Gemini API quota or rate limit exceeded. "
                        "Please wait a minute or set MOCK=1 to continue offline."
                    ) from exc
                raise GeminiError(f"Gemini service temporarily overloaded: {exc}") from exc

            # Other API errors
            logger.error(f"Gemini API error: {exc}")
            raise GeminiError(f"Gemini API communication failed: {exc}") from exc

        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning(
                f"Schema validation failure on attempt {attempt + 1}/{MAX_RETRIES + 1}: {exc}"
            )
            if attempt < MAX_RETRIES:
                # Append a strict JSON correction request on retry
                prompt += (
                    f"\n\n[Correction Notice: Previous output failed schema validation: {exc}. "
                    "Ensure output strictly adheres to the requested JSON schema without extra keys.]"
                )
                time.sleep(1.0)
                last_error = exc
                continue
            raise GeminiValidationError(
                f"Failed to parse Gemini response into {response_model.__name__} after retries: {exc}"
            ) from exc

        except Exception as exc:
            logger.error(f"Unexpected error in Gemini generation: {exc}")
            raise GeminiError(f"Gemini request failed: {exc}") from exc

    raise GeminiError(f"Request failed after {MAX_RETRIES + 1} attempts: {last_error}")

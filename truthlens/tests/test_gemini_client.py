"""
tests/test_gemini_client.py
---------------------------
Unit tests for the centralized Gemini client (core/gemini_client.py).
Tests mock mode, config checks, JSON fence stripping, and error handling.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from core.gemini_client import (
    GeminiConfigError,
    GeminiError,
    GeminiQuotaError,
    GeminiValidationError,
    _clean_json_text,
    _get_mock_fixture,
    generate_structured,
    get_gemini_api_key,
    is_mock_mode,
)
from core.models import ClaimsAnalysis, ExtractedClaim


class SampleModel(BaseModel):
    message: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)


# ── Mock mode tests ───────────────────────────────────────────────────────────

def test_is_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK", "1")
    assert is_mock_mode() is True

    monkeypatch.setenv("MOCK", "true")
    assert is_mock_mode() is True

    monkeypatch.setenv("MOCK", "0")
    assert is_mock_mode() is False


def test_get_gemini_api_key_valid(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "real_test_key_12345")
    assert get_gemini_api_key() == "real_test_key_12345"


def test_get_gemini_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is not configured"):
        get_gemini_api_key()

    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is not configured"):
        get_gemini_api_key()


def test_clean_json_text_removes_fences():
    fenced = "```json\n{\"title\": \"test\"}\n```"
    assert _clean_json_text(fenced) == '{"title": "test"}'

    fenced_no_lang = "```\n{\"title\": \"test\"}\n```"
    assert _clean_json_text(fenced_no_lang) == '{"title": "test"}'

    plain = '{"title": "test"}'
    assert _clean_json_text(plain) == '{"title": "test"}'


def test_mock_fixture_claims_analysis():
    prompt = 'Headline: "Scientists discover water on Mars"\nArticle Body: "NASA confirmed findings today."'
    fixture = _get_mock_fixture(ClaimsAnalysis, prompt)

    assert isinstance(fixture, ClaimsAnalysis)
    assert len(fixture.claims) <= 3
    assert 0.0 <= fixture.clickbait_rating <= 1.0
    assert 0.0 <= fixture.incongruity <= 1.0
    assert len(fixture.claims) > 0


def test_generate_structured_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK", "1")
    res = generate_structured(
        prompt='Headline: "Economy grows by 3 percent"',
        response_model=ClaimsAnalysis,
    )
    assert isinstance(res, ClaimsAnalysis)
    assert len(res.claims) >= 1
    assert 0.0 <= res.clickbait_rating <= 1.0


# ── Live client simulation & Error handling tests ─────────────────────────────

def test_generate_structured_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is not configured"):
        generate_structured(
            prompt="Analyze headline",
            response_model=ClaimsAnalysis,
        )


def test_generate_structured_handles_quota_error(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")

    from google.genai import errors

    quota_error = errors.APIError(
        code=429,
        response_json={"error": {"message": "RESOURCE_EXHAUSTED: quota exceeded"}},
    )

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.side_effect = quota_error

        with patch("time.sleep"):  # Speed up retries
            with pytest.raises(GeminiQuotaError, match="quota or rate limit exceeded"):
                generate_structured(
                    prompt="test",
                    response_model=SampleModel,
                )


def test_generate_structured_validates_json_text(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")

    mock_response = MagicMock()
    mock_response.text = '{"message": "Verified fact", "score": 0.85}'
    mock_response.parsed = None

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = generate_structured(
            prompt="test prompt",
            response_model=SampleModel,
        )
        assert result.message == "Verified fact"
        assert result.score == 0.85

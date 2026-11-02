"""
tests/test_claims.py
--------------------
Unit tests for claim extraction and query building (core/claims.py).
"""

from __future__ import annotations

import pytest

from core.claims import build_claim_queries, extract_claims
from core.models import ClaimsAnalysis, ExtractedClaim


@pytest.fixture(autouse=True)
def enable_mock_mode(monkeypatch):
    """Enable MOCK mode for unit tests to avoid requiring an active API key."""
    monkeypatch.setenv("MOCK", "1")


def test_extract_claims_basic():
    headline = "RBI cuts repo rate by 25 basis points amid cooling inflation"
    body = "The Reserve Bank of India announced a reduction in key lending rates on Friday."

    result = extract_claims(headline=headline, body=body)

    assert isinstance(result, ClaimsAnalysis)
    assert 1 <= len(result.claims) <= 3
    assert 0.0 <= result.clickbait_rating <= 1.0
    assert 0.0 <= result.incongruity <= 1.0

    for claim in result.claims:
        assert isinstance(claim, ExtractedClaim)
        assert len(claim.claim_text) >= 3
        assert claim.claim_type in ["event", "statistic", "quote", "opinion", "other"]


def test_extract_claims_headline_only():
    headline = "ISRO successfully launches Chandrayaan-4 lunar sample return mission"
    result = extract_claims(headline=headline)

    assert isinstance(result, ClaimsAnalysis)
    assert len(result.claims) >= 1
    assert result.clickbait_rating <= 0.5  # Factual headline


def test_extract_claims_clickbait_detection():
    headline = "You WON'T BELIEVE what this minister exposed in SHOCKING secret video!"
    result = extract_claims(headline=headline)

    assert isinstance(result, ClaimsAnalysis)
    assert result.clickbait_rating >= 0.7  # Clickbait flag detected in fixture


def test_extract_claims_empty_headline_raises_value_error():
    with pytest.raises(ValueError, match="Headline cannot be empty"):
        extract_claims(headline="")

    with pytest.raises(ValueError, match="Headline cannot be empty"):
        extract_claims(headline="   ")


def test_build_claim_queries():
    claim = ExtractedClaim(
        claim_text="WHO declares end to global public health emergency",
        claim_type="event",
        entities=["WHO", "World Health Organization"],
    )

    queries = build_claim_queries(claim)

    assert isinstance(queries, list)
    assert len(queries) >= 1
    assert len(queries) <= 2
    # Ensure query 1 is based on the claim text
    assert any("WHO" in q or "emergency" in q for q in queries)


def test_extract_claims_ten_sample_headlines():
    """Verify claim extraction across 10 diverse sample news headlines (PRD Week 2 milestone)."""
    samples = [
        "NASA James Webb Space Telescope discovers oldest known galaxy in universe",
        "UN Climate Summit concludes with historic loss and damage fund pact",
        "European Central Bank raises deposit rate to record four percent",
        "Doctors perform groundbreaking total artificial heart transplant in Louisville",
        "SHOCKING miracle diet cures diabetes overnight with one simple fruit!",
        "Government approves 10,000 electric buses for major urban corridors",
        "Tech giant reports 20 percent surge in cloud revenue beating Wall Street forecasts",
        "Archaeologists unearth 3,000-year-old tomb complex in Luxor Egypt",
        "Global stock markets tumble following unexpected central bank rate hike",
        "You will NEVER believe who was spotted secretly dining with world leaders!",
    ]

    for headline in samples:
        res = extract_claims(headline=headline)
        assert isinstance(res, ClaimsAnalysis), f"Failed for headline: {headline}"
        assert 1 <= len(res.claims) <= 3
        assert 0.0 <= res.clickbait_rating <= 1.0
        assert 0.0 <= res.incongruity <= 1.0


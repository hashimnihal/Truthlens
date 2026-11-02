"""
tests/test_retrieval.py
-----------------------
Unit tests for news retrieval, trust tier resolution, and deduplication (core/retrieval.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import EvidenceCandidate, ExtractedClaim
from core.retrieval import (
    _mock_search_news,
    extract_domain,
    fetch_article_text,
    normalize_title,
    normalize_url,
    resolve_source_tier,
    retrieve_evidence_candidates,
    search_news,
)


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    """Enable MOCK mode for retrieval tests."""
    monkeypatch.setenv("MOCK", "1")


# ── Domain extraction & Normalization ─────────────────────────────────────────

def test_extract_domain():
    assert extract_domain("https://www.reuters.com/article/123") == "reuters.com"
    assert extract_domain("http://bbc.co.uk/news") == "bbc.co.uk"
    assert extract_domain("https://sub.domain.example.com:8080/test") == "sub.domain.example.com"
    assert extract_domain("not-a-valid-url") == ""


def test_normalize_title():
    t1 = "Breaking News: President Signs Major Bill!!!"
    t2 = "breaking news president signs major bill"
    assert normalize_title(t1) == normalize_title(t2)


def test_normalize_url():
    u1 = "https://example.com/news/article/?utm_source=twitter&ref=abc#section"
    u2 = "https://example.com/news/article"
    assert normalize_url(u1) == normalize_url(u2)


# ── Trust Tier Resolution ─────────────────────────────────────────────────────

def test_resolve_source_tier_tier1():
    tier, name = resolve_source_tier("https://www.reuters.com/world/article")
    assert tier == 1
    assert "Reuters" in name

    tier, name = resolve_source_tier("https://apnews.com/article/123")
    assert tier == 1

    tier, name = resolve_source_tier("https://www.bbc.com/news/world")
    assert tier == 1


def test_resolve_source_tier_tier2():
    tier, name = resolve_source_tier("https://www.theprint.in/politics/story")
    assert tier == 2
    assert "Print" in name

    tier, name = resolve_source_tier("https://www.snopes.com/fact-check/test")
    assert tier == 2


def test_resolve_source_tier_tier3_unknown():
    tier, name = resolve_source_tier("https://unknown-viral-blog-2026.net/rumors")
    assert tier == 3
    assert tier not in (1, 2)


# ── Search & Deduplication ───────────────────────────────────────────────────

def test_search_news_mock_mode():
    results = search_news("ISRO weather satellite launch", max_results=5)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "title" in r
        assert "url" in r
        assert r["url"].startswith("http")


def test_retrieve_evidence_candidates_deduplication():
    # Claims that will generate overlapping queries
    claims = [
        ExtractedClaim(
            claim_text="ISRO launched weather satellite",
            claim_type="event",
            entities=["ISRO", "satellite"],
        ),
        ExtractedClaim(
            claim_text="ISRO weather satellite orbital telemetry verified",
            claim_type="event",
            entities=["ISRO", "satellite"],
        ),
    ]

    candidates = retrieve_evidence_candidates(claims, max_per_query=5)
    assert isinstance(candidates, list)
    assert len(candidates) > 0

    # Verify no duplicate URLs exist
    urls = [c.url for c in candidates]
    assert len(urls) == len(set(urls)), "Duplicate URLs detected in candidates"

    # Verify tier tagging
    for c in candidates:
        assert isinstance(c, EvidenceCandidate)
        assert c.tier in (1, 2, 3)
        assert len(c.snippet) > 0


def test_retrieve_evidence_candidates_empty_claims():
    candidates = retrieve_evidence_candidates([])
    assert candidates == []


# ── Web Extraction & Error Handling ──────────────────────────────────────────

def test_fetch_article_text_success():
    sample_html = "<html><body><article><p>Crucial fact about climate policy.</p></article></body></html>"
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_html
        mock_get.return_value = mock_resp

        text = fetch_article_text("https://example.com/news")
        assert text is not None
        assert "Crucial fact" in text


def test_fetch_article_text_handles_network_error():
    with patch("requests.get", side_effect=Exception("Connection refused")):
        text = fetch_article_text("https://broken.domain.com")
        assert text is None


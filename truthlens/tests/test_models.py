"""
tests/test_models.py
--------------------
Unit tests for Pydantic v2 data models in core/models.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import (
    AnalysisRecord,
    AnalysisResult,
    CheckRequest,
    EvidenceItem,
    FeedbackRecord,
    StressResult,
)


# ── EvidenceItem tests ────────────────────────────────────────────────────────

def test_evidence_item_valid():
    item = EvidenceItem(
        title="Breaking News",
        source="reuters.com",
        url="https://reuters.com/news/123",
        snippet="Snippet text describing the event.",
        tier=1,
        stance="supports",
        reason="Corroborates date and event details.",
    )
    assert item.title == "Breaking News"
    assert item.tier == 1
    assert item.stance == "supports"
    assert item.published is None


def test_evidence_item_invalid_url():
    with pytest.raises(ValidationError, match="URL must start with http"):
        EvidenceItem(
            title="Title",
            source="src",
            url="ftp://example.com/file",
            snippet="Snippet",
            tier=1,
            stance="supports",
            reason="Reason",
        )


def test_evidence_item_invalid_tier():
    with pytest.raises(ValidationError):
        EvidenceItem(
            title="Title",
            source="src",
            url="https://example.com",
            snippet="Snippet",
            tier=4,  # Out of range 1-3
            stance="supports",
            reason="Reason",
        )


def test_evidence_item_invalid_stance():
    with pytest.raises(ValidationError):
        EvidenceItem(
            title="Title",
            source="src",
            url="https://example.com",
            snippet="Snippet",
            tier=2,
            stance="agree",  # Not in Literal['supports', 'contradicts', 'unrelated']
            reason="Reason",
        )


# ── StressResult tests ────────────────────────────────────────────────────────

def test_stress_result_valid():
    sr = StressResult(
        variants=["Claim variant 1", "Claim variant 2"],
        rejected=2,
        robustness=1.0,
        bias_flag=False,
    )
    assert sr.rejected == 2
    assert sr.robustness == 1.0


def test_stress_result_rejected_exceeds_variants():
    with pytest.raises(ValidationError, match="cannot exceed number of variants"):
        StressResult(
            variants=["Claim variant 1"],
            rejected=2,
            robustness=0.5,
            bias_flag=False,
        )


def test_stress_result_robustness_bounds():
    with pytest.raises(ValidationError):
        StressResult(
            variants=["A"],
            rejected=1,
            robustness=1.5,  # > 1.0
            bias_flag=False,
        )


# ── AnalysisResult tests ──────────────────────────────────────────────────────

def test_analysis_result_valid_deep():
    ar = AnalysisResult(
        verdict="likely_authentic",
        credibility=85,
        confidence="high",
        clickbait=0.15,
        incongruity=0.05,
        image_risk=None,
        evidence=[
            EvidenceItem(
                title="T",
                source="s",
                url="https://s.com",
                snippet="snip",
                tier=1,
                stance="supports",
                reason="r",
            )
        ],
        stress=StressResult(
            variants=["V1"],
            rejected=1,
            robustness=1.0,
            bias_flag=False,
        ),
        rationale="Solid consensus across sources.",
        mode="deep",
    )
    assert ar.credibility == 85
    assert len(ar.relevant_evidence()) == 1
    assert len(ar.supporting_evidence()) == 1


def test_analysis_result_insufficient_requires_none_credibility():
    # Valid insufficient
    ar = AnalysisResult(
        verdict="insufficient",
        credibility=None,
        confidence="low",
        clickbait=0.2,
        incongruity=0.1,
        image_risk=None,
        evidence=[],
        rationale="Not enough independent sources found.",
        mode="fast",
    )
    assert ar.credibility is None

    # Invalid: insufficient with credibility score set
    with pytest.raises(ValidationError, match="credibility must be None"):
        AnalysisResult(
            verdict="insufficient",
            credibility=50,
            confidence="low",
            clickbait=0.2,
            incongruity=0.1,
            rationale="Rationale",
            mode="fast",
        )


def test_analysis_result_non_insufficient_requires_credibility():
    with pytest.raises(ValidationError, match="credibility must be set"):
        AnalysisResult(
            verdict="likely_false",
            credibility=None,
            confidence="medium",
            clickbait=0.5,
            incongruity=0.5,
            rationale="Contradicted by major wires.",
            mode="fast",
        )


def test_analysis_result_stress_not_allowed_in_fast():
    with pytest.raises(ValidationError, match="stress result must be None in fast mode"):
        AnalysisResult(
            verdict="likely_authentic",
            credibility=80,
            confidence="medium",
            clickbait=0.1,
            incongruity=0.1,
            rationale="Good sources.",
            mode="fast",
            stress=StressResult(variants=["V1"], rejected=1, robustness=1.0),
        )


# ── CheckRequest tests ────────────────────────────────────────────────────────

def test_check_request_with_headline():
    req = CheckRequest(headline="Major policy announced by ministry", mode="fast")
    assert req.headline is not None
    assert len(req.input_hash()) == 64  # sha256 hex length


def test_check_request_with_url():
    req = CheckRequest(url="https://news.example.com/article/1", mode="deep")
    assert req.url is not None
    assert len(req.input_hash()) == 64


def test_check_request_missing_both_headline_and_url():
    with pytest.raises(ValidationError, match="Provide at least a headline or a URL"):
        CheckRequest(body="Some body text without headline or URL")


def test_check_request_invalid_url():
    with pytest.raises(ValidationError, match="URL must start with http"):
        CheckRequest(url="ftp://example.com/not-http")


def test_check_request_image_size_limit():
    large_image = b"0" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ValidationError, match="maximum allowed is 5 MB"):
        CheckRequest(headline="Headline", image_bytes=large_image)


def test_check_request_hash_stability():
    req1 = CheckRequest(headline="Breaking story", body="Body content", mode="fast")
    req2 = CheckRequest(headline="Breaking story", body="Body content", mode="fast")
    assert req1.input_hash() == req2.input_hash()

    # Mode change alters hash
    req_deep = CheckRequest(headline="Breaking story", body="Body content", mode="deep")
    assert req1.input_hash() != req_deep.input_hash()


# ── AnalysisRecord roundtrip ──────────────────────────────────────────────────

def test_analysis_record_roundtrip():
    req = CheckRequest(headline="Test headline", mode="fast")
    res = AnalysisResult(
        verdict="likely_authentic",
        credibility=90,
        confidence="high",
        clickbait=0.1,
        incongruity=0.0,
        rationale="Verified.",
        mode="fast",
    )
    record = AnalysisRecord.from_request_and_result(req, res)
    assert record.headline == "Test headline"
    assert record.input_hash == req.input_hash()

    deserialized = record.to_result()
    assert deserialized.verdict == "likely_authentic"
    assert deserialized.credibility == 90
    assert deserialized.confidence == "high"


# ── FeedbackRecord tests ──────────────────────────────────────────────────────

def test_feedback_record_valid():
    fb = FeedbackRecord(analysis_id=1, rating="right", note="Accurate verdict")
    assert fb.rating == "right"
    assert fb.analysis_id == 1


def test_feedback_record_invalid_rating():
    with pytest.raises(ValidationError):
        FeedbackRecord(analysis_id=1, rating="maybe")


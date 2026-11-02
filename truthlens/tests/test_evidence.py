"""
tests/test_evidence.py
----------------------
Unit tests for evidence stance judgement and anti-hallucination defenses (core/evidence.py).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.evidence import judge_evidence
from core.models import (
    BatchedEvidenceJudgement,
    EvidenceCandidate,
    EvidenceItem,
    ExtractedClaim,
    SourceStanceJudgement,
)


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    """Enable MOCK mode for evidence tests."""
    monkeypatch.setenv("MOCK", "1")


def test_judge_evidence_empty_candidates():
    claims = [
        ExtractedClaim(
            claim_text="Mars rover discovers ancient riverbed",
            claim_type="event",
            entities=["NASA", "Mars"],
        )
    ]
    items, summary = judge_evidence(claims=claims, candidates=[])
    assert items == []
    assert "Insufficient evidence" in summary


def test_judge_evidence_mock_mode_success():
    claims = [
        ExtractedClaim(
            claim_text="ISRO launched weather satellite",
            claim_type="event",
            entities=["ISRO"],
        )
    ]
    candidates = [
        EvidenceCandidate(
            title="Satellite successfully deployed in orbit",
            source="Reuters",
            url="https://reuters.com/article/1",
            snippet="Ground controllers confirm satellite is operational in orbit.",
            tier=1,
        ),
        EvidenceCandidate(
            title="Space agency reports mission success",
            source="AP News",
            url="https://apnews.com/article/2",
            snippet="Engineers confirmed launch and separation milestones were achieved.",
            tier=1,
        ),
    ]

    items, summary = judge_evidence(claims, candidates)

    assert isinstance(items, list)
    assert len(items) == 2
    assert len(summary) > 0

    for item in items:
        assert isinstance(item, EvidenceItem)
        assert item.stance in ["supports", "contradicts", "unrelated"]
        assert len(item.reason) > 0
        assert item.url in ["https://reuters.com/article/1", "https://apnews.com/article/2"]


def test_judge_evidence_anti_hallucination_defense(monkeypatch):
    """Verify that any URL returned by the LLM that was not in the retrieved set is dropped."""
    claims = [
        ExtractedClaim(
            claim_text="Test claim",
            claim_type="event",
            entities=["Entity"],
        )
    ]
    candidates = [
        EvidenceCandidate(
            title="Legitimate Article",
            source="BBC",
            url="https://bbc.com/legitimate-article",
            snippet="Legitimate snippet content.",
            tier=1,
        )
    ]

    # Model returns 1 legitimate URL and 1 hallucinated/invented URL
    fake_judgement = BatchedEvidenceJudgement(
        judgements=[
            SourceStanceJudgement(
                url="https://bbc.com/legitimate-article",
                stance="supports",
                reason="Corroborates event.",
            ),
            SourceStanceJudgement(
                url="https://hallucinated-invented-source.com/fake-news",
                stance="supports",
                reason="Invented hallucinated citation.",
            ),
        ],
        evidence_summary="Summary with hallucinated source.",
    )

    with patch("core.evidence.generate_structured", return_value=fake_judgement):
        items, summary = judge_evidence(claims, candidates)

        # Hallucinated URL must be dropped!
        urls_in_items = [i.url for i in items]
        assert "https://bbc.com/legitimate-article" in urls_in_items
        assert "https://hallucinated-invented-source.com/fake-news" not in urls_in_items
        assert len(items) == 1
        assert items[0].title == "Legitimate Article"


def test_judge_evidence_unjudged_candidate_preserved(monkeypatch):
    """Verify that if Gemini omits one candidate from its response, it is still preserved."""
    claims = [ExtractedClaim(claim_text="Claim", claim_type="event")]
    candidates = [
        EvidenceCandidate(
            title="Article 1",
            source="Source 1",
            url="https://s1.com/1",
            snippet="Snip 1",
            tier=1,
        ),
        EvidenceCandidate(
            title="Article 2",
            source="Source 2",
            url="https://s2.com/2",
            snippet="Snip 2",
            tier=2,
        ),
    ]

    # Model only judged Article 1
    partial_judgement = BatchedEvidenceJudgement(
        judgements=[
            SourceStanceJudgement(
                url="https://s1.com/1",
                stance="supports",
                reason="Confirmed.",
            )
        ],
        evidence_summary="Partial evaluation.",
    )

    with patch("core.evidence.generate_structured", return_value=partial_judgement):
        items, summary = judge_evidence(claims, candidates)

        assert len(items) == 2
        # First was judged
        assert items[0].url == "https://s1.com/1"
        assert items[0].stance == "supports"
        # Second was preserved with fallback
        assert items[1].url == "https://s2.com/2"
        assert items[1].stance == "unrelated"


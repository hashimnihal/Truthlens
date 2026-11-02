"""
tests/test_scoring.py
---------------------
Unit tests for credibility scoring, tier weighting, weight rescaling,
verdict thresholds, and confidence rules (core/scoring.py).

PRD Reference: §7.2
"""

from __future__ import annotations

import pytest

from core.models import AnalysisResult, EvidenceItem, StressResult
from core.scoring import (
    assign_verdict,
    compute_confidence,
    compute_credibility,
    compute_evidence_support,
    score_and_assemble_result,
)


def _make_evidence_item(
    source: str,
    url: str,
    stance: str,
    tier: int,
) -> EvidenceItem:
    return EvidenceItem(
        title="Sample News Title",
        source=source,
        url=url,
        snippet="Sample article snippet describing facts.",
        tier=tier,
        stance=stance,  # type: ignore[arg-type]
        reason="Stance explanation.",
    )


# ── Evidence Support Calculation & Tier Weighting (PRD §7.2) ──────────────────

def test_evidence_support_all_supporting():
    evidence = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", tier=1),
        _make_evidence_item("AP News", "https://apnews.com/2", "supports", tier=1),
    ]
    e_score, relevant = compute_evidence_support(evidence)
    assert len(relevant) == 2
    assert e_score == 1.0


def test_evidence_support_all_contradicting():
    evidence = [
        _make_evidence_item("BBC", "https://bbc.com/1", "contradicts", tier=1),
        _make_evidence_item("The Hindu", "https://thehindu.com/2", "contradicts", tier=1),
    ]
    e_score, relevant = compute_evidence_support(evidence)
    assert len(relevant) == 2
    assert e_score == 0.0


def test_evidence_support_balanced():
    evidence = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", tier=1),
        _make_evidence_item("AP News", "https://apnews.com/2", "contradicts", tier=1),
    ]
    e_score, relevant = compute_evidence_support(evidence)
    assert len(relevant) == 2
    assert e_score == 0.5


def test_evidence_support_tier_weighting():
    # Tier 1 (1.0) supports vs Tier 3 (0.4) contradicts
    # net = +1.0 * 1.0 + (-1.0 * 0.4) = +0.6
    # total_weight = 1.0 + 0.4 = 1.4
    # E = (0.6 + 1.4) / (2 * 1.4) = 2.0 / 2.8 ≈ 0.714
    evidence = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", tier=1),
        _make_evidence_item("Blog", "https://randomblog.com/2", "contradicts", tier=3),
    ]
    e_score, _ = compute_evidence_support(evidence)
    assert pytest.approx(e_score, abs=1e-3) == (2.0 / 2.8)


def test_evidence_support_unrelated_sources_do_not_inflate():
    evidence_with_unrelated = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", tier=1),
        _make_evidence_item("AP News", "https://apnews.com/2", "contradicts", tier=1),
        _make_evidence_item("Other", "https://other.com/3", "unrelated", tier=1),
        _make_evidence_item("Other2", "https://other2.com/4", "unrelated", tier=2),
    ]
    e_score, relevant = compute_evidence_support(evidence_with_unrelated)
    assert len(relevant) == 2
    assert e_score == 0.5


# ── Credibility Formula & Rescaling Rules (PRD §7.2) ──────────────────────────

def test_credibility_full_formula():
    # credibility = 100 * (0.45*E + 0.20*(1-H) + 0.15*(1-V) + 0.10*(1-C) + 0.10*R)
    # E=1.0, H=0.0, V=0.0, C=0.0, R=1.0 -> 100 * (0.45 + 0.20 + 0.15 + 0.10 + 0.10) = 100
    score = compute_credibility(
        e_support=1.0,
        incongruity=0.0,
        clickbait=0.0,
        image_risk=0.0,
        robustness=1.0,
    )
    assert score == 100

    # Opposite extreme -> 0
    score_zero = compute_credibility(
        e_support=0.0,
        incongruity=1.0,
        clickbait=1.0,
        image_risk=1.0,
        robustness=0.0,
    )
    assert score_zero == 0


def test_credibility_no_image_rescaling():
    # Dropping V: remaining weights are E(0.45), H(0.20), C(0.10), R(0.10) -> sum = 0.85
    # If all terms are 1.0 -> 100 * (0.85 / 0.85) = 100
    score = compute_credibility(
        e_support=1.0,
        incongruity=0.0,
        clickbait=0.0,
        image_risk=None,  # No image
        robustness=1.0,
    )
    assert score == 100


def test_credibility_fast_mode_rescaling():
    # Fast mode drops R: remaining weights are E(0.45), H(0.20), V(0.15), C(0.10) -> sum = 0.90
    score = compute_credibility(
        e_support=1.0,
        incongruity=0.0,
        clickbait=0.0,
        image_risk=0.0,
        robustness=None,  # Fast mode
    )
    assert score == 100


def test_credibility_both_missing_rescaling():
    # Both V and R missing: active weights are E(0.45), H(0.20), C(0.10) -> sum = 0.75
    # w'_E = 0.45/0.75 = 0.60
    # w'_H = 0.20/0.75 ≈ 0.2667
    # w'_C = 0.10/0.75 ≈ 0.1333
    # If E=0.8, H=0.1, C=0.2:
    # raw = 100 * (0.60*0.8 + (0.20/0.75)*0.9 + (0.10/0.75)*0.8)
    # raw = 100 * (0.48 + 0.24 + 0.10667) = 82.667 -> round = 83
    score = compute_credibility(
        e_support=0.8,
        incongruity=0.1,
        clickbait=0.2,
        image_risk=None,
        robustness=None,
    )
    assert score == 83


# ── Verdict Thresholds (PRD §7.2) ─────────────────────────────────────────────

def test_verdict_thresholds():
    assert assign_verdict(credibility_score=85, relevant_source_count=3) == "likely_authentic"
    assert assign_verdict(credibility_score=70, relevant_source_count=2) == "likely_authentic"
    assert assign_verdict(credibility_score=69, relevant_source_count=2) == "mixed"
    assert assign_verdict(credibility_score=40, relevant_source_count=2) == "mixed"
    assert assign_verdict(credibility_score=39, relevant_source_count=2) == "likely_false"
    assert assign_verdict(credibility_score=15, relevant_source_count=2) == "likely_false"


def test_verdict_insufficient_evidence():
    # Fewer than 2 relevant sources forces 'insufficient'
    assert assign_verdict(credibility_score=95, relevant_source_count=1) == "insufficient"
    assert assign_verdict(credibility_score=95, relevant_source_count=0) == "insufficient"
    assert assign_verdict(credibility_score=None, relevant_source_count=0) == "insufficient"


# ── Confidence Rules & Bias Flag Downgrade (PRD §7.2) ──────────────────────────

def test_confidence_high():
    # 3 sources from 2 distinct domains that agree, no bias flag
    sources = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", 1),
        _make_evidence_item("AP News", "https://apnews.com/2", "supports", 1),
        _make_evidence_item("BBC News", "https://bbc.com/3", "supports", 1),
    ]
    conf = compute_confidence(sources, bias_flag=False)
    assert conf == "high"


def test_confidence_medium_on_two_sources():
    sources = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", 1),
        _make_evidence_item("AP News", "https://apnews.com/2", "supports", 1),
    ]
    conf = compute_confidence(sources, bias_flag=False)
    assert conf == "medium"


def test_confidence_low_on_insufficient_sources():
    sources = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", 1)
    ]
    conf = compute_confidence(sources, bias_flag=False)
    assert conf == "low"


def test_confidence_bias_flag_downgrade():
    # High confidence sources
    sources_high = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", 1),
        _make_evidence_item("AP News", "https://apnews.com/2", "supports", 1),
        _make_evidence_item("BBC News", "https://bbc.com/3", "supports", 1),
    ]
    # Bias flag drops High -> Medium
    assert compute_confidence(sources_high, bias_flag=True) == "medium"

    # Medium confidence sources
    sources_med = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", 1),
        _make_evidence_item("AP News", "https://apnews.com/2", "supports", 1),
    ]
    # Bias flag drops Medium -> Low
    assert compute_confidence(sources_med, bias_flag=True) == "low"


# ── Full Result Assembly Integration ──────────────────────────────────────────

def test_score_and_assemble_result_authentic():
    evidence = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", tier=1),
        _make_evidence_item("AP News", "https://apnews.com/2", "supports", tier=1),
        _make_evidence_item("BBC", "https://bbc.com/3", "supports", tier=1),
    ]
    res = score_and_assemble_result(
        evidence=evidence,
        incongruity=0.05,
        clickbait=0.10,
        image_risk=None,
        stress=None,
        mode="fast",
    )
    assert isinstance(res, AnalysisResult)
    assert res.verdict == "likely_authentic"
    assert res.credibility is not None
    assert res.credibility >= 70
    assert res.confidence == "high"
    assert res.stress is None
    assert len(res.rationale) > 20


def test_score_and_assemble_result_insufficient():
    # Only 1 relevant source -> verdict must be 'insufficient', credibility must be None
    evidence = [
        _make_evidence_item("Reuters", "https://reuters.com/1", "supports", tier=1),
        _make_evidence_item("Other", "https://other.com/2", "unrelated", tier=3),
    ]
    res = score_and_assemble_result(
        evidence=evidence,
        incongruity=0.2,
        clickbait=0.2,
        mode="fast",
    )
    assert isinstance(res, AnalysisResult)
    assert res.verdict == "insufficient"
    assert res.credibility is None
    assert "Insufficient" in res.rationale


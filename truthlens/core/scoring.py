"""
core/scoring.py
---------------
Credibility scoring, verdict assignment, and confidence evaluation.

PRD reference: §7.2 (Scoring and verdict rules)
Formula:
    credibility = 100 * (0.45*E + 0.20*(1 - H) + 0.15*(1 - V) + 0.10*(1 - C) + 0.10*R)

Rescaling rules:
    - No image provided: Drop V term, rescale remaining weights to sum to 1.0.
    - Fast mode: Drop R term, rescale remaining weights to sum to 1.0.
    - Both missing: Drop both V and R, rescale E, H, C proportionally.

Verdict thresholds:
    - Fewer than 2 relevant sources -> 'insufficient' (credibility is None)
    - Score >= 70 -> 'likely_authentic'
    - Score 40 to 69 -> 'mixed'
    - Score < 40 -> 'likely_false'

Confidence:
    - High: >= 3 relevant sources from >= 2 distinct domains agree, no bias flag
    - Medium: >= 2 relevant sources found
    - Low: Otherwise
    - If bias_flag is True: Lower confidence by one level
"""

from __future__ import annotations

import logging
from typing import Literal
from urllib.parse import urlparse

from core.models import AnalysisResult, EvidenceItem, StressResult

logger = logging.getLogger(__name__)

# Base weights from PRD §7.2
BASE_WEIGHT_E = 0.45  # Evidence support
BASE_WEIGHT_H = 0.20  # Headline-body consistency (1 - H)
BASE_WEIGHT_V = 0.15  # Visual fit/manipulation risk (1 - V)
BASE_WEIGHT_C = 0.10  # Clickbait penalty (1 - C)
BASE_WEIGHT_R = 0.10  # Stress-test robustness (R)

# Tier weighting values (PRD §7.2)
TIER_WEIGHTS = {
    1: 1.0,  # Major wire / top-tier broadcaster
    2: 0.7,  # Known regional / specialist outlet
    3: 0.4,  # Unknown / unclassified source
}


# ── Domain & Evidence Helper Functions ────────────────────────────────────────

def _extract_source_domain(item: EvidenceItem) -> str:
    """Extract registered domain name from evidence URL for diversity checks."""
    try:
        netloc = urlparse(item.url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return item.source.lower().strip()


def compute_evidence_support(evidence: list[EvidenceItem]) -> tuple[float, list[EvidenceItem]]:
    """Calculate tier-weighted evidence support score E in [0.0, 1.0].

    Rules (PRD §7.2):
      - Stances: 'supports' = +1, 'contradicts' = -1, 'unrelated' = 0
      - Tiers: Tier 1 = 1.0, Tier 2 = 0.7, Tier 3 = 0.4
      - Scaled to [0.0, 1.0] based on relevant sources
      - Unrelated sources do not count toward relevant weight or distort support

    Returns:
      (E: float, relevant_sources: list[EvidenceItem])
    """
    relevant = [e for e in evidence if e.stance in ("supports", "contradicts")]
    if not relevant:
        # Default neutral midpoint when no relevant sources are present
        return 0.5, []

    net_weighted_score = 0.0
    total_relevant_weight = 0.0

    for item in relevant:
        tier_weight = TIER_WEIGHTS.get(item.tier, 0.4)
        stance_val = 1.0 if item.stance == "supports" else -1.0

        net_weighted_score += stance_val * tier_weight
        total_relevant_weight += tier_weight

    if total_relevant_weight == 0.0:
        return 0.5, relevant

    # Normalization formula mapping [-total_weight, +total_weight] -> [0.0, 1.0]:
    # E = (net_weighted_score + total_relevant_weight) / (2 * total_relevant_weight)
    e_score = (net_weighted_score + total_relevant_weight) / (2.0 * total_relevant_weight)
    e_score = min(max(e_score, 0.0), 1.0)

    return e_score, relevant


# ── Confidence Evaluation (PRD §7.2) ──────────────────────────────────────────

def compute_confidence(
    relevant_sources: list[EvidenceItem],
    bias_flag: bool = False,
) -> Literal["high", "medium", "low"]:
    """Determine confidence level based on source volume, domain diversity, and bias flag.

    Rules:
      - High: >= 3 relevant sources from >= 2 distinct domains agree, no bias flag
      - Medium: >= 2 relevant sources found
      - Low: Otherwise
      - If bias_flag is True: Lower confidence by one level (High -> Medium, Medium -> Low)
    """
    count = len(relevant_sources)
    if count < 2:
        return "low"

    # Distinct domains among relevant sources
    domains = {_extract_source_domain(s) for s in relevant_sources}

    # Check if sources predominantly agree (all supporting or all contradicting)
    stances = {s.stance for s in relevant_sources}
    unanimous_agreement = len(stances) == 1

    # Determine base confidence from source count, domain diversity, and agreement
    base_confidence: Literal["high", "medium", "low"]
    if count >= 3 and len(domains) >= 2 and unanimous_agreement:
        base_confidence = "high"
    elif count >= 2:
        base_confidence = "medium"
    else:
        base_confidence = "low"

    # Apply bias flag penalty (PRD §7.2: "If an altered false claim is supported, set the bias flag and lower confidence one level")
    if bias_flag:
        if base_confidence == "high":
            return "medium"
        elif base_confidence == "medium":
            return "low"
        return "low"

    return base_confidence


# ── Credibility & Verdict Assignment (PRD §7.2) ───────────────────────────────

def compute_credibility(
    e_support: float,
    incongruity: float,
    clickbait: float,
    image_risk: float | None = None,
    robustness: float | None = None,
) -> int:
    """Compute 0–100 credibility score with dynamic proportional weight rescaling.

    Base weights:
      E = 0.45, (1-H) = 0.20, (1-V) = 0.15, (1-C) = 0.10, R = 0.10

    Rescaling rules:
      - If image_risk is None: Drop V term and rescale remaining weights to sum to 1.
      - If robustness is None: Drop R term and rescale remaining weights to sum to 1.
    """
    # Clamp all inputs to [0.0, 1.0]
    e_val = min(max(float(e_support), 0.0), 1.0)
    h_term = min(max(1.0 - float(incongruity), 0.0), 1.0)
    c_term = min(max(1.0 - float(clickbait), 0.0), 1.0)

    # Active weight dictionary
    active_weights: dict[str, tuple[float, float]] = {
        "E": (BASE_WEIGHT_E, e_val),
        "H": (BASE_WEIGHT_H, h_term),
        "C": (BASE_WEIGHT_C, c_term),
    }

    if image_risk is not None:
        v_term = min(max(1.0 - float(image_risk), 0.0), 1.0)
        active_weights["V"] = (BASE_WEIGHT_V, v_term)

    if robustness is not None:
        r_term = min(max(float(robustness), 0.0), 1.0)
        active_weights["R"] = (BASE_WEIGHT_R, r_term)

    total_weight = sum(w for w, _ in active_weights.values())

    # Proportional weighted sum
    raw_score = sum((w / total_weight) * val for w, val in active_weights.values()) * 100.0

    return int(round(min(max(raw_score, 0.0), 100.0)))


def assign_verdict(
    credibility_score: int | None,
    relevant_source_count: int,
) -> Literal["likely_authentic", "mixed", "likely_false", "insufficient"]:
    """Determine authenticity verdict from relevant source volume and credibility score.

    Thresholds (PRD §7.2):
      - Fewer than 2 relevant sources -> 'insufficient'
      - Score >= 70 -> 'likely_authentic'
      - Score 40 to 69 -> 'mixed'
      - Score < 40 -> 'likely_false'
    """
    if relevant_source_count < 2 or credibility_score is None:
        return "insufficient"

    if credibility_score >= 70:
        return "likely_authentic"
    elif credibility_score >= 40:
        return "mixed"
    else:
        return "likely_false"


# ── Rationale Generator (PRD §9.3) ────────────────────────────────────────────

def generate_rationale(
    verdict: Literal["likely_authentic", "mixed", "likely_false", "insufficient"],
    credibility: int | None,
    confidence: Literal["high", "medium", "low"],
    relevant_sources: list[EvidenceItem],
    clickbait: float,
    incongruity: float,
    evidence_summary: str = "",
) -> str:
    """Synthesize a 3-5 sentence plain-English rationale for the verdict card."""
    if verdict == "insufficient":
        return (
            f"Insufficient independent evidence could be retrieved to substantiate this story. "
            f"Only {len(relevant_sources)} directly relevant source(s) were found, which falls below "
            f"the required threshold of 2 verified sources. We recommend cross-checking with official "
            f"statements or trying again as more reporting becomes available."
        )

    sentences: list[str] = []

    # 1. Verdict & score summary
    verdict_labels = {
        "likely_authentic": "likely authentic",
        "mixed": "unverified or mixed",
        "likely_false": "likely false or misleading",
    }
    label = verdict_labels.get(verdict, verdict)
    sentences.append(
        f"This check received an evidence credibility score of {credibility}/100 and a verdict of {label} "
        f"with {confidence} confidence."
    )

    # 2. Evidence corroboration statement
    supports = sum(1 for s in relevant_sources if s.stance == "supports")
    contradicts = sum(1 for s in relevant_sources if s.stance == "contradicts")
    if supports > 0 and contradicts == 0:
        sentences.append(
            f"Live news search identified {supports} reputable source(s) supporting the key claims."
        )
    elif contradicts > 0 and supports == 0:
        sentences.append(
            f"Live news search identified {contradicts} reputable source(s) that directly refute or contradict this story."
        )
    else:
        sentences.append(
            f"Reporting across sources is conflicting, with {supports} supporting and {contradicts} contradicting source(s)."
        )

    # 3. Text & clickbait characteristics
    if clickbait >= 0.6:
        sentences.append(
            f"The headline exhibits strong clickbait signals (intensity: {clickbait:.2f}), "
            f"utilizing sensationalized wording to evoke curiosity."
        )
    if incongruity >= 0.5:
        sentences.append(
            f"Significant headline-body incongruity ({incongruity:.2f}) was detected, indicating the body text does not "
            f"fully deliver on the headline's claims."
        )

    # 4. Summary closure
    if evidence_summary and len(sentences) < 4:
        sentences.append(evidence_summary)

    sentences.append("Assessments are evidence-based and do not constitute a definitive ruling of absolute truth.")

    return " ".join(sentences)


# ── Full Assembly Function ────────────────────────────────────────────────────

def score_and_assemble_result(
    evidence: list[EvidenceItem],
    incongruity: float,
    clickbait: float,
    image_risk: float | None = None,
    stress: StressResult | None = None,
    mode: Literal["fast", "deep"] = "fast",
    evidence_summary: str = "",
) -> AnalysisResult:
    """End-to-end scoring pipeline that creates a validated AnalysisResult.

    Integrates evidence weighting, weight rescaling, confidence, verdict, and rationale.
    """
    # 1. Evidence calculation
    e_support, relevant = compute_evidence_support(evidence)
    relevant_count = len(relevant)

    # 2. Robustness extraction
    robustness = stress.robustness if (stress is not None and mode == "deep") else None
    bias_flag = stress.bias_flag if (stress is not None and mode == "deep") else False

    # 3. Credibility calculation
    if relevant_count < 2:
        credibility: int | None = None
        verdict = "insufficient"
    else:
        credibility = compute_credibility(
            e_support=e_support,
            incongruity=incongruity,
            clickbait=clickbait,
            image_risk=image_risk,
            robustness=robustness,
        )
        verdict = assign_verdict(credibility, relevant_count)

    # 4. Confidence calculation
    confidence = compute_confidence(relevant, bias_flag=bias_flag)

    # 5. Rationale synthesis
    rationale = generate_rationale(
        verdict=verdict,
        credibility=credibility,
        confidence=confidence,
        relevant_sources=relevant,
        clickbait=clickbait,
        incongruity=incongruity,
        evidence_summary=evidence_summary,
    )

    return AnalysisResult(
        verdict=verdict,
        credibility=credibility,
        confidence=confidence,
        clickbait=round(min(max(clickbait, 0.0), 1.0), 4),
        incongruity=round(min(max(incongruity, 0.0), 1.0), 4),
        image_risk=round(image_risk, 4) if image_risk is not None else None,
        evidence=evidence,
        stress=stress if mode == "deep" else None,
        rationale=rationale,
        mode=mode,
    )

"""
core/evidence.py
----------------
Evidence stance judgement using a single batched Gemini call.

PRD reference: §6 (F8), §7.1 (Evidence judgement), §15 (hallucination defense)
Features:
  - Batched evaluation: Gemini labels all retrieved sources in a single prompt
  - Label stances: 'supports', 'contradicts', or 'unrelated', with a one-line rationale
  - Anti-hallucination defense: Strictly verifies URLs returned by Gemini against
    the retrieved set; drops any URL that was not present in the search candidates
  - Generates an evidence synthesis summary for the final rationale
"""

from __future__ import annotations

import logging
from typing import Sequence

from core.gemini_client import generate_structured
from core.models import (
    BatchedEvidenceJudgement,
    EvidenceCandidate,
    EvidenceItem,
    ExtractedClaim,
)

logger = logging.getLogger(__name__)

EVIDENCE_SYSTEM_INSTRUCTION = """\
You are an objective evidentiary investigator for TruthLens fact-checking.
Your task is to judge whether news sources support, contradict, or are unrelated to specific claims.

CRITICAL RULES:
1. Base your judgement EXCLUSIVELY on the provided article titles, snippets, and context.
2. DO NOT recall outside knowledge or assume unstated facts.
3. If an article snippet does not contain sufficient direct evidence to confirm or refute the claim, mark it as 'unrelated'.
4. Label each source as one of:
   - 'supports': The article provides clear evidence verifying the claim.
   - 'contradicts': The article directly refutes, debunks, or contradicts the claim.
   - 'unrelated': The article discusses general background, different topics, or lacks direct evidence.
5. Provide a crisp, single-sentence reason explaining why the source supports, contradicts, or is unrelated.
6. MANDATORY ANTI-HALLUCINATION REQUIREMENT:
   You must ONLY cite and return URLs that exist verbatim in the provided list. Never invent or hallucinate URLs.
"""


def _construct_evidence_prompt(
    claims: list[ExtractedClaim],
    candidates: list[EvidenceCandidate],
) -> str:
    """Format prompt with claims and retrieved source snippets for batched evaluation."""
    claims_text = "\n".join(
        f"- [{c.claim_type.upper()}] {c.claim_text} (Entities: {', '.join(c.entities) if c.entities else 'None'})"
        for c in claims
    )

    sources_text_list = []
    for i, item in enumerate(candidates, start=1):
        sources_text_list.append(
            f"Source #{i}:\n"
            f"  Title: {item.title}\n"
            f"  Outlet: {item.source} (Trust Tier: {item.tier})\n"
            f"  URL: {item.url}\n"
            f"  Snippet: {item.snippet}\n"
        )
    sources_text = "\n".join(sources_text_list)

    prompt = (
        f"Claims to verify:\n"
        f"{claims_text}\n\n"
        f"Retrieved Evidence Sources ({len(candidates)} total):\n"
        f"{sources_text}\n\n"
        f"For each source above, evaluate its stance toward the claims ('supports', 'contradicts', 'unrelated'), "
        f"provide a 1-sentence reason, and write a 2-3 sentence evidence consensus summary. "
        f"Remember to cite ONLY the exact URLs listed above."
    )
    return prompt


def judge_evidence(
    claims: list[ExtractedClaim],
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceItem], str]:
    """Evaluate retrieved evidence candidates against claims.

    Args:
        claims: The extracted claims to fact-check.
        candidates: List of retrieved EvidenceCandidate objects.

    Returns:
        tuple of (list of EvidenceItem with stances and reasons, evidence_summary)
    """
    if not candidates:
        return (
            [],
            "Insufficient evidence: No relevant sources could be retrieved for the specified claims.",
        )

    # Build URL lookup for strict anti-hallucination verification
    url_to_candidate = {c.url.strip(): c for c in candidates}

    prompt = _construct_evidence_prompt(claims, candidates)

    batched_response: BatchedEvidenceJudgement = generate_structured(
        prompt=prompt,
        response_model=BatchedEvidenceJudgement,
        temperature=0.0,  # Zero temperature for strictly grounded evidentiary judgements
        system_instruction=EVIDENCE_SYSTEM_INSTRUCTION,
    )

    # Process and validate judgements against genuine candidate URLs
    evidence_items: list[EvidenceItem] = []
    judged_urls: set[str] = set()

    for item in batched_response.judgements:
        cited_url = item.url.strip()

        # PRD §7.1 & §15: Anti-hallucination guard
        # Any URL returned that was NOT retrieved must be dropped!
        if cited_url not in url_to_candidate:
            logger.warning(
                f"Dropping hallucinated URL returned by model that was not in retrieved candidates: {cited_url}"
            )
            continue

        candidate = url_to_candidate[cited_url]
        judged_urls.add(cited_url)

        evidence_items.append(
            EvidenceItem(
                title=candidate.title,
                source=candidate.source,
                url=candidate.url,
                published=candidate.published,
                snippet=candidate.snippet,
                tier=candidate.tier,
                stance=item.stance,
                reason=item.reason,
            )
        )

    # For any candidate not explicitly returned by Gemini, include with 'unrelated'
    for candidate in candidates:
        if candidate.url.strip() not in judged_urls:
            evidence_items.append(
                EvidenceItem(
                    title=candidate.title,
                    source=candidate.source,
                    url=candidate.url,
                    published=candidate.published,
                    snippet=candidate.snippet,
                    tier=candidate.tier,
                    stance="unrelated",
                    reason="Source did not contain sufficient direct evidence to substantiate or refute the claim.",
                )
            )

    summary = (
        batched_response.evidence_summary
        or "Retrieved sources were cross-checked against the core claims."
    )
    return evidence_items, summary

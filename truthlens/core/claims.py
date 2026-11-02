"""
core/claims.py
--------------
Claim extraction and text analysis using Gemini 2.5 Flash.

PRD reference: §6 (F5, F9, F10), §7.1 (Claim extraction & Query building), §15
Features:
  - Single Gemini call reads headline and optional article body.
  - Extracts up to 3 checkable factual claims (event, statistic, quote, other, opinion).
  - Rates clickbait intensity (0.0 to 1.0) based on sensationalism and curiosity gaps.
  - Rates headline-body incongruity (0.0 to 1.0) to spot bait-and-switch articles.
  - Generates retrieval queries per claim for live search (PRD §7.1).
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from core.gemini_client import generate_structured
from core.models import ClaimsAnalysis, ExtractedClaim

logger = logging.getLogger(__name__)

CLAIMS_SYSTEM_INSTRUCTION = """\
You are an expert news analyst and fact-checking investigator for TruthLens.
Your task is to analyze a news headline and optional body text with extreme precision.

Instructions:
1. Extract up to 3 specific, verifiable, factual claims from the text.
   - Categorize each claim as: 'event', 'statistic', 'quote', 'opinion', or 'other'.
   - Extract the key named entities (people, organizations, locations) relevant to each claim.
   - Ignore subjective opinions unless they are reported as verifiable statements.
2. Rate clickbait intensity (0.0 = completely objective/factual, 1.0 = extreme sensationalism, misleading curiosity gap, ALL-CAPS hyperbole).
3. Rate headline-body incongruity (0.0 = body perfectly aligns with and substantiates headline, 1.0 = headline directly contradicts or is completely absent from body). If body is not provided, evaluate headline self-containment (default 0.0).
4. Provide a brief 1-2 sentence rationale summarizing your assessment.
"""


def _construct_claim_extraction_prompt(headline: str, body: str | None = None) -> str:
    """Format prompt for claim extraction and text analysis."""
    prompt = f'Headline:\n"{headline.strip()}"\n\n'
    if body and body.strip():
        # Truncate extremely long body to maintain prompt efficiency
        body_snippet = body.strip()[:10_000]
        prompt += f"Article Body:\n{body_snippet}\n\n"
    else:
        prompt += "Article Body:\n[No article body provided - headline only]\n\n"

    prompt += (
        "Extract up to 3 checkable claims, rate clickbait (0.0 to 1.0), "
        "and rate headline-body incongruity (0.0 to 1.0)."
    )
    return prompt


def extract_claims(headline: str, body: str | None = None) -> ClaimsAnalysis:
    """Extract checkable claims, clickbait rating, and incongruity score.

    Args:
        headline: News headline text.
        body: Optional article text.

    Returns:
        ClaimsAnalysis Pydantic model with up to 3 claims, clickbait, and incongruity.
    """
    if not headline or not headline.strip():
        raise ValueError("Headline cannot be empty")

    prompt = _construct_claim_extraction_prompt(headline, body)

    analysis = generate_structured(
        prompt=prompt,
        response_model=ClaimsAnalysis,
        temperature=0.1,
        system_instruction=CLAIMS_SYSTEM_INSTRUCTION,
    )

    # Ensure claims list doesn't exceed 3 (PRD §7.1 requirement)
    if len(analysis.claims) > 3:
        analysis.claims = analysis.claims[:3]

    return analysis


def build_claim_queries(claim: ExtractedClaim) -> list[str]:
    """Generate up to 2 search queries per claim for live news retrieval (PRD §7.1).

    Query 1: Direct claim text (clean, concise).
    Query 2: Key entities combined with salient keywords.
    """
    queries: list[str] = []

    # Clean claim text of punctuation for query 1
    clean_claim = re.sub(r'["\']', "", claim.claim_text).strip()
    if clean_claim:
        # Cap query length for search engine ergonomics
        query_words = clean_claim.split()[:12]
        queries.append(" ".join(query_words))

    # Query 2: Entities combined
    if claim.entities:
        entity_query = " ".join(claim.entities[:4])
        if entity_query and entity_query not in queries:
            queries.append(entity_query)

    # Fallback if no entity query
    if len(queries) < 2 and clean_claim:
        queries.append(f"{clean_claim} news")

    return queries[:2]

"""
core/models.py
--------------
Pydantic v2 data contracts shared across the entire TruthLens engine.

These models are the single source of truth between the engine and the UI.
Every engine function that produces data returns one of these models.
The UI reads these models and nothing else.

PRD reference: Section 8 — Data models and storage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Claims & Text Analysis ────────────────────────────────────────────────────

class ClickbaitSignal(BaseModel):
    """A detected linguistic clickbait indicator for UI chips (PRD §6 F11)."""

    name: str = Field(..., description="Signal identifier e.g. 'all_caps', 'curiosity_gap'")
    label: str = Field(..., description="Human-readable chip text e.g. 'ALL CAPS', 'Curiosity Gap'")
    description: str = Field(..., description="Explanation of what triggered the signal")


class ClickbaitResult(BaseModel):
    """Complete clickbait assessment combining classical ML and Gemini (PRD §7.2)."""

    score: float = Field(..., ge=0.0, le=1.0, description="Blended clickbait score: 0.4*sklearn + 0.6*gemini")
    sklearn_prob: float = Field(..., ge=0.0, le=1.0, description="Classical TF-IDF baseline probability")
    gemini_rating: float = Field(..., ge=0.0, le=1.0, description="Gemini rating from text analysis")
    signals: list[ClickbaitSignal] = Field(
        default_factory=list, description="List of detected linguistic pattern signals"
    )


class ExtractedClaim(BaseModel):
    """A single factual claim extracted from news text."""

    claim_text: str = Field(..., min_length=3, description="A single, specific, checkable claim")
    claim_type: Literal["event", "statistic", "quote", "opinion", "other"] = Field(
        ..., description="The classification of the claim"
    )
    entities: list[str] = Field(
        default_factory=list, description="Key named entities (people, organizations, places)"
    )


class ClaimsAnalysis(BaseModel):
    """Result of Gemini text analysis on headline and article body.

    PRD §7.1:
      - up to 3 checkable claims
      - Gemini clickbait rating (0.0 to 1.0)
      - headline-body incongruity (0.0 to 1.0)
    """

    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 checkable claims extracted from the headline and body",
    )
    clickbait_rating: float = Field(
        ..., ge=0.0, le=1.0, description="Gemini rating of clickbait intensity (0=clean, 1=clickbait)"
    )
    incongruity: float = Field(
        ..., ge=0.0, le=1.0, description="Headline-body mismatch (0=consistent, 1=contradictory)"
    )
    rationale: str = Field(
        default="", description="Brief reasoning for claims, clickbait rating, and incongruity score"
    )


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceCandidate(BaseModel):
    """A retrieved news source before stance evaluation."""

    title: str = Field(..., min_length=1, description="Article headline")
    source: str = Field(..., min_length=1, description="Domain or outlet name")
    url: str = Field(..., min_length=1, description="Full article URL")
    published: str | None = Field(
        default=None, description="ISO-8601 publication date or None if unknown"
    )
    snippet: str = Field(..., min_length=1, description="Excerpt or lead text from article")
    tier: int = Field(default=3, ge=1, le=3, description="Trust tier (1=major, 2=known, 3=unknown)")

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class SourceStanceJudgement(BaseModel):
    """Gemini stance judgement for a single cited URL."""

    url: str = Field(..., description="The exact URL of the evaluated source")
    stance: Literal["supports", "contradicts", "unrelated"] = Field(
        ..., description="Stance of the source toward the claim(s)"
    )
    reason: str = Field(..., min_length=1, description="One-line explanation of the stance")


class BatchedEvidenceJudgement(BaseModel):
    """Structured response from Gemini batched evidence evaluation."""

    judgements: list[SourceStanceJudgement] = Field(
        default_factory=list, description="List of source judgements citing provided URLs"
    )
    evidence_summary: str = Field(
        default="", description="1-3 sentence summary of overall evidence consensus"
    )


class EvidenceItem(BaseModel):
    """A single retrieved news source with its stance toward the claim.

    Tier values:
        1 — major wire service or top-tier outlet
        2 — known regional or specialist outlet
        3 — unknown or unclassified domain
    """

    title: str = Field(..., min_length=1, description="Article headline")
    source: str = Field(..., min_length=1, description="Domain or outlet name")
    url: str = Field(..., min_length=1, description="Full article URL")
    published: str | None = Field(
        default=None, description="ISO-8601 publication date or None if unknown"
    )
    snippet: str = Field(..., min_length=1, description="Short excerpt from the article")
    tier: int = Field(..., ge=1, le=3, description="Source trust tier (1=major, 2=known, 3=unknown)")
    stance: Literal["supports", "contradicts", "unrelated"] = Field(
        ..., description="How this source relates to the main claim"
    )
    reason: str = Field(..., min_length=1, description="One-line explanation of the stance")

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        """URLs must use http or https scheme (PRD §13 security rule)."""
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator("tier")
    @classmethod
    def tier_in_range(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError("tier must be 1, 2, or 3")
        return v


# ── Stress test ───────────────────────────────────────────────────────────────

class StressResult(BaseModel):
    """Result of the counterfactual stress test (Deep mode only).

    robustness = share of altered claims that the evidence correctly rejects.
    bias_flag  = True when the evidence supported an obviously false altered claim.
    """

    variants: list[str] = Field(
        ..., description="The altered claim strings generated by Gemini"
    )
    rejected: int = Field(
        ..., ge=0, description="Number of altered claims the evidence rejected"
    )
    robustness: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of variants correctly rejected (0–1)"
    )
    bias_flag: bool = Field(
        default=False,
        description="True when a false variant was supported — indicates possible bias",
    )

    @model_validator(mode="after")
    def rejected_le_variants(self) -> "StressResult":
        if self.rejected > len(self.variants):
            raise ValueError(
                f"rejected ({self.rejected}) cannot exceed number of variants "
                f"({len(self.variants)})"
            )
        return self


# ── Main analysis result ───────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """The complete output of one credibility check.

    credibility is None only when verdict == 'insufficient'.
    image_risk  is None when no image was provided.
    stress      is None in Fast mode.
    """

    verdict: Literal["likely_authentic", "mixed", "likely_false", "insufficient"] = Field(
        ..., description="Overall credibility verdict"
    )
    credibility: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="0–100 credibility score; None when verdict is 'insufficient'",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level based on source count and domain diversity"
    )
    clickbait: float = Field(
        ..., ge=0.0, le=1.0, description="Clickbait score (0=clean, 1=extreme clickbait)"
    )
    incongruity: float = Field(
        ..., ge=0.0, le=1.0, description="Headline–body mismatch score (0=consistent, 1=contradictory)"
    )
    image_risk: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Image manipulation/mismatch risk (0–1); None when no image given",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list, description="Retrieved and stance-labelled sources"
    )
    stress: StressResult | None = Field(
        default=None, description="Stress-test result; None in Fast mode"
    )
    rationale: str = Field(
        ..., min_length=1, description="3–5 sentence plain-English explanation of the verdict"
    )
    mode: Literal["fast", "deep"] = Field(
        ..., description="Pipeline mode used for this check"
    )

    @model_validator(mode="after")
    def credibility_matches_verdict(self) -> "AnalysisResult":
        """credibility must be None iff verdict is 'insufficient'."""
        if self.verdict == "insufficient" and self.credibility is not None:
            raise ValueError(
                "credibility must be None when verdict is 'insufficient'"
            )
        if self.verdict != "insufficient" and self.credibility is None:
            raise ValueError(
                "credibility must be set (0–100) for non-insufficient verdicts"
            )
        return self

    @model_validator(mode="after")
    def stress_only_in_deep(self) -> "AnalysisResult":
        """stress result is only produced in Deep mode."""
        if self.mode == "fast" and self.stress is not None:
            raise ValueError("stress result must be None in fast mode")
        return self

    def relevant_evidence(self) -> list[EvidenceItem]:
        """Return only sources labelled supports or contradicts."""
        return [e for e in self.evidence if e.stance != "unrelated"]

    def supporting_evidence(self) -> list[EvidenceItem]:
        """Return only sources that support the claim."""
        return [e for e in self.evidence if e.stance == "supports"]


# ── Input request ─────────────────────────────────────────────────────────────

class CheckRequest(BaseModel):
    """Validated input for a single credibility check.

    At least one of (headline, url) must be provided.
    body is optional but improves claim extraction.
    image_bytes is optional raw image data (JPG or PNG, max 5 MB).
    """

    headline: str | None = Field(
        default=None,
        min_length=3,
        max_length=512,
        description="News headline text",
    )
    body: str | None = Field(
        default=None,
        max_length=50_000,
        description="Article body text (optional but improves accuracy)",
    )
    url: str | None = Field(
        default=None,
        description="Article URL — headline and body will be extracted automatically",
    )
    image_bytes: bytes | None = Field(
        default=None,
        description="Raw image data (JPG or PNG, max 5 MB)",
    )
    mode: Literal["fast", "deep"] = Field(
        default="fast",
        description="Fast skips the stress test; Deep adds it (up to 5 Gemini calls)",
    )

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str | None) -> str | None:
        if v is not None and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator("image_bytes")
    @classmethod
    def image_size_limit(cls, v: bytes | None) -> bytes | None:
        max_bytes = 5 * 1024 * 1024  # 5 MB
        if v is not None and len(v) > max_bytes:
            raise ValueError(
                f"Image is {len(v) / 1_048_576:.1f} MB — maximum allowed is 5 MB"
            )
        return v

    @model_validator(mode="after")
    def at_least_headline_or_url(self) -> "CheckRequest":
        if not self.headline and not self.url:
            raise ValueError("Provide at least a headline or a URL")
        return self

    def input_hash(self) -> str:
        """SHA-256 of the canonical inputs — used as the cache key."""
        payload = json.dumps(
            {
                "headline": self.headline or "",
                "body": (self.body or "")[:2000],  # cap body for hash stability
                "url": self.url or "",
                "mode": self.mode,
                # image hash (not raw bytes, to keep the JSON small)
                "image_hash": hashlib.sha256(self.image_bytes).hexdigest()
                if self.image_bytes
                else "",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


# ── Storage helpers ───────────────────────────────────────────────────────────

class AnalysisRecord(BaseModel):
    """Row in the `analyses` SQLite table."""

    id: int | None = Field(default=None, description="Auto-assigned primary key")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the check was run",
    )
    input_hash: str = Field(..., description="SHA-256 of the canonical inputs")
    headline: str | None = Field(default=None, description="Headline at time of check")
    url: str | None = Field(default=None, description="URL at time of check")
    mode: Literal["fast", "deep"] = Field(..., description="Pipeline mode")
    result_json: str = Field(..., description="JSON-serialised AnalysisResult")

    @classmethod
    def from_request_and_result(
        cls, request: CheckRequest, result: AnalysisResult
    ) -> "AnalysisRecord":
        """Construct a record from a completed check."""
        return cls(
            input_hash=request.input_hash(),
            headline=request.headline,
            url=request.url,
            mode=result.mode,
            result_json=result.model_dump_json(),
        )

    def to_result(self) -> AnalysisResult:
        """Deserialise stored JSON back to an AnalysisResult."""
        return AnalysisResult.model_validate_json(self.result_json)


class CacheEntry(BaseModel):
    """Row in the `cache` SQLite table."""

    key: str = Field(..., description="Cache key (input_hash or arbitrary string)")
    value_json: str = Field(..., description="JSON-serialised cached value")
    expires_at: datetime = Field(..., description="UTC expiry timestamp")

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class FeedbackRecord(BaseModel):
    """Row in the `feedback` SQLite table."""

    id: int | None = Field(default=None, description="Auto-assigned primary key")
    analysis_id: int = Field(..., description="FK to analyses.id")
    rating: Literal["right", "wrong"] = Field(
        ..., description="User rating: right = verdict was correct, wrong = it was not"
    )
    note: str | None = Field(
        default=None, max_length=1000, description="Optional free-text comment"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when feedback was submitted",
    )

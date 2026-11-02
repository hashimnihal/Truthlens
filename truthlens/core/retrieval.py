"""
core/retrieval.py
-----------------
DuckDuckGo News search and source trust-tier classification.

PRD reference: §6 (F6, F7), §7.1 (Retrieval & Query building), §15
Features:
  - Searches DuckDuckGo News per claim query (up to 8 results per query)
  - De-duplicates results by domain, normalized title, and URL
  - Classifies source outlets into trust tiers (Tier 1 = wire/major, Tier 2 = known, Tier 3 = unknown)
    using curated allowlist at data/trusted_sources.csv
  - Extracts clean webpage content via trafilatura and requests when snippets need enrichment
  - Supports MOCK=1 mode with deterministic fixtures without external network requests
  - Encapsulated behind a clean interface so search backends can be swapped if needed
"""

from __future__ import annotations

import csv
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

from core.gemini_client import is_mock_mode
from core.models import EvidenceCandidate, ExtractedClaim

logger = logging.getLogger(__name__)

# Default path to trusted sources allowlist
DEFAULT_TRUSTED_SOURCES_PATH = Path(__file__).parent.parent / "data" / "trusted_sources.csv"
DEFAULT_SEARCH_TIMEOUT = 10.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


# ── Domain & Trust Tier Resolution ────────────────────────────────────────────

def extract_domain(url: str) -> str:
    """Extract registered domain name from URL in lowercase without www prefix."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        # Remove port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Remove www prefix
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


@lru_cache(maxsize=1)
def load_trusted_sources(csv_path: str | Path | None = None) -> dict[str, tuple[int, str]]:
    """Load curated domain allowlist from data/trusted_sources.csv into a lookup dict.

    Returns:
        dict mapping domain -> (tier: int, outlet_name: str)
    """
    path = Path(csv_path) if csv_path else DEFAULT_TRUSTED_SOURCES_PATH
    lookup: dict[str, tuple[int, str]] = {}

    if not path.exists():
        logger.warning(f"trusted_sources.csv not found at {path}; using empty tier table.")
        return lookup

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = row.get("domain", "").strip().lower()
                name = row.get("name", "").strip()
                tier_str = row.get("tier", "3").strip()
                if domain:
                    tier = int(tier_str) if tier_str in ("1", "2", "3") else 3
                    lookup[domain] = (tier, name or domain)
    except Exception as exc:
        logger.error(f"Error reading trusted_sources.csv: {exc}")

    return lookup


def resolve_source_tier(url: str, custom_csv_path: str | Path | None = None) -> tuple[int, str]:
    """Determine trust tier and outlet name for a URL.

    Matches domain and subdomains against trusted_sources.csv.
    Tier 1 = Wire service / major international or national outlet
    Tier 2 = Known regional, tech, or fact-checking outlet
    Tier 3 = Unknown / unclassified outlet
    """
    domain = extract_domain(url)
    if not domain:
        return 3, "Unknown Outlet"

    trusted = load_trusted_sources(custom_csv_path)

    # 1. Exact domain match
    if domain in trusted:
        return trusted[domain]

    # 2. Match parent domain (e.g. news.bbc.co.uk -> bbc.co.uk or edition.cnn.com -> cnn.com)
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in trusted:
            return trusted[parent]

    # Fallback Tier 3
    # Use formatted domain as outlet name
    clean_name = parts[-2].capitalize() if len(parts) >= 2 else domain
    return 3, clean_name


# ── Title & URL Normalization for Deduplication ───────────────────────────────

def normalize_title(title: str) -> str:
    """Normalize news headline for fuzzy duplicate comparison."""
    # Lowercase, strip punctuation and extra spaces
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    return " ".join(title.split())


def normalize_url(url: str) -> str:
    """Normalize URL by stripping tracking parameters, fragments, and trailing slashes."""
    try:
        parsed = urlparse(url)
        # Drop fragments and common tracking query parameters
        clean_path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc.lower()}{clean_path}"
    except Exception:
        return url.strip().rstrip("/")


# ── Web Content Extraction ────────────────────────────────────────────────────

def fetch_article_text(url: str, timeout: float = DEFAULT_SEARCH_TIMEOUT) -> str | None:
    """Extract clean body text from a webpage using requests and trafilatura.

    Returns extracted main text, or None if extraction failed or timed out.
    """
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None

        # Trafilatura extracts clean article body from HTML
        extracted = trafilatura.extract(
            resp.text,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return extracted.strip() if extracted else None
    except Exception as exc:
        logger.debug(f"Article text extraction failed for {url}: {exc}")
        return None


# ── DuckDuckGo News Search ────────────────────────────────────────────────────

def _mock_search_news(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Generate realistic, deterministic search results for mock development."""
    q_lower = query.lower()
    is_space = "mars" in q_lower or "nasa" in q_lower or "space" in q_lower or "isro" in q_lower
    is_health = "health" in q_lower or "who" in q_lower or "cure" in q_lower or "vaccine" in q_lower
    is_economy = "economy" in q_lower or "rate" in q_lower or "bank" in q_lower or "inflation" in q_lower

    if is_space:
        return [
            {
                "title": "Space Agency Confirms Successful Satellite Orbital Insertion",
                "url": "https://www.reuters.com/science/space-satellite-orbit-2026",
                "body": "Official flight data confirmed the payload successfully reached intended geostationary transfer orbit earlier today.",
                "date": "2026-10-31T10:00:00Z",
                "source": "Reuters",
            },
            {
                "title": "Telemetry Confirms Primary Weather Satellite Mission Milestones",
                "url": "https://apnews.com/article/space-satellite-mission-milestone-847",
                "body": "Engineers from the space control facility verified solar array deployment and preliminary communication link integrity.",
                "date": "2026-10-31T11:30:00Z",
                "source": "AP News",
            },
            {
                "title": "Next-Generation Climate Monitoring Array Deployed Successfully",
                "url": "https://www.bbc.com/news/science-environment-68493021",
                "body": "The advanced observation sensors will monitor extreme weather patterns across the hemisphere with unprecedented resolution.",
                "date": "2026-10-31T12:00:00Z",
                "source": "BBC News",
            },
            {
                "title": "Aviation and Maritime Authorities Await New Climate Sensor Feeds",
                "url": "https://timesofindia.indiatimes.com/india/weather-satellite-deployment/articleshow/998472.cms",
                "body": "Meteorological agencies announced ground stations will begin ingesting test data feeds within the upcoming operational cycle.",
                "date": "2026-10-31T13:15:00Z",
                "source": "Times of India",
            },
        ][:max_results]

    if is_health:
        return [
            {
                "title": "Health Authority Debunks Viral Medical Miracle Claims",
                "url": "https://www.who.int/news/item/medical-misinformation-alert-2026",
                "body": "Peer-reviewed clinical trials found zero efficacy for the unapproved treatment circulating across social media platforms.",
                "date": "2026-10-25T08:00:00Z",
                "source": "World Health Organization",
            },
            {
                "title": "Fact Check: Unsubstantiated Dietary Remedy Lacks Scientific Backing",
                "url": "https://www.snopes.com/fact-check/miracle-diet-cure-claims/",
                "body": "Medical experts from national academies confirmed the claims misrepresent preliminary in-vitro cell culture research.",
                "date": "2026-10-26T14:20:00Z",
                "source": "Snopes",
            },
            {
                "title": "Clinical Investigators Warn Consumers Against Unregulated Supplements",
                "url": "https://www.theguardian.com/society/health-claims-scrutiny-2026",
                "body": "Regulatory agencies have dispatched formal warnings to digital distributors marketing unverified therapeutic guarantees.",
                "date": "2026-10-27T09:45:00Z",
                "source": "The Guardian",
            },
        ][:max_results]

    # Default general news mock results
    return [
        {
            "title": f"Independent Corroboration on Developments Concerning {query[:35]}",
            "url": "https://www.reuters.com/world/news-analysis-comprehensive-report-1",
            "body": f"Official briefings and primary source accounts corroborate details surrounding {query[:50]}.",
            "date": "2026-10-30T10:00:00Z",
            "source": "Reuters",
        },
        {
            "title": f"Analysis and Timeline of Statements on {query[:35]}",
            "url": "https://apnews.com/article/breaking-developments-overview-2",
            "body": f"Documentation and verified records confirm the timeline and primary entities engaged in {query[:50]}.",
            "date": "2026-10-30T12:00:00Z",
            "source": "AP News",
        },
        {
            "title": f"Press Briefing and In-Depth Perspective on {query[:35]}",
            "url": "https://www.thehindu.com/news/national/verified-statement-3.ece",
            "body": f"Spokespersons issued comprehensive public clarifications regarding {query[:50]}.",
            "date": "2026-10-30T14:30:00Z",
            "source": "The Hindu",
        },
    ][:max_results]


def search_news(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Execute news search on DuckDuckGo News or return mock fixtures if MOCK=1.

    Returns raw list of result dicts with keys: title, url, body/snippet, date, source.
    """
    if is_mock_mode():
        logger.info(f"[MOCK SEARCH] Returning mock news results for query: '{query}'")
        return _mock_search_news(query, max_results=max_results)

    from duckduckgo_search import DDGS

    clean_query = query.strip()
    if not clean_query:
        return []

    try:
        ddgs = DDGS(timeout=DEFAULT_SEARCH_TIMEOUT)
        raw_results = ddgs.news(
            keywords=clean_query,
            region="wt-wt",  # Worldwide
            safesearch="moderate",
            max_results=max_results,
        )
        return list(raw_results or [])
    except Exception as exc:
        logger.warning(f"DuckDuckGo news search failed for '{query}': {exc}")
        # Fall back gracefully to empty list without crashing pipeline
        return []


# ── End-to-End Retrieval & Deduplication ─────────────────────────────────────

def retrieve_evidence_candidates(
    claims: list[ExtractedClaim],
    max_per_query: int = 8,
    custom_csv_path: str | Path | None = None,
) -> list[EvidenceCandidate]:
    """Retrieve, de-duplicate, and tier-tag news evidence for a list of claims.

    PRD §7.1 Rules:
      - 2 queries per claim (claim text & entity query)
      - Up to 8 results per query
      - De-duplicated by domain and title (and normalized URL)
      - Tagged with trust tier (Tier 1 major, Tier 2 known, Tier 3 unknown)
    """
    from core.claims import build_claim_queries

    candidates: list[EvidenceCandidate] = []
    seen_urls: set[str] = set()
    seen_domain_titles: set[tuple[str, str]] = set()

    # Collect all queries
    queries_to_run: list[str] = []
    for claim in claims:
        queries = build_claim_queries(claim)
        for q in queries:
            if q not in queries_to_run:
                queries_to_run.append(q)

    # If no claims or queries generated, return empty
    if not queries_to_run:
        return []

    # Execute searches and deduplicate
    for query in queries_to_run:
        results = search_news(query, max_results=max_per_query)

        for item in results:
            url = item.get("url", "").strip()
            title = item.get("title", "").strip()
            snippet = item.get("body", "") or item.get("snippet", "") or ""
            date_published = item.get("date", "") or None

            # Must have basic valid fields
            if not url or not title or not (url.startswith("http://") or url.startswith("https://")):
                continue

            # Deduplication 1: Normalized URL
            norm_url = normalize_url(url)
            if norm_url in seen_urls:
                continue

            # Deduplication 2: Domain + Normalized Title
            domain = extract_domain(url)
            norm_title = normalize_title(title)
            domain_title_key = (domain, norm_title)
            if domain_title_key in seen_domain_titles:
                continue

            # Resolve trust tier and outlet name
            tier, outlet_name = resolve_source_tier(url, custom_csv_path=custom_csv_path)

            # Ensure snippet has minimum content
            clean_snippet = snippet.strip()
            if not clean_snippet:
                clean_snippet = title

            candidate = EvidenceCandidate(
                title=title,
                source=item.get("source", "").strip() or outlet_name,
                url=url,
                published=date_published,
                snippet=clean_snippet,
                tier=tier,
            )

            seen_urls.add(norm_url)
            seen_domain_titles.add(domain_title_key)
            candidates.append(candidate)

    return candidates

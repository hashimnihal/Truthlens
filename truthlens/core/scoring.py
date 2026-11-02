"""
core/scoring.py
---------------
Credibility scoring and verdict assignment (PRD §7.2).

Formula:
    credibility = 100 × (0.45·E + 0.20·(1-H) + 0.15·(1-V) + 0.10·(1-C) + 0.10·R)

Rescaling rules:
    No image  → drop V term, renormalise weights to sum to 1.
    Fast mode → drop R term, renormalise weights to sum to 1.

Verdict thresholds:
    < 2 relevant sources → 'insufficient'
    score ≥ 70           → 'likely_authentic'
    score 40–69          → 'mixed'
    score < 40           → 'likely_false'

Confidence:
    High   → ≥3 sources, ≥2 domains, no bias flag
    Medium → ≥2 relevant sources
    Low    → otherwise

Implemented in Phase 5.
"""
# Phase 5 — to be implemented


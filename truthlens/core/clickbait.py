"""
core/clickbait.py
-----------------
Clickbait scorer: scikit-learn TF-IDF baseline blended with Gemini's rating.

Formula (PRD §7.2):
    C = 0.4 × sklearn_probability  +  0.6 × gemini_rating

Also detects linguistic signals for UI chips:
    ALL CAPS, stacked punctuation, curiosity-gap phrases, number-led headlines

Implemented in Phase 5.
"""
# Phase 5 — to be implemented


"""
core/image_check.py
-------------------
Multimodal image credibility check via Gemini Vision.

Given the image bytes, headline and lead paragraph, Gemini returns:
  - fit:               supports | unrelated | contradicts
  - manipulation_risk: 0.0–1.0
  - indicators:        list of risk indicators observed
  - context_note:      plain-text explanation

The verdict is always worded as risk indicators, never as absolute proof.

Implemented in Phase 7.
"""
# Phase 7 — to be implemented


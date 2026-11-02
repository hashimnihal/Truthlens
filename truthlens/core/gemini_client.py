"""
core/gemini_client.py
---------------------
Single point of contact for all Gemini 2.5 Flash API calls.

Features (Phase 3):
  - Structured JSON output validated by Pydantic (retry once on schema failure)
  - Low temperature (0.0–0.2) for all judgement calls
  - Retry with exponential back-off; clear quota-exceeded error
  - MOCK=1 mode: returns fixture JSON without any network call

Implemented in Phase 3.
"""
# Phase 3 — to be implemented


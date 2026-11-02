"""
core/evidence.py
----------------
Evidence stance judgement via a single batched Gemini call.

For each retrieved source the model labels it as:
  supports | contradicts | unrelated
with a one-line reason.

Security: any URL returned by Gemini that was NOT in the retrieved set is dropped.

Implemented in Phase 4.
"""
# Phase 4 — to be implemented


"""
core/stress.py
--------------
Counterfactual stress test (Deep mode only).

Gemini writes up to 3 altered versions of the main claim:
  - Negated claim
  - Changed entity
  - Changed number or date

Each variant is retrieved and judged like the original claim.
Robustness R = share of altered claims the evidence rejects.
If an altered false claim is supported, the bias flag is set and
confidence is lowered one level.

Implemented in Phase 8.
"""
# Phase 8 — to be implemented


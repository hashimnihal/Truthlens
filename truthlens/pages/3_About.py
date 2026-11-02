"""
pages/3_About.py — Method summary, limitations and disclaimer.

Phase 1 (Scaffold): Content is complete (static). No further changes planned.
"""

import streamlit as st

st.set_page_config(page_title="About · TruthLens", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About TruthLens")

st.markdown(
    """
TruthLens is a multimodal news-credibility and clickbait classifier built as a
college mini-project (BAI507P, AY 2026-27, Sahyadri College of Engineering and
Management, CSE — AI & ML).

**Team:** Hashim Nihal (leader) · Pawan P Bangera · Samuel Joy Fernandes · Mohammed Ashqar Sameer  
**Guide:** Mrs. Shruthi Vishwajeeth
"""
)

st.divider()

st.subheader("How it works")
st.markdown(
    """
1. You paste a headline, article body or URL (and optionally an image).
2. **Gemini 2.5 Flash** extracts up to 3 checkable claims and rates clickbait and
   headline–body incongruity.
3. **DuckDuckGo News** retrieves live sources for each claim, tagged with a trust tier.
4. A second Gemini call labels each source as *supports*, *contradicts* or *unrelated*.
5. A **scikit-learn TF-IDF baseline** contributes to the clickbait score.
6. *(Deep mode)* Three counterfactual variants test whether the verdict is robust.
7. All signals feed a weighted formula (PRD §7.2) to produce a **0–100 credibility score**
   and a verdict.
"""
)

st.divider()

st.subheader("Scoring formula")
st.latex(
    r"\text{credibility} = 100 \times "
    r"(0.45E + 0.20(1-H) + 0.15(1-V) + 0.10(1-C) + 0.10R)"
)
st.caption(
    "E = evidence support · H = headline–body incongruity · V = image risk · "
    "C = clickbait score · R = stress-test robustness"
)

st.divider()

st.subheader("Limitations")
st.markdown(
    """
- Results are **evidence-based assessments**, never final rulings.
- Image checks return **risk indicators only** — not proof of manipulation or AI generation.
- Live retrieval depends on DuckDuckGo News; results change over time.
- English only; no video or audio input.
- Free-tier Gemini quota limits throughput — use Fast mode for quick checks.
- Opinions and predictions are flagged as non-verifiable and skipped.
"""
)

st.divider()

st.subheader("Ethics")
st.markdown(
    """
- Sources and reasoning are always shown so you can disagree with the verdict.
- The app never labels outlets or people as liars; it describes what the evidence shows.
- Feedback is collected anonymously for error analysis only.
"""
)

st.divider()

st.caption(
    "_TruthLens is an evidence-based assessment tool, not a final arbiter of truth._  \n"
    "BAI507P Mini Project · AY 2026-27 · Sahyadri College · CSE (AI & ML)"
)


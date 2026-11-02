"""
app.py — TruthLens Analyse page (home).

Phase 1 (Scaffold): Renders a "hello" page that confirms the project is wired
correctly. Full Analyse UI is implemented in Phase 9.

Architecture rule: this page calls only pipeline.run_check — never any engine
module directly.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**TruthLens** · Online News Credibility & Clickbait Classifier\n\n"
            "BAI507P Mini Project · CSE (AI & ML) · Sahyadri College · AY 2026-27"
        )
    },
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/magnifier--v1.png",
        width=64,
    )
    st.markdown("## 🔍 TruthLens")
    st.caption("Online News Credibility & Clickbait Classifier")
    st.divider()
    st.markdown(
        "**Pages**\n"
        "- 🏠 Analyse ← *you are here*\n"
        "- 📜 History\n"
        "- 📊 Evaluation\n"
        "- ℹ️ About"
    )
    st.divider()
    mock_env = os.environ.get("MOCK", "0")
    if mock_env == "1":
        st.success("🧪 MOCK mode active — no API calls")
    else:
        st.info("🌐 Live mode")

# ── Main content ─────────────────────────────────────────────────────────────
st.title("🔍 TruthLens")
st.subheader("Online News Credibility & Clickbait Classifier")
st.caption(
    "BAI507P Mini Project · CSE (AI & ML) · Sahyadri College of Engineering and Management"
)

st.divider()

# Phase badge
st.info(
    "**Phase 1 — Scaffold complete.** "
    "The full Analyse UI will be wired here in Phase 9. "
    "Navigate to History, Evaluation and About pages in the sidebar to confirm routing.",
    icon="🏗️",
)

# System check panel
with st.expander("🔧 System check", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    with col2:
        try:
            import pydantic
            st.metric("Pydantic", pydantic.__version__, delta="v2 ✅" if int(pydantic.__version__[0]) >= 2 else "needs v2")
        except ImportError:
            st.metric("Pydantic", "❌ not installed")

    with col3:
        st.metric("MOCK mode", "ON 🧪" if mock_env == "1" else "OFF 🌐")

    col4, col5, col6 = st.columns(3)

    with col4:
        try:
            import sklearn
            st.metric("scikit-learn", sklearn.__version__, delta="✅")
        except ImportError:
            st.metric("scikit-learn", "❌ not installed")

    with col5:
        try:
            import streamlit as _st
            st.metric("Streamlit", _st.__version__, delta="✅")
        except ImportError:
            st.metric("Streamlit", "❌")

    with col6:
        db_path = os.environ.get("DB_PATH", "truthlens.db")
        st.metric("DB path", db_path)

# Project structure preview
with st.expander("📁 Project structure (Phase 1 scaffold)"):
    st.code(
        """\
truthlens/
├── app.py                  ← you are here (Analyse / home page)
├── pages/
│   ├── 1_History.py
│   ├── 2_Evaluation.py
│   └── 3_About.py
├── core/                   ← engine (Phases 2–8)
│   ├── models.py           ← Pydantic v2 schemas
│   ├── pipeline.py         ← run_check() — the only function pages call
│   ├── gemini_client.py    ← Gemini 2.5 Flash, retry, mock
│   ├── claims.py
│   ├── retrieval.py
│   ├── evidence.py
│   ├── clickbait.py
│   ├── image_check.py
│   ├── stress.py
│   ├── scoring.py
│   └── storage.py
├── ui/
│   ├── components.py
│   └── styles.py
├── data/
│   ├── trusted_sources.csv
│   └── eval/
├── models/
│   └── clickbait_lr.joblib
├── extension/
├── tests/
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
└── README.md""",
        language="text",
    )

st.divider()
st.caption(
    "TruthLens · BAI507P Mini Project · AY 2026-27 · "
    "Team: Hashim Nihal, Pawan P Bangera, Samuel Joy Fernandes, Mohammed Ashqar Sameer · "
    "Guide: Mrs. Shruthi Vishwajeeth\n\n"
    "_This is an evidence-based assessment tool, not a final arbiter of truth._"
)


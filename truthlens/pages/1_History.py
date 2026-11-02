"""
pages/1_History.py — Past checks browser.

Phase 1 (Scaffold): Renders a placeholder. Full implementation in Phase 10.
"""

import streamlit as st

st.set_page_config(page_title="History · TruthLens", page_icon="📜", layout="wide")

st.title("📜 History")
st.caption("Past checks — reopen, export or delete")

st.info(
    "**Phase 10** will wire this page to the SQLite `analyses` table.\n\n"
    "You will be able to filter checks by date and verdict, reopen any check, "
    "export it as JSON or Markdown, and delete it.",
    icon="🏗️",
)


"""
pages/2_Evaluation.py — Model evaluation and ablation results.

Phase 1 (Scaffold): Renders a placeholder. Full implementation in Phase 12.
"""

import streamlit as st

st.set_page_config(page_title="Evaluation · TruthLens", page_icon="📊", layout="wide")

st.title("📊 Evaluation")
st.caption("Ablation study · confusion matrices · dataset notes")

st.info(
    "**Phase 12** will populate this page with evaluation results from `data/eval/`.\n\n"
    "Ablation configurations (Baseline, A, B, C, D/Full) and metric tables "
    "(accuracy, macro-F1, ROC-AUC) will be displayed here for evaluators and the paper.",
    icon="🏗️",
)

with st.expander("Ablation configurations"):
    import pandas as pd

    df = pd.DataFrame(
        {
            "Config": ["Baseline", "A", "B", "C", "D (Full)"],
            "Components": [
                "sklearn TF-IDF",
                "Gemini text only, no retrieval",
                "A + live retrieval",
                "B + image check",
                "C + stress test",
            ],
            "Research question": [
                "Cost of a classic approach",
                "Starting point for RQ1",
                "RQ1",
                "RQ2",
                "RQ3",
            ],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


"""
tests/test_clickbait.py
-----------------------
Unit tests for scikit-learn clickbait baseline, linguistic signal extraction,
and blended scoring formula (core/clickbait.py).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.clickbait import (
    compute_clickbait_score,
    detect_linguistic_signals,
    load_clickbait_model,
    predict_sklearn_baseline,
    train_clickbait_baseline,
)
from core.models import ClickbaitResult


# ── Linguistic Signal Detection ───────────────────────────────────────────────

def test_detect_linguistic_signals_normal_headline():
    headline = "Central Bank announces 25 basis point reduction in benchmark interest rate"
    signals = detect_linguistic_signals(headline)
    assert len(signals) == 0


def test_detect_linguistic_signals_all_caps():
    headline = "SHOCKING video shows minister admitting massive fraud"
    signals = detect_linguistic_signals(headline)
    names = [s.name for s in signals]
    assert "all_caps" in names


def test_detect_linguistic_signals_ignores_standard_acronyms():
    headline = "NASA and ISRO sign agreement for joint earth observation mission"
    signals = detect_linguistic_signals(headline)
    names = [s.name for s in signals]
    assert "all_caps" not in names


def test_detect_linguistic_signals_stacked_punctuation():
    headline1 = "Is this the end of traditional banking???"
    signals1 = detect_linguistic_signals(headline1)
    assert any(s.name == "stacked_punctuation" for s in signals1)

    headline2 = "You will not believe what was found in the vault!?!"
    signals2 = detect_linguistic_signals(headline2)
    assert any(s.name == "stacked_punctuation" for s in signals2)


def test_detect_linguistic_signals_curiosity_gap():
    headlines = [
        "She opened the mysterious package and you won't believe what was inside",
        "The airline cancelled all flights and here's why",
        "He invested five hundred dollars and what happened next stunned everyone",
    ]
    for h in headlines:
        signals = detect_linguistic_signals(h)
        assert any(s.name == "curiosity_gap" for s in signals), f"Failed to detect in: {h}"


def test_detect_linguistic_signals_number_led():
    headline = "10 simple habits that will transform your daily productivity"
    signals = detect_linguistic_signals(headline)
    assert any(s.name == "number_led" for s in signals)

    # Ordinary headline with numbers shouldn't be tagged
    normal_numbered = "2026 fiscal policy framework approved by committee"
    signals_normal = detect_linguistic_signals(normal_numbered)
    assert not any(s.name == "number_led" for s in signals_normal)


# ── Model Training & Fallback ─────────────────────────────────────────────────

def test_train_and_load_clickbait_model(tmp_path: Path):
    # Create small synthetic training dataset
    data = {
        "headline": [
            "You will never believe this shocking celebrity secret!",
            "10 crazy things doctors do not want you to know",
            "This simple trick will blow your mind completely",
            "Reserve Bank of India publishes annual monetary report",
            "Prime Minister delivers keynote address at global trade forum",
            "Agricultural university researchers develop drought resistant wheat",
        ],
        "label": [1, 1, 1, 0, 0, 0],
    }
    csv_file = tmp_path / "train_clickbait.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    model_file = tmp_path / "test_model.joblib"
    pipeline = train_clickbait_baseline(csv_file, save_path=model_file)
    assert pipeline is not None
    assert model_file.is_file()

    # Test safe loading
    loaded_model = load_clickbait_model(model_file)
    assert loaded_model is not None

    # Test prediction with trained model
    prob = predict_sklearn_baseline(
        "SHOCKING miracle secret that will blow your mind!", model_path=model_file
    )
    assert 0.0 <= prob <= 1.0


def test_missing_model_fallback():
    # Pointing to nonexistent model triggers deterministic lexical fallback
    missing_path = Path("/nonexistent/clickbait_model.joblib")
    prob_clean = predict_sklearn_baseline(
        "Ministry of Transport inaugurates new national highway bypass", model_path=missing_path
    )
    assert 0.0 <= prob_clean <= 0.30

    prob_clickbait = predict_sklearn_baseline(
        "SHOCKING: You won't believe what happened next!!!", model_path=missing_path
    )
    assert prob_clickbait >= 0.60


# ── Final Blended Clickbait Score (PRD §7.2: C = 0.4*sklearn + 0.6*gemini) ────

def test_compute_clickbait_score_formula():
    headline = "Factual economic update"

    with patch("core.clickbait.predict_sklearn_baseline", return_value=0.20):
        # C = 0.4 * 0.20 + 0.6 * 0.50 = 0.08 + 0.30 = 0.38
        result = compute_clickbait_score(headline, gemini_rating=0.50)
        assert isinstance(result, ClickbaitResult)
        assert pytest.approx(result.score, abs=1e-3) == 0.38
        assert result.sklearn_prob == 0.20
        assert result.gemini_rating == 0.50


def test_compute_clickbait_score_bounds():
    result_min = compute_clickbait_score("Clean headline", gemini_rating=-0.5)
    assert 0.0 <= result_min.score <= 1.0
    assert result_min.gemini_rating == 0.0

    result_max = compute_clickbait_score("Extreme headline", gemini_rating=1.5)
    assert 0.0 <= result_max.score <= 1.0
    assert result_max.gemini_rating == 1.0


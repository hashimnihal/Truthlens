"""
core/clickbait.py
-----------------
Clickbait detection module combining classical scikit-learn TF-IDF baseline,
Gemini rating, and linguistic signal extraction.

PRD reference: §6 (F9, F11), §7.2 (Scoring formula C = 0.4*sklearn + 0.6*gemini)
Features:
  - Scikit-learn TF-IDF + Logistic Regression baseline loader and predictor
  - Graceful rule-based lexical fallback if models/clickbait_lr.joblib is not yet trained
  - Linguistic clickbait signal detector for UI chips:
      * ALL-CAPS words or phrases
      * Stacked punctuation (!!!, ???, !?, ?!)
      * Curiosity-gap phrases ("You won't believe", "What happened next", etc.)
      * Number-led listicle headlines
  - Weighted clickbait score blending: C = 0.4 * sklearn + 0.6 * gemini
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from core.models import ClickbaitResult, ClickbaitSignal

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "clickbait_lr.joblib"

# Common legitimate acronyms that shouldn't trigger ALL-CAPS warnings
COMMON_ACRONYMS = {
    "NASA", "ISRO", "WHO", "UN", "US", "UK", "EU", "NATO", "FBI", "CIA",
    "RBI", "IMF", "AI", "ML", "COVID", "SARS", "CEO", "CFO", "CTO", "PM",
    "MP", "BJP", "INC", "AAP", "GDP", "CPI", "IPL", "ICC", "BCCI", "ISPR",
    "DRDO", "BARC", "IIT", "IIM", "AIIMS", "UGC", "CBSE", "ICSE",
}

# Regex for stacked punctuation
STACKED_PUNCT_REGEX = re.compile(r"([!?]{2,}|!\?|\?!|;\?|\.{3,})")

# Curiosity-gap expressions
CURIOSITY_GAP_PATTERNS = [
    r"\byou\s+wo\s*n'?t\s+believe\b",
    r"\byou\s+will\s+never\s+believe\b",
    r"\bwhat\s+happened\s+next\b",
    r"\bthis\s+is\s+why\b",
    r"\bthe\s+reason\s+why\b",
    r"\bhere'?s\s+why\b",
    r"\bhere'?s\s+what\b",
    r"\bwill\s+blow\s+your\s+mind\b",
    r"\bwill\s+shock\s+you\b",
    r"\bwill\s+stun\s+you\b",
    r"\bsee\s+what\s+happens\b",
    r"\bsecret\s+revealed\b",
    r"\bshocking\s+(?:reason|truth|secret|details)\b",
    r"\bleft\s+(?:everyone|viewers|fans)\s+(?:speechless|stunned|in\s+tears)\b",
]

# Number-led listicle pattern (e.g. "10 Reasons Why...", "7 Signs That...")
# Avoid matching simple years (e.g. "2026 Budget Announced") or factual casualty numbers ("5 Rescued")
NUMBER_LED_REGEX = re.compile(
    r"^(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s+"
    r"(?:reasons|ways|things|signs|habits|tricks|secrets|facts|photos|simple|crazy|top|worst|best|shocking)\b",
    re.IGNORECASE,
)


# ── Linguistic Signal Detection (PRD §6 F11) ──────────────────────────────────

def detect_linguistic_signals(headline: str) -> list[ClickbaitSignal]:
    """Identify sensational linguistic signals in a news headline."""
    signals: list[ClickbaitSignal] = []
    if not headline or not headline.strip():
        return signals

    text = headline.strip()

    # 1. All-caps words detection (words >= 3 letters, all caps, not standard acronyms)
    words = re.findall(r"\b[A-Z]{3,}\b", text)
    caps_words = [w for w in words if w not in COMMON_ACRONYMS]
    if len(caps_words) >= 1:
        signals.append(
            ClickbaitSignal(
                name="all_caps",
                label="ALL CAPS",
                description=f"Sensational capitalized emphasis: {', '.join(caps_words[:3])}",
            )
        )

    # 2. Stacked punctuation (!!!, ???, !?, etc.)
    punct_matches = STACKED_PUNCT_REGEX.findall(text)
    if punct_matches:
        signals.append(
            ClickbaitSignal(
                name="stacked_punctuation",
                label="Stacked Punctuation",
                description=f"Emotional punctuation marks: '{punct_matches[0]}'",
            )
        )

    # 3. Curiosity-gap phrases
    for pattern in CURIOSITY_GAP_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            signals.append(
                ClickbaitSignal(
                    name="curiosity_gap",
                    label="Curiosity Gap",
                    description=f"Withholding critical context: '{match.group(0)}'",
                )
            )
            break  # Record one curiosity gap signal

    # 4. Number-led listicle headline
    if NUMBER_LED_REGEX.search(text):
        signals.append(
            ClickbaitSignal(
                name="number_led",
                label="Listicle Pattern",
                description="Listicle headline structuring designed to maximize virality",
            )
        )

    return signals


# ── Classical scikit-learn Baseline ───────────────────────────────────────────

def load_clickbait_model(model_path: Path | str | None = None) -> Pipeline | None:
    """Safely load the trained TF-IDF + LogisticRegression model from disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.is_file():
        return None

    try:
        model = joblib.load(path)
        if hasattr(model, "predict_proba"):
            return model
        logger.warning(f"Loaded object from {path} lacks predict_proba; ignoring.")
        return None
    except Exception as exc:
        logger.warning(f"Could not load clickbait model from {path}: {exc}")
        return None


def train_clickbait_baseline(
    dataset_path: Path | str, save_path: Path | str | None = None
) -> Pipeline:
    """Train a TF-IDF + LogisticRegression clickbait baseline on labelled data.

    Expects a CSV with 'headline' (str) and 'label' (int: 1=clickbait, 0=clean).
    """
    import pandas as pd

    data_file = Path(dataset_path)
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    df = pd.read_csv(data_file)
    if "headline" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset CSV must contain 'headline' and 'label' columns")

    X = df["headline"].astype(str)
    y = df["label"].astype(int)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, lowercase=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ]
    )

    pipeline.fit(X, y)

    target_path = Path(save_path) if save_path else DEFAULT_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target_path)
    logger.info(f"Trained clickbait baseline saved to {target_path}")

    return pipeline


def _fallback_lexical_clickbait_prob(headline: str) -> float:
    """Deterministic lexical baseline used when no pre-trained joblib model exists.

    Evaluates linguistic signals, sensational word density, and question hooks.
    """
    signals = detect_linguistic_signals(headline)
    text_lower = headline.lower()

    # Base baseline probability
    prob = 0.10

    # Weight by detected signals
    for s in signals:
        if s.name == "curiosity_gap":
            prob += 0.40
        elif s.name == "all_caps":
            prob += 0.25
        elif s.name == "stacked_punctuation":
            prob += 0.20
        elif s.name == "number_led":
            prob += 0.15

    # Check common hyperbole tokens
    hyperbole = [
        "shocking", "insane", "miracle", "epic", "viral", "unbelievable",
        "jaw-dropping", "must-see", "breaks the internet", "destroyed",
    ]
    if any(h in text_lower for h in hyperbole):
        prob += 0.20

    return min(max(prob, 0.0), 1.0)


def predict_sklearn_baseline(headline: str, model_path: Path | str | None = None) -> float:
    """Predict clickbait probability using scikit-learn model, with fallback."""
    if not headline or not headline.strip():
        return 0.0

    model = load_clickbait_model(model_path)
    if model is not None:
        try:
            # Predict class 1 (clickbait)
            probs = model.predict_proba([headline.strip()])[0]
            # Assumes binary classifier [prob_clean, prob_clickbait]
            return float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception as exc:
            logger.warning(f"Error during model prediction: {exc}; using fallback.")

    # Explicit fallback when model file is not yet trained
    return _fallback_lexical_clickbait_prob(headline)


# ── Final Blended Clickbait Score (PRD §7.2) ──────────────────────────────────

def compute_clickbait_score(
    headline: str,
    gemini_rating: float,
    model_path: Path | str | None = None,
) -> ClickbaitResult:
    """Compute blended clickbait score: C = 0.4 * sklearn + 0.6 * gemini.

    Args:
        headline: News headline text.
        gemini_rating: Clickbait intensity from Gemini claim analysis (0.0 to 1.0).
        model_path: Optional custom path to trained joblib model.

    Returns:
        ClickbaitResult Pydantic model with score, components, and signals.
    """
    # Validate gemini rating is in range [0, 1]
    g_rating = min(max(float(gemini_rating), 0.0), 1.0)

    # Scikit-learn probability
    sk_prob = predict_sklearn_baseline(headline, model_path=model_path)
    sk_prob = min(max(sk_prob, 0.0), 1.0)

    # Formula from PRD §7.2:
    # C = 0.4 * scikit-learn probability + 0.6 * Gemini rating
    final_score = (0.4 * sk_prob) + (0.6 * g_rating)
    final_score = round(min(max(final_score, 0.0), 1.0), 4)

    # Extract linguistic signals
    signals = detect_linguistic_signals(headline)

    return ClickbaitResult(
        score=final_score,
        sklearn_prob=round(sk_prob, 4),
        gemini_rating=round(g_rating, 4),
        signals=signals,
    )

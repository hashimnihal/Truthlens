"""
tests/test_scaffold.py
----------------------
Phase 1 smoke tests — verify directory structure and imports.
These tests contain NO business logic; they confirm the scaffold is correct.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

# Root of the truthlens package
ROOT = Path(__file__).parent.parent


# ── Directory structure ───────────────────────────────────────────────────────

EXPECTED_DIRS = [
    "core",
    "ui",
    "pages",
    "data",
    "data/eval",
    "models",
    "extension",
    "tests",
    ".streamlit",
]

EXPECTED_FILES = [
    "app.py",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "README.md",
    ".streamlit/config.toml",
    "data/trusted_sources.csv",
    "core/__init__.py",
    "core/models.py",
    "core/pipeline.py",
    "core/gemini_client.py",
    "core/extract.py",
    "core/claims.py",
    "core/retrieval.py",
    "core/evidence.py",
    "core/clickbait.py",
    "core/image_check.py",
    "core/stress.py",
    "core/scoring.py",
    "core/storage.py",
    "ui/__init__.py",
    "ui/components.py",
    "ui/styles.py",
    "pages/1_History.py",
    "pages/2_Evaluation.py",
    "pages/3_About.py",
    "tests/__init__.py",
]


def test_expected_directories_exist():
    """Every required directory must be present."""
    for d in EXPECTED_DIRS:
        path = ROOT / d
        assert path.is_dir(), f"Missing directory: {d}"


def test_expected_files_exist():
    """Every required file must be present."""
    for f in EXPECTED_FILES:
        path = ROOT / f
        assert path.is_file(), f"Missing file: {f}"


# ── Python imports ────────────────────────────────────────────────────────────

IMPORTABLE_MODULES = [
    "core",
    "core.models",
    "core.pipeline",
    "core.gemini_client",
    "core.extract",
    "core.claims",
    "core.retrieval",
    "core.evidence",
    "core.clickbait",
    "core.image_check",
    "core.stress",
    "core.scoring",
    "core.storage",
    "ui",
    "ui.components",
    "ui.styles",
]


def test_core_modules_importable():
    """All core stub modules must be importable without errors."""
    import sys

    # Ensure ROOT is on the path so `import core` works
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for mod in IMPORTABLE_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            raise AssertionError(f"Cannot import {mod}: {exc}") from exc


# ── requirements.txt sanity ───────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    "streamlit",
    "google-genai",
    "pydantic",
    "duckduckgo-search",
    "scikit-learn",
    "trafilatura",
    "requests",
    "python-dotenv",
    "Pillow",
    "pytest",
]


def test_requirements_lists_all_packages():
    """requirements.txt must mention every approved dependency."""
    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in REQUIRED_PACKAGES:
        assert pkg.lower() in req_text, f"requirements.txt missing: {pkg}"


# ── .env.example sanity ───────────────────────────────────────────────────────

def test_env_example_has_gemini_key():
    """.env.example must define GEMINI_API_KEY (without a real value)."""
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in env_text
    # Must NOT contain a real key (no 'AIza' Google API key prefix)
    assert "AIza" not in env_text, ".env.example contains what looks like a real API key!"


def test_env_example_has_mock_flag():
    """.env.example must define the MOCK flag."""
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MOCK" in env_text


# ── trusted_sources.csv sanity ────────────────────────────────────────────────

def test_trusted_sources_csv_structure():
    """trusted_sources.csv must have domain, name, tier columns and tier values 1–3."""
    import csv

    csv_path = ROOT / "data" / "trusted_sources.csv"
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert "domain" in reader.fieldnames, "Missing 'domain' column"
    assert "name" in reader.fieldnames, "Missing 'name' column"
    assert "tier" in reader.fieldnames, "Missing 'tier' column"

    assert len(rows) > 0, "trusted_sources.csv is empty"

    for row in rows:
        assert row["tier"] in ("1", "2", "3"), (
            f"Invalid tier '{row['tier']}' for domain '{row['domain']}'"
        )


# ── config.toml sanity ───────────────────────────────────────────────────────

def test_streamlit_config_has_theme():
    """.streamlit/config.toml must define primaryColor."""
    config_text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "primaryColor" in config_text
    assert "#1F4E79" in config_text, "Primary colour should match PRD §9.5"


"""
tests/test_storage.py
---------------------
Unit tests for core/storage.py SQLite operations.
Uses an isolated temporary database for each test.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.models import AnalysisRecord, AnalysisResult, CheckRequest, FeedbackRecord
from core.storage import (
    clean_expired_cache,
    delete_analysis,
    delete_cache,
    get_analysis_by_id,
    get_cache,
    get_feedback_by_analysis_id,
    get_latest_analysis_by_hash,
    init_db,
    list_analyses,
    list_feedback,
    save_analysis,
    save_feedback,
    set_cache,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a clean, initialized temporary database file for a test."""
    db_file = tmp_path / "test_truthlens.db"
    init_db(db_file)
    return db_file


def _dummy_result(mode: str = "fast", credibility: int = 80) -> AnalysisResult:
    return AnalysisResult(
        verdict="likely_authentic",
        credibility=credibility,
        confidence="high",
        clickbait=0.2,
        incongruity=0.1,
        rationale="Tested by evidence.",
        mode=mode,
    )


# ── Initialization tests ──────────────────────────────────────────────────────

def test_init_db_creates_tables(temp_db: Path):
    assert temp_db.exists()
    # Calling init_db again should be idempotent
    init_db(temp_db)


# ── Analysis CRUD tests ───────────────────────────────────────────────────────

def test_save_and_get_analysis_by_id(temp_db: Path):
    req = CheckRequest(headline="Headline 1", mode="fast")
    res = _dummy_result()
    record = AnalysisRecord.from_request_and_result(req, res)

    row_id = save_analysis(record, db_path=temp_db)
    assert row_id > 0

    fetched = get_analysis_by_id(row_id, db_path=temp_db)
    assert fetched is not None
    assert fetched.id == row_id
    assert fetched.headline == "Headline 1"
    assert fetched.mode == "fast"
    assert fetched.input_hash == req.input_hash()

    res_restored = fetched.to_result()
    assert res_restored.verdict == "likely_authentic"
    assert res_restored.credibility == 80


def test_get_analysis_by_nonexistent_id(temp_db: Path):
    fetched = get_analysis_by_id(99999, db_path=temp_db)
    assert fetched is None


def test_get_latest_analysis_by_hash(temp_db: Path):
    req = CheckRequest(headline="Common Headline", mode="fast")
    res1 = _dummy_result(credibility=70)
    res2 = _dummy_result(credibility=95)

    rec1 = AnalysisRecord.from_request_and_result(req, res1)
    rec2 = AnalysisRecord.from_request_and_result(req, res2)

    save_analysis(rec1, db_path=temp_db)
    # Small sleep to ensure distinct created_at timestamp
    time.sleep(0.01)
    save_analysis(rec2, db_path=temp_db)

    latest = get_latest_analysis_by_hash(req.input_hash(), db_path=temp_db)
    assert latest is not None
    assert latest.to_result().credibility == 95


def test_list_analyses_pagination(temp_db: Path):
    for i in range(5):
        req = CheckRequest(headline=f"Headline #{i}", mode="fast")
        res = _dummy_result(credibility=50 + i)
        save_analysis(AnalysisRecord.from_request_and_result(req, res), db_path=temp_db)

    all_rows = list_analyses(limit=10, db_path=temp_db)
    assert len(all_rows) == 5

    page1 = list_analyses(limit=2, offset=0, db_path=temp_db)
    assert len(page1) == 2

    page2 = list_analyses(limit=2, offset=2, db_path=temp_db)
    assert len(page2) == 2
    assert page1[0].id != page2[0].id


def test_delete_analysis(temp_db: Path):
    req = CheckRequest(headline="To Delete", mode="fast")
    res = _dummy_result()
    row_id = save_analysis(
        AnalysisRecord.from_request_and_result(req, res), db_path=temp_db
    )

    # Delete existing
    deleted = delete_analysis(row_id, db_path=temp_db)
    assert deleted is True

    # Confirm it's gone
    assert get_analysis_by_id(row_id, db_path=temp_db) is None

    # Delete non-existing
    assert delete_analysis(row_id, db_path=temp_db) is False


# ── Cache CRUD tests ──────────────────────────────────────────────────────────

def test_set_and_get_cache(temp_db: Path):
    set_cache("key1", '{"status": "ok"}', ttl_seconds=60, db_path=temp_db)
    val = get_cache("key1", db_path=temp_db)
    assert val == '{"status": "ok"}'


def test_cache_overwrite(temp_db: Path):
    set_cache("key1", '{"v": 1}', ttl_seconds=60, db_path=temp_db)
    set_cache("key1", '{"v": 2}', ttl_seconds=60, db_path=temp_db)
    val = get_cache("key1", db_path=temp_db)
    assert val == '{"v": 2}'


def test_cache_miss_and_expiration(temp_db: Path):
    # Non-existent key
    assert get_cache("non_existent_key", db_path=temp_db) is None

    # Expired key (negative TTL)
    set_cache("expired_key", '{"old": true}', ttl_seconds=-1, db_path=temp_db)
    assert get_cache("expired_key", db_path=temp_db) is None


def test_delete_cache(temp_db: Path):
    set_cache("del_key", "value", ttl_seconds=60, db_path=temp_db)
    assert delete_cache("del_key", db_path=temp_db) is True
    assert get_cache("del_key", db_path=temp_db) is None
    assert delete_cache("del_key", db_path=temp_db) is False


def test_clean_expired_cache(temp_db: Path):
    set_cache("k_active", "val", ttl_seconds=300, db_path=temp_db)
    set_cache("k_exp1", "val", ttl_seconds=-5, db_path=temp_db)
    set_cache("k_exp2", "val", ttl_seconds=-10, db_path=temp_db)

    cleaned = clean_expired_cache(db_path=temp_db)
    assert cleaned == 2
    assert get_cache("k_active", db_path=temp_db) == "val"


# ── Feedback CRUD tests ───────────────────────────────────────────────────────

def test_save_and_get_feedback(temp_db: Path):
    req = CheckRequest(headline="Headline Feedback", mode="fast")
    res = _dummy_result()
    analysis_id = save_analysis(
        AnalysisRecord.from_request_and_result(req, res), db_path=temp_db
    )

    fb1 = FeedbackRecord(
        analysis_id=analysis_id, rating="right", note="Verdict matched fact-checkers"
    )
    fb2 = FeedbackRecord(
        analysis_id=analysis_id, rating="wrong", note="Missed recent update"
    )

    id1 = save_feedback(fb1, db_path=temp_db)
    id2 = save_feedback(fb2, db_path=temp_db)
    assert id1 > 0
    assert id2 > 0

    records = get_feedback_by_analysis_id(analysis_id, db_path=temp_db)
    assert len(records) == 2
    assert records[0].rating == "right"
    assert records[1].rating == "wrong"

    all_fb = list_feedback(limit=10, db_path=temp_db)
    assert len(all_fb) == 2


def test_feedback_cascading_delete(temp_db: Path):
    req = CheckRequest(headline="Headline to delete with feedback", mode="fast")
    analysis_id = save_analysis(
        AnalysisRecord.from_request_and_result(req, _dummy_result()), db_path=temp_db
    )

    save_feedback(
        FeedbackRecord(analysis_id=analysis_id, rating="right", note="Good"),
        db_path=temp_db,
    )
    assert len(get_feedback_by_analysis_id(analysis_id, db_path=temp_db)) == 1

    # Deleting analysis should cascade and delete its feedback
    delete_analysis(analysis_id, db_path=temp_db)
    assert len(get_feedback_by_analysis_id(analysis_id, db_path=temp_db)) == 0


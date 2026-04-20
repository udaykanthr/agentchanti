"""Tests for agentchanti.testing.locator_cache."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentchanti.testing.locator_cache import CACHE_VERSION, CacheEntry, LocatorCache


def test_empty_cache_returns_none(tmp_path: Path):
    cache = LocatorCache(tmp_path / "cache.json")
    assert cache.get("flow", "step-1") is None
    assert len(cache) == 0


def test_set_then_get_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    cache = LocatorCache(cache_path)
    cache.set("flow", "step-1", "button[data-testid=submit]")

    # Re-open from disk — persistence must survive process restart
    cache2 = LocatorCache(cache_path)
    assert cache2.get("flow", "step-1") == "button[data-testid=submit]"
    assert len(cache2) == 1


def test_invalidate_removes_entry(tmp_path: Path):
    cache = LocatorCache(tmp_path / "cache.json")
    cache.set("flow", "step-1", "#btn")
    cache.invalidate("flow", "step-1")
    assert cache.get("flow", "step-1") is None


def test_ttl_expiry_returns_none(tmp_path: Path):
    cache = LocatorCache(tmp_path / "cache.json")
    cache.set("flow", "step-1", "#btn")

    # Forge an expired timestamp by editing the file directly
    raw = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    stale = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    raw["entries"]["flow::step-1"]["last_verified_at"] = stale
    (tmp_path / "cache.json").write_text(json.dumps(raw), encoding="utf-8")

    cache2 = LocatorCache(tmp_path / "cache.json")
    assert cache2.get("flow", "step-1", ttl_days=14) is None
    # But without a TTL, the cached selector is still served
    assert cache2.get("flow", "step-1") == "#btn"


def test_corrupt_cache_starts_fresh(tmp_path: Path):
    bad = tmp_path / "cache.json"
    bad.write_text("{not valid json", encoding="utf-8")
    cache = LocatorCache(bad)
    assert len(cache) == 0
    # Can still write after recovery
    cache.set("flow", "step-1", "#btn")
    assert cache.get("flow", "step-1") == "#btn"


def test_version_mismatch_starts_fresh(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "version": "0-from-the-future",
        "entries": {"flow::step-1": {"selector": "#old", "last_verified_at": "x"}},
    }), encoding="utf-8")
    cache = LocatorCache(path)
    assert cache.get("flow", "step-1") is None


def test_atomic_save_leaves_no_tmp_file(tmp_path: Path):
    cache = LocatorCache(tmp_path / "cache.json")
    cache.set("flow", "step-1", "#btn")
    assert (tmp_path / "cache.json").exists()
    assert not (tmp_path / "cache.json.tmp").exists()


def test_saved_payload_declares_current_version(tmp_path: Path):
    cache = LocatorCache(tmp_path / "cache.json")
    cache.set("flow", "step-1", "#btn")
    raw = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert raw["version"] == CACHE_VERSION


def test_clear_drops_everything(tmp_path: Path):
    cache = LocatorCache(tmp_path / "cache.json")
    cache.set("flow", "step-1", "#a")
    cache.set("flow", "step-2", "#b")
    cache.clear()
    assert len(cache) == 0
    # Persisted clear survives reload
    assert len(LocatorCache(tmp_path / "cache.json")) == 0


def test_is_expired_with_corrupt_timestamp():
    entry = CacheEntry(selector="#x", last_verified_at="not-a-date")
    assert entry.is_expired(ttl_days=14) is True

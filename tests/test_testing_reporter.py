"""Tests for agentchanti.testing.reporter."""

from __future__ import annotations

import json
from pathlib import Path

from agentchanti.testing.reporter import Reporter
from agentchanti.testing.validator import AssertionResult


def _results():
    return [
        AssertionResult(id="s1::net::0", kind="network", passed=True,
                        detail="POST /api/orders → 201 (ok)"),
        AssertionResult(id="a1", kind="url_equals", passed=False,
                        detail="expected '/dashboard', got '/login'"),
        AssertionResult(id="a2", kind="natural_language", passed=False,
                        skipped=True, detail="no llm_client supplied"),
    ]


def test_console_header_counts_are_correct():
    out = Reporter().render_console(_results())
    assert "1 passed, 1 failed, 1 skipped (3 total)" in out


def test_console_marks_pass_fail_skip():
    out = Reporter().render_console(_results())
    assert "PASS [network] s1::net::0" in out
    assert "FAIL [url_equals] a1" in out
    assert "SKIP [natural_language] a2" in out


def test_console_shows_detail_on_failure():
    out = Reporter().render_console(_results())
    assert "expected '/dashboard', got '/login'" in out


def test_empty_results_produces_clear_message():
    out = Reporter().render_console([])
    assert "no assertions evaluated" in out


def test_json_payload_has_stable_shape(tmp_path: Path):
    path = Reporter().render_json(_results(), tmp_path / "report.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["summary"] == {"total": 3, "passed": 1, "failed": 1, "skipped": 1}
    assert {a["id"] for a in raw["assertions"]} == {"s1::net::0", "a1", "a2"}


def test_json_creates_parent_directory(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "report.json"
    Reporter().render_json(_results(), nested)
    assert nested.exists()

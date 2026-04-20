"""Tests for agentchanti.testing.trace — raw trace JSONL read/write."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentchanti.testing.trace import (
    INTERACTION,
    NAVIGATE,
    NETWORK,
    SESSION_END,
    SESSION_START,
    ElementContext,
    TraceWriter,
    read_trace,
)


def test_roundtrip_captures_full_session(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write_session_start(start_url="http://localhost:3000",
                              viewport={"width": 1280, "height": 720})
        w.write_navigate(url="/products/42", status=200)
        w.write_interaction(
            action="click",
            selector_used="button.add-to-cart",
            element=ElementContext(
                tag="button", text="Add to cart", data_testid="add-to-cart",
                classes=["add-to-cart", "btn-primary"],
            ),
        )
        w.write_network(
            request_id="r1", method="POST", url="/api/cart/items",
            status=201, request_body={"product_id": 42},
            response_body={"id": "c1", "product_id": 42, "qty": 1},
            duration_ms=42,
        )
        w.write_session_end(reason="user_stopped")

    events = list(read_trace(path))
    assert [e["type"] for e in events] == [
        SESSION_START, NAVIGATE, INTERACTION, NETWORK, SESSION_END,
    ]
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
    assert events[2]["element"]["data_testid"] == "add-to-cart"
    assert events[3]["response_body"]["qty"] == 1


def test_context_manager_emits_session_end_on_exception(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with pytest.raises(RuntimeError):
        with TraceWriter(path) as w:
            w.write_session_start(start_url="/")
            raise RuntimeError("boom")

    events = list(read_trace(path))
    assert events[-1]["type"] == SESSION_END
    assert events[-1]["reason"] == "error"


def test_reader_tolerates_truncated_final_line(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write_session_start(start_url="/")
        w.write_navigate(url="/home")
        w.write_session_end()

    # Append a partial line, simulating a crash mid-write
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 99, "type": "interaction", "action":')

    events = list(read_trace(path))
    # Three complete events survive; the partial line is dropped silently.
    assert len(events) == 3
    assert events[-1]["type"] == SESSION_END


def test_writer_flushes_per_event(tmp_path: Path):
    """Each write should be visible on disk immediately — crash-resilience."""
    path = tmp_path / "trace.jsonl"
    w = TraceWriter(path)
    w.__enter__()
    try:
        w.write_session_start(start_url="/")
        # Read via a separate handle before the writer closes
        mid_run = path.read_text(encoding="utf-8")
        assert mid_run.count("\n") == 1
        w.write_navigate(url="/home")
        mid_run2 = path.read_text(encoding="utf-8")
        assert mid_run2.count("\n") == 2
    finally:
        w.__exit__(None, None, None)


def test_interaction_without_value_omits_key(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write_interaction(
            action="click", selector_used="#b",
            element=ElementContext(tag="button"),
        )
    events = list(read_trace(path))
    interaction = next(e for e in events if e["type"] == INTERACTION)
    assert "value" not in interaction


def test_interaction_with_value_preserves_it(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write_interaction(
            action="fill", selector_used="input[name=email]",
            element=ElementContext(tag="input"),
            value="user@example.com",
        )
    events = list(read_trace(path))
    assert events[0]["value"] == "user@example.com"


def test_reader_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"seq": 1, "type": SESSION_START, "start_url": "/"}) + "\n"
        "\n"
        + json.dumps({"seq": 2, "type": SESSION_END, "reason": "ok"}) + "\n",
        encoding="utf-8",
    )
    events = list(read_trace(path))
    assert len(events) == 2


def test_reader_skips_unknown_event_types(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"seq": 1, "type": "never_heard_of_this"}) + "\n"
        + json.dumps({"seq": 2, "type": SESSION_START, "start_url": "/"}) + "\n",
        encoding="utf-8",
    )
    events = list(read_trace(path))
    assert len(events) == 1
    assert events[0]["type"] == SESSION_START


def test_writer_rejects_use_outside_context_manager(tmp_path: Path):
    w = TraceWriter(tmp_path / "trace.jsonl")
    with pytest.raises(RuntimeError, match="outside of a `with` block"):
        w.write_session_start(start_url="/")

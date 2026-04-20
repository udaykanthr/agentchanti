"""Tests for agentchanti.testing.recorder.

Uses a FakeMCPClient — exercising the real BrowserMCPClient requires a
running Playwright MCP server, which isn't available in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentchanti.testing.mcp_client import ActionResult, NetworkEvent
from agentchanti.testing.recorder import Recorder
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


class FakeMCPClient:
    """Minimal BrowserMCPClient substitute that records every call."""

    def __init__(self, navigate_result: ActionResult | None = None):
        self.navigate_result = navigate_result or ActionResult(success=True)
        self.calls: list[tuple[str, tuple, dict]] = []
        self.entered = False
        self.exited = False

    def __enter__(self) -> FakeMCPClient:
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def navigate(self, url: str) -> ActionResult:
        self.calls.append(("navigate", (url,), {}))
        return self.navigate_result


def _make_recorder(tmp_path: Path, mcp: FakeMCPClient | None = None) -> Recorder:
    return Recorder(
        mcp_client=mcp or FakeMCPClient(),
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
    )


def test_context_manager_opens_and_closes_both_collaborators(tmp_path: Path):
    mcp = FakeMCPClient()
    rec = _make_recorder(tmp_path, mcp)
    with rec:
        assert mcp.entered is True
    assert mcp.exited is True


def test_start_writes_session_start_and_navigate(tmp_path: Path):
    mcp = FakeMCPClient()
    with _make_recorder(tmp_path, mcp) as rec:
        rec.start("http://localhost:3000/login",
                  viewport={"width": 1280, "height": 720})
        rec.stop()
    assert mcp.calls == [("navigate", ("http://localhost:3000/login",), {})]
    events = list(read_trace(tmp_path / "trace.jsonl"))
    assert [e["type"] for e in events[:2]] == [SESSION_START, NAVIGATE]
    assert events[0]["viewport"] == {"width": 1280, "height": 720}


def test_start_captures_network_events_from_initial_nav(tmp_path: Path):
    mcp = FakeMCPClient(
        navigate_result=ActionResult(
            success=True,
            network_events=[
                NetworkEvent(method="GET", url="/api/session", status=200,
                             response_body={"user_id": 42}),
            ],
        )
    )
    with _make_recorder(tmp_path, mcp) as rec:
        rec.start("/home")
        rec.stop()
    events = list(read_trace(tmp_path / "trace.jsonl"))
    nets = [e for e in events if e["type"] == NETWORK]
    assert len(nets) == 1
    assert nets[0]["url"] == "/api/session"
    assert nets[0]["response_body"] == {"user_id": 42}


def test_record_interaction_writes_event(tmp_path: Path):
    with _make_recorder(tmp_path) as rec:
        rec.start("/")
        rec.record_interaction(
            action="click", selector_used="#btn",
            element=ElementContext(tag="button", text="Go"),
        )
        rec.stop()
    events = list(read_trace(tmp_path / "trace.jsonl"))
    interactions = [e for e in events if e["type"] == INTERACTION]
    assert interactions[0]["action"] == "click"
    assert interactions[0]["element"]["text"] == "Go"


def test_record_network_synthesizes_request_id_when_missing(tmp_path: Path):
    class MinimalEvent:
        method = "POST"
        url = "/api/x"
        status = 201
        # no request_id attr
    with _make_recorder(tmp_path) as rec:
        rec.start("/")
        rec.record_network(MinimalEvent())
        rec.stop()
    events = list(read_trace(tmp_path / "trace.jsonl"))
    net = next(e for e in events if e["type"] == NETWORK)
    assert net["request_id"].startswith("auto-")


def test_double_start_raises(tmp_path: Path):
    with _make_recorder(tmp_path) as rec:
        rec.start("/")
        with pytest.raises(RuntimeError, match="called twice"):
            rec.start("/again")
        rec.stop()


def test_exception_during_recording_still_writes_session_end(tmp_path: Path):
    with pytest.raises(ValueError):
        with _make_recorder(tmp_path) as rec:
            rec.start("/")
            raise ValueError("boom")
    events = list(read_trace(tmp_path / "trace.jsonl"))
    assert events[-1]["type"] == SESSION_END
    # TraceWriter.__exit__ records reason="error" when an exception propagates
    assert events[-1]["reason"] == "error"


def test_from_url_builds_concrete_collaborators(tmp_path: Path):
    rec = Recorder.from_url("http://localhost:8931", tmp_path / "t.jsonl")
    # Constructor must not have tried to connect to MCP yet
    from agentchanti.testing.mcp_client import BrowserMCPClient
    assert isinstance(rec.mcp, BrowserMCPClient)
    assert rec.mcp.server_url == "http://localhost:8931"
    assert rec.writer.path == tmp_path / "t.jsonl"


def test_subscribe_to_live_events_is_stubbed(tmp_path: Path):
    with _make_recorder(tmp_path) as rec:
        rec.start("/")
        with pytest.raises(NotImplementedError, match="Playwright MCP"):
            rec.subscribe_to_live_events()
        rec.stop()

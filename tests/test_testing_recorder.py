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
    """Minimal BrowserMCPClient substitute that records every call.

    ``_call_tool`` is the same private method the real client exposes —
    we need it because the live recorder loop talks directly to it for
    inject + drain via ``browser_evaluate``. Tests can pre-program a
    queue of evaluate response strings to simulate the live JS buffer.
    """

    def __init__(self, navigate_result: ActionResult | None = None):
        self.navigate_result = navigate_result or ActionResult(success=True)
        self.calls: list[tuple[str, tuple, dict]] = []
        self.entered = False
        self.exited = False
        self.evaluate_responses: list[str] = []

    def __enter__(self) -> FakeMCPClient:
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def navigate(self, url: str) -> ActionResult:
        self.calls.append(("navigate", (url,), {}))
        return self.navigate_result

    def _call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append(("_call_tool", (name,), arguments))
        if name != "browser_evaluate":
            return ""
        if self.evaluate_responses:
            return self.evaluate_responses.pop(0)
        # Default: nothing buffered, page hasn't navigated.
        return (
            '### Result\n{"events": [], "url": "about:blank", "missing": false}\n'
            '### Ran Playwright code\n```js\n```'
        )


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


def test_subscribe_drains_buffered_js_events_into_trace(tmp_path: Path):
    """Live polling path with a fake MCP — verifies the wiring between
    JS buffer payload and trace.write_interaction without needing a live
    Playwright MCP server."""
    mcp = FakeMCPClient()
    # Pre-program three drain responses: install (any reply), one with
    # buffered events, then steady-state empties.
    buffered = (
        '### Result\n'
        '{"events": ['
        '{"type":"click","clientX":42,"clientY":99,"button":0,'
        '"element":{"tag":"button","data_testid":"go","text":"Go",'
        '"id":null,"classes":[],"aria_label":null,"role":"button","name":null}}'
        '], "url": "/", "missing": false}\n'
        '### Ran Playwright code\n```js\n```'
    )
    mcp.evaluate_responses = ['### Result\n"installed"\n### Ran Playwright code\n```js\n```',
                              buffered]
    with _make_recorder(tmp_path, mcp) as rec:
        rec.start("/")
        rec.subscribe_to_live_events(poll_interval_s=0.05)
        # Give the polling thread a moment to consume the buffered drain.
        import time as _t
        _t.sleep(0.25)
        rec.stop()
    events = list(read_trace(tmp_path / "trace.jsonl"))
    interactions = [e for e in events if e["type"] == INTERACTION]
    assert len(interactions) == 1
    assert interactions[0]["action"] == "click"
    assert interactions[0]["selector_used"] == "[data-testid=go]"
    assert interactions[0]["coord"] == {"x": 42, "y": 99}


def test_subscribe_emits_navigate_event_on_url_change(tmp_path: Path):
    """A drain that reports a different URL than the last one should
    produce a navigate event so the trace records the page change."""

    class StickyURLMCP(FakeMCPClient):
        # Override default drain so once the test programs /page-2, every
        # subsequent steady-state poll keeps reporting /page-2.
        current_url = "/page-2"

        def _call_tool(self, name, arguments):
            self.calls.append(("_call_tool", (name,), arguments))
            if name != "browser_evaluate":
                return ""
            if self.evaluate_responses:
                return self.evaluate_responses.pop(0)
            return (
                f'### Result\n{{"events": [], "url": "{self.current_url}",'
                f' "missing": false}}\n'
                '### Ran Playwright code\n```js\n```'
            )

    mcp = StickyURLMCP()
    mcp.evaluate_responses = [
        '### Result\n"installed"\n### Ran Playwright code\n```js\n```',
    ]
    with _make_recorder(tmp_path, mcp) as rec:
        rec.start("/page-1")
        rec.subscribe_to_live_events(poll_interval_s=0.05)
        import time as _t
        _t.sleep(0.25)
        rec.stop()
    nav_urls = [e["url"] for e in read_trace(tmp_path / "trace.jsonl")
                if e["type"] == NAVIGATE]
    # /page-1 from start() and ONE /page-2 from URL-change detection.
    # Subsequent polls also see /page-2 but must not duplicate.
    assert nav_urls == ["/page-1", "/page-2"]


def test_subscribe_is_idempotent(tmp_path: Path):
    """Calling subscribe twice must not start two polling threads."""
    with _make_recorder(tmp_path) as rec:
        rec.start("/")
        rec.subscribe_to_live_events(poll_interval_s=1.0)
        first_thread = rec._live_thread
        rec.subscribe_to_live_events(poll_interval_s=1.0)
        assert rec._live_thread is first_thread
        rec.stop()

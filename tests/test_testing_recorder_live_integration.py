"""
Integration test for the live Recorder.

Loads a small data-URL page, starts the live recorder, then drives clicks/
fills/selects/key presses *via* BrowserMCPClient on the same session.
Playwright dispatches real synthetic UI events with proper clientX/clientY,
which our injected JS hook captures into the buffer; the polling loop
then drains them into the trace.

Opt-in: ``pytest -m integration`` (deselected by default ``addopts``).
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from urllib.parse import quote

import pytest

from agentchanti.testing.mcp_client import (
    DEFAULT_MCP_SERVER_URL,
    BrowserMCPClient,
)
from agentchanti.testing.recorder import Recorder
from agentchanti.testing.trace import (
    INTERACTION,
    NAVIGATE,
    SESSION_END,
    SESSION_START,
    TraceWriter,
    read_trace,
)

pytestmark = pytest.mark.integration


PAGE_HTML = """
<button data-testid="go-btn" onclick="this.textContent='CLICKED';">Click me</button>
<input id="email" data-testid="email-input" type="text">
<input id="pw" type="password">
<select id="country" data-testid="country-select">
  <option value="us">US</option>
  <option value="uk">UK</option>
</select>
"""


def _server_reachable(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture
def trace_path(tmp_path: Path) -> Path:
    return tmp_path / "live-trace.jsonl"


def _make_recorder(trace_path: Path) -> Recorder:
    if not _server_reachable(DEFAULT_MCP_SERVER_URL):
        pytest.skip(f"Playwright MCP not reachable at {DEFAULT_MCP_SERVER_URL}")
    return Recorder(
        mcp_client=BrowserMCPClient(DEFAULT_MCP_SERVER_URL),
        trace_writer=TraceWriter(trace_path),
    )


def test_live_records_click_with_coords_and_metadata(trace_path: Path):
    rec = _make_recorder(trace_path)
    with rec:
        rec.start("data:text/html," + quote(PAGE_HTML),
                  viewport={"width": 1280, "height": 720})
        rec.subscribe_to_live_events(poll_interval_s=0.1)
        time.sleep(0.3)  # allow install to settle

        rec.mcp.click("text=Click me")
        time.sleep(0.5)  # let the polling loop drain

        rec.stop()

    events = list(read_trace(trace_path))
    interactions = [e for e in events if e["type"] == INTERACTION]
    clicks = [e for e in interactions if e["action"] == "click"]
    assert clicks, f"no click recorded; got events {[e['type'] for e in events]}"

    click = clicks[0]
    assert click["element"]["tag"] == "button"
    assert click["element"]["data_testid"] == "go-btn"
    # Synthesized selector via translator priority: data-testid wins
    assert click["selector_used"] == "[data-testid=go-btn]"
    # Coordinates should be present (Playwright synthetic events fill them)
    assert "coord" in click and "x" in click["coord"] and "y" in click["coord"]


def test_live_records_input_value(trace_path: Path):
    rec = _make_recorder(trace_path)
    with rec:
        rec.start("data:text/html," + quote(PAGE_HTML))
        rec.subscribe_to_live_events(poll_interval_s=0.1)
        time.sleep(0.3)

        rec.mcp.fill("#email", "user@example.com")
        time.sleep(0.5)

        rec.stop()

    fills = [e for e in read_trace(trace_path)
             if e["type"] == INTERACTION and e["action"] == "fill"]
    assert fills, "no fill recorded"
    last = fills[-1]
    assert last["element"]["id"] == "email"
    assert last["value"] == "user@example.com"


def test_live_redacts_password_input(trace_path: Path):
    rec = _make_recorder(trace_path)
    with rec:
        rec.start("data:text/html," + quote(PAGE_HTML))
        rec.subscribe_to_live_events(poll_interval_s=0.1)
        time.sleep(0.3)

        rec.mcp.fill("#pw", "supersecret")
        time.sleep(0.5)

        rec.stop()

    fills = [e for e in read_trace(trace_path)
             if e["type"] == INTERACTION and e["action"] == "fill"]
    pw_events = [e for e in fills if e["element"].get("id") == "pw"]
    assert pw_events, "no password input recorded"
    # Plaintext must NOT appear anywhere — that's the whole point of the
    # redaction. Final value should be the marker.
    for ev in pw_events:
        assert "supersecret" not in (ev.get("value") or "")
    assert pw_events[-1]["value"] == "***REDACTED***"


def test_live_records_select_change(trace_path: Path):
    rec = _make_recorder(trace_path)
    with rec:
        rec.start("data:text/html," + quote(PAGE_HTML))
        rec.subscribe_to_live_events(poll_interval_s=0.1)
        time.sleep(0.3)

        rec.mcp.select("[data-testid=country-select]", "uk")
        time.sleep(0.5)

        rec.stop()

    selects = [e for e in read_trace(trace_path)
               if e["type"] == INTERACTION and e["action"] == "select"]
    assert selects, "no select event recorded"
    assert selects[-1]["value"] == "uk"


def test_live_session_emits_start_and_end(trace_path: Path):
    rec = _make_recorder(trace_path)
    with rec:
        rec.start("data:text/html," + quote(PAGE_HTML),
                  viewport={"width": 1024, "height": 768})
        rec.subscribe_to_live_events(poll_interval_s=0.1)
        time.sleep(0.2)
        rec.stop()

    events = list(read_trace(trace_path))
    assert events[0]["type"] == SESSION_START
    assert events[0]["viewport"] == {"width": 1024, "height": 768}
    assert events[1]["type"] == NAVIGATE
    assert events[-1]["type"] == SESSION_END

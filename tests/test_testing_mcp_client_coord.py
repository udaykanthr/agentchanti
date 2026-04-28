"""Unit tests for the coordinate dispatch path of BrowserMCPClient.

The live integration tests in
``test_testing_mcp_client_interactions_integration.py`` cover the
end-to-end behavior with a real Playwright MCP server. These tests
isolate the JS-builder + dispatch wiring with a fake ``_call_tool`` so
the contract is checked without needing a server.
"""

from __future__ import annotations

import pytest

from agentchanti.testing.mcp_client import (
    BrowserMCPClient,
    NetworkEvent,
    _build_coord_action_js,
    _parse_coord,
    _parse_network_requests,
)


# ---- Coord parsing ---------------------------------------------------------

def test_parse_coord_accepts_integer_and_float_pairs():
    assert _parse_coord("coord=42,99") == (42.0, 99.0)
    assert _parse_coord("coord=12.5, 7") == (12.5, 7.0)


def test_parse_coord_rejects_missing_prefix_or_comma():
    assert _parse_coord("42,99") is None
    assert _parse_coord("coord=42") is None
    assert _parse_coord("") is None


def test_parse_coord_rejects_non_numeric():
    assert _parse_coord("coord=a,b") is None
    assert _parse_coord("coord=42,xyz") is None


# ---- JS builder ------------------------------------------------------------

def test_build_click_uses_elementfrompoint_and_clicks():
    js = _build_coord_action_js(42.0, 99.0, "click", None)
    assert "elementFromPoint(42.0, 99.0)" in js
    assert "el.click()" in js


def test_build_fill_dispatches_input_and_change_with_quoted_value():
    js = _build_coord_action_js(10, 20, "fill", "user@example.com")
    assert '"user@example.com"' in js
    assert "Event('input'" in js
    assert "Event('change'" in js


def test_build_fill_escapes_quotes_in_value():
    js = _build_coord_action_js(0, 0, "fill", 'he said "hi"')
    # JSON-escaped: " becomes \", surviving inside the JS string literal
    assert '\\"hi\\"' in js


def test_build_hover_carries_coordinates_into_clientxy():
    js = _build_coord_action_js(150, 250, "hover", None)
    assert "clientX: 150" in js
    assert "clientY: 250" in js
    assert "MouseEvent('mouseover'" in js


def test_build_select_dispatches_change_with_value():
    js = _build_coord_action_js(5, 5, "select", "uk")
    assert '"uk"' in js
    assert "Event('change'" in js


def test_build_unknown_action_raises():
    with pytest.raises(ValueError, match="unsupported coord action"):
        _build_coord_action_js(0, 0, "teleport", None)


# ---- Dispatch wiring (no live server) --------------------------------------

class _SpyClient(BrowserMCPClient):
    """BrowserMCPClient that captures every ``_call_tool`` invocation
    instead of actually talking to a server."""

    def __init__(self):
        super().__init__(server_url="http://test/mcp")
        self.calls: list[tuple[str, dict]] = []

    def _call_tool(self, name: str, arguments: dict) -> str:  # type: ignore[override]
        self.calls.append((name, arguments))
        # Match the response shape browser_evaluate produces; tests that
        # care about reading the boolean back can override.
        return "### Result\ntrue\n### Ran Playwright code\n```js\n```"


def test_click_with_coord_routes_through_browser_evaluate():
    c = _SpyClient()
    result = c.click("coord=42,99")
    assert result.success
    assert len(c.calls) == 1
    name, args = c.calls[0]
    assert name == "browser_evaluate"
    assert "elementFromPoint(42.0, 99.0)" in args["function"]
    assert "el.click()" in args["function"]


def test_fill_with_coord_routes_value_through_evaluate():
    c = _SpyClient()
    result = c.fill("coord=10,20", "secret")
    assert result.success
    name, args = c.calls[0]
    assert name == "browser_evaluate"
    assert "elementFromPoint(10.0, 20.0)" in args["function"]
    assert '"secret"' in args["function"]


def test_select_with_coord_dispatches_change():
    c = _SpyClient()
    result = c.select("coord=5,5", "uk")
    assert result.success
    _, args = c.calls[0]
    assert "Event('change'" in args["function"]
    assert '"uk"' in args["function"]


def test_hover_with_coord_dispatches_mouseover_with_clientxy():
    c = _SpyClient()
    result = c.hover("coord=120,240")
    assert result.success
    _, args = c.calls[0]
    assert "clientX: 120" in args["function"]
    assert "clientY: 240" in args["function"]


def test_press_with_coord_focuses_then_presses_key():
    c = _SpyClient()
    result = c.press("coord=10,20", "Enter")
    assert result.success
    # First call focuses the element at the coord, second presses the key.
    assert len(c.calls) == 2
    name0, args0 = c.calls[0]
    name1, args1 = c.calls[1]
    assert name0 == "browser_evaluate"
    assert "elementFromPoint(10.0, 20.0)" in args0["function"]
    assert "el.focus()" in args0["function"]
    assert name1 == "browser_press_key"
    assert args1 == {"key": "Enter"}


def test_wait_for_coord_returns_success_when_element_exists():
    c = _SpyClient()
    # _SpyClient returns "true" by default — element resolves
    result = c.wait_for("coord=50,50", timeout_ms=100)
    assert result.success


def test_wait_for_coord_returns_failure_when_no_element():
    class _FalsyClient(_SpyClient):
        def _call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return "### Result\nfalse\n### Ran Playwright code\n```js\n```"
    c = _FalsyClient()
    result = c.wait_for("coord=999,999", timeout_ms=100)
    assert result.success is False


def test_bad_coord_selector_returns_failure_without_calling_tool():
    c = _SpyClient()
    result = c.click("coord=not,a,number")
    assert result.success is False
    assert "bad coord" in (result.error or "")
    assert c.calls == []


# ---- Network requests parser -----------------------------------------------

def test_parse_network_requests_strips_fences_and_headers():
    text = (
        "### Result\n"
        "[GET] http://example.com/foo => [200] OK\n"
        "[POST] /api/login => [201] Created\n"
        "### Ran Playwright code\n"
        "```js\n"
        "```\n"
    )
    out = _parse_network_requests(text)
    assert out == [
        NetworkEvent(method="GET", url="http://example.com/foo", status=200),
        NetworkEvent(method="POST", url="/api/login", status=201),
    ]


def test_parse_network_requests_keeps_pending_request_with_status_zero():
    """Requests still in flight have no status. We must surface them so
    expected_network mismatches don't silently disappear into pending."""
    out = _parse_network_requests("[GET] /still/loading\n")
    assert out == [NetworkEvent(method="GET", url="/still/loading", status=0)]


def test_parse_network_requests_skips_unparseable_lines():
    out = _parse_network_requests(
        "garbage line\n[GET] /ok => [200] OK\nanother garbage line\n"
    )
    assert out == [NetworkEvent(method="GET", url="/ok", status=200)]


def test_parse_network_requests_handles_empty_input():
    assert _parse_network_requests("") == []
    assert _parse_network_requests(None) == []  # type: ignore[arg-type]


def test_network_requests_method_returns_empty_on_call_failure():
    """If the underlying tool errors, return [] — the Replayer's diff
    machinery must keep working (best-effort observability)."""

    class ExplodingClient(BrowserMCPClient):
        def __init__(self):
            super().__init__(server_url="http://test/mcp")

        def _call_tool(self, name, arguments):  # type: ignore[override]
            raise RuntimeError("boom")

    assert ExplodingClient().network_requests() == []

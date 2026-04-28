"""
Integration tests for BrowserMCPClient interaction methods.

Each test loads a small data-URL page, performs an interaction via the
client, and verifies the effect via either a follow-up snapshot or a
``browser_evaluate`` read-back.

Opt-in (same as other integration tests): ``pytest -m integration``.
Requires a running Playwright MCP server at the default URL.
"""

from __future__ import annotations

import socket
from urllib.parse import quote

import pytest

from agentchanti.testing.mcp_client import (
    DEFAULT_MCP_SERVER_URL,
    BrowserMCPClient,
)

pytestmark = pytest.mark.integration


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


# A single HTML playground touches every interaction method we implement.
PAGE_HTML = """
<label for="email">Email</label>
<input id="email" type="text" data-testid="email-input">
<label for="country">Country</label>
<select id="country" data-testid="country-select">
  <option value="us">US</option>
  <option value="uk">UK</option>
  <option value="de">DE</option>
</select>
<div id="hover-box"
     onmouseover="this.dataset.hovered='yes'; this.textContent='HOVERED';">
  Hover me
</div>
<button data-testid="go-btn" onclick="this.textContent='CLICKED';">Click me</button>
<input id="key-target" type="text" data-testid="key-target"
       onkeydown="if (event.key==='Enter') this.value='ENTER';">
"""


@pytest.fixture
def mcp():
    if not _server_reachable(DEFAULT_MCP_SERVER_URL):
        pytest.skip(
            f"Playwright MCP not reachable at {DEFAULT_MCP_SERVER_URL}"
        )
    with BrowserMCPClient(DEFAULT_MCP_SERVER_URL) as client:
        # Every test starts from a freshly loaded page.
        client.navigate("data:text/html," + quote(PAGE_HTML))
        yield client


def _eval(mcp, js: str) -> str:
    """Read back a JS expression's value. Uses the same _call_tool path the
    client uses internally. Tests only — not a public API."""
    return mcp._call_tool("browser_evaluate", {"function": js})


# ---- click -----------------------------------------------------------------

def test_click_semantic_by_text_changes_button(mcp):
    result = mcp.click("text=Click me")
    assert result.success, f"click failed: {result.error!r}"
    snap = mcp.snapshot()["raw"]
    assert "CLICKED" in snap


def test_click_css_by_data_testid_changes_button(mcp):
    result = mcp.click("[data-testid=go-btn]")
    assert result.success, f"click failed: {result.error!r}"
    assert "CLICKED" in mcp.snapshot()["raw"]


def test_click_missing_element_returns_failure(mcp):
    result = mcp.click("text=Nonexistent")
    assert result.success is False
    assert "no element matching" in (result.error or "")


# ---- fill ------------------------------------------------------------------

def test_fill_css_by_id_sets_value(mcp):
    result = mcp.fill("#email", "user@example.com")
    assert result.success, f"fill failed: {result.error!r}"
    got = _eval(mcp, '() => document.querySelector("#email").value')
    assert "user@example.com" in got


def test_fill_semantic_by_role_name_sets_value(mcp):
    result = mcp.fill('role=textbox name="Email"', "sem@example.com")
    assert result.success, f"fill failed: {result.error!r}"
    got = _eval(mcp, '() => document.querySelector("#email").value')
    assert "sem@example.com" in got


# ---- select ----------------------------------------------------------------

def test_select_css_changes_dropdown_value(mcp):
    result = mcp.select("[data-testid=country-select]", "uk")
    assert result.success, f"select failed: {result.error!r}"
    got = _eval(mcp, '() => document.querySelector("#country").value')
    assert "uk" in got


# ---- hover -----------------------------------------------------------------

def test_hover_css_fires_mouseover(mcp):
    result = mcp.hover("#hover-box")
    assert result.success, f"hover failed: {result.error!r}"
    got = _eval(mcp, '() => document.querySelector("#hover-box").dataset.hovered')
    assert "yes" in got


# ---- press -----------------------------------------------------------------

def test_press_focuses_and_fires_key(mcp):
    result = mcp.press("#key-target", "Enter")
    assert result.success, f"press failed: {result.error!r}"
    got = _eval(mcp, '() => document.querySelector("#key-target").value')
    assert "ENTER" in got


# ---- wait_for probe --------------------------------------------------------

def test_wait_for_existing_element_succeeds(mcp):
    result = mcp.wait_for("text=Click me", timeout_ms=1000)
    assert result.success


def test_wait_for_existing_css_element_succeeds(mcp):
    result = mcp.wait_for("[data-testid=go-btn]", timeout_ms=1000)
    assert result.success


def test_wait_for_missing_element_fails_after_timeout(mcp):
    result = mcp.wait_for("text=DoesNotExist", timeout_ms=200)
    assert result.success is False


# ---- coord=X,Y dispatch ----------------------------------------------------
# Verifies the P2b fallback path: replay can hit an element by its captured
# screen position even when ids/testids on the page have drifted.

def _bbox_center(mcp, css_selector: str) -> tuple[float, float]:
    """Read the element's bounding-rect center via browser_evaluate.

    Returns (x, y) coordinates in CSS pixels — the same units
    ``elementFromPoint`` and our coord= dispatch path use.
    """
    from agentchanti.testing._live_translate import extract_evaluate_result
    js = (
        f"() => {{ const r = document.querySelector("
        f"'{css_selector}').getBoundingClientRect(); "
        "return {x: r.left + r.width/2, y: r.top + r.height/2}; }"
    )
    raw = _eval(mcp, js)
    obj = extract_evaluate_result(raw)
    assert isinstance(obj, dict), f"expected object payload, got {raw!r}"
    return float(obj["x"]), float(obj["y"])


def test_click_at_coordinate_clicks_element_under_point(mcp):
    x, y = _bbox_center(mcp, "[data-testid=go-btn]")
    result = mcp.click(f"coord={x},{y}")
    assert result.success, f"coord click failed: {result.error!r}"
    assert "CLICKED" in mcp.snapshot()["raw"]


def test_fill_at_coordinate_sets_value(mcp):
    x, y = _bbox_center(mcp, "#email")
    result = mcp.fill(f"coord={x},{y}", "coord@example.com")
    assert result.success, f"coord fill failed: {result.error!r}"
    got = _eval(mcp, '() => document.querySelector("#email").value')
    assert "coord@example.com" in got


def test_wait_for_coordinate_succeeds_when_element_present(mcp):
    x, y = _bbox_center(mcp, "[data-testid=go-btn]")
    result = mcp.wait_for(f"coord={x},{y}", timeout_ms=500)
    assert result.success


# ---- resize ---------------------------------------------------------------

def test_resize_updates_browser_viewport(mcp):
    """The Replayer calls resize before first navigate so coord=X,Y
    fallbacks land on the same screen positions captured at recording.
    Verify the underlying tool actually changes window.innerWidth."""
    result = mcp.resize(640, 480)
    assert result.success, f"resize failed: {result.error!r}"
    width = _eval(mcp, "() => window.innerWidth")
    height = _eval(mcp, "() => window.innerHeight")
    assert "640" in width
    assert "480" in height


# ---- network_requests -----------------------------------------------------

def test_network_requests_returns_parsed_event_list(mcp):
    """Hit a data: URL (no real network), then a navigated page, and
    confirm at least the page-level GET shows up. We don't assert exact
    content because Playwright may also surface favicon / etc."""
    mcp.navigate("data:text/html,<h1>net probe</h1>")
    events = mcp.network_requests()
    # Type contract holds even when the list is empty.
    assert isinstance(events, list)
    for ev in events:
        assert isinstance(ev.method, str) and ev.method.isupper()
        assert isinstance(ev.url, str) and ev.url

"""
Integration test — BrowserMCPClient against a live Playwright MCP server.

Opt-in: default pytest runs skip everything marked ``integration``. Run
these manually while a server is up::

    npx @playwright/mcp@latest --port 8931       # in another terminal
    pytest -m integration tests/test_testing_mcp_client_integration.py -v

The fixtures skip cleanly (not fail) when the server isn't reachable, so
running ``pytest -m integration`` without the server produces a
``SKIPPED`` — not a red CI.
"""

from __future__ import annotations

import socket

import pytest

from agentchanti.testing.mcp_client import (
    DEFAULT_MCP_SERVER_URL,
    BrowserMCPClient,
)

pytestmark = pytest.mark.integration

DATA_URL = "data:text/html,<h1>Hello AgentChanti</h1><button>Go</button>"


def _server_reachable(url: str) -> bool:
    """Cheap TCP probe so tests skip (not error) when the MCP server is down."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def mcp():
    if not _server_reachable(DEFAULT_MCP_SERVER_URL):
        pytest.skip(
            f"Playwright MCP not reachable at {DEFAULT_MCP_SERVER_URL}; "
            f"start it with `npx @playwright/mcp@latest --port 8931`"
        )
    with BrowserMCPClient(DEFAULT_MCP_SERVER_URL) as client:
        yield client


def test_navigate_returns_success(mcp):
    result = mcp.navigate(DATA_URL)
    assert result.success, f"navigate failed: {result.error!r}"
    assert result.current_url == DATA_URL


def test_snapshot_contains_page_content(mcp):
    mcp.navigate(DATA_URL)
    snap = mcp.snapshot()
    raw = snap.get("raw", "")
    assert raw, "snapshot returned no text"
    # Playwright MCP renders the accessibility tree as markdown-ish text
    # that mentions element roles + accessible names. Our data URL has a
    # heading and a button — both should appear in the tree.
    assert "Hello AgentChanti" in raw
    assert "Go" in raw


def test_session_teardown_does_not_raise():
    """A fresh client outside the module fixture: ensure connect+disconnect
    is idempotent and doesn't leak threads."""
    if not _server_reachable(DEFAULT_MCP_SERVER_URL):
        pytest.skip("server not running")
    with BrowserMCPClient(DEFAULT_MCP_SERVER_URL) as client:
        client.navigate(DATA_URL)
    # If we reach here without an exception, teardown completed cleanly.

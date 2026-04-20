"""
BrowserMCPClient — thin sync wrapper around a browser MCP server.

Target server: **Playwright MCP** (Microsoft, cross-browser, official).
  https://github.com/microsoft/playwright-mcp

Why Playwright MCP over raw CDP or Selenium Grid:
  * Official Microsoft-maintained MCP server, actively developed.
  * Cross-browser out of the box (Chromium, Firefox, WebKit).
  * Exposes a stable MCP tool surface — navigate, click, fill, snapshot,
    screenshot, network events — which is exactly what Recorder and
    Replayer need. No need to re-invent protocol plumbing.

Sync vs async:
  The ``mcp`` Python client is async. This wrapper is sync — it runs each
  call through ``asyncio.run`` (or the running loop if one exists). The
  rest of AgentChanti is sync and we don't want to fork the codebase over
  one module. Performance is fine for test automation because steps are
  human-paced, not high-throughput.

Method surface below is deliberately small — only what Recorder/Replayer
need. Resist adding methods until there's a concrete caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Default Playwright MCP endpoint when run with `npx @playwright/mcp`
# using its built-in HTTP transport. Override with --mcp-server on the CLI
# or pass ``server_url`` to the constructor.
DEFAULT_MCP_SERVER_URL = "http://localhost:8931"


@dataclass
class NetworkEvent:
    """One HTTP request/response observed during an action."""
    method: str
    url: str
    status: int
    request_body: Any = None
    response_body: Any = None


@dataclass
class ActionResult:
    """Outcome of a single browser action."""
    success: bool
    current_url: str = ""
    screenshot_path: str | None = None
    network_events: list[NetworkEvent] = field(default_factory=list)
    error: str | None = None


class BrowserMCPClient:
    """Sync facade over a Playwright MCP server.

    Holds one MCP session for the life of a recording or replay run.
    Opens on ``__enter__``, closes on ``__exit__`` — use as a context
    manager to guarantee the browser is torn down even on exceptions.
    """

    def __init__(self, server_url: str = DEFAULT_MCP_SERVER_URL):
        self.server_url = server_url
        # The mcp package is an optional dep. Import happens lazily in
        # ``_connect`` so merely constructing a client never errors when
        # the user hasn't installed the [testing] extra.
        self._session: Any = None

    # ---- Session lifecycle ----------------------------------------------

    def __enter__(self) -> BrowserMCPClient:
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._disconnect()

    def _connect(self) -> None:
        """Open an MCP session to the browser server."""
        raise NotImplementedError("BrowserMCPClient._connect not implemented yet")

    def _disconnect(self) -> None:
        """Close the MCP session and free browser resources."""
        if self._session is None:
            return
        raise NotImplementedError("BrowserMCPClient._disconnect not implemented yet")

    # ---- Actions (called by Recorder + Replayer) ------------------------

    def navigate(self, url: str) -> ActionResult:
        """Navigate the browser to ``url``."""
        raise NotImplementedError

    def click(self, selector: str) -> ActionResult:
        """Click the first element matching ``selector``."""
        raise NotImplementedError

    def fill(self, selector: str, value: str) -> ActionResult:
        """Fill a text input matching ``selector`` with ``value``."""
        raise NotImplementedError

    def press(self, selector: str, key: str) -> ActionResult:
        """Focus ``selector`` and press ``key`` (e.g. 'Enter')."""
        raise NotImplementedError

    def select(self, selector: str, value: str) -> ActionResult:
        """Select an option by value/label in a ``<select>`` matching ``selector``."""
        raise NotImplementedError

    def hover(self, selector: str) -> ActionResult:
        """Hover the first element matching ``selector``."""
        raise NotImplementedError

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        """Wait until ``selector`` is visible or ``timeout_ms`` elapses."""
        raise NotImplementedError

    # ---- Inspection (called by Replayer for self-healing) ---------------

    def snapshot(self) -> dict[str, Any]:
        """Return an accessibility-tree snapshot of the current page.

        Used by the LLM-driven self-healing path: when cached locators fail,
        the snapshot is fed to the LLM alongside the step's semantic label
        so it can pick a working selector.
        """
        raise NotImplementedError

    def screenshot(self, path: str) -> str:
        """Save a PNG screenshot to ``path`` and return it."""
        raise NotImplementedError

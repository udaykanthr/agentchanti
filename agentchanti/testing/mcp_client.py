"""
BrowserMCPClient — thin sync wrapper around a browser MCP server.

Target server: **Playwright MCP** (Microsoft, cross-browser, official).
  https://github.com/microsoft/playwright-mcp

Why Playwright MCP over raw CDP or Selenium Grid:
  * Official Microsoft-maintained MCP server, actively developed.
  * Cross-browser out of the box (Chromium, Firefox, WebKit).
  * Exposes a stable MCP tool surface — navigate, click, type, snapshot,
    network events — which is exactly what Recorder and Replayer need.

Ref-based addressing (important architectural note)
---------------------------------------------------
Every interaction tool Playwright MCP exposes (browser_click,
browser_type, browser_hover, browser_select_option, browser_drag) takes
a ``ref`` — an opaque string ("e7", "s1e42") scoped to the most recent
``browser_snapshot``. Refs are ephemeral: they regenerate on every
snapshot.

The P1 transport implemented here covers only ``navigate`` + ``snapshot``
— enough to prove the round-trip works end-to-end. ``click`` / ``fill`` /
``hover`` / ``select`` / ``press`` / ``wait_for`` will land in P2 once
snapshot→ref resolution is wired.

Sync vs async
-------------
The ``mcp`` Python client is async-only. We run a dedicated asyncio event
loop on a background thread and submit each call via
``run_coroutine_threadsafe``. This keeps the public API synchronous so
the rest of AgentChanti — and user code driving the Recorder — stays
sync without a library-wide rewrite. Performance is fine because test
steps are human-paced, not high-throughput.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

# Default Playwright MCP endpoint. Note the `/mcp` suffix — that's the
# streamable HTTP transport path. The bare `http://localhost:8931` root
# responds 404. Override with --mcp-server on the CLI or pass
# ``server_url`` to the constructor.
DEFAULT_MCP_SERVER_URL = "http://localhost:8931/mcp"

_CALL_TIMEOUT_S = 60.0
_CONNECT_TIMEOUT_S = 30.0


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
    # Raw text payload from the MCP tool call. Useful for snapshot() where
    # the accessibility tree arrives as a markdown-ish string; callers that
    # want structured access can parse it. Kept loose on purpose — we don't
    # want a Playwright-version-specific parser baked in here.
    raw: str | None = None


class BrowserMCPClient:
    """Sync facade over a Playwright MCP server.

    Holds one MCP session for the life of a recording or replay run.
    Use as a context manager so the browser is torn down even on errors.
    """

    def __init__(self, server_url: str = DEFAULT_MCP_SERVER_URL):
        self.server_url = server_url
        self._engine: _AsyncEngine | None = None
        self._session: Any = None
        self._client_cm: Any = None   # the streamablehttp_client async CM

    # ---- Session lifecycle ----------------------------------------------

    def __enter__(self) -> BrowserMCPClient:
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._disconnect()

    def _connect(self) -> None:
        # Lazy import — mcp is an optional dep. Users without the
        # [testing] extra can still import agentchanti.testing.* safely.
        from mcp import ClientSession
        # Prefer the non-deprecated name when available (mcp>=1.26); fall
        # back to the older alias for slightly older SDK installs.
        try:
            from mcp.client.streamable_http import (  # type: ignore
                streamable_http_client as _http_client,
            )
        except ImportError:
            from mcp.client.streamable_http import (  # type: ignore
                streamablehttp_client as _http_client,
            )

        self._engine = _AsyncEngine()
        self._engine.start()

        async def _open() -> None:
            self._client_cm = _http_client(self.server_url)
            read, write, _get_sid = await self._client_cm.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()

        self._engine.run(_open(), timeout=_CONNECT_TIMEOUT_S)

    def _disconnect(self) -> None:
        if self._engine is None:
            return

        async def _close() -> None:
            if self._session is not None:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception:
                    pass
            if self._client_cm is not None:
                try:
                    await self._client_cm.__aexit__(None, None, None)
                except Exception:
                    pass

        try:
            self._engine.run(_close(), timeout=_CONNECT_TIMEOUT_S)
        finally:
            self._engine.stop()
            self._engine = None
            self._session = None
            self._client_cm = None

    # ---- Actions (P1: navigate + snapshot only) --------------------------

    def navigate(self, url: str) -> ActionResult:
        """Navigate the browser to ``url``."""
        try:
            text = self._call_tool("browser_navigate", {"url": url})
        except Exception as e:
            return ActionResult(success=False, error=f"navigate failed: {e}")
        return ActionResult(success=True, current_url=url, raw=text)

    def snapshot(self) -> dict[str, Any]:
        """Return the current accessibility snapshot.

        Playwright MCP returns the snapshot as markdown-formatted text
        (the "YAML-ish" accessibility tree). We wrap it in a dict with a
        ``raw`` key so callers can evolve toward structured parsing
        without a breaking signature change.
        """
        try:
            text = self._call_tool("browser_snapshot", {})
        except Exception:
            return {}
        return {"raw": text or ""}

    # ---- P2 scope: resolve ref then dispatch -----------------------------
    # These stay NotImplementedError until snapshot->ref resolution is
    # wired. The fakes in tests still satisfy Replayer's contract; a live
    # replay against Playwright MCP will exercise the implementations below
    # when they land.

    def click(self, selector: str) -> ActionResult:
        raise NotImplementedError("click lands in P2 (snapshot->ref resolution)")

    def fill(self, selector: str, value: str) -> ActionResult:
        raise NotImplementedError("fill lands in P2")

    def press(self, selector: str, key: str) -> ActionResult:
        raise NotImplementedError("press lands in P2")

    def select(self, selector: str, value: str) -> ActionResult:
        raise NotImplementedError("select lands in P2")

    def hover(self, selector: str) -> ActionResult:
        raise NotImplementedError("hover lands in P2")

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        # Replayer uses this as a "does this selector match right now?"
        # probe. Playwright MCP's browser_wait_for is text/time-based, so
        # the real implementation is: take a snapshot and query it in
        # memory. Lands alongside the ref-resolution work in P2.
        raise NotImplementedError("wait_for probe lands in P2 (via snapshot)")

    def screenshot(self, path: str) -> str:
        try:
            self._call_tool("browser_take_screenshot", {"filename": path})
        except Exception as e:
            raise RuntimeError(f"screenshot failed: {e}") from e
        return path

    # ---- Internals -------------------------------------------------------

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke an MCP tool and return the concatenated text content."""
        if self._session is None or self._engine is None:
            raise RuntimeError("BrowserMCPClient is not connected")

        async def _do() -> str:
            result = await self._session.call_tool(name, arguments)
            # Each result has a list of content parts; we concatenate the
            # text of any TextContent parts. Images/resources ignored here.
            parts: list[str] = []
            for c in result.content or []:
                text = getattr(c, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)

        return self._engine.run(_do(), timeout=_CALL_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Background-thread asyncio engine
# ---------------------------------------------------------------------------

class _AsyncEngine:
    """Runs an asyncio loop on a daemon thread so a sync caller can submit
    coroutines via ``run``. The engine owns its loop end-to-end — callers
    don't touch asyncio directly.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="agentchanti-mcp-loop", daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("asyncio loop failed to start within 5s")

    def _run(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def run(self, coro, *, timeout: float) -> Any:
        if self._loop is None:
            raise RuntimeError("engine not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
        self._loop = None
        self._thread = None

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
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .snapshot import (
    SelectorKind,
    classify_selector,
    parse_snapshot,
    resolve_selector,
)

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

    # ---- Interactions (P2) ----------------------------------------------
    # Two dispatch paths:
    #   SEMANTIC selector (text=, role=, bare name) -> snapshot + resolve
    #     to a ref, then call browser_{click,type,hover,select_option}
    #     with {ref, element}. Best-case path: debuggable, self-healable.
    #   CSS selector (#id, [data-testid=X], tag[attr]) -> dispatch via
    #     browser_evaluate using document.querySelector. Escape hatch for
    #     the many DOM attributes the accessibility tree hides.

    def click(self, selector: str) -> ActionResult:
        kind = classify_selector(selector)
        if kind is SelectorKind.SEMANTIC:
            return self._dispatch_semantic(selector, "browser_click", extra={})
        if kind is SelectorKind.CSS:
            return self._dispatch_css(selector, "click")
        return _unsupported_selector(selector, kind)

    def fill(self, selector: str, value: str) -> ActionResult:
        kind = classify_selector(selector)
        if kind is SelectorKind.SEMANTIC:
            return self._dispatch_semantic(
                selector, "browser_type", extra={"text": value},
            )
        if kind is SelectorKind.CSS:
            return self._dispatch_css(selector, "fill", value=value)
        return _unsupported_selector(selector, kind)

    def press(self, selector: str, key: str) -> ActionResult:
        # Two-step: focus the element, then press the key. browser_press_key
        # doesn't take a ref — it presses on whatever has focus.
        focus_result = self._focus(selector)
        if not focus_result.success:
            return focus_result
        try:
            self._call_tool("browser_press_key", {"key": key})
        except Exception as e:
            return ActionResult(success=False, error=f"press failed: {e}")
        return ActionResult(success=True)

    def select(self, selector: str, value: str) -> ActionResult:
        kind = classify_selector(selector)
        if kind is SelectorKind.SEMANTIC:
            return self._dispatch_semantic(
                selector, "browser_select_option", extra={"values": [value]},
            )
        if kind is SelectorKind.CSS:
            return self._dispatch_css(selector, "select", value=value)
        return _unsupported_selector(selector, kind)

    def hover(self, selector: str) -> ActionResult:
        kind = classify_selector(selector)
        if kind is SelectorKind.SEMANTIC:
            return self._dispatch_semantic(selector, "browser_hover", extra={})
        if kind is SelectorKind.CSS:
            return self._dispatch_css(selector, "hover")
        return _unsupported_selector(selector, kind)

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        """Probe whether ``selector`` matches something on the page now.

        Replayer uses this to test locator candidates cheaply. We poll
        because the element may render just after the action that
        triggered its appearance — the caller supplies ``timeout_ms``.
        """
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        while True:
            try:
                if self._selector_exists(selector):
                    return ActionResult(success=True)
            except Exception as e:
                return ActionResult(success=False, error=str(e))
            if time.monotonic() >= deadline:
                return ActionResult(success=False)
            time.sleep(0.1)

    # ---- Dispatch helpers -----------------------------------------------

    def _dispatch_semantic(
        self, selector: str, tool: str, *, extra: dict[str, Any],
    ) -> ActionResult:
        ref = self._resolve_ref(selector)
        if ref is None:
            return ActionResult(
                success=False,
                error=f"no element matching {selector!r} in current snapshot",
            )
        args = {"ref": ref, "element": selector, **extra}
        try:
            self._call_tool(tool, args)
        except Exception as e:
            return ActionResult(success=False, error=f"{tool} failed: {e}")
        return ActionResult(success=True)

    def _dispatch_css(
        self, selector: str, action: str, *, value: str | None = None,
    ) -> ActionResult:
        js = _build_css_action_js(selector, action, value)
        try:
            self._call_tool("browser_evaluate", {"function": js})
        except Exception as e:
            return ActionResult(success=False, error=f"{action} via evaluate failed: {e}")
        return ActionResult(success=True)

    def _focus(self, selector: str) -> ActionResult:
        kind = classify_selector(selector)
        if kind is SelectorKind.SEMANTIC:
            ref = self._resolve_ref(selector)
            if ref is None:
                return ActionResult(
                    success=False,
                    error=f"no element matching {selector!r} in current snapshot",
                )
            try:
                # `(element) => element.focus()` runs scoped to the ref —
                # no side-effect click, no accidental form submit.
                self._call_tool("browser_evaluate", {
                    "function": "(element) => element.focus()",
                    "ref": ref,
                    "element": selector,
                })
            except Exception as e:
                return ActionResult(success=False, error=f"focus failed: {e}")
            return ActionResult(success=True)
        if kind is SelectorKind.CSS:
            js = (
                f"() => {{ const el = document.querySelector({json.dumps(selector)}); "
                f"if (!el) throw new Error('selector not found: ' + {json.dumps(selector)}); "
                f"el.focus(); }}"
            )
            try:
                self._call_tool("browser_evaluate", {"function": js})
            except Exception as e:
                return ActionResult(success=False, error=f"focus failed: {e}")
            return ActionResult(success=True)
        return _unsupported_selector(selector, kind)

    def _selector_exists(self, selector: str) -> bool:
        kind = classify_selector(selector)
        if kind is SelectorKind.SEMANTIC:
            return self._resolve_ref(selector) is not None
        if kind is SelectorKind.CSS:
            js = f"() => document.querySelector({json.dumps(selector)}) !== null"
            try:
                result_text = self._call_tool("browser_evaluate", {"function": js})
            except Exception:
                return False
            # browser_evaluate surfaces the boolean as text; "true" literal
            # is the success case. The result includes the ### Result
            # header — simple substring check is robust.
            return "true" in (result_text or "").lower()
        return False

    def _resolve_ref(self, selector: str) -> str | None:
        snap = self.snapshot()
        elements = parse_snapshot(snap.get("raw", ""))
        return resolve_selector(elements, selector)

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
# CSS-action JS builders
# ---------------------------------------------------------------------------

def _unsupported_selector(selector: str, kind: SelectorKind) -> ActionResult:
    return ActionResult(
        success=False,
        error=f"cannot dispatch selector {selector!r} of kind {kind.value}",
    )


def _build_css_action_js(selector: str, action: str, value: str | None) -> str:
    """Return a JS function string that performs ``action`` on the first
    element matching ``selector``. Values are JSON-escaped to survive
    arbitrary user input without injection risk.
    """
    q = json.dumps(selector)
    if action == "click":
        body = "el.click();"
    elif action == "fill":
        v = json.dumps(value or "")
        body = (
            f"el.focus(); el.value = {v}; "
            f"el.dispatchEvent(new Event('input', {{bubbles: true}})); "
            f"el.dispatchEvent(new Event('change', {{bubbles: true}}));"
        )
    elif action == "select":
        v = json.dumps(value or "")
        body = (
            f"el.value = {v}; "
            f"el.dispatchEvent(new Event('change', {{bubbles: true}}));"
        )
    elif action == "hover":
        # No native `.hover()` — dispatch mouse events. Imperfect (doesn't
        # trigger CSS :hover the same way a real mouse does) but good
        # enough for UI effects that listen to pointer events.
        body = (
            "const r = el.getBoundingClientRect();"
            "const opts = {bubbles: true, clientX: r.left + r.width/2, "
            "clientY: r.top + r.height/2};"
            "el.dispatchEvent(new MouseEvent('mouseover', opts));"
            "el.dispatchEvent(new MouseEvent('mouseenter', opts));"
            "el.dispatchEvent(new MouseEvent('mousemove', opts));"
        )
    else:
        raise ValueError(f"unsupported CSS action {action!r}")
    return (
        f"() => {{ const el = document.querySelector({q}); "
        f"if (!el) throw new Error('selector not found: ' + {q}); "
        f"{body} }}"
    )


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

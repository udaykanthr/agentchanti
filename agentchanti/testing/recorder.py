"""
Recorder — drives a BrowserMCPClient and writes events to a TraceWriter.

Two usage paths:

**Scripted** — caller invokes ``record_interaction`` / ``record_network``
explicitly. Used for programmatic test-generation flows and by the test
suite. Fully working today.

**Live** — ``subscribe_to_live_events`` injects a small JS recorder into
the page (via ``browser_evaluate``) and starts a background thread that
drains user interactions from a buffer at ~5 Hz. Designed for the
"open a browser, click around, walk away with a trace" workflow.

Live recording overview
-----------------------
  1. We inject a JS function (``_recorder_script.INSTALL_SCRIPT``) that
     adds capture-phase listeners for click / input / change / keydown
     to ``document``. Each event pushes a small payload to a buffer on
     ``window.__agentchantiRecorder.events``. Idempotent — re-installing
     is a no-op.
  2. A daemon thread polls every ``poll_interval_s`` (default 0.2s) by
     calling ``browser_evaluate`` with ``DRAIN_SCRIPT``. The script
     splices the buffer, returns the events plus current URL plus a
     ``missing`` flag.
  3. ``missing == True`` means the page navigated and our hooks are
     gone. The poller re-injects, then continues.
  4. On URL change we emit a ``navigate`` trace event before draining.
  5. ``stop()`` signals the thread, joins it, performs one final drain
     (so events buffered during teardown survive), then emits
     ``session_end``.

Dependency-injected: the constructor takes pre-built ``mcp_client`` and
``trace_writer``. Use ``Recorder.from_url`` for the common case, or pass
fakes directly in tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..cli_display import log
from ._live_translate import extract_evaluate_result, translate_js_event
from ._recorder_script import DRAIN_SCRIPT, INSTALL_SCRIPT
from .mcp_client import BrowserMCPClient, NetworkEvent
from .trace import ElementContext, TraceWriter


class Recorder:
    """Orchestrates an MCP browser session and a JSONL trace writer."""

    def __init__(self, mcp_client, trace_writer: TraceWriter):
        self.mcp = mcp_client
        self.writer = trace_writer
        self._started = False
        # Live-recording state (only set once subscribe_to_live_events runs)
        self._live_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._last_url: str | None = None

    @classmethod
    def from_url(
        cls,
        mcp_server_url: str,
        output_path: str | Path,
    ) -> Recorder:
        """Build a Recorder with concrete BrowserMCPClient + TraceWriter.

        The returned Recorder is not yet "open" — use it as a context
        manager so the MCP session and trace file are opened and closed
        in the right order.
        """
        return cls(
            mcp_client=BrowserMCPClient(mcp_server_url),
            trace_writer=TraceWriter(Path(output_path)),
        )

    # ---- Session lifecycle ----------------------------------------------

    def __enter__(self) -> Recorder:
        self.mcp.__enter__()
        self.writer.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Stop any live-recording thread first so it can't write more
        # events after we've torn down the writer.
        if self._stop_event is not None and not self._stop_event.is_set():
            try:
                self._stop_live_thread()
            except Exception:
                pass
        # Always close the writer before MCP — the writer's __exit__
        # emits session_end, and we want it captured before the browser
        # tears down.
        self.writer.__exit__(exc_type, exc, tb)
        self.mcp.__exit__(exc_type, exc, tb)

    def start(
        self,
        start_url: str,
        *,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Navigate to ``start_url`` and emit session_start + navigate events."""
        if self._started:
            raise RuntimeError("Recorder.start called twice")
        self.writer.write_session_start(
            start_url=start_url, viewport=viewport, user_agent=user_agent,
        )
        result = self.mcp.navigate(start_url)
        self.writer.write_navigate(url=start_url, status=None)
        for ev in getattr(result, "network_events", ()) or ():
            self._write_network_event(ev)
        self._last_url = start_url
        self._started = True

    def stop(self, reason: str = "user_stopped") -> Path:
        """Stop live recording (if running), emit session_end, return path."""
        if self._stop_event is not None and not self._stop_event.is_set():
            self._stop_live_thread()
        self.writer.write_session_end(reason=reason)
        return self.writer.path

    # ---- Scripted recording (immediate) ---------------------------------

    def record_interaction(
        self,
        action: str,
        selector_used: str,
        element: ElementContext,
        value: str | None = None,
        coord: dict[str, int] | None = None,
    ) -> None:
        """Append an interaction event to the trace."""
        self.writer.write_interaction(
            action=action,
            selector_used=selector_used,
            element=element,
            value=value,
            coord=coord,
        )

    def record_network(self, event: NetworkEvent) -> None:
        """Append a network event to the trace."""
        self._write_network_event(event)

    # ---- Live recording (stream from MCP) -------------------------------

    def subscribe_to_live_events(
        self, *, poll_interval_s: float = 0.2,
    ) -> None:
        """Inject the JS recorder and start the background poll loop.

        Returns immediately — the loop runs on a daemon thread until
        ``stop()`` (or the context manager) tears it down. Calls after
        the first are no-ops.
        """
        if self._stop_event is not None:
            return  # already subscribed

        # Install once up front so events from the first user interaction
        # are captured even if the first poll lands later.
        try:
            self._install_recorder()
        except Exception as e:
            log.warning(f"[Recorder] initial JS recorder install failed: {e}")

        self._stop_event = threading.Event()
        self._live_thread = threading.Thread(
            target=self._live_loop,
            name="agentchanti-live-recorder",
            kwargs={"poll_interval_s": poll_interval_s},
            daemon=True,
        )
        self._live_thread.start()

    def _stop_live_thread(self) -> None:
        assert self._stop_event is not None
        self._stop_event.set()
        thread = self._live_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        # Final drain — capture anything that lands between the last
        # poll and the stop signal so we don't lose tail events.
        try:
            self._poll_once()
        except Exception as e:
            log.debug(f"[Recorder] final drain failed (non-fatal): {e}")
        self._live_thread = None

    # ---- Live polling loop ---------------------------------------------

    def _live_loop(self, *, poll_interval_s: float) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as e:
                # Poll failures must not kill the thread — the network
                # / MCP server can have hiccups. Log and keep trying.
                log.debug(f"[Recorder] poll error (non-fatal): {e}")
            self._stop_event.wait(timeout=poll_interval_s)

    def _poll_once(self) -> None:
        """One drain cycle: fetch buffered events, write them out,
        re-inject if the page navigated."""
        text = self.mcp._call_tool("browser_evaluate", {"function": DRAIN_SCRIPT})
        payload = extract_evaluate_result(text)
        if not isinstance(payload, dict):
            return

        current_url = payload.get("url")
        missing = payload.get("missing", False)
        events = payload.get("events", []) or []

        # Detect navigation: if the URL changed since the last poll,
        # emit a navigate event so the trace records the page change
        # before any interactions on the new page.
        if current_url and current_url != self._last_url:
            self.writer.write_navigate(url=current_url, status=None)
            self._last_url = current_url

        for js_ev in events:
            self._handle_js_event(js_ev)

        # Re-inject AFTER processing tail events from the old page.
        if missing:
            try:
                self._install_recorder()
            except Exception as e:
                log.debug(f"[Recorder] re-install failed (will retry): {e}")

    def _handle_js_event(self, js_ev: Any) -> None:
        translated = translate_js_event(js_ev)
        if translated is None:
            return
        # Pop fields that aren't kwargs of write_interaction.
        coord = translated.pop("coord", None)
        self.writer.write_interaction(
            action=translated["action"],
            selector_used=translated["selector_used"],
            element=translated["element"],
            value=translated.get("value"),
            coord=coord,
        )

    def _install_recorder(self) -> None:
        """Inject the JS event hook into the current page. Idempotent."""
        self.mcp._call_tool("browser_evaluate", {"function": INSTALL_SCRIPT})

    # ---- Internals -------------------------------------------------------

    def _write_network_event(self, ev: Any) -> None:
        self.writer.write_network(
            request_id=getattr(ev, "request_id", None) or f"auto-{id(ev):x}",
            method=ev.method,
            url=ev.url,
            status=ev.status,
            request_body=getattr(ev, "request_body", None),
            response_body=getattr(ev, "response_body", None),
            duration_ms=getattr(ev, "duration_ms", None),
        )

"""
Recorder — drives a BrowserMCPClient and writes events to a TraceWriter.

Two usage paths:

**Scripted** — caller invokes ``record_interaction`` / ``record_network``
explicitly. Used for programmatic test-generation flows and by the test
suite. Fully working today.

**Live** — ``subscribe_to_live_events`` attaches listeners to the MCP
server so a human user's clicks/fills/network traffic stream into the
trace automatically. This is the UX-critical path but depends on the
Playwright MCP server's specific event tool names and subscription
semantics; implementation is stubbed with a clear pointer until we wire
it against a live server.

Dependency-injected: the constructor takes pre-built ``mcp_client`` and
``trace_writer``. Use ``Recorder.from_url`` for the common case, or pass
fakes directly in tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .mcp_client import BrowserMCPClient, NetworkEvent
from .trace import ElementContext, TraceWriter


class Recorder:
    """Orchestrates an MCP browser session and a JSONL trace writer."""

    def __init__(self, mcp_client, trace_writer: TraceWriter):
        self.mcp = mcp_client
        self.writer = trace_writer
        self._started = False

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
        # Tear down in reverse construction order. Always close the
        # writer first so the session_end event lands before the MCP
        # session is torn down (the close itself can emit network events
        # we'd then be unable to record).
        self.writer.__exit__(exc_type, exc, tb)
        self.mcp.__exit__(exc_type, exc, tb)

    def start(
        self,
        start_url: str,
        *,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Navigate to ``start_url`` and emit session_start + navigate events.

        Any network traffic observed during the initial navigation is
        attached to the navigate event in the trace.
        """
        if self._started:
            raise RuntimeError("Recorder.start called twice")
        self.writer.write_session_start(
            start_url=start_url, viewport=viewport, user_agent=user_agent,
        )
        result = self.mcp.navigate(start_url)
        self.writer.write_navigate(url=start_url, status=None)
        for ev in getattr(result, "network_events", ()) or ():
            self._write_network_event(ev)
        self._started = True

    def stop(self, reason: str = "user_stopped") -> Path:
        """Emit session_end and return the trace file path."""
        self.writer.write_session_end(reason=reason)
        return self.writer.path

    # ---- Scripted recording (immediate) ---------------------------------

    def record_interaction(
        self,
        action: str,
        selector_used: str,
        element: ElementContext,
        value: str | None = None,
    ) -> None:
        """Append an interaction event to the trace."""
        self.writer.write_interaction(
            action=action,
            selector_used=selector_used,
            element=element,
            value=value,
        )

    def record_network(self, event: NetworkEvent) -> None:
        """Append a network event to the trace."""
        self._write_network_event(event)

    # ---- Live recording (stream from MCP) -------------------------------

    def subscribe_to_live_events(self) -> None:
        """Attach listeners so user clicks + network traffic stream to the trace.

        This is the "record a real human session" path. Requires wiring
        against Playwright MCP's specific tool surface for:
          * DOM interaction notifications (click, fill, type, press, ...)
          * Network request/response notifications

        Playwright MCP exposes these via its ``browser.subscribe_*`` tools
        but the exact shapes aren't stable enough to hard-code before we
        have a live server to test against. Until that lands, use the
        scripted ``record_interaction`` / ``record_network`` path.
        """
        raise NotImplementedError(
            "Live event subscription requires Playwright MCP wiring. "
            "Use record_interaction / record_network for scripted recording."
        )

    # ---- Internals -------------------------------------------------------

    def _write_network_event(self, ev: Any) -> None:
        # Accept NetworkEvent dataclass or a duck-typed object with the
        # same attributes. request_id is synthesized when absent so
        # scripted recordings don't have to invent one.
        self.writer.write_network(
            request_id=getattr(ev, "request_id", None) or f"auto-{id(ev):x}",
            method=ev.method,
            url=ev.url,
            status=ev.status,
            request_body=getattr(ev, "request_body", None),
            response_body=getattr(ev, "response_body", None),
            duration_ms=getattr(ev, "duration_ms", None),
        )

"""
Raw trace format — the contract between Recorder and Normalizer.

A trace is a plain JSONL file: one event per line, append-only, flushed
per-write so a crashed recorder still yields a usable file up to the last
complete event. The reader tolerates a truncated final line.

Event types
-----------
Every line is a JSON object with a discriminator ``type`` and a monotonic
``seq``. Use ``seq`` (not ``ts``) to reason about ordering — timestamps
have ties and timezone gotchas; the integer sequence does not.

    session_start   {start_url, viewport, user_agent}
    navigate        {url, status}
    interaction     {action, selector_used, element, value?}
    network         {request_id, method, url, status,
                     request_body?, response_body?, duration_ms}
    session_end     {reason}

``interaction.element`` carries the DOM metadata the Normalizer's LLM
needs to produce a stable semantic label: tag, visible text, aria-label,
role, id, data-testid, nearby label text, classes. Capture everything
cheap up front — missing context at normalize time means a brittle spec.

``network.request_id`` lets requests and responses be correlated when
they arrive as separate events; the Recorder emits a single combined
``network`` event whenever it has both halves in hand, but the schema
allows either ordering.

Design rationale: flat, discriminated, monotonically ordered. No nesting
beyond one level. Keeps grep/jq-friendly and bounds LLM prompt size for
the Normalizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Event type discriminators
SESSION_START = "session_start"
NAVIGATE = "navigate"
INTERACTION = "interaction"
NETWORK = "network"
SESSION_END = "session_end"

ALL_EVENT_TYPES = frozenset({
    SESSION_START, NAVIGATE, INTERACTION, NETWORK, SESSION_END,
})


@dataclass
class ElementContext:
    """DOM metadata captured at the moment of an interaction.

    Consumed by the Normalizer's LLM call to synthesize a semantic label
    and locator fallbacks. More context here = fewer brittle specs.
    """
    tag: str
    text: str | None = None
    aria_label: str | None = None
    role: str | None = None
    id: str | None = None
    data_testid: str | None = None
    nearby_label: str | None = None
    classes: list[str] = field(default_factory=list)


class TraceWriter:
    """Append-only JSONL writer with per-event flush for crash safety.

    Use as a context manager so the session_end event is always emitted
    and the file handle is always closed::

        with TraceWriter(path) as w:
            w.write_session_start(start_url="...", viewport={...})
            w.write_interaction(action="click", selector_used="#btn",
                                element=ElementContext(tag="button", text="Submit"))
            # ...
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._seq = 0
        self._fh = None  # type: ignore[assignment]
        self._ended = False

    def __enter__(self) -> TraceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._ended and self._fh is not None:
            reason = "error" if exc_type is not None else "closed"
            try:
                self._emit(SESSION_END, {"reason": reason})
            except (OSError, ValueError):
                # Best-effort: don't mask the original exception.
                pass
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---- Public write helpers -------------------------------------------

    def write_session_start(
        self,
        start_url: str,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
    ) -> None:
        self._emit(SESSION_START, {
            "start_url": start_url,
            "viewport": viewport or {},
            "user_agent": user_agent,
        })

    def write_navigate(self, url: str, status: int | None = None) -> None:
        self._emit(NAVIGATE, {"url": url, "status": status})

    def write_interaction(
        self,
        action: str,
        selector_used: str,
        element: ElementContext,
        value: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "action": action,
            "selector_used": selector_used,
            "element": _element_to_dict(element),
        }
        if value is not None:
            payload["value"] = value
        self._emit(INTERACTION, payload)

    def write_network(
        self,
        *,
        request_id: str,
        method: str,
        url: str,
        status: int,
        request_body: Any = None,
        response_body: Any = None,
        duration_ms: int | None = None,
    ) -> None:
        self._emit(NETWORK, {
            "request_id": request_id,
            "method": method,
            "url": url,
            "status": status,
            "request_body": request_body,
            "response_body": response_body,
            "duration_ms": duration_ms,
        })

    def write_session_end(self, reason: str = "user_stopped") -> None:
        self._emit(SESSION_END, {"reason": reason})
        self._ended = True

    # ---- Internals -------------------------------------------------------

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("TraceWriter used outside of a `with` block")
        self._seq += 1
        line = {"seq": self._seq, "ts": _now_iso(), "type": event_type, **payload}
        self._fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._fh.flush()


def read_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield events from a JSONL trace.

    Tolerant of a truncated final line (common when a recorder crashed
    mid-write) and of blank lines.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                # Truncated tail — stop cleanly; no partial event is ever
                # emitted to callers.
                return
            if event.get("type") in ALL_EVENT_TYPES:
                yield event


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _element_to_dict(el: ElementContext) -> dict[str, Any]:
    return {
        "tag": el.tag,
        "text": el.text,
        "aria_label": el.aria_label,
        "role": el.role,
        "id": el.id,
        "data_testid": el.data_testid,
        "nearby_label": el.nearby_label,
        "classes": list(el.classes),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

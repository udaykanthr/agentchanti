"""
Pure-Python helpers for the live recorder. Kept off the Recorder class
so they're trivial to unit-test without an MCP server in the loop.

Two responsibilities:

  1. ``extract_evaluate_result(text)`` — parse Playwright MCP's
     ``browser_evaluate`` text response into a Python dict. The response
     looks like::

        ### Result
        {
          "events": [...],
          "url": "...",
          "missing": false
        }
        ### Ran Playwright code
        ```js ...```

     We slice out the block between the headers and JSON-load it. Robust
     to trailing whitespace and to the result being a primitive (string,
     number) rather than an object.

  2. ``translate_js_event(js_ev)`` — convert one JS-side event dict into
     the inputs the trace writer expects: action, synthesized
     ``selector_used``, an ``ElementContext``, optional value, optional
     ``(coord_x, coord_y)``. Selector synthesis priority mirrors the
     resolver: data-testid > id > role+name > tag.

     We synthesize a selector here (rather than at normalize time)
     because (a) the Replayer needs *something* concrete in the trace as
     a last-resort fallback, and (b) the Normalizer's LLM needs an
     anchor to ground its semantic-label decision against.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .trace import ElementContext


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_RESULT_BLOCK_RE = re.compile(
    r"###\s*Result\s*\n(?P<body>.*?)\n\s*###\s*Ran Playwright code",
    re.DOTALL,
)


def extract_evaluate_result(text: str) -> Any:
    """Pull the JSON-ish value out of a ``browser_evaluate`` response.

    Returns whatever the JS function returned: dict, list, str, int, etc.
    Returns an empty dict when no Result block is present (defensive — a
    malformed response shouldn't crash the polling loop).
    """
    if not text:
        return {}
    match = _RESULT_BLOCK_RE.search(text)
    if not match:
        # Some response paths may omit the trailing header; try a softer
        # grab — everything after the first Result header.
        idx = text.find("### Result")
        if idx < 0:
            return {}
        body = text[idx + len("### Result"):].strip()
    else:
        body = match.group("body").strip()

    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Plain string returns come through as `"hello"` — try to strip
        # one layer of surrounding quotes and return the raw string.
        if body.startswith('"') and body.endswith('"'):
            try:
                return json.loads(body)  # already attempted; defensive
            except json.JSONDecodeError:
                return body[1:-1]
        return body


# ---------------------------------------------------------------------------
# JS event -> trace inputs
# ---------------------------------------------------------------------------


# What write_interaction expects, packaged for one return value.
class TranslatedEvent(dict):
    """Type alias dict with: action, selector_used, element, value?, coord?.

    Using a dict (not a dataclass) so callers can pass it straight to
    ``writer.write_interaction(**event)`` once we drop the keys writer
    doesn't accept (only ``coord`` for now).
    """


def translate_js_event(js_ev: dict[str, Any]) -> TranslatedEvent | None:
    """Translate one JS-side event into trace.write_interaction kwargs.

    Returns ``None`` for events we deliberately drop (e.g. clicks with
    no element, malformed payloads).
    """
    if not isinstance(js_ev, dict):
        return None
    js_type = js_ev.get("type")
    el_meta = js_ev.get("element")
    if not js_type or not isinstance(el_meta, dict) or not el_meta:
        # No element metadata = un-replayable event. Drop it rather than
        # record a phantom step the Replayer can never resolve.
        return None

    element = ElementContext(
        tag=str(el_meta.get("tag") or "unknown"),
        text=el_meta.get("text"),
        aria_label=el_meta.get("aria_label"),
        role=el_meta.get("role"),
        id=el_meta.get("id"),
        data_testid=el_meta.get("data_testid"),
        nearby_label=None,
        classes=list(el_meta.get("classes") or []),
    )

    selector_used = synthesize_selector(el_meta)

    out: TranslatedEvent = TranslatedEvent(
        action=_map_action(js_type, el_meta),
        selector_used=selector_used,
        element=element,
    )

    # Action-specific extras.
    if js_type == "input" or (js_type == "change" and (el_meta.get("tag") == "select")):
        out["value"] = js_ev.get("value")
    elif js_type == "keydown":
        # Carry the key in the same `value` slot the existing trace
        # schema uses for press actions — Replayer's press() expects it.
        out["value"] = js_ev.get("key")

    # Coords: only present on click events.
    if js_type == "click":
        x = js_ev.get("clientX")
        y = js_ev.get("clientY")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            out["coord"] = {"x": int(x), "y": int(y)}

    return out


def _map_action(js_type: str, el_meta: dict[str, Any]) -> str:
    """Translate a JS event type into one of the trace's interaction
    actions (matches Spec.ALLOWED_ACTIONS values)."""
    if js_type == "click":
        return "click"
    if js_type == "input":
        return "fill"
    if js_type == "change":
        return "select" if el_meta.get("tag") == "select" else "fill"
    if js_type == "keydown":
        return "press"
    # Unknown / future event types — fall through as "click" so the
    # Replayer at least tries something rather than silently dropping.
    return "click"


def synthesize_selector(el_meta: dict[str, Any]) -> str:
    """Best-effort selector for an element captured live.

    Priority mirrors the resolver: data-testid > id > role+name > tag.
    Returns a string the existing classify_selector + dispatch logic can
    consume — CSS for the first two, semantic for the rest.
    """
    if not isinstance(el_meta, dict):
        return "unknown"
    testid = el_meta.get("data_testid")
    if testid:
        return f"[data-testid={testid}]"
    el_id = el_meta.get("id")
    if el_id:
        return f"#{el_id}"
    role = el_meta.get("role")
    text = el_meta.get("text")
    aria = el_meta.get("aria_label")
    name = aria or text
    if role and name:
        return f'role={role} name="{name}"'
    if name:
        return f"text={name}"
    tag = el_meta.get("tag") or "unknown"
    return tag

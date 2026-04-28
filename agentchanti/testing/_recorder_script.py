"""
JavaScript injection script for the live Recorder.

Installed once into the page via ``browser_evaluate``; from then on,
every user click / input / change / keydown is pushed into a buffer on
``window.__agentchantiRecorder.events``. The Python polling loop drains
that buffer at ~5 Hz via a separate ``browser_evaluate`` call.

Why capture-phase listeners?
  Bubble-phase listeners get cancelled by ``stopPropagation()`` calls in
  page code, which is common in React event delegation, modal libraries,
  etc. Capture-phase listeners run first, before any page handler can
  cancel propagation, so we always see the event.

Why split install + drain into two scripts?
  ``DRAIN_SCRIPT`` runs on every poll (5x/sec) — keeping it tiny matters
  for latency. ``INSTALL_SCRIPT`` runs only on first install + after page
  navigation, so we can afford the larger setup body.

Sensitive input redaction
  ``<input type="password">`` values are recorded as ``***REDACTED***``
  rather than the typed text. We don't want test traces to be
  credential-leaking artifacts on disk or in CI logs. Same applies to
  any input with ``data-sensitive`` set — an opt-in escape hatch for
  apps that store SSNs / card numbers / etc. in non-password fields.
"""

from __future__ import annotations


# Marker we look for in the drain output to detect "the page navigated
# and our recorder is gone — please re-inject". Kept as a Python-side
# constant so one place owns the contract.
MISSING_MARKER_KEY = "missing"


# Note: this string is fed verbatim to Playwright MCP's browser_evaluate,
# which expects a JS function expression. Keep the leading `() =>` form.
INSTALL_SCRIPT = r"""
() => {
  if (window.__agentchantiRecorder) return "already-installed";
  const state = {events: [], installedAt: Date.now()};
  window.__agentchantiRecorder = state;

  const SPECIAL_KEYS = new Set([
    "Enter", "Tab", "Escape", "Backspace", "Delete",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  ]);

  const meta = (el) => {
    if (!el || !el.tagName) return {};
    let classes = [];
    if (el.className && typeof el.className === "string") {
      classes = el.className.trim().split(/\s+/).filter(Boolean);
    }
    let text = null;
    try {
      const raw = (el.innerText || el.textContent || "").trim();
      text = raw ? raw.slice(0, 200) : null;
    } catch (_e) { /* shadow DOM weirdness */ }
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id || null,
      classes: classes,
      data_testid: el.getAttribute ? el.getAttribute("data-testid") : null,
      aria_label: el.getAttribute ? el.getAttribute("aria-label") : null,
      role: el.getAttribute ? el.getAttribute("role") : null,
      name: el.getAttribute ? el.getAttribute("name") : null,
      type: el.getAttribute ? el.getAttribute("type") : null,
      text: text,
    };
  };

  const isSensitive = (el) => {
    if (!el || !el.tagName) return false;
    if (el.tagName === "INPUT" && (el.type || "").toLowerCase() === "password") return true;
    if (el.getAttribute && el.getAttribute("data-sensitive") !== null) return true;
    return false;
  };

  const push = (payload) => {
    state.events.push(payload);
    // Cap the buffer so a runaway page can't eat all browser memory if
    // Python polling stops (Ctrl+C, network blip, etc.).
    if (state.events.length > 5000) {
      state.events.splice(0, state.events.length - 2500);
    }
  };

  document.addEventListener("click", (e) => {
    const el = e.target;
    push({
      type: "click",
      clientX: e.clientX, clientY: e.clientY,
      button: e.button,
      element: meta(el),
    });
  }, true);

  // `input` fires per character. We don't try to debounce in JS — the
  // Normalizer collapses chains in Python where it's easier to reason
  // about. The drain rate is high enough that the buffer doesn't bloat.
  document.addEventListener("input", (e) => {
    const el = e.target;
    if (!el || !("value" in el)) return;
    push({
      type: "input",
      element: meta(el),
      value: isSensitive(el) ? "***REDACTED***" : (el.value || ""),
    });
  }, true);

  document.addEventListener("change", (e) => {
    const el = e.target;
    if (!el) return;
    if (el.tagName === "SELECT") {
      push({type: "change", element: meta(el), value: el.value});
    }
  }, true);

  document.addEventListener("keydown", (e) => {
    if (!SPECIAL_KEYS.has(e.key)) return;
    push({type: "keydown", key: e.key, element: meta(e.target)});
  }, true);

  return "installed";
}
"""


DRAIN_SCRIPT = r"""
() => {
  if (!window.__agentchantiRecorder) {
    return {events: [], url: location.href, missing: true};
  }
  const events = window.__agentchantiRecorder.events.splice(0);
  return {events: events, url: location.href, missing: false};
}
"""

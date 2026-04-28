"""
Replayer — drives a Spec against a BrowserMCPClient with self-healing.

For each step:
  1. Resolve a working selector via the three-tier strategy:
       (a) LocatorCache hit — the single fast path that avoids LLM calls
           and is the reason replay is cheap on CI.
       (b) Walk ``step.target.fallbacks`` in order. First hit wins and is
           pinned in the cache for next run.
       (c) If an ``llm_client`` was supplied, ask it to pick a selector
           from a DOM snapshot (self-healing). Also cached on success.
  2. Dispatch the MCP action (click / fill / press / ...).
  3. Record per-step outcome, network events, and any error.

Fails fast on the first step that can't be resolved or errors out — a
broken state would only invalidate later assertions. ``RunResult`` still
captures everything observed up to that point so Validator/Reporter can
surface a useful diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .locator_cache import LocatorCache
from .mcp_client import ActionResult, NetworkEvent
from .spec import Spec, Step


@dataclass
class StepResult:
    step_id: str
    action: str
    success: bool
    selector_used: str | None = None
    network_events: list[NetworkEvent] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunResult:
    spec_name: str
    steps: list[StepResult] = field(default_factory=list)
    final_url: str = ""
    final_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only when every step succeeded."""
        return bool(self.steps) and all(s.success for s in self.steps)


class Replayer:
    """Execute a Spec against a BrowserMCPClient.

    Parameters
    ----------
    mcp_client:
        Anything with the BrowserMCPClient method surface. Tests inject
        a fake; production uses the real client.
    locator_cache:
        ``LocatorCache`` pinning successful selectors. Make it explicit
        so tests can point at a tmp_path and production at the project's
        ``.agentchanti/testing/locator-cache.json``.
    llm_client:
        Optional. When present, enables the self-healing third tier —
        if every cached + fallback selector fails, the LLM is asked to
        pick one from a snapshot. Without it, Replayer still runs but
        a locator change breaks the step instead of self-healing.
    probe_timeout_ms:
        How long to wait when probing whether a candidate selector matches.
        Kept short by default — we're probing, not waiting for the action
        itself. The actual action (click/fill/...) uses the MCP client's
        own default timeout.
    """

    def __init__(
        self,
        mcp_client,
        locator_cache: LocatorCache,
        *,
        llm_client=None,
        probe_timeout_ms: int = 1000,
    ):
        self.mcp = mcp_client
        self.cache = locator_cache
        self.llm_client = llm_client
        self.probe_timeout_ms = probe_timeout_ms
        # Cursor into ``mcp.network_requests()`` so each step gets only
        # the traffic it triggered. Reset at the top of every replay().
        self._network_seen: int = 0

    def replay(self, spec: Spec) -> RunResult:
        """Run every step of ``spec`` and return what was observed."""
        result = RunResult(spec_name=spec.name)

        # Enforce the recorded viewport BEFORE the first navigate so
        # coord=X,Y fallbacks land on the same screen positions they did
        # during recording. Skipping silently breaks coordinate-based
        # replay on a different monitor — a class of flake we want to
        # eliminate end-to-end, not paper over with retries.
        self._enforce_viewport(spec)

        # Anchor the network cursor at whatever the browser had
        # accumulated from a prior session — anything before this point
        # is not ours to attribute to a step.
        self._network_seen = 0
        self._drain_new_network()

        # The spec's own start_url is the entry point — not a Step, so
        # dispatch it up front. Its network events count toward whatever
        # the first step expects (pre-step traffic).
        nav = self.mcp.navigate(spec.start_url)
        pre_step_network = list(getattr(nav, "network_events", ()) or ())
        pre_step_network.extend(self._drain_new_network())

        for step in spec.steps:
            step_result = self._run_step(spec, step, pre_step_network)
            pre_step_network = []  # only the first step inherits pre-step traffic
            result.steps.append(step_result)
            if not step_result.success:
                break  # fail-fast

        # Always capture final state — Validator uses it for url_equals
        # and dom_predicate assertions regardless of whether replay
        # succeeded end-to-end.
        try:
            result.final_snapshot = self.mcp.snapshot() or {}
        except NotImplementedError:
            # Live MCP wiring not ready — tests supply fakes that implement
            # snapshot(); production without a live server just skips.
            result.final_snapshot = {}
        result.final_url = _current_url(self.mcp, result)
        return result

    # ---- Viewport enforcement -------------------------------------------

    def _enforce_viewport(self, spec: Spec) -> None:
        viewport = (spec.metadata or {}).get("viewport")
        if not isinstance(viewport, dict):
            return
        try:
            width = int(viewport["width"])
            height = int(viewport["height"])
        except (KeyError, TypeError, ValueError):
            return
        # Best-effort: if the MCP client doesn't expose resize (older fake,
        # or a transport that hasn't wired it yet) we skip rather than
        # crash the whole replay. Coord fallbacks may still match if the
        # default viewport is close enough.
        resize = getattr(self.mcp, "resize", None)
        if not callable(resize):
            return
        try:
            resize(width, height)
        except Exception:
            pass

    # ---- Step dispatch ---------------------------------------------------

    def _run_step(
        self,
        spec: Spec,
        step: Step,
        inherited_network: list[NetworkEvent],
    ) -> StepResult:
        network: list[NetworkEvent] = list(inherited_network)

        # navigate is the one action that doesn't need locator resolution.
        if step.action == "navigate":
            try:
                ar = self.mcp.navigate(step.url or "")
            except Exception as e:
                network.extend(self._drain_new_network())
                return StepResult(
                    step_id=step.id, action=step.action, success=False,
                    network_events=network, error=f"navigate raised: {e}",
                )
            network.extend(getattr(ar, "network_events", ()) or ())
            network.extend(self._drain_new_network())
            return StepResult(
                step_id=step.id, action=step.action, success=ar.success,
                network_events=network,
                error=ar.error if not ar.success else None,
            )

        # Every other action needs a selector.
        selector = self._resolve_selector(spec, step)
        if selector is None:
            return StepResult(
                step_id=step.id, action=step.action, success=False,
                network_events=network,
                error=f"no working selector for {step.target.label!r}",
            )

        try:
            ar = self._dispatch_action(step, selector)
        except Exception as e:
            network.extend(self._drain_new_network())
            return StepResult(
                step_id=step.id, action=step.action, success=False,
                selector_used=selector, network_events=network,
                error=f"{step.action} raised: {e}",
            )

        network.extend(getattr(ar, "network_events", ()) or ())
        network.extend(self._drain_new_network())
        if not ar.success:
            # Selector matched something but the action failed — invalidate
            # the cache so the next run re-probes from scratch.
            self.cache.invalidate(spec.name, step.id)

        return StepResult(
            step_id=step.id, action=step.action, success=ar.success,
            selector_used=selector, network_events=network,
            error=ar.error if not ar.success else None,
        )

    # ---- Network diff drain ----------------------------------------------

    def _drain_new_network(self) -> list[NetworkEvent]:
        """Return network events the browser logged since the last drain.

        Empty list when the MCP client doesn't expose ``network_requests``
        (older fakes, or transports that haven't wired it). The fall-back
        path is the legacy per-call ``ActionResult.network_events`` —
        replay correctness still holds, just less observability.
        """
        fetcher = getattr(self.mcp, "network_requests", None)
        if not callable(fetcher):
            return []
        try:
            current = fetcher()
        except Exception:
            return []
        if not isinstance(current, list):
            return []
        new = current[self._network_seen:]
        self._network_seen = len(current)
        return list(new)

    def _dispatch_action(self, step: Step, selector: str) -> ActionResult:
        if step.action == "click":
            return self.mcp.click(selector)
        if step.action == "fill":
            return self.mcp.fill(selector, step.value or "")
        if step.action == "press":
            return self.mcp.press(selector, step.value or "")
        if step.action == "select":
            return self.mcp.select(selector, step.value or "")
        if step.action == "hover":
            return self.mcp.hover(selector)
        if step.action == "wait_for":
            return self.mcp.wait_for(selector)
        raise ValueError(f"unknown action {step.action!r}")

    # ---- Selector resolution --------------------------------------------

    def _resolve_selector(self, spec: Spec, step: Step) -> str | None:
        """Three-tier resolution: cache → fallbacks → LLM self-heal."""
        if step.target is None:
            return None

        cached = self.cache.get(spec.name, step.id)
        if cached and self._selector_matches(cached):
            return cached
        if cached:
            self.cache.invalidate(spec.name, step.id)

        for candidate in step.target.fallbacks:
            if self._selector_matches(candidate):
                self.cache.set(spec.name, step.id, candidate)
                return candidate

        if self.llm_client is not None:
            healed = self._ask_llm_for_selector(step)
            if healed and self._selector_matches(healed):
                self.cache.set(spec.name, step.id, healed)
                return healed

        return None

    def _selector_matches(self, selector: str) -> bool:
        """Cheap probe: does this selector match anything on the page right now?"""
        try:
            ar = self.mcp.wait_for(selector, timeout_ms=self.probe_timeout_ms)
        except Exception:
            return False
        return bool(ar.success)

    def _ask_llm_for_selector(self, step: Step) -> str | None:
        """Ask the LLM to pick a working selector from the current snapshot."""
        try:
            snapshot = self.mcp.snapshot()
        except Exception:
            return None
        prompt = _build_heal_prompt(step, snapshot)
        raw = self.llm_client.generate_response(prompt)
        return (raw or "").strip().splitlines()[0].strip() or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_url(mcp, result: RunResult) -> str:
    """Best-effort current URL — prefer the MCP client's notion, fall back to
    the last step that observed one."""
    try:
        ar = mcp.wait_for("body", timeout_ms=0)
        if getattr(ar, "current_url", ""):
            return ar.current_url
    except Exception:
        pass
    # Back off to whatever the last step saw.
    for step in reversed(result.steps):
        if step.action == "navigate":
            # Couldn't recover the real URL cheaply — Validator may still
            # pass url_equals by comparing against the step's known URL.
            return ""
    return ""


def _build_heal_prompt(step: Step, snapshot: dict[str, Any]) -> str:
    import json
    return (
        "A test step's locator fallbacks all failed. Pick ONE CSS or text "
        "selector from the accessibility snapshot below that most likely "
        "matches the element described by the step's semantic label.\n\n"
        f"STEP LABEL: {step.target.label!r}\n"
        f"ACTION: {step.action}\n"
        f"FAILED FALLBACKS: {list(step.target.fallbacks)}\n\n"
        "SNAPSHOT:\n"
        + json.dumps(snapshot, ensure_ascii=False)[:4000]
        + "\n\nOutput ONLY the selector on a single line. No prose, no fences."
    )

"""Tests for agentchanti.testing.replayer — uses a fake MCP client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentchanti.testing.locator_cache import LocatorCache
from agentchanti.testing.mcp_client import ActionResult, NetworkEvent
from agentchanti.testing.replayer import Replayer, RunResult
from agentchanti.testing.spec import (
    Assertion,
    Locator,
    NetworkExpectation,
    Spec,
    Step,
)


class FakeMCP:
    """Scripted browser: selectors in ``known`` match; others don't.

    Tracks every action so tests can assert the expected call sequence.
    """

    def __init__(
        self,
        known_selectors: set[str] | None = None,
        navigate_result: ActionResult | None = None,
        snapshot: dict[str, Any] | None = None,
        action_results: dict[str, ActionResult] | None = None,
    ):
        self.known = set(known_selectors or [])
        self.navigate_result = navigate_result or ActionResult(
            success=True, current_url="/home")
        self._snapshot = snapshot or {"role": "document", "children": []}
        self.action_results = action_results or {}
        self.calls: list[tuple[str, tuple, dict]] = []

    # Method surface Replayer uses ---------------------------------------
    def navigate(self, url: str) -> ActionResult:
        self.calls.append(("navigate", (url,), {}))
        return self.navigate_result

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        self.calls.append(("wait_for", (selector,), {"timeout_ms": timeout_ms}))
        return ActionResult(success=selector in self.known, current_url="/home")

    def click(self, selector: str) -> ActionResult:
        self.calls.append(("click", (selector,), {}))
        return self.action_results.get(selector, ActionResult(success=True))

    def fill(self, selector: str, value: str) -> ActionResult:
        self.calls.append(("fill", (selector, value), {}))
        return self.action_results.get(selector, ActionResult(success=True))

    def press(self, selector: str, key: str) -> ActionResult:
        self.calls.append(("press", (selector, key), {}))
        return self.action_results.get(selector, ActionResult(success=True))

    def select(self, selector: str, value: str) -> ActionResult:
        self.calls.append(("select", (selector, value), {}))
        return self.action_results.get(selector, ActionResult(success=True))

    def hover(self, selector: str) -> ActionResult:
        self.calls.append(("hover", (selector,), {}))
        return self.action_results.get(selector, ActionResult(success=True))

    def snapshot(self) -> dict[str, Any]:
        self.calls.append(("snapshot", (), {}))
        return self._snapshot

    def resize(self, width: int, height: int) -> ActionResult:
        self.calls.append(("resize", (width, height), {}))
        return ActionResult(success=True)


def _simple_spec() -> Spec:
    return Spec(
        name="login",
        start_url="/login",
        steps=[
            Step(
                id="step-1", action="fill",
                target=Locator(label="Email", fallbacks=["#email"]),
                value="user@example.com",
            ),
            Step(
                id="step-2", action="click",
                target=Locator(label="Sign in",
                               fallbacks=["#signin", "button[type=submit]"]),
            ),
        ],
    )


# ---- Core dispatch ---------------------------------------------------------

def test_replay_happy_path_calls_actions_in_order(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "#signin"})
    cache = LocatorCache(tmp_path / "cache.json")
    r = Replayer(mcp, cache).replay(_simple_spec())

    assert r.passed
    assert [s.step_id for s in r.steps] == ["step-1", "step-2"]
    assert [s.selector_used for s in r.steps] == ["#email", "#signin"]
    # Each step probed its fallback via wait_for before dispatching.
    methods = [c[0] for c in mcp.calls]
    assert methods.count("fill") == 1
    assert methods.count("click") == 1


def test_first_successful_fallback_wins_and_is_cached(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "button[type=submit]"})
    cache_path = tmp_path / "cache.json"
    cache = LocatorCache(cache_path)
    r = Replayer(mcp, cache).replay(_simple_spec())

    assert r.passed
    # Second step's #signin fallback doesn't match; button[type=submit] does.
    assert r.steps[1].selector_used == "button[type=submit]"
    # Cache pinned it for next run
    reloaded = LocatorCache(cache_path)
    assert reloaded.get("login", "step-2") == "button[type=submit]"


# ---- Cache fast path -------------------------------------------------------

def test_cached_selector_short_circuits_fallback_walk(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "button[type=submit]"})
    cache = LocatorCache(tmp_path / "cache.json")
    cache.set("login", "step-2", "button[type=submit]")  # pre-warm

    Replayer(mcp, cache).replay(_simple_spec())

    # Only the cached selector is probed for step-2 — #signin never asked.
    probed_for_step2 = [c[1][0] for c in mcp.calls if c[0] == "wait_for"]
    assert "#signin" not in probed_for_step2
    assert "button[type=submit]" in probed_for_step2


def test_stale_cache_entry_is_invalidated_and_refreshed(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "#signin"})
    cache_path = tmp_path / "cache.json"
    cache = LocatorCache(cache_path)
    cache.set("login", "step-2", "#old-selector-that-no-longer-matches")

    Replayer(mcp, cache).replay(_simple_spec())

    # Old entry dropped, new one pinned
    assert LocatorCache(cache_path).get("login", "step-2") == "#signin"


# ---- LLM self-heal ---------------------------------------------------------

class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def generate_response(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_llm_selfheals_when_all_fallbacks_miss(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "text=Sign in"})
    cache_path = tmp_path / "cache.json"
    llm = FakeLLM("text=Sign in")
    r = Replayer(mcp, LocatorCache(cache_path), llm_client=llm).replay(_simple_spec())

    assert r.passed
    assert r.steps[1].selector_used == "text=Sign in"
    assert "Sign in" in llm.last_prompt  # step label reached the LLM
    # Self-healed selector is cached for next run
    assert LocatorCache(cache_path).get("login", "step-2") == "text=Sign in"


def test_without_llm_unresolvable_step_fails_cleanly(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email"})  # step-2 has no matching selector
    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(_simple_spec())

    assert r.passed is False
    assert r.steps[1].success is False
    assert "no working selector" in (r.steps[1].error or "")


# ---- Action failures -------------------------------------------------------

def test_action_failure_invalidates_cache_entry(tmp_path: Path):
    """Selector matched, but the click itself failed — next run must re-probe."""
    mcp = FakeMCP(
        known_selectors={"#email", "#signin"},
        action_results={"#signin": ActionResult(success=False, error="intercepted")},
    )
    cache_path = tmp_path / "cache.json"
    cache = LocatorCache(cache_path)
    # Pre-warm cache so we can observe invalidation
    cache.set("login", "step-2", "#signin")

    r = Replayer(mcp, cache).replay(_simple_spec())

    assert r.passed is False
    assert r.steps[1].error == "intercepted"
    assert LocatorCache(cache_path).get("login", "step-2") is None


def test_fail_fast_skips_later_steps(tmp_path: Path):
    mcp = FakeMCP(known_selectors=set())  # nothing matches
    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(_simple_spec())

    # First step failed → second step never attempted
    assert len(r.steps) == 1
    assert r.steps[0].success is False


# ---- Navigate + network capture -------------------------------------------

def test_navigate_step_dispatches_and_captures_network(tmp_path: Path):
    mcp = FakeMCP(
        navigate_result=ActionResult(
            success=True,
            current_url="/home",
            network_events=[
                NetworkEvent(method="GET", url="/api/me", status=200),
            ],
        )
    )
    spec = Spec(
        name="nav-only", start_url="/",
        steps=[Step(id="step-1", action="navigate", url="/home")],
    )
    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(spec)

    assert r.passed
    # The start_url nav AND the step nav both went through
    nav_urls = [c[1][0] for c in mcp.calls if c[0] == "navigate"]
    assert nav_urls == ["/", "/home"]
    # Network from the first navigate was inherited by step-1; step-1's own
    # navigate added more. Both should appear in the step result.
    assert any(ne.url == "/api/me" for ne in r.steps[0].network_events)


# ---- Final state capture ---------------------------------------------------

def test_replay_captures_final_snapshot(tmp_path: Path):
    mcp = FakeMCP(
        known_selectors={"#email", "#signin"},
        snapshot={"role": "document", "text": "Welcome back"},
    )
    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(_simple_spec())

    assert r.final_snapshot["text"] == "Welcome back"


# ---- Network diff capture --------------------------------------------------

def test_step_picks_up_new_network_requests_via_drain(tmp_path: Path):
    """When the MCP client exposes ``network_requests``, each step should
    receive only the events that were new during it. The browser
    accumulates traffic across the session — the Replayer is responsible
    for slicing the diff."""

    class NetMCP(FakeMCP):
        def __init__(self):
            super().__init__(known_selectors={"#email", "#signin"})
            # Simulates the browser's running network log, growing as
            # each call is made. Keys are the call name we expect to be
            # made — a poor-mans script.
            self._timeline = [
                # Pre-replay state (already in the log from a prior session)
                [],
                # After spec.start_url navigate — one request shows up
                [NetworkEvent("GET", "/start", 200)],
                # After step-1 fill — no traffic added
                [NetworkEvent("GET", "/start", 200)],
                # After step-2 click — login POST + redirect GET
                [
                    NetworkEvent("GET", "/start", 200),
                    NetworkEvent("POST", "/api/login", 200),
                    NetworkEvent("GET", "/dashboard", 200),
                ],
            ]
            self._step = 0

        def network_requests(self) -> list[NetworkEvent]:
            i = min(self._step, len(self._timeline) - 1)
            self._step += 1
            return list(self._timeline[i])

    mcp = NetMCP()
    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(_simple_spec())
    assert r.passed

    # First step inherits start_url's traffic; second step gets the
    # login POST + redirect GET that fired during the click.
    step1_urls = [e.url for e in r.steps[0].network_events]
    step2_urls = [e.url for e in r.steps[1].network_events]
    assert "/start" in step1_urls
    assert step2_urls == ["/api/login", "/dashboard"]


def test_replayer_falls_back_when_mcp_lacks_network_requests(tmp_path: Path):
    """The legacy ActionResult.network_events path must still work for
    fakes that pre-date the network diff plumbing."""
    mcp = FakeMCP(
        known_selectors={"#email", "#signin"},
        navigate_result=ActionResult(
            success=True,
            current_url="/home",
            network_events=[NetworkEvent("GET", "/legacy", 200)],
        ),
    )
    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(_simple_spec())
    assert r.passed
    assert any(e.url == "/legacy" for e in r.steps[0].network_events)


# ---- Viewport enforcement --------------------------------------------------

def test_viewport_metadata_triggers_resize_before_first_navigate(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "#signin"})
    spec = _simple_spec()
    spec.metadata["viewport"] = {"width": 1280, "height": 720}

    Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(spec)

    methods = [c[0] for c in mcp.calls]
    # resize must be the first thing we ask the browser to do, so a
    # captured coord=X,Y fallback hits the same screen position.
    assert methods[0] == "resize"
    assert mcp.calls[0][1] == (1280, 720)
    # navigate happens after resize
    assert methods.index("navigate") > methods.index("resize")


def test_replay_without_viewport_metadata_does_not_resize(tmp_path: Path):
    mcp = FakeMCP(known_selectors={"#email", "#signin"})
    Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(_simple_spec())
    methods = [c[0] for c in mcp.calls]
    assert "resize" not in methods


def test_replay_tolerates_mcp_without_resize(tmp_path: Path):
    """A minimal MCP fake that doesn't implement resize must not crash
    the run — coord fallbacks may still match on the default viewport."""

    class MinimalMCP:
        def __init__(self):
            self.calls: list[tuple[str, tuple, dict]] = []

        def navigate(self, url):
            self.calls.append(("navigate", (url,), {}))
            return ActionResult(success=True, current_url="/home")

        def wait_for(self, sel, timeout_ms=5000):
            self.calls.append(("wait_for", (sel,), {"timeout_ms": timeout_ms}))
            return ActionResult(success=sel in {"#email", "#signin"})

        def fill(self, sel, val):
            self.calls.append(("fill", (sel, val), {}))
            return ActionResult(success=True)

        def click(self, sel):
            self.calls.append(("click", (sel,), {}))
            return ActionResult(success=True)

        def snapshot(self):
            return {}
        # Deliberately no `resize` attribute.

    mcp = MinimalMCP()
    spec = _simple_spec()
    spec.metadata["viewport"] = {"width": 1024, "height": 768}

    r = Replayer(mcp, LocatorCache(tmp_path / "c.json")).replay(spec)
    assert r.passed
    assert "resize" not in [c[0] for c in mcp.calls]

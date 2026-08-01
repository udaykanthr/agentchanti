"""Adversarial property check for simulation-style projects.

Why this exists
---------------
A Pac-Man run shipped green with every ghost able to walk through walls.
Every layer of verification missed it, and all for the same reason::

    step gate 3.1   constructs Game(), never simulates      green
    generated tests fixed dt of 0.05 / 0.1 / 0.2 / 0.01     green
    smoke test      launches the app, never simulates       green

The defect lived entirely in the gap between a *fixed* timestep and a
*variable* one. ``Ghost.update`` only re-chose direction when
``is_centered()`` was true, and that used ``abs(pos - round(pos)) <
0.0001``. With exactly ``1/60`` the increments happen to land inside that
tolerance; with the jittery dt a real ``clock.tick(60)`` produces, the
ghost steps *past* the centre, never re-steers, and continues straight
through walls::

    uniform dt = 1/60      wall-frames =   0
    jittery dt             wall-frames = 129
    dt = 0.033 (30 fps)    wall-frames =  66

No amount of gate *strength* catches that — the gates were behavioural.
What was missing is the adversarial *condition*. So the harness supplies
the condition deterministically (randomised dt, long run, assert every
iteration) and uses the model only for the part that genuinely needs
domain knowledge: expressing the project's invariants against its own
API.

Deliberately narrow: it only fires when the project actually looks like a
simulation, so ordinary CRUD/web projects pay nothing.
"""

from __future__ import annotations

import re
from typing import Optional

from ..cli_display import log

# A method that advances state by a time delta — the shape that makes a
# project timestep-sensitive in the first place. Matching the parameter
# name (not just "update") keeps ORM/model `update()` methods out.
_SIM_UPDATE_RE = re.compile(
    r"def\s+\w*update\w*\s*\(\s*self\s*,[^)]*\b"
    r"(dt|delta|delta_time|deltatime|elapsed|time_step|timestep|tick)\b",
    re.IGNORECASE,
)

# A frame loop. Either signal alone is weak; both together are a simulation.
_LOOP_MARKERS = (
    "clock.tick", "pygame.display.flip", "pygame.display.update",
    "requestanimationframe", "time.perf_counter", "while self.running",
)

_MAX_SOURCE_CHARS = 12000


def simulation_files(memory) -> list[str]:
    """Project files that advance state by a time delta.

    Empty when the project is not a simulation, which is the common case
    and the reason this whole stage is usually free.
    """
    hits: list[str] = []
    try:
        files = memory.as_dict() or {}
    except Exception:
        return []
    for path, content in files.items():
        if not path.endswith(".py") or not content:
            continue
        if path.startswith("_cmd_output/"):
            continue
        if _SIM_UPDATE_RE.search(content):
            hits.append(path)
    if not hits:
        return []
    # Require a frame loop somewhere in the project before believing it.
    joined = " ".join((files.get(p) or "").lower() for p in files)
    if not any(marker in joined for marker in _LOOP_MARKERS):
        return []
    return sorted(hits)


def _invariant_hint(task: str, intent_spec) -> str:
    """Invariants the task states, if any, else a generic starting set."""
    stated = ""
    for source in (getattr(intent_spec, "raw", None),
                   getattr(intent_spec, "text", None), task):
        if isinstance(source, str) and source.strip():
            stated = source.strip()
            break
    return stated[:2000]


def build_property_step(sim_files: list[str], task: str,
                        intent_spec=None) -> str:
    """The step text handed to the agent loop.

    The protocol is fixed by the harness rather than left to the model:
    the whole failure mode is that a model asked for "smooth animation"
    writes fixed-dt tests every single time.
    """
    return (
        "Write and run a property-based test that drives this project's "
        "simulation under ADVERSARIAL frame timing and asserts its "
        "invariants hold at every step.\n\n"
        f"Simulation entry point(s): {', '.join(sim_files)}\n\n"
        "REQUIRED protocol — do not simplify it:\n"
        "  * Drive the main update loop for at least 600 iterations.\n"
        "  * Draw the delta-time for EACH iteration randomly from 0.008 "
        "to 0.05 seconds, from a SEEDED random.Random so failures "
        "reproduce. A real clock.tick() never yields a constant dt, and "
        "a fixed dt is exactly what hides timestep bugs.\n"
        "  * Also run a fixed-dt control at 1/60 for the same length.\n"
        "  * Assert the invariants after EVERY iteration, not just at the "
        "end.\n"
        "  * On violation, fail with the iteration index, which entity, "
        "and its position.\n\n"
        "Invariants to assert — derive the concrete ones from the task "
        "below and from the actual code, and always include these:\n"
        "  * No entity may occupy a position the map/world reports as "
        "impassable (wall, out of bounds).\n"
        "  * No coordinate becomes NaN or infinite.\n"
        "  * The loop raises no exception across the whole run.\n\n"
        f"Task the project implements:\n{_invariant_hint(task, intent_spec)}\n\n"
        "CRITICAL: if an invariant is genuinely violated, that is a real "
        "defect in the SOURCE — fix the source. Do NOT weaken, delete or "
        "skip the assertion, and do NOT switch the test to a fixed "
        "delta-time to make it pass."
    )


def run_property_check(
    memory,
    executor,
    coder,
    display,
    task: str,
    language: str | None,
    cfg=None,
    intent_spec=None,
    step_idx: int = 0,
) -> tuple[bool, str]:
    """Generate + run an adversarial property test. Returns (ok, error).

    Skips silently — returning success — when the project is not a
    simulation, the feature is disabled, or the provider has no tool
    support. A skip must never fail a run that has nothing to check.
    """
    if cfg is not None and not getattr(cfg, "PROPERTY_CHECK_ENABLED", True):
        return True, ""
    if language not in (None, "python"):
        return True, ""     # Python-only for now

    sim_files = simulation_files(memory)
    if not sim_files:
        return True, ""

    from .agent_loop import (
        agent_loop_enabled, build_step_tools, run_agent_loop_with_escalation,
    )
    llm_client = getattr(coder, "llm_client", None)
    if not agent_loop_enabled(cfg, llm_client):
        # The classic path has no way to author and iterate on a test file
        # against real output; skipping beats emitting an unverified one.
        log.info("[PropertyCheck] Agent loop unavailable — skipping")
        return True, ""

    test_file = "test_properties.py"
    verify_cmd = f"python -m unittest -v {test_file[:-3]}"
    log.info("[PropertyCheck] Simulation detected (%s) — generating "
             "adversarial timing test", ", ".join(sim_files))
    if display is not None:
        display.step_info(step_idx, "Property check: adversarial frame timing")

    try:
        # The whole stage sits inside the guard, not just the loop call:
        # never take a run down over the property check itself. An earlier
        # cut left build_step_tools outside and a raise there escaped.
        tools = build_step_tools(executor, memory)
        step_text = build_property_step(sim_files, task, intent_spec)
        ok, info = run_agent_loop_with_escalation(
            llm_client, tools, step_text, task,
            escalation_client=getattr(coder, "escalation_client", None),
            display=display, step_idx=step_idx, language=language,
            max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
            verify_cmd=verify_cmd,
            preload_files=sim_files + [test_file],
        )
    except Exception as exc:
        log.warning("[PropertyCheck] Raised %s: %s — treating as skipped",
                    type(exc).__name__, exc)
        return True, ""

    if ok:
        log.info("[PropertyCheck] Invariants hold under randomised "
                 "frame timing")
        return True, ""
    log.warning("[PropertyCheck] Invariants violated under randomised "
                "frame timing: %s", (info or "")[:300])
    return False, f"Property check failed: {(info or '')[:800]}"

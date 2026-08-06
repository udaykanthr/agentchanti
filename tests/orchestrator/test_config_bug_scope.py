"""A config fix may only touch config.

Observed live (classic mode, Pac-Man task, 2026-08-05, classic_r5). Step 8
had a suite of 61 tests with a single failing assertion. The triage called
it CONFIG_BUG — "the test framework or environment is misconfigured" — and
the fix branch then wrote whatever the model returned:

    Step 8: Triage result: CONFIG_BUG                                   x5
    Step 8: CONFIG_BUG fix applied: ['map.py', 'player.py', 'ghost.py',
                                     'game.py', 'main.py']              x5

Five source modules, five times. The suite went 61 tests / 1 failure ->
62 / 1 -> 64 / 4 failures + 3 errors: a misfiled triage rewrote a
nearly-passing game and moved it further from passing each round.

Triage will always be fallible — run-wide it returned 15 CONFIG_BUG
against 4 SOURCE_BUG and 3 TEST_BUG. What must not be fallible is the
blast radius: a CONFIG_BUG fix that cannot reach source files does no
damage when it fires by mistake, and the next attempt re-triages.
"""

from __future__ import annotations

import os

import pytest

from agentchanti.language_backend import get_backend


def _apply_scope_filter(fix_files: dict[str, str],
                        candidates: list[str]) -> tuple[dict, list]:
    """The filter as applied in the CONFIG_BUG branch."""
    allowed = {c.lower() for c in candidates}
    kept = {p: c for p, c in fix_files.items()
            if os.path.basename(p).lower() in allowed}
    dropped = [p for p in fix_files if p not in kept]
    return kept, dropped


PY_CANDIDATES = get_backend("python").get_config_candidates()


def test_the_observed_source_rewrite_is_dropped_entirely():
    fix_files = {p: "..." for p in
                 ["map.py", "player.py", "ghost.py", "game.py", "main.py"]}
    kept, dropped = _apply_scope_filter(fix_files, PY_CANDIDATES)
    assert kept == {}
    assert sorted(dropped) == ["game.py", "ghost.py", "main.py",
                               "map.py", "player.py"]


def test_real_config_files_are_kept():
    fix_files = {"conftest.py": "import sys\n", "pytest.ini": "[pytest]\n"}
    kept, dropped = _apply_scope_filter(fix_files, PY_CANDIDATES)
    assert kept == fix_files
    assert dropped == []


def test_a_mixed_response_keeps_only_the_config_half():
    fix_files = {"conftest.py": "x", "player.py": "y", "setup.cfg": "z"}
    kept, dropped = _apply_scope_filter(fix_files, PY_CANDIDATES)
    assert set(kept) == {"conftest.py", "setup.cfg"}
    assert dropped == ["player.py"]


def test_a_config_file_inside_a_subproject_is_kept():
    # Paths are subproject-prefixed before filtering, so the check is on
    # the basename rather than the full path.
    fix_files = {"backend/conftest.py": "x"}
    kept, _ = _apply_scope_filter(fix_files, PY_CANDIDATES)
    assert set(kept) == {"backend/conftest.py"}


@pytest.mark.parametrize("lang,cfg,src", [
    ("python", "conftest.py", "game.py"),
    ("javascript", "jest.config.js", "src/App.jsx"),
    ("typescript", "vitest.config.ts", "src/App.tsx"),
])
def test_the_scope_follows_the_language_backend(lang, cfg, src):
    candidates = get_backend(lang).get_config_candidates()
    kept, dropped = _apply_scope_filter({cfg: "a", src: "b"}, candidates)
    assert set(kept) == {cfg}
    assert dropped == [src]


def test_an_empty_result_is_not_an_error():
    """Dropping everything is the safe outcome, not a failure.

    The step re-runs its tests afterwards and re-triages; doing nothing is
    strictly better than rewriting five source files on a bad guess.
    """
    kept, dropped = _apply_scope_filter({"main.py": "x"}, PY_CANDIDATES)
    assert kept == {}
    assert dropped == ["main.py"]


# ── a config verdict that changed nothing must not repeat ────────────

def test_a_repeat_config_verdict_is_routed_to_source():
    """Believing CONFIG_BUG twice makes the step loop without acting.

    Observed three times consecutively in classic_r6 on a step whose real
    defect was one missing method (`Ghost.get_tile_position`): triage said
    CONFIG_BUG, the model answered with source files, the scope guard
    dropped them, the attempt was spent, and triage said CONFIG_BUG again.
    Nothing was written — safe — but the retry budget drained without the
    defect ever being touched.
    """
    from agentchanti.orchestrator.step_handlers import effective_bug_origin

    # First verdict is honoured: missing config is a real thing.
    assert effective_bug_origin("CONFIG_BUG", False) == "CONFIG_BUG"
    # Repeat, after a config fix that changed nothing, goes to source.
    assert effective_bug_origin("CONFIG_BUG", True) == "SOURCE_BUG"


def test_other_verdicts_are_never_rewritten():
    from agentchanti.orchestrator.step_handlers import effective_bug_origin

    for verdict in ("SOURCE_BUG", "TEST_BUG"):
        assert effective_bug_origin(verdict, False) == verdict
        assert effective_bug_origin(verdict, True) == verdict

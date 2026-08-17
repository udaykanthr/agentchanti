"""A declared export in the wrong naming convention is the same symbol.

Every ``exports:`` example in the planner prompt is JavaScript
(``exports: app, startServer``), so a planner naming a *Python* file's
symbols answers in camelCase. Measured 2026-08-17 on the Pac-Man task:
the plan declared ``exports: main, runHeadlessSimulation,
runInteractiveGame`` for main.py, the coder correctly wrote snake_case,
and the run produced two ``violated-exports`` findings against an
artifact that passed all nine external behavioural probes.

`_export_satisfied` exists to stop exactly that: per its own history, a
finding that is always wrong is worse than none, because it trains the
reader to skip the line that will one day be right.

The bound matters as much as the fix. Only case and separators are
forgiven; a planner that invented a *different* name is still a real
disagreement and must still be reported.
"""

import pytest

from agentchanti.orchestrator.plan_graph import (
    _canonical_name, _export_satisfied,
)


# ── the same name, spelled by a different convention ─────────────────────

@pytest.mark.parametrize("spec,actual", [
    ("runHeadless", {"run_headless"}),          # the measured shape
    ("run_headless", {"runHeadless"}),          # and its mirror
    ("RunHeadless", {"run_headless"}),
    ("run-headless", {"run_headless"}),
    ("pelletsRemaining", {"pellets_remaining"}),
    ("TILE_SIZE", {"tileSize"}),
    ("GameMap", {"game_map"}),
])
def test_convention_differences_are_satisfied(spec, actual):
    assert _export_satisfied(spec, actual)


def test_prose_identifiers_also_get_convention_matching():
    """A planner answering in English AND in the wrong convention."""
    assert _export_satisfied(
        "the runHeadless helper that drives the simulation",
        {"run_headless", "main"})


# ── the bound: a different name is still a real disagreement ─────────────

@pytest.mark.parametrize("spec,actual", [
    # The other half of the measured plan: the planner did not merely
    # re-case the name, it invented a longer one.
    ("runHeadlessSimulation", {"run_headless", "main"}),
    ("runInteractiveGame", {"run_interactive", "main"}),
    ("startServer", {"run_headless", "main"}),
])
def test_invented_names_are_still_reported(spec, actual):
    assert not _export_satisfied(spec, actual)


def test_an_absent_name_in_an_empty_looking_file_is_still_reported():
    """Underscore-only actual names canonicalise to "" and must not
    become a wildcard that satisfies every declaration."""
    assert not _export_satisfied("runHeadless", {"_", "__"})


# ── the canonical key itself ─────────────────────────────────────────────

def test_canonical_name_collapses_only_case_and_separators():
    assert _canonical_name("runHeadless") == "runheadless"
    assert _canonical_name("run_headless") == "runheadless"
    assert _canonical_name("RUN_HEADLESS") == "runheadless"
    assert _canonical_name("run-headless") == "runheadless"
    # Distinct names never collide.
    assert _canonical_name("runHeadless") != _canonical_name("runHeadlessSim")
    assert _canonical_name("") == ""


# ── nothing the function already handled may regress ─────────────────────

@pytest.mark.parametrize("spec,actual", [
    ("Footer", {"Footer"}),                     # exact match
    ("default Footer", {"default"}),            # JS default export
    ("(none)", set()),                          # "no exports"
    ("Footer", {"default"}),                    # default-only file
    ("main function that prints Hello", {"main"}),   # prose
])
def test_existing_behaviour_is_unchanged(spec, actual):
    assert _export_satisfied(spec, actual)


def test_a_genuinely_missing_symbol_still_warns():
    assert not _export_satisfied("Sidebar", {"Footer", "Header"})

"""Wiring verification writes source, so its edits must be gate-checked.

Measured 2026-09-13, a single-file pygame pinball run::

    13:58:01  gate 2.1 passes (wave 2 snapshot ecfb83ab211d)
    13:58:35  [WiringVerification] Applying fixes for: ['pinball.py']
    13:58:38  [SmokeTest] App launched successfully (pinball.py)
    13:58:38  gate 2.1 fails: pygame.error: font not initialized
    13:58:39  [Monotonic] smoke-test fixes broke 1 previously-passing gate
    13:58:39  [Snapshots] Rolled back workdir to ecfb83ab211d
    13:58:39  [SmokeTest] The launch fix left a previously-passing gate red
    13:58:42  [Ghost] failed-but-clean
    13:58:45  Evidence: independent (pre-existing-tests) ... passed
              Pipeline failed.

The smoke test had fixed nothing. WiringVerification's LLM fix introduced
the regression, nothing checked it under its own name, and the run was
failed after the rollback had already restored a tree that passed the gate,
launched, and passed its independent contract 3/3.
"""
import inspect
import logging

import pytest

from agentchanti.orchestrator import cli

GATE = 'python -c "from pinball import PinballGame; PinballGame()"'
FAILURE = "pygame.error: font not initialized"


class _Snapshots:
    managed = True

    def __init__(self, rollback_ok=True):
        self.rollback_ok = rollback_ok
        self.rolled_back = False
        self.committed = []

    def commit_wave(self, stage):
        self.committed.append(stage)

    def mark_green(self):
        pass

    def rollback_to_last(self):
        if not self.rollback_ok:
            return False, "no snapshot available"
        self.rolled_back = True
        return True, "ok"


def _ledger(monkeypatch, snaps, red_after_rollback=False):
    """Red until the snapshot is rolled back; then as *red_after_rollback*."""

    class _Ledger:
        def gates(self):
            return {GATE: "2.1"}

        def recheck(self, executor, timeout=300):
            if snaps.rolled_back and not red_after_rollback:
                return []
            return [(GATE, "2.1", FAILURE)]

    monkeypatch.setattr("agentchanti.orchestrator.wave_snapshots"
                        ".get_gate_ledger", lambda: _Ledger())


def test_the_measured_run_is_no_longer_failed(monkeypatch, caplog):
    snaps = _Snapshots()
    _ledger(monkeypatch, snaps)
    with caplog.at_level(logging.WARNING):
        ok = cli._check_advisory_stage(snaps, None, "wiring fixes")
    assert snaps.rolled_back, "the wiring fix's regression must still be undone"
    assert ok is True, "a run whose rolled-back tree is green was failed"


def test_the_regression_is_blamed_on_the_stage_that_caused_it(monkeypatch,
                                                              caplog):
    snaps = _Snapshots()
    _ledger(monkeypatch, snaps)
    with caplog.at_level(logging.WARNING):
        cli._check_advisory_stage(snaps, None, "wiring fixes")
    assert "wiring fixes left 1 gate(s) red" in caplog.text
    assert "smoke-test" not in caplog.text
    assert FAILURE in caplog.text, "the rollback must still say why"


def test_a_rollback_that_could_not_happen_still_fails(monkeypatch):
    """A red gate is never reported as success."""
    snaps = _Snapshots(rollback_ok=False)
    _ledger(monkeypatch, snaps)
    assert cli._check_advisory_stage(snaps, None, "wiring fixes") is False


def test_a_tree_still_red_after_rollback_fails(monkeypatch):
    """The regression predates this stage — rolling back cannot clear it."""
    snaps = _Snapshots()
    _ledger(monkeypatch, snaps, red_after_rollback=True)
    assert cli._check_advisory_stage(snaps, None, "wiring fixes") is False


def test_a_gate_conflict_still_fails(monkeypatch):
    """Nothing is rolled back on a conflict, so the gate is still red."""
    snaps = _Snapshots()
    _ledger(monkeypatch, snaps)
    monkeypatch.setattr(cli, "_green_suites_contradicting",
                        lambda regs: [("python -m unittest discover", "4.1")])
    assert cli._check_advisory_stage(snaps, None, "wiring fixes") is False
    assert not snaps.rolled_back


def test_a_clean_stage_is_committed_and_passes(monkeypatch):
    snaps = _Snapshots()

    class _Green:
        def gates(self):
            return {GATE: "2.1"}

        def recheck(self, executor, timeout=300):
            return []

    monkeypatch.setattr("agentchanti.orchestrator.wave_snapshots"
                        ".get_gate_ledger", lambda: _Green())
    assert cli._check_advisory_stage(snaps, None, "wiring fixes") is True
    assert snaps.committed == ["wiring fixes"]
    assert not snaps.rolled_back


def test_the_pipeline_checks_wiring_before_the_smoke_test():
    """The seam is the call site: after wiring runs, before smoke launches.

    Checked in source order because `_main_impl` is the whole pipeline and
    cannot be driven in a unit test. Before, the smoke test launched the
    wiring-modified tree and its re-check inherited the blame.
    """
    src = inspect.getsource(cli._main_impl)
    wiring = src.index("run_wiring_verification(")
    check = src.index('_check_advisory_stage(snapshots, executor, "wiring fixes"')
    smoke = src.index("run_smoke_verification(")
    assert wiring < check < smoke

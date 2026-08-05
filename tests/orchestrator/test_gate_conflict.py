"""A gate that contradicts the task must not cost the work that satisfies it.

Observed live (classic mode, Pac-Man task). The plan gated step 3.1 on
`assert p.can_move()`, and entities.py implemented

    def can_move(self) -> bool:
        return self._state.direction != STOP     # "is currently MOVING"

so the gate demanded a freshly-constructed player already be in motion,
while the brief demanded "2000+ frames without the player moving". The
sequence:

  08:04:56  gate 3.1 fails -- a correct Pac-Man starts stationary
  08:05:03  diagnosis edits Player.__init__ to auto-start, annotating it
            "# Ensure the player starts in a valid moving state for
             acceptance/tests"
  08:05:42  gate 3.1 green
  08:06:5x  the suite's test_idle_run_2000_frames_without_player_moving
            fails: Tuples differ: (2, 1) != (1, 1)
  08:06:59  its fix removes the auto-start -> gate 3.1 REGRESSES
  08:07:00  rollback discards wave 6, restoring the artifact that
            violates the brief; pipeline failed

The suite is where the task's own invariants live, so when it is green and
only inline gates are red, the gate is the suspect. Keep the work, name
both sides, and still fail the run -- an unresolved red gate is never a
success.
"""

from __future__ import annotations

import unittest

from agentchanti.orchestrator.cli import (
    _enforce_monotonic_gates, _green_suites_contradicting,
)
from agentchanti.orchestrator.wave_snapshots import (
    get_gate_ledger, is_suite_gate,
)

# The two real commands from the observed run.
CAN_MOVE_GATE = (
    'python -c "from map import Map; from entities import Player, Ghost; '
    "m=Map(); p=Player(m); g=Ghost(m,0,'chase'); "
    'assert m.is_walkable(*p.tile_pos()) and p.can_move()"'
)
SUITE_GATE = "python -m unittest -v"


class TestSuiteGateDetection(unittest.TestCase):

    def test_suite_runners_are_recognised(self):
        for cmd in ("python -m unittest -v",
                    "python -m unittest discover -s tests",
                    "python -m pytest",
                    "python3.13 -m pytest -q",
                    "pytest tests/",
                    "npm test", "npm run test", "yarn test", "pnpm test",
                    "npx vitest run", "npx jest --ci",
                    "go test ./...", "cargo test"):
            self.assertTrue(is_suite_gate(cmd), cmd)

    def test_inline_assertion_gates_are_not_suites(self):
        for cmd in (CAN_MOVE_GATE,
                    'python -c "import main; assert callable(main.main)"',
                    'python -c "from map import Map; m=Map()"'):
            self.assertFalse(is_suite_gate(cmd), cmd)

    def test_hyphenated_packages_are_not_suite_runs(self):
        """`npm i jest-dom` must not read as running jest."""
        for cmd in ("npm install --save-dev jest-dom",
                    "npm i vitest-dev", "pip install pytest-cov"):
            self.assertFalse(is_suite_gate(cmd), cmd)


class TestConflictDetection(unittest.TestCase):

    def setUp(self):
        get_gate_ledger().reset()
        self.addCleanup(get_gate_ledger().reset)

    @staticmethod
    def _reg(cmd, label="3.1"):
        return [(cmd, label, "AssertionError")]

    def test_the_observed_conflict_is_detected(self):
        get_gate_ledger().record(CAN_MOVE_GATE, "3.1")
        get_gate_ledger().record(SUITE_GATE, "6.1")
        conflict = _green_suites_contradicting(self._reg(CAN_MOVE_GATE))
        self.assertEqual([c for c, _l in conflict], [SUITE_GATE])

    def test_no_suite_recorded_means_ordinary_rollback(self):
        """Without a suite there is no higher authority to prefer."""
        get_gate_ledger().record(CAN_MOVE_GATE, "3.1")
        self.assertEqual(
            _green_suites_contradicting(self._reg(CAN_MOVE_GATE)), [])

    def test_a_red_suite_means_ordinary_rollback(self):
        """A failing suite means the stage broke real behaviour."""
        get_gate_ledger().record(CAN_MOVE_GATE, "3.1")
        get_gate_ledger().record(SUITE_GATE, "6.1")
        regs = self._reg(CAN_MOVE_GATE) + self._reg(SUITE_GATE, "6.1")
        self.assertEqual(_green_suites_contradicting(regs), [],
                         "a red suite must never suppress rollback")

    def test_suite_red_alone_means_ordinary_rollback(self):
        get_gate_ledger().record(SUITE_GATE, "6.1")
        self.assertEqual(
            _green_suites_contradicting(self._reg(SUITE_GATE, "6.1")), [])


class _FakeSnapshots:
    managed = True

    def __init__(self):
        self.committed = []
        self.greens = 0
        self.rollbacks = 0

    def commit_wave(self, stage):
        self.committed.append(stage)

    def mark_green(self):
        self.greens += 1

    def rollback_to_last(self):
        self.rollbacks += 1
        return True, "ok"


class _FakeExecutor:
    """Fails exactly the commands named in *failing*."""

    last_exit_code = 1

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.ran = []

    def run_command(self, cmd, timeout=None):
        self.ran.append(cmd)
        if cmd in self.failing:
            return False, "AssertionError"
        return True, ""


class TestEnforceMonotonicGatesEndToEnd(unittest.TestCase):
    """The behaviour the loop path depends on must be untouched."""

    def setUp(self):
        get_gate_ledger().reset()
        self.addCleanup(get_gate_ledger().reset)
        get_gate_ledger().record(CAN_MOVE_GATE, "3.1")
        get_gate_ledger().record(SUITE_GATE, "6.1")

    def test_all_green_commits_and_never_consults_the_conflict_path(self):
        """r1/r3/r5 recorded ZERO regressions — this is their path."""
        snaps, ex = _FakeSnapshots(), _FakeExecutor()
        ok = _enforce_monotonic_gates(snaps, ex, "wave 3")
        self.assertTrue(ok)
        self.assertEqual(snaps.committed, ["wave 3"])
        self.assertEqual(snaps.greens, 1)
        self.assertEqual(snaps.rollbacks, 0)

    def test_the_observed_conflict_preserves_the_work(self):
        snaps = _FakeSnapshots()
        ex = _FakeExecutor(failing=[CAN_MOVE_GATE])
        ok = _enforce_monotonic_gates(snaps, ex, "wave 6")
        self.assertFalse(ok, "an unresolved red gate is never a success")
        self.assertEqual(snaps.rollbacks, 0,
                         "the fix that satisfies the suite must survive")
        self.assertEqual(snaps.committed, [])

    def test_a_red_suite_still_rolls_back(self):
        """Ordinary regressions keep the original safety behaviour."""
        snaps = _FakeSnapshots()
        ex = _FakeExecutor(failing=[CAN_MOVE_GATE, SUITE_GATE])
        ok = _enforce_monotonic_gates(snaps, ex, "wave 6")
        self.assertFalse(ok)
        self.assertEqual(snaps.rollbacks, 1,
                         "a broken suite means the code regressed, not the gate")

    def test_unmanaged_repo_is_left_alone(self):
        snaps = _FakeSnapshots()
        snaps.managed = False
        ex = _FakeExecutor(failing=[CAN_MOVE_GATE])
        self.assertTrue(_enforce_monotonic_gates(snaps, ex, "wave 6"))
        self.assertEqual(snaps.rollbacks, 0)


class TestHealthyLoopGatesAreUnaffected(unittest.TestCase):
    """The real gates from the passing loop runs must not be misread.

    r3 and r5 both assert `g.state == 'START'` on a freshly-constructed
    Game -- correct, because it asserts the INITIAL state -- and drive the
    behaviour before asserting it elsewhere (`g.start_game()`,
    `p.set_direction(...); p.update(...)`). None of them is a suite gate,
    so none can be mistaken for the task's authority.
    """

    R5_GATES = [
        'python -c "from src.map import Map, PLAYER_SPAWN; m=Map(); '
        'assert m.is_walkable(*PLAYER_SPAWN)"',
        'python -c "from src.game import Game; g=Game(headless=True); '
        "assert g.state == 'START'; g.start_game(); "
        "assert g.state == 'PLAYING'\"",
        'python -c "import main; assert hasattr(main, \'main\')"',
    ]
    R3_GATES = [
        'python -c "from map import Map; from player import Player; '
        'm=Map(); p=Player(m.player_spawn); p.set_direction((1,0)); '
        'p.update(0.05, m); assert not m.is_wall(*p.current_tile())"',
        'python -c "from game import Game; g=Game(); '
        "assert g.state=='start' and len(g.ghosts)==4\"",
    ]

    def test_none_of_the_passing_runs_gates_read_as_a_suite(self):
        for cmd in self.R5_GATES + self.R3_GATES:
            self.assertFalse(is_suite_gate(cmd), cmd)

    def test_their_unittest_gate_does_read_as_a_suite(self):
        """Both runs also record `python -m unittest -v` for the TEST step."""
        self.assertTrue(is_suite_gate("python -m unittest -v"))


if __name__ == "__main__":
    unittest.main()

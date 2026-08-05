"""A TEST step must leave behind a suite the project's runner can find.

A plan-declared gate may name a single file — observed:
``python -m unittest -v tests/test_game.py``. That proves the file runs.
It says nothing about discovery, and a benchmark run shipped exactly that
gap: the gate was green, the pipeline reported success, and the delivered
project answered ``python -m unittest -v`` with "Ran 0 tests" because
nothing made ``tests/`` a package.

The check re-runs the gate with its file scoping stripped, deliberately
NOT the language default: the Python default is pytest, which collects a
``tests/`` directory with no ``__init__.py`` at all and would have called
the broken project green.
"""

from __future__ import annotations

import unittest

from agentchanti.agent_tools import NO_TESTS_MARKER
from agentchanti.orchestrator.step_handlers import _tests_are_discoverable


class _Ex:
    """Executor whose run_command replays one scripted (ok, out, code)."""

    def __init__(self, ok=True, out="Ran 3 tests\nOK", code=0):
        self._reply = (ok, out)
        self.last_exit_code = code
        self.commands: list[str] = []

    def run_command(self, cmd, timeout=None):
        self.commands.append(cmd)
        return self._reply


class TestsDiscoverableTest(unittest.TestCase):

    def test_a_path_scoped_gate_that_collects_nothing_is_flagged(self):
        ex = _Ex(ok=True, out="Ran 0 tests in 0.000s\n\nNO TESTS RAN")
        ok, cmd, _ = _tests_are_discoverable(
            ex, "python", None, "python -m unittest -v tests/test_game.py")
        self.assertFalse(ok)
        self.assertEqual(cmd, "python -m unittest -v")

    def test_a_path_scoped_gate_that_still_collects_is_fine(self):
        ex = _Ex(ok=True, out="Ran 3 tests in 0.1s\n\nOK")
        ok, _, _ = _tests_are_discoverable(
            ex, "python", None, "python -m unittest -v tests/test_game.py")
        self.assertTrue(ok)

    def test_an_already_project_wide_gate_is_not_rerun(self):
        """It just ran green — asking again proves nothing and costs a
        whole suite run."""
        ex = _Ex()
        ok, _, _ = _tests_are_discoverable(ex, "python", None,
                                           "python -m unittest -v")
        self.assertTrue(ok)
        self.assertEqual(ex.commands, [])

    def test_a_gate_that_names_no_test_runner_is_left_alone(self):
        ex = _Ex()
        ok, _, _ = _tests_are_discoverable(
            ex, "python", None, 'python -c "import main"')
        self.assertTrue(ok)
        self.assertEqual(ex.commands, [])

    def test_a_failing_but_collecting_suite_is_not_a_discovery_problem(self):
        """The step's own gate already ruled on correctness; this check
        only ever answers "can the runner find them"."""
        ex = _Ex(ok=False, out="Ran 4 tests\n\nFAILED (failures=1)", code=1)
        ok, _, _ = _tests_are_discoverable(
            ex, "python", None, "python -m unittest -v tests/test_game.py")
        self.assertTrue(ok)

    def test_the_no_tests_marker_shape_is_also_caught(self):
        ex = _Ex(ok=True, out=f"something {NO_TESTS_MARKER} here", code=5)
        ok, _, _ = _tests_are_discoverable(
            ex, "python", None, "python -m pytest -q tests/test_game.py")
        self.assertFalse(ok)

    def test_no_gate_at_all_is_not_a_verdict(self):
        ex = _Ex()
        self.assertTrue(_tests_are_discoverable(ex, "python", None, None)[0])
        self.assertEqual(ex.commands, [])


if __name__ == "__main__":
    unittest.main()

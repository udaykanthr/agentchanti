"""A green gate is not the same as a finished step.

Observed 2026-08-12 (gpt-5.6-terra Pac-Man run, step 9). The plan declared

    target: tests/__init__.py, tests/test_map.py, tests/test_movement_invariants.py

The loop wrote the first two, `python -m unittest -v` went green on those
alone — three static map assertions — and `verified-early` ended the step on
turn 3 of 8. The run reported success and auto-committed with
`tests/test_movement_invariants.py` never created: no randomised-dt run, no
600 frames, no 2000-frame idle run, no ghost tile-centre check. The task's
own stated acceptance ("includes a test that runs the game loop with
randomised dt") was simply absent, and it was the CHEAPEST run of the four
precisely because the early exit skipped the work.

The guard only suppresses the EARLY exit. The other exits are untouched, so
it can spend turns the step already had but can never turn a passing step
into a failing one.
"""

import os
import tempfile
import unittest

from agentchanti.agent_tools import AgentTools
from agentchanti.orchestrator.agent_loop import _missing_required


class MissingRequiredTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.tools = AgentTools(project_root=self.root)

    def _touch(self, rel):
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write("x\n")

    def test_the_incident_the_third_declared_file_is_reported(self):
        self._touch("tests/__init__.py")
        self._touch("tests/test_map.py")
        self.assertEqual(
            _missing_required(self.tools, {
                "tests/__init__.py", "tests/test_map.py",
                "tests/test_movement_invariants.py"}),
            ["tests/test_movement_invariants.py"])

    def test_all_present_is_no_obstacle_to_exiting_early(self):
        self._touch("tests/__init__.py")
        self._touch("tests/test_map.py")
        self.assertEqual(
            _missing_required(self.tools,
                              {"tests/__init__.py", "tests/test_map.py"}),
            [])

    def test_no_declared_targets_never_blocks(self):
        """Most steps declare none — they must be completely unaffected."""
        for spec in (None, set(), {""}, {"   "}):
            with self.subTest(spec=spec):
                self.assertEqual(_missing_required(self.tools, spec), [])

    def test_an_empty_file_still_counts_as_written(self):
        """Existence, not content — judging content is the gate's job."""
        full = os.path.join(self.root, "empty.py")
        open(full, "w").close()
        self.assertEqual(_missing_required(self.tools, {"empty.py"}), [])

    def test_a_path_escaping_the_root_is_ignored_not_raised(self):
        """A malformed `target:` must not be able to hold a step open."""
        self.assertEqual(
            _missing_required(self.tools, {"../outside.py", "C:/etc/x.py"}),
            [])

    def test_results_are_sorted_for_a_stable_message(self):
        self.assertEqual(
            _missing_required(self.tools, {"b.py", "a.py", "c.py"}),
            ["a.py", "b.py", "c.py"])


class WiringTest(unittest.TestCase):
    """The helper is worthless if nothing calls it, and the call site is
    the thing a refactor silently drops."""

    def test_the_early_exit_consults_the_guard(self):
        import inspect
        from agentchanti.orchestrator import agent_loop
        src = inspect.getsource(agent_loop.run_agent_loop)
        self.assertIn("_missing_required(tools, required_files)", src)
        # It must gate the early exit specifically.
        early = src[src.index("Early gate:"):]
        self.assertLess(early.index("_missing_required"),
                        early.index('_finish("verified-early"'),
                        "the guard must be consulted BEFORE exiting early")

    def test_both_step_handlers_pass_the_declared_targets(self):
        import inspect
        from agentchanti.orchestrator import step_handlers
        src = inspect.getsource(step_handlers)
        self.assertEqual(
            src.count("required_files=set("), 2,
            "both the CODE and TEST loop call sites must pass targets")

    def test_run_agent_loop_accepts_the_parameter(self):
        import inspect
        from agentchanti.orchestrator.agent_loop import run_agent_loop
        self.assertIn("required_files",
                      inspect.signature(run_agent_loop).parameters)


if __name__ == "__main__":
    unittest.main()

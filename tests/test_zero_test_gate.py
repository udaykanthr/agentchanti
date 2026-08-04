"""A verification gate that collected NOTHING must not pass.

`exit: success` is not proof. unittest only gained a non-zero status for a
zero-test run in CPython 3.12, so on 3.10/3.11 a project whose test
discovery quietly broke satisfies its own gate having executed no tests —
a green verdict backed by nothing, arrived at *through* the gate rather
than around it. Both execution paths are covered here so they cannot
disagree about what "verified" means.
"""

import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import NO_TESTS_MARKER
from agentchanti.orchestrator.agent_loop import verify_passed


# What `python -m unittest -v` prints for an empty project on 3.10/3.11:
# a zero exit status and a body that says nothing ran.
ZERO_TEST_BODY = "Ran 0 tests in 0.000s\n\nOK\n"


class VerifyPassedTest(unittest.TestCase):
    def test_real_pass_is_accepted(self):
        self.assertTrue(verify_passed("exit: success\n2 passed in 0.10s"))

    def test_failure_is_rejected(self):
        self.assertFalse(verify_passed("exit: FAILED\nAssertionError"))

    def test_zero_tests_is_rejected_despite_success(self):
        """The whole point: exit 0 with nothing collected is not a pass."""
        self.assertFalse(verify_passed(
            f"exit: success\n{ZERO_TEST_BODY}\n\nNOTE: the runner exited "
            f"having {NO_TESTS_MARKER}. ..."))

    def test_marker_anywhere_in_the_body_counts(self):
        self.assertFalse(verify_passed(
            "exit: success\n... " + NO_TESTS_MARKER + " ..."))


class ToolResultCarriesTheMarkerTest(unittest.TestCase):
    """The gate can only see it because run_command labels it."""

    def _result(self, ok, out, code):
        from agentchanti.agent_tools import AgentTools
        from agentchanti.llm.chat_types import ToolCall
        ex = MagicMock()
        ex.run_command.return_value = (ok, out)
        ex.last_exit_code = code
        tools = AgentTools(project_root=".", executor=ex)
        return tools.execute(ToolCall(name="run_command",
                                      arguments={"command":
                                                 "python -m unittest -v"}))

    def test_zero_tests_on_a_zero_exit_is_labelled(self):
        """The 3.10/3.11 shape — exits 0, ran nothing."""
        result = self._result(True, ZERO_TEST_BODY, 0)
        self.assertIn(NO_TESTS_MARKER, result)
        self.assertFalse(verify_passed(result))

    def test_zero_tests_on_exit_5_is_labelled(self):
        """The 3.12+ shape — exits 5, ran nothing."""
        result = self._result(False, "no tests ran in 0.01s", 5)
        self.assertIn(NO_TESTS_MARKER, result)
        self.assertFalse(verify_passed(result))

    def test_a_genuine_pass_is_untouched(self):
        result = self._result(True, "2 passed in 0.10s", 0)
        self.assertNotIn(NO_TESTS_MARKER, result)
        self.assertTrue(verify_passed(result))

    def test_a_genuine_failure_is_not_mislabelled(self):
        result = self._result(False, "FAILED (failures=1)", 1)
        self.assertNotIn(NO_TESTS_MARKER, result)
        self.assertFalse(verify_passed(result))


class DeclaredVerifyGateTest(unittest.TestCase):
    """The classic path runs its gate through Executor, not AgentTools,
    so it needs the same check or the two paths disagree."""

    def _gate(self, ok, out, code):
        from agentchanti.orchestrator.step_handlers import (
            _gate_on_declared_verify)
        executor = MagicMock()
        executor.run_command.return_value = (ok, out)
        executor.last_exit_code = code
        plan_step = MagicMock()
        plan_step.verify_cmd = "python -m unittest -v"
        display = MagicMock()
        memory = MagicMock()
        memory.all_files.return_value = {}
        return _gate_on_declared_verify(
            True, "", plan_step, executor, memory, display, 0)

    def test_zero_tests_on_a_zero_exit_fails_the_gate(self):
        ok, info = self._gate(True, ZERO_TEST_BODY, 0)
        self.assertFalse(ok)
        self.assertIn("COLLECTED NO TESTS", info)

    def test_a_genuine_pass_still_passes(self):
        ok, _ = self._gate(True, "2 passed in 0.10s", 0)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()

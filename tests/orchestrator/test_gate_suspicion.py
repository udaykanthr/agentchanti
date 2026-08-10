"""When a step's own gate is the defect, believe the command that passes.

Twice on hello-world runs the plan's gate named something that did not
exist::

    gate: python -m pytest test_hello.py -q          -> exit 4, no such file
    ran : python -m pytest tests/test_hello_world.py -> exit 0, 2 passed

The tester had written a conventionally-named file, so the gate pointed at a
path nobody created. Diagnosis identified this correctly every round and
proposed the working command; the pipeline RAN that command, saw it pass,
then re-ran the gate and failed the step. The evidence needed to save the run
was produced and thrown away three times before the run halted.

The danger of the cure is a gate quietly replaced by something weaker, so the
substitution is behavioural and narrow: the candidate must drive the same
instrument, and BOTH commands are re-run at decision time — the gate must
still fail and the candidate must still pass. Those guards are what most of
this file pins down.
"""

import unittest
from unittest.mock import MagicMock

from agentchanti.orchestrator.gate_integrity import (
    gate_operation, prove_gate_superseded, repaired_gate, reset_repairs,
    same_gate_operation)
from agentchanti.orchestrator.pipeline import _consider_gate_superseded

GATE = "python -m pytest test_hello.py -q"
WORKING = "python -m pytest tests/test_hello_world.py"


def _runner(results):
    """A run(cmd) -> (ok, output) stub over a {cmd: passes} mapping."""
    return lambda cmd: (results.get(cmd, False), "")


class OperationIdentityTest(unittest.TestCase):
    def test_same_instrument_different_argument(self):
        self.assertEqual(gate_operation(GATE), {"pytest"})
        self.assertTrue(same_gate_operation(GATE, WORKING))

    def test_a_command_naming_no_runner_has_no_operation(self):
        """This is what stops `echo ok` becoming a test suite."""
        self.assertEqual(gate_operation("echo ok"), set())
        self.assertFalse(same_gate_operation(GATE, "echo ok"))

    def test_package_manager_scripts(self):
        self.assertEqual(gate_operation("npm test --silent"), {"npm:test"})
        self.assertTrue(same_gate_operation("npm test --silent", "npm run test"))
        self.assertFalse(same_gate_operation("npm test", "npm run build"))
        self.assertFalse(same_gate_operation("npm test", "npm install"))

    def test_different_instruments_never_match(self):
        self.assertFalse(same_gate_operation(GATE, "python -m unittest -v"))


class ProofTest(unittest.TestCase):
    def test_gate_fails_and_candidate_passes(self):
        self.assertTrue(prove_gate_superseded(
            GATE, WORKING, _runner({GATE: False, WORKING: True})))

    def test_a_gate_that_passes_is_left_alone(self):
        """Nothing to supersede — and replacing it could only weaken it."""
        self.assertFalse(prove_gate_superseded(
            GATE, WORKING, _runner({GATE: True, WORKING: True})))

    def test_a_candidate_that_also_fails_proves_nothing(self):
        self.assertFalse(prove_gate_superseded(
            GATE, WORKING, _runner({GATE: False, WORKING: False})))

    def test_a_weaker_command_is_refused(self):
        self.assertFalse(prove_gate_superseded(
            GATE, "echo ok", _runner({GATE: False, "echo ok": True})))

    def test_the_same_command_is_not_its_own_replacement(self):
        self.assertFalse(prove_gate_superseded(
            GATE, GATE, _runner({GATE: False})))

    def test_decision_uses_a_fresh_run_not_the_earlier_observation(self):
        """Files changed since; a stale pass is exactly the wrong evidence."""
        seen = []

        def run(cmd):
            seen.append(cmd)
            return (cmd == WORKING), ""

        prove_gate_superseded(GATE, WORKING, run)
        self.assertIn(GATE, seen)
        self.assertIn(WORKING, seen)


class DiagnosisLoopIntegrationTest(unittest.TestCase):
    """The wiring: a failed round hands its passing commands to the check."""

    def setUp(self):
        reset_repairs()
        self.addCleanup(reset_repairs)
        self.plan_step = MagicMock(verify_cmd=GATE)
        self.executor = MagicMock()
        self.executor.run_command.side_effect = \
            lambda cmd, **kw: ((cmd == WORKING), "")

    def test_a_repair_is_recorded_after_repeated_failure(self):
        recorded = _consider_gate_superseded(
            self.plan_step, [WORKING], self.executor, 0, diag_attempt=2)
        self.assertTrue(recorded)
        self.assertEqual(repaired_gate(GATE), WORKING)

    def test_the_first_failure_is_just_a_red_test(self):
        """One failing round is normal; only a survivor is suspicious."""
        self.assertFalse(_consider_gate_superseded(
            self.plan_step, [WORKING], self.executor, 0, diag_attempt=1))
        self.assertIsNone(repaired_gate(GATE))

    def test_nothing_passed_means_nothing_to_believe(self):
        self.assertFalse(_consider_gate_superseded(
            self.plan_step, [], self.executor, 0, diag_attempt=3))

    def test_a_step_with_no_gate_is_skipped(self):
        self.assertFalse(_consider_gate_superseded(
            MagicMock(verify_cmd=None), [WORKING], self.executor, 0,
            diag_attempt=3))

    def test_an_unrelated_passing_command_is_not_adopted(self):
        """`pip install x` succeeding says nothing about the suite."""
        self.assertFalse(_consider_gate_superseded(
            self.plan_step, ["pip install pytest"], self.executor, 0,
            diag_attempt=3))
        self.assertIsNone(repaired_gate(GATE))

    def test_an_executor_that_raises_never_breaks_the_loop(self):
        boom = MagicMock()
        boom.run_command.side_effect = OSError("gone")
        self.assertFalse(_consider_gate_superseded(
            self.plan_step, [WORKING], boom, 0, diag_attempt=3))


class WiringTest(unittest.TestCase):
    """The tests above call the helper directly, so they stay green even if
    the diagnosis loop never calls it — which would restore the original bug
    in full. Pin the call site itself."""

    def test_the_diagnosis_loop_consults_it_on_a_failed_round(self):
        import inspect

        from agentchanti.orchestrator import pipeline
        src = inspect.getsource(pipeline._run_diagnosis_loop)
        self.assertIn("_consider_gate_superseded(", src,
                      "the diagnosis loop no longer consults the gate check")
        self.assertIn("_fix_cmds_passed", src,
                      "the passing fix commands are not reaching the check")

    def test_apply_fix_still_reports_which_commands_passed(self):
        """The whole check is fed by that list; a 4-tuple return breaks it."""
        import inspect

        from agentchanti.orchestrator import diagnosis
        src = inspect.getsource(diagnosis._apply_fix)
        self.assertIn("passed_cmds", src)


if __name__ == "__main__":
    unittest.main()

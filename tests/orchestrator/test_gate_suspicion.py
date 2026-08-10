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

    def test_an_installer_is_never_an_operation(self):
        """`pip install pytest` mentions the runner but verifies NOTHING.

        An earlier version scanned every token for a known runner name, so
        the package being installed was read as the instrument — and since
        that command exits 0 whenever pytest is already present, it could
        have been adopted as a stand-in for the suite.
        """
        for cmd in ("pip install pytest",
                    "pip3 install pytest",
                    "pip install -r requirements.txt",
                    "python -m pip install --upgrade pip",
                    "python -m venv venv",
                    "npm install", "npm ci", "yarn add jest"):
            with self.subTest(cmd=cmd):
                self.assertEqual(gate_operation(cmd), set())
                self.assertFalse(same_gate_operation(GATE, cmd))

    def test_only_a_positional_runner_counts(self):
        """A runner NAMED in passing is not a runner being invoked."""
        self.assertEqual(gate_operation("echo pytest"), set())

    def test_subcommand_runners_distinguish_their_verbs(self):
        self.assertEqual(gate_operation("go test ./..."), {"go:test"})
        self.assertFalse(same_gate_operation("go test ./...", "go build"))

    def test_run_wrappers_resolve_to_the_runner(self):
        for cmd in ("poetry run pytest", "uv run pytest -q", "pipenv run pytest"):
            with self.subTest(cmd=cmd):
                self.assertEqual(gate_operation(cmd), {"pytest"})

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

    def test_a_passing_installer_is_not_adopted(self):
        """`pip install pytest` exits 0 once pytest exists — and proves
        nothing. Note the stub makes it PASS, so this fails unless the
        operation check itself rejects it; an earlier version of this test
        let the command fail and so passed for the wrong reason."""
        everything_passes = MagicMock()
        everything_passes.run_command.side_effect = \
            lambda cmd, **kw: ((cmd != GATE), "")
        self.assertFalse(_consider_gate_superseded(
            self.plan_step, ["pip install pytest"], everything_passes, 0,
            diag_attempt=3))
        self.assertIsNone(repaired_gate(GATE))

    def test_a_passing_echo_is_not_adopted(self):
        everything_passes = MagicMock()
        everything_passes.run_command.side_effect = \
            lambda cmd, **kw: ((cmd != GATE), "")
        self.assertFalse(_consider_gate_superseded(
            self.plan_step, ["echo ok"], everything_passes, 0, diag_attempt=3))
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

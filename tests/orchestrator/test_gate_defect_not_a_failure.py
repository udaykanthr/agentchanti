"""A gate proven not to measure the artifact cannot fail the run.

`observe_gate_verdict` trips only on repeated byte-identical failing
verdicts across distinct artifact digests, where the failure never
reached the code. That is a proof that the gate's answer does not depend
on what the step wrote — so the red verdict is not evidence about the
code in EITHER direction, and must not be the thing that fails a run.

Measured 2026-09-25, pre-scaffolded Next.js, two of three runs::

    [GateIntegrity] gate STALLED - 8 identical failing verdicts over 2
                    different versions of the code
    [Evidence] independent (pre-existing-tests) - ... passed
    Pipeline failed.

Both artifacts build, serve 200, and have had every line of Next's
starter copy removed — the exact property the gates were written to
assert and could not express.

The exclusion matters as much as the rule: a step refused BEFORE turn 1
carries the same marker, and there nothing was built, so there is no
artifact for the run to stand on.
"""
from agentchanti.orchestrator.agent_loop import (GATE_STALLED_MARKER,
                                                 GATE_UNSTARTED_NOTE,
                                                 gate_proven_not_measuring)

STALLED_AFTER_WORK = (
    f"{GATE_STALLED_MARKER} the gate returned the IDENTICAL failure 8 times "
    f"while the code changed 2 times, so it is not measuring the artifact\n\n"
    f"exit: 1")
NEVER_STARTED = (
    f"{GATE_STALLED_MARKER} {GATE_UNSTARTED_NOTE}: its verify command cannot "
    f"run on this platform, so no edit could change the verdict.")


class TestTheMeasuredCase:

    def test_a_stalled_gate_after_real_work_is_not_a_failure(self):
        assert gate_proven_not_measuring(STALLED_AFTER_WORK)


class TestTheExclusions:

    def test_a_step_that_never_started_still_fails(self):
        """Nothing was built, so there is no artifact to stand on."""
        assert not gate_proven_not_measuring(NEVER_STARTED)

    def test_an_ordinary_failure_still_fails(self):
        assert not gate_proven_not_measuring(
            "Verification still failing after 10 turns:\nAssertionError")

    def test_a_recovery_failure_still_fails(self):
        assert not gate_proven_not_measuring(
            "[agent-loop-recovery-failed] the command still fails")

    def test_no_error_text_is_not_a_gate_defect(self):
        assert not gate_proven_not_measuring("")
        assert not gate_proven_not_measuring(None)


class TestTheCliActsOnIt:
    """The rule is worthless if nothing consumes it — the mistake
    `protect_acceptance_files` made, where the guard existed and was
    called nowhere outside a test."""

    def test_cli_imports_and_calls_it(self):
        import inspect

        from agentchanti.orchestrator import cli
        src = inspect.getsource(cli)
        assert "gate_proven_not_measuring" in src
        # ...and on the failure path, not merely imported.
        assert src.count("gate_proven_not_measuring") >= 2
        assert "gate_defect_steps" in src

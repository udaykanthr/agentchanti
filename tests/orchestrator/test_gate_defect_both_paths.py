"""A gate proven not to measure must not halt EITHER failure path.

`gate_proven_not_measuring` says a gate's red verdict is not evidence
about the artifact in either direction, so it cannot be the thing that
fails a run. The first release of that guard was wired into one of the two
places `_main_impl` handles a failed step, and the other one still called
`_run_diagnosis_loop` — which has its own `GATE_STALLED_MARKER` branch
that returns False, failing the step and halting the pipeline.

Measured 2026-09-30 on a C benchmark run. The plan's gate carried Makefile
escaping into a shell::

    d=$$(mktemp -d); gcc ... && (cd "$$d" && /tmp/todo-test add 'Buy milk')

`$$` is the PID in a shell, so `d=$$(mktemp -d)` is a syntax error and no
output of the step could ever change the verdict. Every layer did its job:
`GateIntegrity` reported "gate STALLED — 3 identical failing verdicts over
2 different versions of the code", and recovery was correctly skipped
because it would be held to the same gate. Then the run stopped at step 3
of 5 regardless, leaving 37 of 43 postconditions never evaluated.

The same shape as `phantom_root_manifest_reason`: a guard is only as
strong as the weakest path that reaches the thing it guards.

These tests read the source, because the behaviour they pin is *which
call sites are protected* — a structural fact about `_main_impl`, which
takes a live LLM, an executor and a plan to run. A helper nothing calls is
the mistake `protect_acceptance_files` already made once.
"""
import inspect
import re

from agentchanti.orchestrator import cli


def _source():
    return inspect.getsource(cli)


class TestBothFailurePathsAreGuarded:

    def test_there_are_exactly_two_diagnosis_call_sites(self):
        """If a third appears, it needs the guard too — and this test is
        how anyone adding one finds that out."""
        src = _source()
        n = len(re.findall(r"^\s*fixed = _run_diagnosis_loop\(", src,
                           re.MULTILINE))
        assert n == 2, (
            f"{n} call sites to _run_diagnosis_loop; each one needs the "
            f"gate-defect check ahead of it")

    def test_every_diagnosis_call_is_preceded_by_the_guard(self):
        """The guard must come BEFORE the diagnosis loop at each site.

        Order is the whole point: `_run_diagnosis_loop` has its own
        GATE_STALLED_MARKER branch that returns False, so reaching it at
        all means the step fails.
        """
        src = _source()
        guard_positions = [m.start() for m in
                           re.finditer(r"_gate_defect_continue\(idx, error_info\)",
                                       src)]
        call_positions = [m.start() for m in
                          re.finditer(r"fixed = _run_diagnosis_loop\(", src)]
        assert len(guard_positions) == 2, guard_positions
        assert len(call_positions) == 2, call_positions
        for call in call_positions:
            earlier = [g for g in guard_positions if g < call]
            assert earlier, (
                "a _run_diagnosis_loop call with no gate-defect check "
                "before it — a stalled gate would halt the run there")
            # The nearest guard must be close by, not one from far above.
            assert call - max(earlier) < 1200, (
                "the nearest gate-defect check is too far above this "
                "_run_diagnosis_loop call to be guarding it")

    def test_the_guard_is_one_helper_not_two_copies(self):
        """Two copies drift. The codebase's own resolution for this is a
        single shared object — `references_subproject`, `EMPTY_SUITE_RE`."""
        src = _source()
        assert src.count("def _gate_defect_continue(") == 1

    def test_the_helper_defers_to_gate_proven_not_measuring(self):
        """It must not re-implement the condition, which would let the two
        disagree about the same error string."""
        src = _source()
        body = src[src.index("def _gate_defect_continue("):]
        body = body[:body.index("\n    # Clear any lingering")]
        assert "gate_proven_not_measuring(error_info)" in body
        assert "GATE_STALLED_MARKER" not in body, (
            "the marker check belongs in gate_proven_not_measuring, not "
            "duplicated here")

    def test_the_guarded_step_is_recorded_and_marked_done(self):
        """Marked done so the run continues, and recorded so the summary
        can name the verify: lines to fix — silence would make this
        indistinguishable from a step that genuinely passed."""
        src = _source()
        body = src[src.index("def _gate_defect_continue("):]
        body = body[:body.index("\n    # Clear any lingering")]
        assert "gate_defect_steps.append((idx, error_info))" in body
        assert 'display.complete_step(idx, "done")' in body
        assert 'step_results[idx] = "done"' in body

    def test_the_sequential_path_checkpoints_like_the_success_path(self):
        """A step marked done that is not checkpointed would be re-run on
        resume, against the same unsatisfiable gate."""
        src = _source()
        branch = src[src.index("elif _gate_defect_continue(idx, error_info):"):]
        branch = branch[:branch.index("            else:")]
        assert "save_checkpoint(" in branch

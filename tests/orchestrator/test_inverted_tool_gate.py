"""A precondition written into the `verify:` slot.

Measured 2026-10-07 on a live Blender run. The task was *"add a cube named
Spinner and animate it"*; the planner's gate was::

    verify: mcp:blender__execute_code {"code":"import bpy;
      assert bpy.data.objects.get('Spinner') is None,
      'Spinner already exists; preserve it and do not create a replacement'"}

It asserts the Spinner must NOT exist, while the step's whole job is to
create it. It passes exactly when the step has failed and fails exactly when
the step has succeeded. The artifact was **correct** -- an independent
acceptance check scored 13/13 on it -- and the run reported `Pipeline
failed` after 10 turns and 73k tokens, with the acceptance command never
reached (`acceptance-commands-not-run`).

Nothing existing could see it:

* `_always_fails_error` proves inversion from cmd.exe's own operator
  semantics (`&&`, `||`, `exit /b`). This is a Python assertion inside a
  tool payload.
* `observe_gate_verdict` correctly abstained: an `AssertionError` counts as
  "reached the code", a condition added after a measured false positive.
* `shallow_tool_gate_reason` asks whether the gate CAN fail. It can. This is
  the opposite problem.

Underneath sat three file-shaped assumptions in the loop's bookkeeping, of
which this is the third (`_artifact_digest` was the first two):

* the early gate is guarded on `edited_files`, which a tool-only step never
  appends to -- so it never ran, and the gate was first evaluated only
  AFTER the work;
* `_dirty_since_gate` was set by `run_command` and the file writers only, so
  an MCP call could not invalidate a cached GREEN verdict.

The proof is empirical rather than textual: **a gate that passed before the
step acted and fails once it has acted is measuring the absence of the
step's own work.** Taking that baseline is deliberately NOT an early exit --
a gate passing before any work is either a step already satisfied or a
precondition, and exiting green on it would turn a false RED into a false
GREEN, which is worse.
"""
import pytest

from agentchanti.orchestrator import agent_loop


class TestTheBookkeepingIsNoLongerFileShaped:
    def _src(self):
        import inspect
        return inspect.getsource(agent_loop.run_agent_loop)

    def test_an_mcp_call_marks_the_gate_dirty(self):
        """Without this a cached GREEN verdict outlives the tool call that
        invalidated it -- a stale pass, the worst direction."""
        src = self._src()
        i = src.index("MCP_NAME_SEPARATOR in _tc.name")
        window = src[i:i + 700]
        assert "_dirty_since_gate = True" in window

    def test_an_mcp_call_counts_as_the_step_acting(self):
        src = self._src()
        i = src.index("MCP_NAME_SEPARATOR in _tc.name")
        assert "_tool_acted = True" in src[i:i + 700]

    def test_the_early_gate_no_longer_requires_a_file_edit(self):
        """`edited_files` answered 'did the step do anything' with 'did it
        write a file'. A tool-only step writes none."""
        src = self._src()
        assert "(edited_files or _tool_acted)" in src

    def test_run_command_and_writes_are_left_alone(self):
        """Widening `_tool_acted` to run_command would change the early-gate
        behaviour of ordinary runs, which is a separate question from the
        defect measured here. Exactly one place sets it."""
        src = self._src()
        assert src.count("_tool_acted = True") == 1


class TestTheBaselineIsOnlyTakenWhereItIsNeeded:
    def _src(self):
        import inspect
        return inspect.getsource(agent_loop.run_agent_loop)

    def test_the_baseline_runs_only_for_a_tool_gate(self):
        """An unconditional baseline would mean a full pytest suite at the
        start of every CODE step -- changing the cost profile of every
        ordinary run to fix a defect only ever observed on tool gates, which
        a shell gate's `_always_fails_error` already covers without running
        anything."""
        src = self._src()
        assert "if verify_cmd and _is_tool_gate(verify_cmd):" in src

    def test_passing_before_the_work_is_not_an_early_exit(self):
        """The load-bearing refusal. A gate passing before any work is
        either a step already satisfied or a precondition, and the two are
        indistinguishable until the step acts -- so exiting green on it
        would turn the measured false RED into a false GREEN."""
        src = self._src()
        i = src.index("_gate_before_work = verify_passed(")
        window = src[i:i + 1200]
        assert "_finish(" not in window, (
            "the baseline gate must not end the step")
        assert "continuing" in window

    def test_the_baseline_does_not_poison_the_cache(self):
        """The baseline measures the PRE-work state, which is not the state
        any later question is about."""
        src = self._src()
        i = src.index("_gate_before_work = verify_passed(")
        window = src[i:i + 1200]
        assert "_gate_cache = None" in window
        assert "_dirty_since_gate = True" in window


class TestInversionIsProvenFromTwoVerdicts:
    def _src(self):
        import inspect
        return inspect.getsource(agent_loop.run_agent_loop)

    def test_passed_before_and_fails_after_is_the_proof(self):
        src = self._src()
        assert "_gate_before_work is True and not verify_passed(_early)" in src

    def test_it_sets_the_stalled_reason_not_a_failure(self):
        """A failing measurement from an instrument proven not to be
        measuring is not evidence of failure, so it must not fail the run --
        the reasoning `gate_proven_not_measuring` already carries."""
        src = self._src()
        i = src.index("_gate_before_work is True and not verify_passed(_early)")
        window = src[i:i + 900]
        assert "_stalled_reason = (" in window

    def test_it_does_not_overwrite_an_existing_stall_reason(self):
        src = self._src()
        i = src.index("_gate_before_work is True and not verify_passed(_early)")
        assert "_stalled_reason is None" in src[i - 200:i + 200]

    def test_the_reason_names_the_gate_so_it_can_be_fixed(self):
        src = self._src()
        i = src.index("_gate_before_work is True and not verify_passed(_early)")
        window = src[i:i + 900]
        assert "truncate_middle(verify_cmd" in window

    def test_a_stalled_gate_still_reads_as_not_measuring(self):
        """The marker the rest of the pipeline keys on, so an inverted gate
        is handled exactly as a stalled one: step done, escalation
        suppressed, verdict left to the ghost and the evidence check."""
        marker = agent_loop.GATE_STALLED_MARKER
        assert agent_loop.gate_proven_not_measuring(f"{marker} inverted gate")
        assert not agent_loop.gate_proven_not_measuring("ordinary failure")


class TestTheMeasuredGateIsRecognisable:
    """The real gate from the incident, parsed by the real parser."""

    def test_it_is_a_tool_gate(self):
        import json

        from agentchanti.orchestrator import tool_gates

        class _Bridge:
            def owns(self, name):
                return name == "blender__execute_code"

        gate = ("mcp:blender__execute_code " + json.dumps({
            "code": "import bpy; assert bpy.data.objects.get('Spinner')"
                    " is None, 'Spinner already exists'"}))
        gates = tool_gates.parse(gate, _Bridge())
        assert gates is not None and gates[0].tool == "blender__execute_code"

    def test_the_falsifiability_check_does_flag_it(self):
        """Worth recording because I expected the opposite. `assert x is
        None` is the EXISTENCE shape `_falsifiable_assertions` already
        refuses, so this gate is reported shallow -- the operator gets a
        warning naming it.

        That warning is not a fix, though: it is advisory and the run still
        failed on the gate. The inversion proof is what stops a correct
        artifact being failed by it, and the two are independent -- a gate
        can be inverted while carrying a perfectly falsifiable assertion
        (`assert len(scene.objects) == 5` before a step that adds one)."""
        import json

        from agentchanti.orchestrator import tool_gates

        class _Bridge:
            def owns(self, name):
                return name == "blender__execute_code"

        gate = ("mcp:blender__execute_code " + json.dumps({
            "code": "import bpy; assert bpy.data.objects.get('Spinner')"
                    " is None, 'Spinner already exists'"}))
        gates = tool_gates.parse(gate, _Bridge())
        assert tool_gates.shallow_tool_gate_reason(gates)

    def test_an_inverted_gate_can_still_be_falsifiable(self):
        """So the shallow check cannot stand in for the inversion proof: a
        count assertion is falsifiable AND inverted when the step's job is
        to change the count."""
        import json

        from agentchanti.orchestrator import tool_gates

        class _Bridge:
            def owns(self, name):
                return name == "blender__execute_code"

        gate = ("mcp:blender__execute_code " + json.dumps({
            "code": "import bpy; assert len(bpy.context.scene.objects) == 5"}))
        gates = tool_gates.parse(gate, _Bridge())
        assert tool_gates.shallow_tool_gate_reason(gates) is None

    @pytest.mark.parametrize("payload,shallow", [
        ("import bpy; assert bpy.data.objects.get('X') is None", True),
        ("import bpy; assert len(bpy.data.objects) == 6", False),
    ])
    def test_bare_is_none_is_existence_and_does_not_count(self, payload,
                                                          shallow):
        """`assert x is None` is the existence shape the falsifiability
        analysis already refuses -- so a gate made ONLY of that is caught
        earlier; the incident's gate escaped because it carries a message
        and the analysis reads the test, not the message."""
        import json

        from agentchanti.orchestrator import tool_gates

        class _Bridge:
            def owns(self, name):
                return name == "blender__execute_code"

        gates = tool_gates.parse(
            "mcp:blender__execute_code " + json.dumps({"code": payload}),
            _Bridge())
        got = tool_gates.shallow_tool_gate_reason(gates)
        assert bool(got) is shallow

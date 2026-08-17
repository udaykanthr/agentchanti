"""The three defects behind the 2026-08-17 467k-token failure.

One run failed 20 times across two models on a gate that could never
have passed. Three separate mechanisms had to miss it for that to
happen, and each is tested here against the real gate:

1. the plan optimizer merged five steps and kept the one unrunnable gate
   of the five, discarding four that worked,
2. no check knew that `> /dev/null`, `timeout N <cmd>` and `wait` are not
   things cmd.exe can do,
3. nothing noticed that twenty identical verdicts about seventeen
   different versions of the file could not all be about the code.
"""

import os

import pytest

from agentchanti.orchestrator import gate_integrity
from agentchanti.orchestrator.gate_integrity import (
    observe_gate_verdict,
    platform_equivalent_variants,
    posix_only_idiom_reason,
    reset_gate_verdicts,
)
from agentchanti.orchestrator.plan_optimizer import _merged_gate
from agentchanti.orchestrator.plan_step import PlanStep, unrunnable_gate_reason


# The gate that burned the run, verbatim from step 2.1 of the 23:46 plan.
BROKEN_GATE = (
    'python main.py > /dev/null 2>&1 & timeout 3 python -c "import time; '
    "time.sleep(1); import psutil; procs = [p for p in psutil.process_iter() "
    "if 'python' in p.name().lower()]; assert len(procs) > 0\" & wait"
)

# Four gates the same run wrote for the same file, all of which ran fine.
GOOD_GATES = [
    'python -c "from main import CubeCollectorGame; g = CubeCollectorGame(); '
    'assert len(g.snake_segments) == 3"',
    'python -c "from main import CubeCollectorGame; g = CubeCollectorGame(); '
    "assert hasattr(g, '_on_game_over')\"",
    'python -c "from main import CubeCollectorGame; g = CubeCollectorGame(); '
    'assert g.camera is not None"',
    'python -c "from main import CubeCollectorGame; g = CubeCollectorGame(); '
    'assert g.food_bob_offset != 0"',
]


@pytest.fixture
def on_windows(monkeypatch):
    """Force the nt branch — these idioms are only fatal under cmd.exe."""
    monkeypatch.setattr(gate_integrity.os, "name", "nt")


@pytest.fixture
def on_posix(monkeypatch):
    monkeypatch.setattr(gate_integrity.os, "name", "posix")


class TestPosixIdiomsUnderCmd:
    @pytest.mark.parametrize("cmd,fragment", [
        ("python x.py > /dev/null 2>&1", "/dev/null"),
        ("python x.py 2> /dev/null", "/dev/null"),
        ('timeout 3 python -c "print(1)"', "timeout"),
        ("python x.py & wait", "wait"),
    ])
    def test_each_idiom_is_caught(self, on_windows, cmd, fragment):
        reason = posix_only_idiom_reason(cmd)
        assert reason is not None, cmd
        assert fragment in reason

    def test_the_measured_gate_is_unrunnable(self, on_windows):
        assert posix_only_idiom_reason(BROKEN_GATE) is not None
        # And the plan-time check that should have refused it now does.
        assert unrunnable_gate_reason(BROKEN_GATE) is not None

    def test_silent_on_posix_where_they_are_correct(self, on_posix):
        assert posix_only_idiom_reason(BROKEN_GATE) is None

    @pytest.mark.parametrize("cmd", GOOD_GATES + [
        "python -m unittest",
        "npm test",
        "python main.py --headless --frames 3",
        "timeout /t 2 /nobreak",           # the real Windows spelling
        'python -c "assert \'timeout 3 x\' in open(\'a\').read()"',
    ])
    def test_working_gates_are_untouched(self, on_windows, cmd):
        assert posix_only_idiom_reason(cmd) is None, cmd

    def test_a_cmd_dialect_variant_is_offered(self, on_windows):
        variants = dict((r, v) for r, v in
                        platform_equivalent_variants(BROKEN_GATE))
        assert "posix-shell-idioms" in variants
        translated = variants["posix-shell-idioms"]
        assert "/dev/null" not in translated
        assert "> nul" in translated
        assert not translated.rstrip().endswith("wait")
        # The variant must itself be runnable, or it proves nothing.
        assert posix_only_idiom_reason(translated) is None


class TestMergeKeepsEveryGate:
    def _group(self):
        steps = [PlanStep(id="2.1", step_type="CODE", verify_cmd=BROKEN_GATE)]
        for i, cmd in enumerate(GOOD_GATES):
            steps.append(PlanStep(id=f"3.{i + 2}", step_type="CODE",
                                  verify_cmd=cmd))
        return steps

    def test_the_four_working_gates_survive_the_merge(self, on_windows):
        merged = _merged_gate(self._group())
        for cmd in GOOD_GATES:
            assert cmd in merged, cmd

    def test_the_unrunnable_gate_is_dropped_not_conjoined(self, on_windows):
        # Conjoining it would poison all four: `A && B` never passes when
        # A cannot. This is the incident made worse, so it must not happen.
        merged = _merged_gate(self._group())
        assert "/dev/null" not in merged
        assert "psutil" not in merged
        assert unrunnable_gate_reason(merged) is None

    def test_gates_are_conjoined_with_and(self):
        group = [PlanStep(id="1.1", step_type="CODE", verify_cmd="cmd-a"),
                 PlanStep(id="1.2", step_type="CODE", verify_cmd="cmd-b")]
        assert _merged_gate(group) == "cmd-a && cmd-b"

    def test_redundant_gates_are_not_repeated(self):
        group = [
            PlanStep(id="1.1", step_type="CODE",
                     verify_cmd="python -m unittest"),
            PlanStep(id="1.2", step_type="CODE",
                     verify_cmd="python -m unittest -v"),
        ]
        merged = _merged_gate(group)
        assert merged.count("unittest") == 1

    def test_steps_without_a_gate_contribute_nothing(self):
        group = [PlanStep(id="1.1", step_type="CODE", verify_cmd=""),
                 PlanStep(id="1.2", step_type="CODE", verify_cmd="cmd-b")]
        assert _merged_gate(group) == "cmd-b"

    def test_all_gates_unrunnable_yields_no_gate(self, on_windows):
        group = [PlanStep(id="1.1", step_type="CODE", verify_cmd=BROKEN_GATE)]
        assert _merged_gate(group) == ""


class TestStallBreaker:
    def setup_method(self):
        reset_gate_verdicts()

    FAIL = "exit: failure\nThe system cannot find the path specified."

    def test_three_identical_failures_over_changed_code_trips_it(self):
        assert observe_gate_verdict("g", self.FAIL, "digest-1") is None
        assert observe_gate_verdict("g", self.FAIL, "digest-2") is None
        reason = observe_gate_verdict("g", self.FAIL, "digest-3")
        assert reason is not None
        assert "not measuring the artifact" in reason

    def test_it_reports_only_once(self):
        for d in ("d1", "d2", "d3"):
            observe_gate_verdict("g", self.FAIL, d)
        assert observe_gate_verdict("g", self.FAIL, "d4") is None

    def test_unchanged_code_is_not_evidence(self):
        # The half that makes this evidence rather than impatience: a
        # model that edited nothing would otherwise look like a broken
        # gate.
        for _ in range(6):
            assert observe_gate_verdict("g", self.FAIL, "same") is None

    def test_a_varying_message_means_the_gate_reads_the_code(self):
        for i in range(6):
            assert observe_gate_verdict(
                "g", f"exit: failure\nAssertionError line {i}",
                f"d{i}") is None

    def test_a_gate_that_ever_passed_is_never_stalled(self):
        observe_gate_verdict("g", "exit: success", "d0")
        for i in range(5):
            assert observe_gate_verdict("g", self.FAIL, f"d{i}") is None

    def test_gates_are_tracked_independently(self):
        for d in ("d1", "d2", "d3"):
            observe_gate_verdict("gate-a", self.FAIL, d)
        assert observe_gate_verdict("gate-b", self.FAIL, "d1") is None

    def test_reset_clears_observations(self):
        for d in ("d1", "d2", "d3"):
            observe_gate_verdict("g", self.FAIL, d)
        reset_gate_verdicts()
        assert observe_gate_verdict("g", self.FAIL, "d1") is None

    def test_no_command_is_never_stalled(self):
        for d in ("d1", "d2", "d3"):
            assert observe_gate_verdict(None, self.FAIL, d) is None


class TestRefusedBeforeTurnOne:
    """The advisory that nothing consumed.

    Measured 2026-08-18: the plan-time check named this gate unrunnable
    at 00:38:48, and the run then spent 180k tokens and 14 turns across
    two models proving it right. The same conclusion the stall breaker
    reaches after three verdicts is available before the first token.
    """

    def setup_method(self):
        from agentchanti.orchestrator.gate_integrity import reset_repairs
        reset_gate_verdicts()
        reset_repairs()

    def _run(self, monkeypatch, gate, on_windows=True):
        """Run the loop against stubs that answer with no tool calls.

        Anything past the refusal therefore ends at "no-tools", which is
        enough to tell "the loop was entered" from "the step was refused
        before turn 1" without a live model.
        """
        from agentchanti.llm.chat_types import ChatResponse
        from agentchanti.orchestrator import agent_loop

        monkeypatch.setattr(gate_integrity.os, "name",
                            "nt" if on_windows else "posix")
        started = []

        def _spy(tools, *a, **kw):
            started.append(True)
            return ""

        monkeypatch.setattr(agent_loop, "_preload_target_files", _spy)
        monkeypatch.setattr(agent_loop, "_preload_listing",
                            lambda *_a, **_kw: "")

        class _Tools:
            project_root = "."

            def definitions(self):
                return []

        class _Client:
            def chat(self, messages, tools=None):
                return ChatResponse(text="nothing to do")

        ok, info = agent_loop.run_agent_loop(
            _Client(), _Tools(), "step", "task", verify_cmd=gate, step_idx=0)
        return ok, info, started

    def test_a_gate_with_no_runnable_reading_ends_the_step_at_once(
            self, monkeypatch):
        from agentchanti.orchestrator import agent_loop

        # `wait` alone: nothing to translate to, so no runnable variant.
        ok, info, started = self._run(monkeypatch, "wait")
        assert ok is False
        assert agent_loop.GATE_STALLED_MARKER in info
        assert "never started" in info
        assert started == []            # not one turn was taken

    def test_the_refusal_suppresses_escalation_too(self, monkeypatch):
        from agentchanti.orchestrator import agent_loop

        _ok, info, _ = self._run(monkeypatch, "wait")
        # Same marker the escalation wrapper checks, so a stronger model
        # is never sent at a gate its shell cannot run either.
        assert agent_loop.GATE_STALLED_MARKER in info

    def test_a_translatable_gate_is_repaired_and_the_step_runs(
            self, monkeypatch):
        from agentchanti.orchestrator.gate_integrity import repaired_gate

        ok, _info, started = self._run(
            monkeypatch, 'python -c "assert 1" > /dev/null')
        # The loop was entered: an unrunnable gate with a working
        # equivalent is a defective instrument, not a dead step.
        assert started == [True]
        assert repaired_gate('python -c "assert 1" > /dev/null') is not None
        assert ok is False              # no LLM behind the stub

    def test_a_runnable_gate_is_untouched(self, monkeypatch):
        _ok, _info, started = self._run(monkeypatch, "python -m unittest")
        assert started == [True]

    def test_no_gate_at_all_is_not_refused(self, monkeypatch):
        _ok, _info, started = self._run(monkeypatch, None)
        assert started == [True]

    def test_on_posix_the_same_gate_runs(self, monkeypatch):
        _ok, _info, started = self._run(
            monkeypatch, "python x.py > /dev/null", on_windows=False)
        assert started == [True]

    def test_the_refusal_is_recorded_as_a_zero_turn_run(self, monkeypatch):
        from agentchanti.orchestrator.agent_loop import (
            get_loop_stats, reset_attempt_journal,
        )

        reset_attempt_journal()
        before = len(get_loop_stats())
        self._run(monkeypatch, "wait")
        runs = get_loop_stats()[before:]
        assert len(runs) == 1
        assert runs[0]["turns"] == 0
        assert runs[0]["outcome"] == "gate-unrunnable"


class TestEscalationIsSuppressed:
    def test_a_stalled_gate_does_not_reach_the_stronger_model(self,
                                                              monkeypatch):
        from agentchanti.orchestrator import agent_loop

        calls = []

        def fake_loop(client, tools, step_text, task, **kw):
            calls.append(client)
            return (False, f"{agent_loop.GATE_STALLED_MARKER} gate is broken")

        monkeypatch.setattr(agent_loop, "run_agent_loop", fake_loop)

        class _Client:
            def supports_tools(self):
                return True

        weak, strong = _Client(), _Client()
        ok, info = agent_loop.run_agent_loop_with_escalation(
            weak, object(), "step", "task", escalation_client=strong,
            step_idx=0)
        assert ok is False
        assert calls == [weak]          # the stronger model never ran

    def test_an_ordinary_failure_still_escalates(self, monkeypatch):
        from agentchanti.orchestrator import agent_loop

        calls = []

        def fake_loop(client, tools, step_text, task, **kw):
            calls.append(client)
            return (False, "AssertionError: expected 3 got 2")

        monkeypatch.setattr(agent_loop, "run_agent_loop", fake_loop)

        class _Client:
            def supports_tools(self):
                return True

        weak, strong = _Client(), _Client()
        agent_loop.run_agent_loop_with_escalation(
            weak, object(), "step", "task", escalation_client=strong,
            step_idx=0)
        assert calls == [weak, strong]

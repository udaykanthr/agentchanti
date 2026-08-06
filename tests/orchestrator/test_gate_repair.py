"""Gate-quality pressure must land on real gaps, and cost one line to fix.

Observed live (classic mode, Pac-Man task, 2026-08-05). Step 2.5 built
`main.py` and gated it on

    python -c "import main" 2>&1 | findstr /i "error" && exit 1 || exit 0

which fails whenever main.py raises at import — a real defect the step can
have. The quality check judged only the `python -c` payload, called it
"imports and prints but never asserts", and sent the WHOLE plan back to be
regenerated. Twice. Each re-plan cost a full generation (7.7k sent / 2.1k
received) and produced a different step decomposition, so ~20k tokens went
on churn; the third attempt finally satisfied the checker by appending
`assert True` — a gate that cannot fail, arrived at through the machinery
that exists to prevent exactly that.

Three things follow, and this module pins all three:
  - a shell-level error assertion counts as teeth;
  - `assert True` does not;
  - a weak gate is repaired in place, not by regenerating the plan.
"""

from __future__ import annotations

from agentchanti.orchestrator.plan_step import (
    PlanStep,
    repair_verify_commands,
    shallow_gate_reason,
    shell_level_assertion,
)


class _Client:
    """Minimal LLM stand-in recording the single prompt it is given."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _code_step(verify: str) -> PlanStep:
    return PlanStep(id="2.5", step_type="CODE",
                    description="Create main.py, the game entry point",
                    target_files=["main.py"], verify_cmd=verify)


# ── the shell can carry the assertion ────────────────────────────────

def test_output_grep_that_exits_nonzero_is_not_shallow():
    cmd = ('python -c "import sys; sys.argv=[]; import main" 2>&1 '
           '| findstr /i "error" && exit 1 || exit 0')
    assert shell_level_assertion(cmd)
    assert shallow_gate_reason(cmd) is None


def test_posix_grep_form_is_also_recognised():
    cmd = 'python -c "import main" 2>&1 | grep -i error && exit 1 || exit 0'
    assert shallow_gate_reason(cmd) is None


def test_bare_import_without_a_shell_check_is_still_shallow():
    assert shallow_gate_reason('python -c "import main"') is not None


# ── `assert True` is punctuation, not an assertion ───────────────────

def test_constant_assert_does_not_buy_teeth():
    cmd = ('python -c "import pygame; from game import Game; '
           'pygame.init(); g=Game(800, 600); dt=0.016; assert True"')
    assert shallow_gate_reason(cmd) is not None


def test_a_real_assert_alongside_a_constant_one_still_counts():
    cmd = ('python -c "from game import Game; g=Game(800, 600); '
           'assert g.is_running(); assert True"')
    assert shallow_gate_reason(cmd) is None


# ── repair replaces the line, not the plan ───────────────────────────

def test_repair_rewrites_only_the_offending_gate():
    steps = [
        PlanStep(id="2.1", step_type="CODE", verify_cmd='python -c "assert 1 == len([0])"'),
        _code_step('python -c "from game import Game; print(Game)"'),
    ]
    client = _Client('2.5: python -c "from game import Game; '
                     'g=Game(800, 600); assert len(g.ghosts) == 4"')

    repaired = repair_verify_commands(
        steps, [("2.5", "imports and prints but never asserts")], client)

    assert repaired == ["2.5"]
    assert "assert len(g.ghosts) == 4" in steps[1].verify_cmd
    assert steps[0].verify_cmd == 'python -c "assert 1 == len([0])"'
    assert len(client.prompts) == 1


def test_repair_refuses_a_replacement_with_the_same_defect():
    step = _code_step('python -c "from game import Game; print(Game)"')
    client = _Client('2.5: python -c "from game import Game; print(Game.__name__)"')

    assert repair_verify_commands([step], [("2.5", "never asserts")], client) == []
    assert step.verify_cmd == 'python -c "from game import Game; print(Game)"'


def test_repair_refuses_an_unrunnable_replacement():
    step = _code_step('python -c "from game import Game; print(Game)"')
    client = _Client('2.5: python -c "def broken(: pass"')

    assert repair_verify_commands([step], [("2.5", "never asserts")], client) == []


def test_repair_reports_nothing_when_the_provider_fails():
    class _Dead:
        def generate_response(self, prompt):
            raise RuntimeError("connection reset")

    step = _code_step('python -c "from game import Game; print(Game)"')
    # Falling back to the full re-plan is the caller's job; repair just
    # has to say it did nothing rather than propagate the failure.
    assert repair_verify_commands([step], [("2.5", "never asserts")], _Dead()) == []


def test_repair_without_a_client_is_a_no_op():
    step = _code_step('python -c "import main"')
    assert repair_verify_commands([step], [("2.5", "why")], None) == []


def test_the_model_may_echo_the_label_it_was_given():
    # Observed verbatim: the prompt asks for a bare id and the reply comes
    # back as "step 2.5: ...". A parser that insists on the bare form throws
    # away a correct 506/140-token answer and falls back to a re-plan.
    for prefix in ("step 2.5:", "2.5:", "- step 2.5 -", "#2.5.", "  2.5 :"):
        step = _code_step('python -c "import main"')
        client = _Client(
            f'{prefix} python -c "from game import Game; '
            f'assert Game(1, 1).state == \'START\'"')
        assert repair_verify_commands(
            [step], [("2.5", "never asserts")], client) == ["2.5"], prefix

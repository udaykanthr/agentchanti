"""`findstr "a b"` matches a WORD, not the phrase — and hides what it rejects.

Measured 2026-09-16, step 10 of a Snake run on 0.8.2:

    set PYTHONPATH=src&&python -m snake_game --version | findstr /x "snake_game 0.1.0"

findstr splits a quoted search string on spaces and matches ANY of the
words; with /x that means a whole line equal to `snake_game` or to `0.1.0`.
The program printed exactly `snake_game 0.1.0`, the gate exited 1 with
empty output (findstr swallows every rejected line), and the step spent 25
turns — a stall, a recovery and an escalation — and 374k tokens, 79% of the
run, before the run failed with four waves never started. With /c:"..."
the identical gate passes.
"""
import os

import pytest

from agentchanti.executor import Executor
from agentchanti.orchestrator.gate_integrity import (
    _findstr_literal_phrase, findstr_phrase_reason,
    platform_equivalent_variants,
)
from agentchanti.orchestrator.plan_step import unrunnable_gate_reason

pytestmark = pytest.mark.skipif(os.name != "nt",
                                reason="findstr is a Windows command")

MEASURED = ('set PYTHONPATH=src&&python -m snake_game --version | '
            'findstr /x "snake_game 0.1.0"')


class TestDetection:

    def test_the_measured_gate(self):
        assert findstr_phrase_reason(MEASURED)
        assert unrunnable_gate_reason(MEASURED), \
            "the plan-time and turn-zero checks must see it"

    @pytest.mark.parametrize("cmd", [
        'findstr /c:"snake_game 0.1.0" out.txt',     # the literal-phrase form
        'findstr /x "ok"',                           # one word: fine
        'findstr /i /c:"aria-label" src\\App.jsx',
        'python -m pytest -q',
    ])
    def test_correct_uses_are_left_alone(self, cmd):
        assert findstr_phrase_reason(cmd) is None


class TestRewrite:

    def test_the_literal_phrase_form(self):
        assert _findstr_literal_phrase(MEASURED) == (
            'set PYTHONPATH=src&&python -m snake_game --version | '
            'findstr /x /c:"snake_game 0.1.0"')

    def test_turn_zero_adopts_a_runnable_equivalent(self):
        """The loop uses the first variant that is not itself unrunnable."""
        variants = dict(platform_equivalent_variants(MEASURED))
        fixed = variants.get("findstr-literal-phrase")
        assert fixed and unrunnable_gate_reason(fixed) is None


class TestRealShell:
    """Measured behaviour, through the real executor and cmd.exe."""

    def test_as_written_it_can_never_pass(self, tmp_path):
        ok, _ = Executor().run_command(
            'echo snake_game 0.1.0| findstr /x "snake_game 0.1.0"',
            cwd=str(tmp_path))
        assert not ok

    def test_the_rewrite_passes_on_the_same_output(self, tmp_path):
        ok, out = Executor().run_command(
            _findstr_literal_phrase(
                'echo snake_game 0.1.0| findstr /x "snake_game 0.1.0"'),
            cwd=str(tmp_path))
        assert ok, out
        assert "snake_game 0.1.0" in out

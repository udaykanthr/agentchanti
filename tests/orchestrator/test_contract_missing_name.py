"""A missing NAME is not a missing module.

Measured 2026-09-25, gpt-5.6-terra on the snake task. The contract guessed
the class would be `Game`; the code named it `SnakeGame`:

    ImportError: cannot import name 'Game' from 'snake_game.game'

`_not_ready_reason` matched `cannot import name` and deferred - four
times, and then reported at the end of the run that "the contract still
cannot reach the project". The module imported fine every time. The repair
path, which exists for exactly this (a contract that crashes, with both
the contract and the traceback available), never ran because the deferral
sat in front of it.

The discriminator is the plan's own vocabulary: `exports:` says which
symbols some step promises. A name no step ever declared is the contract's
invention, not something to wait for.
"""
import pytest

from agentchanti.orchestrator.acceptance_seed import _not_ready_reason

MISSING_NAME = (
    "ERROR: test_eating_food_grows_snake\n"
    "ImportError: cannot import name 'Game' from 'snake_game.game' "
    "(C:\\p\\snake_game\\game.py). Did you mean: 'game'?\n")
MISSING_MODULE = (
    "ERROR: test_startup\n"
    "ModuleNotFoundError: No module named 'snake_game'\n")


class TestTheMeasuredCase:

    def test_a_name_the_plan_never_promised_is_not_deferred(self):
        """The code exports SnakeGame; no step ever declared `Game`."""
        assert _not_ready_reason(MISSING_NAME, {"SnakeGame", "Renderer"}) \
            is None

    def test_a_name_a_step_declares_is_still_deferred(self):
        """It may yet appear — that IS 'not built yet'."""
        why = _not_ready_reason(MISSING_NAME, {"Game", "Renderer"})
        assert why and "Game" in why


class TestWhatIsUnchanged:

    def test_a_missing_module_always_defers(self):
        assert _not_ready_reason(MISSING_MODULE, {"SnakeGame"})
        assert _not_ready_reason(MISSING_MODULE, set())

    def test_no_plan_facts_keeps_the_old_behaviour(self):
        """Callers that pass nothing must not change meaning."""
        assert _not_ready_reason(MISSING_NAME) == "cannot import name"

    def test_a_plain_assertion_failure_is_still_a_verdict(self):
        out = ("FAIL: test_score\n"
               "AssertionError: 0 != 10 : eating food must score\n")
        assert _not_ready_reason(out, {"SnakeGame"}) is None

    def test_a_missing_source_file_still_defers(self):
        out = "FileNotFoundError: [Errno 2] ... 'snake_game/game.py'"
        assert _not_ready_reason(out, {"SnakeGame"})

    @pytest.mark.parametrize("out", ["", None])
    def test_no_output_is_not_a_deferral(self, out):
        assert _not_ready_reason(out, {"SnakeGame"}) is None

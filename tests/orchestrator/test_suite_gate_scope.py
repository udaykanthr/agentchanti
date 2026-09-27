"""A step's gate measures that step, not the whole directory.

Measured 2026-09-25, gpt-5.6-terra. Step 11 - "create unit tests covering
initial state, movement, direction-reversal prevention, food growth and
scoring, valid food placement, wall collision, self collision, and restart"
- was gated on `python -m unittest -q`, which collects everything in the
project root INCLUDING the seeded acceptance contract, which the step
neither wrote nor may edit. Its own eight tests passed; the contract has
one failing assertion, so the gate could not go green whatever the step
wrote. 10 turns, then 10 more in recovery: 20 of the run's 39 turns and
~190k of its 270k tokens. The comparable run the night before, whose plan
happened not to write that gate, cost 78k.
"""
import pytest

from agentchanti.orchestrator.plan_step import (PlanStep, scope_suite_gates,
                                                scope_suite_gate_to_step)


def _step(verify, targets, sid="11.1", kind="TEST"):
    return PlanStep(id=sid, description="write the tests", step_type=kind,
                    target_files=list(targets), verify_cmd=verify)


class TestTheMeasuredCase:

    def test_a_bare_unittest_is_scoped_to_the_steps_own_tests(self):
        step = _step("python -m unittest -q",
                     ["tests/test_game.py", "tests/test_model.py"])
        assert scope_suite_gate_to_step(step) == \
            "python -m unittest discover -s tests -q"

    def test_the_plan_is_rewritten_in_place(self):
        steps = [_step("python -m unittest -q", ["tests/test_game.py"])]
        changed = scope_suite_gates(steps)
        assert [c[0] for c in changed] == ["11.1"]
        assert steps[0].verify_cmd == "python -m unittest discover -s tests -q"

    def test_root_level_tests_are_named_individually(self):
        """Discovery at the root would sweep the contract back in."""
        step = _step("python -m unittest", ["test_game.py", "test_model.py"])
        assert scope_suite_gate_to_step(step) == \
            "python -m unittest test_game.py test_model.py"

    def test_pytest_too(self):
        step = _step("python -m pytest -q", ["tests/test_game.py"])
        assert scope_suite_gate_to_step(step) == "python -m pytest -q tests"


class TestWhatItLeavesAlone:

    @pytest.mark.parametrize("cmd", [
        "python -m unittest tests/test_game.py",      # already scoped
        "python -m unittest discover -s tests -q",    # already scoped
        "python -m pytest -q tests",                  # already scoped
        "python -m pytest -q --cov=snake tests",      # already scoped
        "npm test",                                   # not python discovery
        "python -c \"import snake; assert snake.VERSION\"",
    ])
    def test_gates_that_already_say_where(self, cmd):
        step = _step(cmd, ["tests/test_game.py"])
        assert scope_suite_gate_to_step(step) is None

    def test_a_step_with_no_tests_of_its_own_is_untouched(self):
        """A release step deliberately gating on the whole suite stays."""
        step = _step("python -m unittest -q", ["snake/game.py"], kind="CODE")
        assert scope_suite_gate_to_step(step) is None

    def test_a_step_with_no_targets_is_untouched(self):
        assert scope_suite_gate_to_step(_step("python -m unittest -q", [])) \
            is None

    def test_non_test_targets_do_not_count(self):
        step = _step("python -m unittest -q",
                     ["tests/conftest.py", "tests/helpers.py"])
        assert scope_suite_gate_to_step(step) is None


class TestTheContractIsExcluded:

    def test_the_scoped_gate_no_longer_reaches_the_root_contract(self):
        """The whole point: `tests/` discovery cannot collect a contract
        that lives at the project root."""
        step = _step("python -m unittest -q", ["tests/test_game.py"])
        scoped = scope_suite_gate_to_step(step)
        assert "discover -s tests" in scoped
        assert "test_acceptance_contract" not in scoped

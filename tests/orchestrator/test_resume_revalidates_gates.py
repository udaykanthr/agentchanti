"""A checkpoint must not be a way around the gate checks.

The plan-time checks — destructive-gate neutralisation, gate quality,
gate merging — all live in the planning branch, and resume skips that
branch entirely: it rebuilds `plan_steps_parsed` straight from the saved
dicts. Measured 2026-08-18 00:11, a resume brought back verbatim the
gate that had already burned the run which wrote the checkpoint.
"""

from agentchanti.orchestrator.gate_safety import (
    destructive_reason,
    neutralize_destructive_gates,
)
from agentchanti.orchestrator.plan_step import PlanStep


DESTRUCTIVE = (
    'python -c "from main import Game; assert Game().score == 0" && '
    "python main.py & timeout /t 2 /nobreak & taskkill /im python.exe /f"
)


def _round_trip(step: PlanStep) -> PlanStep:
    """Exactly what resume does: to_dict on save, from_dict on restore."""
    return PlanStep.from_dict(step.to_dict())


class TestCheckpointRoundTrip:
    def test_a_destructive_gate_survives_the_round_trip_unchanged(self):
        # Establishes the premise: the checkpoint preserves the gate, so
        # something on the restore side has to catch it.
        saved = PlanStep(id="1.1", step_type="CODE", verify_cmd=DESTRUCTIVE)
        restored = _round_trip(saved)
        assert restored.verify_cmd == DESTRUCTIVE
        assert destructive_reason(restored.verify_cmd) is not None

    def test_neutralising_the_restored_plan_disarms_it(self):
        restored = [_round_trip(
            PlanStep(id="1.1", step_type="CODE", verify_cmd=DESTRUCTIVE))]
        changed = neutralize_destructive_gates(restored)
        assert [sid for sid, _, _ in changed] == ["1.1"]
        assert destructive_reason(restored[0].verify_cmd) is None
        # The real assertion the planner wrote is still there.
        assert "Game().score == 0" in restored[0].verify_cmd

    def test_an_ordinary_restored_plan_is_untouched(self):
        restored = [
            _round_trip(PlanStep(id="1.1", step_type="CODE",
                                 verify_cmd="python -m unittest")),
            _round_trip(PlanStep(id="1.2", step_type="CODE", verify_cmd="")),
        ]
        assert neutralize_destructive_gates(restored) == []
        assert restored[0].verify_cmd == "python -m unittest"
        # The round trip normalises an empty gate to None, so a gateless
        # step must be tolerated rather than compared to "".
        assert not restored[1].verify_cmd

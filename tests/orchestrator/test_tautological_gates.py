"""A gate that cannot fail is not a gate.

`assert True` was already rejected as punctuation, but the same intent
survives in forms the constant check cannot see. Observed 2026-08-12 on a
Pac-Man run whose plan gated its entire Game class on:

    assert isinstance(g.player, type(g.player))
    assert isinstance(g.map,    type(g.map))

True for every object that has ever existed. Both went green against a
game where NOTHING moved in 600 frames — 0/600 frames of player motion,
0/600 of ghost motion, 0 pellets collected — and the pipeline reported
success and auto-committed.

`verified-early` sharpens the cost. The loop exits the moment the gate
passes, so a gate that cannot fail ends the step on turn one, and the
model never gets a turn in which it might notice its module does nothing.
"""

import unittest

from agentchanti.orchestrator.plan_step import (
    _always_true,
    shallow_gate_reason,
)


def _test_of(src):
    import ast
    return ast.parse(src).body[0].test


class AlwaysTrueTest(unittest.TestCase):

    def test_the_two_gates_that_shipped_an_inert_game(self):
        for src in ("assert isinstance(g.player, type(g.player))",
                    "assert isinstance(g.map, type(g.map))"):
            with self.subTest(src=src):
                self.assertTrue(_always_true(_test_of(src)))

    def test_other_forms_that_cannot_fail(self):
        for src in ("assert True",
                    "assert 1",
                    "assert 'text'",
                    "assert isinstance(x, object)",
                    "assert x == x",
                    "assert x is x",
                    "assert len(g.ghosts) == len(g.ghosts)",
                    "assert x >= x",
                    "assert True or g.broken",
                    "assert isinstance(x, type(x)) and 1"):
            with self.subTest(src=src):
                self.assertTrue(_always_true(_test_of(src)), src)

    def test_real_assertions_are_untouched(self):
        """The check must keep its hands off gates that can fail."""
        for src in ("assert len(g.ghosts) == 4",
                    "assert isinstance(TILE_SIZE, int)",
                    "assert isinstance(g.player, Player)",
                    "assert x == y",
                    "assert g.state == 'START'",
                    "assert m.is_walkable(*m.player_spawn)",
                    "assert x > 0",
                    "assert not m.is_wall(1, 1)",
                    "assert isinstance(x, type(y))",
                    "assert False"):
            with self.subTest(src=src):
                self.assertFalse(_always_true(_test_of(src)), src)

    def test_and_needs_every_operand_to_be_vacuous(self):
        """One real conjunct makes the whole assertion meaningful."""
        self.assertFalse(_always_true(
            _test_of("assert isinstance(x, type(x)) and len(g.ghosts) == 4")))
        self.assertTrue(_always_true(
            _test_of("assert isinstance(x, type(x)) and x == x")))

    def test_or_needs_only_one(self):
        self.assertTrue(_always_true(
            _test_of("assert len(g.ghosts) == 4 or True")))


class ShallowGateReasonTest(unittest.TestCase):

    def test_the_verbatim_gate_from_the_run(self):
        cmd = ('python -c "from game import Game, GameState; game=Game(); '
               "assert isinstance(game.player, type(game.player)); "
               'assert isinstance(game.map, type(game.map))"')
        reason = shallow_gate_reason(cmd)
        self.assertIsNotNone(reason)
        self.assertIn("true for every possible value", reason)

    def test_the_reason_names_the_offending_expression(self):
        """A planner asked to fix a gate needs to see which part is dead."""
        cmd = ('python -c "import g; assert isinstance(g.x, type(g.x))"')
        self.assertIn("isinstance(g.x, type(g.x))", shallow_gate_reason(cmd))

    def test_a_gate_with_one_real_assertion_still_passes(self):
        cmd = ('python -c "from game import Game; g=Game(); '
               "assert isinstance(g.player, type(g.player)); "
               'assert len(g.ghosts) == 4"')
        self.assertIsNone(shallow_gate_reason(cmd))

    def test_a_substantive_gate_is_unaffected(self):
        cmd = ('python -c "from map import Map; m=Map(); '
               'assert m.is_walkable(*m.player_spawn); '
               'assert m.remaining_pellet_count() > 0"')
        self.assertIsNone(shallow_gate_reason(cmd))

    def test_the_import_only_message_is_unchanged(self):
        """The pre-existing classification must not be reworded."""
        reason = shallow_gate_reason('python -c "import main"')
        self.assertIn("only imports the module", reason)

    def test_a_test_runner_is_still_exempt(self):
        self.assertIsNone(shallow_gate_reason("python -m unittest -v"))


class RepairRejectsTheSameDefectTest(unittest.TestCase):
    """A replacement gate with the same hole must not be accepted, or the
    repair call is burned and the gate stays toothless."""

    def test_wiring(self):
        import inspect
        from agentchanti.orchestrator import plan_step
        src = inspect.getsource(plan_step.repair_verify_commands)
        self.assertIn("shallow_gate_reason(cmd)", src)

    def test_check_gate_quality_reports_it(self):
        """The reason has to reach the plan-quality gaps list, or nothing
        ever asks the planner to fix it."""
        from agentchanti.orchestrator.plan_step import (
            PlanStep, check_gate_quality)
        step = PlanStep(id="2.7", step_type="CODE", description="Game class")
        step.verify_cmd = ('python -c "from game import Game; g=Game(); '
                           'assert isinstance(g.player, type(g.player))"')
        gaps = dict(check_gate_quality([step]))
        self.assertIn("2.7", gaps)
        self.assertIn("true for every possible value", gaps["2.7"])


if __name__ == "__main__":
    unittest.main()

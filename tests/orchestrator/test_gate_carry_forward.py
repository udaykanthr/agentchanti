"""A re-plan must not weaken a gate it was never asked to touch.

Measured on a Pac-Man run. Plan attempt 1 declared, on step 3.1:

    verify: python -c "from game import Game; g=Game(); g.run_frame(0.02);
            assert g.player.pos[0] != g.player.prev_pos[0]"

which fails against exactly the artifact that run shipped — a
``run_frame(dt)`` that never reads dt and a player that never moves. The
re-plan was triggered by step *4.1*'s import-only gate, and attempt 2
returned a step 3.1 asserting only ``len(g.ghosts)==4`` and that the
ghost spawns are walkable, both true of a stub. One weak gate on one
step cost the strongest gate in the plan.
"""

from agentchanti.orchestrator.plan_step import (
    PlanStep, carry_forward_strong_gates, check_gate_quality,
)

_STRONG = ('python -c "from game import Game; g=Game(); g.run_frame(0.02); '
           'assert g.player.pos[0] != g.player.prev_pos[0]"')
_WEAKER = ('python -c "from game import Game; g=Game(); '
           'assert len(g.ghosts)==4"')
_IMPORT_ONLY = 'python -c "import main"'


def _step(sid, target, verify, step_type="CODE", **kw):
    return PlanStep(id=sid, step_type=step_type, target_files=[target],
                    verify_cmd=verify, description=f"build {target}", **kw)


def test_the_measured_loss_is_prevented():
    before = [
        _step("3.1", "game.py", _STRONG),
        _step("4.1", "main.py", _IMPORT_ONLY),
    ]
    after = [
        _step("2.1", "game.py", _WEAKER),        # id churned, target did not
        _step("3.1", "main.py",
              'python -c "import main; assert callable(main.run_game)"'),
    ]
    restored = carry_forward_strong_gates(before, after)
    assert restored == ["2.1"]
    # Both gates could fail on wrong behaviour, so both are kept — the
    # movement assertion is what the shipped artifact would have failed.
    assert _STRONG in after[0].verify_cmd
    assert _WEAKER in after[0].verify_cmd


def test_two_strong_gates_are_conjoined_not_chosen_between():
    """Each was written to catch something; a step that satisfies both is
    strictly better checked than one that satisfies whichever survived."""
    better = ('python -c "from game import Game; g=Game(); '
              'g.run_frame(0.02); assert g.player.moved_this_frame"')
    before = [_step("1.1", "game.py", _STRONG)]
    after = [_step("1.1", "game.py", better)]
    assert carry_forward_strong_gates(before, after) == ["1.1"]
    assert after[0].verify_cmd == f"{_STRONG} && {better}"


def test_a_gate_naming_a_module_the_new_plan_dropped_is_not_carried():
    """`gate_integrity` exists because a bad gate failed working code for
    182k tokens. A stale carried gate would do exactly that."""
    before = [_step("1.1", "game.py", _STRONG),
              _step("1.2", "player.py", 'python -c "import player"')]
    after = [_step("1.1", "engine.py", "")]        # game.py no longer planned
    assert carry_forward_strong_gates(before, after) == []


def test_a_weak_old_gate_is_not_carried():
    """Carrying an import-only gate forward would entrench the defect the
    re-plan was called to fix."""
    before = [_step("1.1", "main.py", _IMPORT_ONLY)]
    after = [_step("1.1", "main.py", "")]
    assert carry_forward_strong_gates(before, after) == []
    assert not after[0].verify_cmd


def test_a_missing_gate_is_filled_from_the_previous_attempt():
    before = [_step("1.1", "game.py", _STRONG)]
    after = [_step("1.1", "game.py", "")]
    assert carry_forward_strong_gates(before, after) == ["1.1"]
    assert after[0].verify_cmd == _STRONG


def test_steps_for_different_files_do_not_swap_gates():
    before = [_step("1.1", "game.py", _STRONG)]
    after = [_step("1.1", "player.py", "")]
    assert carry_forward_strong_gates(before, after) == []
    assert not after[0].verify_cmd


def test_targetless_steps_match_by_id():
    """CMD steps often declare no target; the id is the only handle."""
    strong = 'python -c "import pygame; assert pygame.version.vernum >= (2,)"'
    before = [PlanStep(id="1.1", step_type="CMD", verify_cmd=strong,
                       description="install pygame")]
    after = [PlanStep(id="1.1", step_type="CMD", verify_cmd="",
                      description="install pygame")]
    assert carry_forward_strong_gates(before, after) == ["1.1"]
    assert after[0].verify_cmd == strong


def test_identical_command_is_not_reported_as_restored():
    before = [_step("1.1", "game.py", _STRONG)]
    after = [_step("1.1", "game.py", _STRONG)]
    assert carry_forward_strong_gates(before, after) == []


def test_no_previous_plan_is_a_no_op():
    after = [_step("1.1", "game.py", "")]
    assert carry_forward_strong_gates([], after) == []


def test_carrying_the_gate_also_clears_the_gate_gap():
    """The point of running this BEFORE the quality check: a restored gate
    should end the re-plan churn, not merely survive it."""
    before = [_step("1.1", "game.py", _STRONG)]
    after = [_step("1.1", "game.py", _IMPORT_ONLY)]
    assert [sid for sid, _ in check_gate_quality(after)] == ["1.1"]
    carry_forward_strong_gates(before, after)
    assert [sid for sid, _ in check_gate_quality(after)] == []


def test_a_subset_gate_is_not_conjoined():
    """Replaying the measured plans produced this: the older gate's
    assertion was a prefix of the newer one's, differing only in spacing.
    Conjoining would have doubled the command to re-check one value."""
    before = [_step("1.1", "constants.py",
                    'python -c "import constants; assert constants.TILE_SIZE == 20"')]
    after = [_step("1.1", "constants.py",
                   'python -c "import constants; assert constants.TILE_SIZE==20; '
                   'assert constants.WALL==1"')]
    assert carry_forward_strong_gates(before, after) == []


def test_the_same_test_runner_is_not_run_twice():
    """`unittest -v <path>` and `unittest <path>` are one suite, not two."""
    before = [_step("1.1", "tests/test_pacman.py",
                    "python -m unittest -v tests/test_pacman.py", "TEST")]
    after = [_step("1.1", "tests/test_pacman.py",
                   "python -m unittest tests/test_pacman.py", "TEST")]
    assert carry_forward_strong_gates(before, after) == []

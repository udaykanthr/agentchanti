"""`failed-but-clean` may only speak about the step that actually failed.

The finding tells a reader to suspect the harness before the model. It
already required *some* green acceptance gate before saying so — added
after a 20B run whose eight structurally perfect files hid a real logic
bug. That guard asks "is there any green gate in this run?", which is a
different question from "is there green evidence about the thing that
failed", and the two come apart as soon as a plan has more than one step.

Both measured instances are that gap, and both blamed the harness for the
model:

  2026-08-17 iter1  gates 2.1, 3.1 and 4.1 green; step 5 failed having
                    never recorded a gate at all. Gate 3.1 was also stale
                    — step 5's diagnosis had rewritten the game.py it
                    validated into a TypeError on every advance().
  2026-08-17 iter5  four green gates; step 6 failed `verify` three times.
                    None of the green gates covered it.

The second half of this file covers the staleness: a gate's HOLDS was
folded permanently, so a verdict about a tree that no longer existed kept
counting as behavioural evidence.
"""

from agentchanti.orchestrator.ghost import (
    VIOLATED,
    HOLDS, KIND_GATE_PASSED, UNKNOWN, GhostPlan,
)
from agentchanti.orchestrator.plan_step import PlanStep


def _Step(sid, target, verify=None):
    return PlanStep(id=sid, step_type="CODE",
                    target_files=[target] if target else [],
                    verify_cmd=verify)


def _plan(tmp_path, steps):
    return GhostPlan.build(steps, str(tmp_path))


def _kinds(disagreements):
    return {d.kind for d in disagreements}


def test_a_halted_step_without_its_own_gate_stays_silent(tmp_path):
    """iter1 and iter5: the green gates belonged to steps that finished."""
    steps = [_Step("1.1", "a.py", verify="python -c \"import a\""),
             _Step("2.1", "b.py", verify="python -c \"import b\"")]
    plan = _plan(tmp_path, steps)          # pre-state: neither file exists
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    # Step 1.1's gate went green; step 2.1's never did, and 2.1 is the
    # step the run halted on. Nothing here is VIOLATED — both files were
    # written and both parse — so the ONLY thing that can suppress the
    # finding is the scoping of the evidence.
    plan.resolve(["1.1", "2.1"], gate_cmds=['python -c "import a"'])
    assert plan.tally()[VIOLATED] == 0

    found = plan.disagreements(["1.1"], pipeline_success=False)
    assert "failed-but-clean" not in _kinds(found)


def test_all_steps_complete_and_still_failed_is_the_real_shape(tmp_path):
    """The case the finding was written for is untouched: every step done,
    every postcondition holding, and the pipeline still says failed."""
    steps = [_Step("1.1", "a.py", verify="python -c \"import a\"")]
    plan = _plan(tmp_path, steps)          # pre-state: a.py absent
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    plan.resolve(["1.1"], gate_cmds=['python -c "import a"'])

    found = plan.disagreements(["1.1"], pipeline_success=False)
    assert "failed-but-clean" in _kinds(found)


def test_a_halted_step_with_its_own_green_gate_still_reports(tmp_path):
    """Scoping narrows the evidence; it does not remove the finding."""
    steps = [_Step("1.1", "a.py", verify="python -c \"import a\""),
             _Step("2.1", "b.py", verify="python -c \"import b\"")]
    plan = _plan(tmp_path, steps)          # pre-state: neither file exists
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    plan.resolve(["1.1", "2.1"],
                 gate_cmds=['python -c "import a"', 'python -c "import b"'])

    # 2.1 never completed, but its own gate did go green.
    found = plan.disagreements(["1.1"], pipeline_success=False)
    assert "failed-but-clean" in _kinds(found)


# ── staleness ────────────────────────────────────────────────────────────

def test_a_gate_expires_when_the_file_it_exercised_changes(tmp_path):
    """iter1's gate 3.1: green at wave 3, and the file rewritten at wave 5."""
    target = tmp_path / "game.py"
    target.write_text("def advance(dt):\n    return dt\n", encoding="utf-8")
    steps = [_Step("3.1", "game.py", verify="python -c \"import game\"")]
    plan = _plan(tmp_path, steps)
    gates = ['python -c "import game"']

    plan.resolve(["3.1"], gate_cmds=gates)
    gate = next(e for e in plan.expectations.values()
                if e.kind == KIND_GATE_PASSED)
    assert gate.verdict == HOLDS

    # A later step rewrites the very file the gate validated.
    target.write_text("def advance(dt):\n    raise TypeError\n",
                      encoding="utf-8")
    plan.resolve(["3.1"], gate_cmds=gates)

    assert gate.verdict == UNKNOWN
    assert "no longer exists" in gate.evidence


def test_an_unchanged_file_keeps_its_gate_green(tmp_path):
    target = tmp_path / "game.py"
    target.write_text("def advance(dt):\n    return dt\n", encoding="utf-8")
    steps = [_Step("3.1", "game.py", verify="python -c \"import game\"")]
    plan = _plan(tmp_path, steps)
    gates = ['python -c "import game"']

    plan.resolve(["3.1"], gate_cmds=gates)
    plan.resolve(["3.1"], gate_cmds=gates)

    gate = next(e for e in plan.expectations.values()
                if e.kind == KIND_GATE_PASSED)
    assert gate.verdict == HOLDS


def test_a_stale_gate_no_longer_counts_as_behavioural_evidence(tmp_path):
    """The two halves together — iter1 end to end. The only green gate
    belonged to a completed step AND described a file since rewritten."""
    target = tmp_path / "game.py"
    target.write_text("def advance(dt):\n    return dt\n", encoding="utf-8")
    (tmp_path / "test_game.py").write_text("def test_x():\n    pass\n",
                                           encoding="utf-8")
    steps = [_Step("3.1", "game.py", verify="python -c \"import game\""),
             _Step("5.1", "test_game.py", verify="python -m unittest")]
    plan = _plan(tmp_path, steps)
    gates = ['python -c "import game"']
    plan.resolve(["3.1"], gate_cmds=gates)

    target.write_text("def advance(dt):\n    raise TypeError\n",
                      encoding="utf-8")
    plan.resolve(["3.1"], gate_cmds=gates)

    found = plan.disagreements(["3.1"], pipeline_success=False)
    assert "failed-but-clean" not in _kinds(found)

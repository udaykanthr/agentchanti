"""A declared export nobody imports is drift, not a defect.

`exports:` is a contract. When another step declares it imports one of
those names and the file never defines it, that step is heading for the
`gate_integrity` shape — one bad name rejecting working code for a whole
run. When nothing imports it, the same disagreement is a naming
preference the code won on its own, and it cannot break anything.

The pipeline already draws that line: `_missing_declared_exports` fires
"only on an import/attribute error naming a declared export".

Measured 2026-08-17: a classic run produced three `violated-exports`
findings — `Pellet` where the code wrote `Collectible`, `render_game`
where it wrote `draw_maze`/`draw_ghost`/`draw_panel`, `HeadlessCommandTests`
where it wrote `GameApiTests` — against an artifact that passed all nine
external behavioural probes. An earlier run produced two more of the same
shape. Enough of those bury the one that matters, so the unconsumed ones
collapse into a single run-level `export-drift` note, exactly as six
`unplanned-write` findings collapse into `plan-declares-no-targets`.
"""

from agentchanti.orchestrator.ghost import GhostPlan
from agentchanti.orchestrator.plan_step import PlanStep


def _kinds(found):
    return [d.kind for d in found]


def _one(found, kind):
    return next(d for d in found if d.kind == kind)


def test_an_unconsumed_renamed_export_collapses_to_drift(tmp_path):
    """The measured shape: the plan said `Pellet`, the code said
    `Collectible`, and no step imports either."""
    steps = [PlanStep(id="2.1", step_type="CODE", target_files=["game.py"],
                      exports=["Game", "Pellet"])]
    plan = GhostPlan.build(steps, str(tmp_path))
    (tmp_path / "game.py").write_text(
        "class Game:\n    pass\n\n\nclass Collectible:\n    pass\n",
        encoding="utf-8")
    plan.resolve(["2.1"], language="python")

    found = plan.disagreements(["2.1"])
    assert "violated-exports" not in _kinds(found)
    drift = _one(found, "export-drift")
    assert "Pellet" in drift.detail
    assert "nothing can break" in drift.detail


def test_a_consumed_missing_export_is_still_reported(tmp_path):
    """The case worth keeping: a step declares it imports the name."""
    steps = [
        PlanStep(id="2.1", step_type="CODE", target_files=["game.py"],
                 exports=["Game", "Pellet"]),
        PlanStep(id="3.1", step_type="CODE", target_files=["main.py"],
                 imports_from={"game.py": ["Pellet"]}),
    ]
    plan = GhostPlan.build(steps, str(tmp_path))
    (tmp_path / "game.py").write_text(
        "class Game:\n    pass\n\n\nclass Collectible:\n    pass\n",
        encoding="utf-8")
    (tmp_path / "main.py").write_text("import game\n", encoding="utf-8")
    plan.resolve(["2.1", "3.1"], language="python")

    found = plan.disagreements(["2.1", "3.1"])
    assert "violated-exports" in _kinds(found)
    assert "Pellet" in _one(found, "violated-exports").detail


def test_a_consumer_naming_the_symbol_by_another_convention_still_counts(
        tmp_path):
    """The importer wrote `run_headless`; the planner, shown only
    JavaScript examples, declared `runHeadless`. One name, not two — so
    this export IS consumed and must not be demoted to drift."""
    steps = [
        PlanStep(id="2.1", step_type="CODE", target_files=["engine.py"],
                 exports=["runHeadless"]),
        PlanStep(id="3.1", step_type="CODE", target_files=["cli.py"],
                 imports_from={"engine.py": ["run_headless"]}),
    ]
    plan = GhostPlan.build(steps, str(tmp_path))
    (tmp_path / "engine.py").write_text("def something_else():\n    pass\n",
                                        encoding="utf-8")
    (tmp_path / "cli.py").write_text("import engine\n", encoding="utf-8")
    plan.resolve(["2.1", "3.1"], language="python")

    found = plan.disagreements(["2.1", "3.1"])
    assert "violated-exports" in _kinds(found)


def test_many_drifted_exports_produce_exactly_one_note(tmp_path):
    """Three findings on a nine-of-nine artifact was the problem."""
    steps = [PlanStep(id="2.1", step_type="CODE", target_files=["game.py"],
                      exports=["Pellet", "render_game", "HeadlessTests"])]
    plan = GhostPlan.build(steps, str(tmp_path))
    (tmp_path / "game.py").write_text("class Collectible:\n    pass\n",
                                      encoding="utf-8")
    plan.resolve(["2.1"], language="python")

    found = plan.disagreements(["2.1"])
    assert _kinds(found).count("export-drift") == 1
    assert "3 export(s)" in _one(found, "export-drift").detail


def test_a_satisfied_export_produces_nothing(tmp_path):
    steps = [PlanStep(id="2.1", step_type="CODE", target_files=["game.py"],
                      exports=["Game"])]
    plan = GhostPlan.build(steps, str(tmp_path))
    (tmp_path / "game.py").write_text("class Game:\n    pass\n",
                                      encoding="utf-8")
    plan.resolve(["2.1"], language="python")

    found = plan.disagreements(["2.1"])
    assert "violated-exports" not in _kinds(found)
    assert "export-drift" not in _kinds(found)

"""Tests for the read-only ghost shadow (orchestrator/ghost.py).

The module exists to REPORT, never to decide, so these tests care about
two things in equal measure: that it names real disagreements, and that
it stays silent (UNKNOWN) whenever the evidence is inconclusive.
"""

import os

import pytest

from agentchanti.orchestrator.ghost import (
    HOLDS, INAPPLICABLE, UNKNOWN, VIOLATED, GhostPlan, MIN_STEP_STRENGTH,
)
from agentchanti.orchestrator.plan_step import PlanStep


def _step(sid, **kw):
    kw.setdefault("step_type", "CODE")
    return PlanStep(id=sid, **kw)


def _write(root, rel, text):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ── construction ─────────────────────────────────────────────────────


def test_build_records_pre_state_hashes(tmp_path):
    root = str(tmp_path)
    _write(root, "src/board.py", "OLD = 1\n")
    ghost = GhostPlan.build([_step("1.1", target_files=["src/board.py"])], root)

    gf = ghost.files["src/board.py"]
    assert gf.pre_hash is not None
    assert "1.1" in gf.writers


def test_missing_file_has_no_pre_hash(tmp_path):
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["src/new.py"])], str(tmp_path))
    assert ghost.files["src/new.py"].pre_hash is None


def test_shared_facts_intern_to_one_node(tmp_path):
    """Two steps naming the same file share one expectation, not two."""
    steps = [_step("1.1", target_files=["a.py"]),
             _step("2.1", target_files=["a.py"])]
    ghost = GhostPlan.build(steps, str(tmp_path))

    assert len(ghost.files["a.py"].writers) == 2
    exists = [e for e in ghost.expectations.values()
              if e.kind == "EXISTS" and e.subject == "a.py"]
    assert len(exists) == 1


# ── the two failure classes nothing else catches ─────────────────────


def test_planned_but_untouched_is_violated(tmp_path):
    """A step reports done and its target's bytes never changed."""
    root = str(tmp_path)
    _write(root, "src/board.py", "OLD = 1\n")
    ghost = GhostPlan.build([_step("1.1", target_files=["src/board.py"])], root)

    ghost.resolve(["1.1"], language="python")

    assert ghost.expectations["file:src/board.py#touched"].verdict == VIOLATED
    gaps = ghost.disagreements(["1.1"])
    assert any(g.kind == "violated-touched" for g in gaps)


def test_modified_file_holds(tmp_path):
    root = str(tmp_path)
    _write(root, "src/board.py", "OLD = 1\n")
    ghost = GhostPlan.build([_step("1.1", target_files=["src/board.py"])], root)
    _write(root, "src/board.py", "NEW = 2\n")

    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:src/board.py#touched"].verdict == HOLDS


def test_unplanned_write_is_reported(tmp_path):
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["a.py"])], str(tmp_path))
    assert ghost.unplanned_writes(["a.py", "sneaky.py"]) == ["sneaky.py"]


def test_unplanned_write_ignores_agentchanti_internals(tmp_path):
    ghost = GhostPlan.build([_step("1.1", target_files=["a.py"])],
                            str(tmp_path))
    assert ghost.unplanned_writes([".agentchanti/log.txt", "CLAUDE.md"]) == []


# ── per-kind checks ──────────────────────────────────────────────────


def test_prose_target_is_ignored(tmp_path):
    """`produces: pygame package` is an English answer, not a path.

    Observed on a 20B-model run: that line became a "planned target"
    which could never exist on disk, producing a missing-file finding
    and then a zero-evidence finding on top of it — two fabricated
    disagreements from one prose line.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["pygame package", "pacman/config.py"])],
        root)

    assert "file:pygame package#exists" not in ghost.expectations
    assert "file:pacman/config.py#exists" in ghost.expectations


def test_bare_directory_target_is_still_kept(tmp_path):
    """The prose filter must not throw away `venv` or `tests`."""
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["venv", "tests"])], str(tmp_path))
    assert "file:venv#exists" in ghost.expectations
    assert "file:tests#exists" in ghost.expectations


def test_absent_target_violates_exists(tmp_path):
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["src/never.py"])], str(tmp_path))
    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:src/never.py#exists"].verdict == VIOLATED


def test_case_mismatch_is_its_own_finding(tmp_path):
    """A wrong-case filename resolves on Windows and breaks on Linux.

    EXISTS must still HOLD — the file genuinely works on this machine,
    and failing it would be false. The portability defect is reported
    separately instead of distorting that verdict.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["src/board.py"])], root)
    _write(root, "src/Board.py", "class Board:\n    pass\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["file:src/board.py#exists"].verdict == HOLDS

    if ghost.case_mismatches:            # case-insensitive filesystem
        assert ghost.case_mismatches["src/board.py"] == "Board.py"
        gaps = ghost.disagreements(["2.1"])
        assert any(g.kind == "filename-case-mismatch" for g in gaps)


def test_extension_mismatch_is_named_in_the_evidence(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["src/app.tsx"])], root)
    _write(root, "src/app.jsx", "export const App = () => null;\n")

    ghost.resolve(["2.1"], language="typescript")
    exp = ghost.expectations["file:src/app.tsx#exists"]
    assert exp.verdict == VIOLATED
    assert "app.jsx" in exp.evidence


def test_plain_absence_says_so_without_guessing(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("2.1", target_files=["src/gone.py"])], root)
    _write(root, "src/unrelated.py", "x = 1\n")

    ghost.resolve(["2.1"], language="python")
    exp = ghost.expectations["file:src/gone.py#exists"]
    assert exp.verdict == VIOLATED
    assert "mismatch" not in exp.evidence


def test_directory_target_exists(tmp_path):
    """`produces: venv` names a directory, not a file.

    Observed on a real run: the plan's first CMD step declared
    ``produces: venv``, the venv was created, and the shadow reported
    "planned target does not exist on disk" — then compounded it by
    scoring the step at zero evidence and calling it a step that
    asserted nothing.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("1.1", target_files=["venv"])], root)
    os.makedirs(os.path.join(root, "venv"))

    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:venv#exists"].verdict == HOLDS
    assert ghost.disagreements(["1.1"]) == []


def test_directory_target_is_not_judged_on_content(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("1.1", target_files=["assets"])], root)
    os.makedirs(os.path.join(root, "assets"))

    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:assets#touched"].verdict == INAPPLICABLE


def test_absent_directory_still_violates(tmp_path):
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["venv"])], str(tmp_path))
    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:venv#exists"].verdict == VIOLATED


def test_syntax_error_violates_parses(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("1.1", target_files=["bad.py"])], root)
    _write(root, "bad.py", "def broken(:\n")

    ghost.resolve(["1.1"], language="python")
    exp = ghost.expectations["file:bad.py#parses"]
    assert exp.verdict == VIOLATED
    assert "SyntaxError" in exp.evidence


def test_valid_json_parses(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("1.1", target_files=["cfg.json"])], root)
    _write(root, "cfg.json", '{"a": 1}')
    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:cfg.json#parses"].verdict == HOLDS


def test_missing_declared_export_is_violated(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["m.py"], exports=["Board"])], root)
    _write(root, "m.py", "class Other:\n    pass\n")

    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:m.py#exports:Board"].verdict == VIOLATED


def test_present_declared_export_holds(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["m.py"], exports=["Board"])], root)
    _write(root, "m.py", "class Board:\n    pass\n")

    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:m.py#exports:Board"].verdict == HOLDS


def test_class_attribute_satisfies_declared_export(tmp_path):
    """A constant on the class it belongs to is still exported.

    Observed on a real run: `map.py` defined `class Map:` with
    `TILE_SIZE = 32`, `game.py` used `Map.TILE_SIZE`, and the shadow
    called the plan's `exports: TILE_SIZE` a broken promise because the
    backend reports module-level names only.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["map.py"], exports=["TILE_SIZE"])], root)
    _write(root, "map.py", "class Map:\n    TILE_SIZE = 32\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["file:map.py#exports:TILE_SIZE"].verdict == HOLDS


def test_qualified_class_attribute_also_matches(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["map.py"], exports=["Map.TILE_SIZE"])],
        root)
    _write(root, "map.py", "class Map:\n    TILE_SIZE: int = 32\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "file:map.py#exports:Map.TILE_SIZE"].verdict == HOLDS


def test_genuinely_absent_export_still_violated_with_classes(tmp_path):
    """The class-member widening must not swallow a real mismatch.

    Observed alongside the TILE_SIZE false positive: the plan declared
    `OPPOSITE_DIRECTION` while the file defined `OPPOSITE_DIRECTIONS`.
    That one is a true finding and must survive.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.2", target_files=["entities.py"],
               exports=["OPPOSITE_DIRECTION"])], root)
    _write(root, "entities.py",
           "OPPOSITE_DIRECTIONS = {}\n\n\nclass Ghost:\n    SPEED = 1\n")

    ghost.resolve(["2.2"], language="python")
    assert ghost.expectations[
        "file:entities.py#exports:OPPOSITE_DIRECTION"].verdict == VIOLATED


def test_unwired_import_edge_is_violated(tmp_path):
    root = str(tmp_path)
    steps = [
        _step("1.1", target_files=["board.py"], exports=["Board"]),
        _step("2.1", target_files=["main.py"],
              imports_from={"board.py": ["Board"]}),
    ]
    ghost = GhostPlan.build(steps, root)
    _write(root, "board.py", "class Board:\n    pass\n")
    _write(root, "main.py", "print('hi')\n")   # never imports Board

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["edge:board.py->step:2.1"].verdict == VIOLATED


def test_edge_satisfied_by_any_of_the_steps_targets(tmp_path):
    """`imports:` is a step-level claim, not a per-file one.

    Observed on a real run: a TEST step targeting
    ``tests/__init__.py, tests/test_game_invariants.py`` produced three
    "import edge was never wired" findings against the package marker,
    while the test module beside it imported every declared symbol.
    """
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/__init__.py",
                                      "tests/test_game.py"],
                 imports_from={"game.py": ["Game", "GameState"]})
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", '"""Test package."""\n')
    _write(root, "tests/test_game.py", "from game import Game, GameState\n")

    ghost.resolve(["5.1"], language="python")
    exp = ghost.expectations["edge:game.py->step:5.1"]
    assert exp.verdict == HOLDS
    assert "test_game.py" in exp.evidence
    assert ghost.disagreements(["5.1"]) == []


def test_edge_violated_only_when_no_target_wires_it(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/__init__.py",
                                      "tests/test_game.py"],
                 imports_from={"game.py": ["Game"]})
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", '"""Test package."""\n')
    _write(root, "tests/test_game.py", "import unittest\n")

    ghost.resolve(["5.1"], language="python")
    assert ghost.expectations["edge:game.py->step:5.1"].verdict == VIOLATED


def test_edge_unknown_when_a_target_is_unreadable(tmp_path):
    """A missing sibling makes the step-level claim unjudgeable."""
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/__init__.py", "tests/gone.py"],
                 imports_from={"game.py": ["Game"]})
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", '"""Test package."""\n')

    ghost.resolve(["5.1"], language="python")
    assert ghost.expectations["edge:game.py->step:5.1"].verdict == UNKNOWN


def test_gate_matches_despite_respelled_whitespace(tmp_path):
    """The pipeline respells gates before running them.

    Observed: the plan declared `set SDL_VIDEODRIVER=dummy && python -m
    unittest -v`; the ledger recorded `...dummy&& python -m unittest -v`
    (the space is removed on purpose — cmd.exe would otherwise assign a
    trailing space into the variable). The suite had just passed.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("5.1", target_files=["m.py"],
               verify_cmd="set SDL_VIDEODRIVER=dummy && python -m unittest -v")],
        root)
    _write(root, "m.py", "x = 1\n")

    ghost.resolve(["5.1"], language="python",
                  gate_cmds=["set SDL_VIDEODRIVER=dummy&& python -m unittest -v"])
    gate = next(e for e in ghost.expectations.values()
                if e.kind == "GATE_PASSED")
    assert gate.verdict == HOLDS


def test_wired_import_edge_holds(tmp_path):
    root = str(tmp_path)
    steps = [
        _step("1.1", target_files=["board.py"], exports=["Board"]),
        _step("2.1", target_files=["main.py"],
              imports_from={"board.py": ["Board"]}),
    ]
    ghost = GhostPlan.build(steps, root)
    _write(root, "board.py", "class Board:\n    pass\n")
    _write(root, "main.py", "from board import Board\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["edge:board.py->step:2.1"].verdict == HOLDS


def test_gate_absent_from_ledger_is_inconclusive(tmp_path):
    """The ledger is not a complete log of every gate that ever ran.

    Observed on a real run: a CMD step's `python -m unittest -v` passed
    inside the agent loop's recovery path and again in BulkTest, and
    never entered the ledger. Calling that "never passed" was a
    confident falsehood about a suite that had just gone green twice.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["m.py"], verify_cmd="pytest -q")], root)
    _write(root, "m.py", "x = 1\n")

    ghost.resolve(["1.1"], language="python", gate_cmds=["npm test"])
    gate = next(e for e in ghost.expectations.values()
                if e.kind == "GATE_PASSED")
    assert gate.verdict == UNKNOWN
    assert not [g for g in ghost.disagreements(["1.1"])
                if g.kind == "violated-gate-passed"]


def test_cmd_step_suite_command_counts_as_its_gate(tmp_path):
    """A CMD step whose body is the acceptance suite asserts plenty.

    Observed: a step whose entire body was `set SDL_VIDEODRIVER=dummy &&
    python -m unittest -v` declared no target and no verify, carried no
    expectations at all, and was reported as asserting nothing that
    could have failed — while running the project's whole suite.
    """
    root = str(tmp_path)
    step = PlanStep(
        id="5.1", step_type="CMD",
        command="set SDL_VIDEODRIVER=dummy && python -m unittest -v")
    ghost = GhostPlan.build([step], root)

    gates = [e for e in ghost.expectations.values()
             if e.kind == "GATE_PASSED"]
    assert len(gates) == 1
    ghost.resolve(
        ["5.1"], language="python",
        gate_cmds=["set SDL_VIDEODRIVER=dummy&& python -m unittest -v"])
    assert gates[0].verdict == HOLDS
    assert ghost.step_strength("5.1") >= MIN_STEP_STRENGTH
    assert ghost.disagreements(["5.1"]) == []


def test_cmd_step_with_a_plain_command_is_not_a_gate(tmp_path):
    """`mkdir build` is not an assertion about anything."""
    ghost = GhostPlan.build(
        [PlanStep(id="1.1", step_type="CMD", command="mkdir build")],
        str(tmp_path))
    assert not [e for e in ghost.expectations.values()
                if e.kind == "GATE_PASSED"]


def test_gate_matches_through_cd_prefix(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["m.py"], verify_cmd="pytest -q")], root)
    _write(root, "m.py", "x = 1\n")

    ghost.resolve(["1.1"], language="python",
                  gate_cmds=["cd app && pytest -q"])
    gate = next(e for e in ghost.expectations.values()
                if e.kind == "GATE_PASSED")
    assert gate.verdict == HOLDS


def test_no_ledger_is_unknown_not_violated(tmp_path):
    """Absence of a ledger is absence of evidence, not a failed gate."""
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["m.py"], verify_cmd="pytest -q")], root)
    _write(root, "m.py", "x = 1\n")

    ghost.resolve(["1.1"], language="python", gate_cmds=[])
    gate = next(e for e in ghost.expectations.values()
                if e.kind == "GATE_PASSED")
    assert gate.verdict == UNKNOWN


# ── declared dependencies vs. the app's real environment ─────────────


def _make_venv(root, packages=()):
    """Minimal venv skeleton the Executor's detector will accept."""
    scripts = os.path.join(root, "venv", "Scripts")
    os.makedirs(scripts, exist_ok=True)
    open(os.path.join(scripts, "python.exe"), "wb").close()
    site = os.path.join(root, "venv", "Lib", "site-packages")
    os.makedirs(site, exist_ok=True)
    for pkg in packages:
        os.makedirs(os.path.join(site, f"{pkg}-1.0.dist-info"), exist_ok=True)
    return site


def test_dependency_missing_from_project_venv_is_violated(tmp_path):
    """The defect that shipped a non-starting app with every gate green.

    Observed on both benchmark arms: the plan ran
    ``python -m venv venv && python -m pip install pygame``, which creates
    the venv but never activates it, so pygame installed into the
    pipeline's interpreter. The suite passed because no tested module
    imported pygame; `main.py` did, and crashed on launch.
    """
    root = str(tmp_path)
    _make_venv(root, packages=[])          # venv exists, pygame absent
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")

    ghost.resolve(["2.1"], language="python")
    exp = ghost.expectations["pkg:requirements.txt#deps-installed"]
    assert exp.verdict == VIOLATED
    assert "pygame" in exp.evidence
    assert any(g.kind == "violated-pkg-present"
               for g in ghost.disagreements(["2.1"]))


def test_dependency_present_in_project_venv_holds(tmp_path):
    root = str(tmp_path)
    _make_venv(root, packages=["pygame"])
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame>=2.0\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == HOLDS


def test_no_project_venv_is_unknown_not_violated(tmp_path):
    """No venv means the ambient interpreter — not a missing dependency."""
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == UNKNOWN
    assert ghost.disagreements(["2.1"]) == []


def test_requirement_specifiers_and_comments_are_stripped(tmp_path):
    root = str(tmp_path)
    _make_venv(root, packages=["pygame", "requests"])
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt",
           "# runtime deps\n"
           "pygame==2.6.1  # pinned\n"
           "requests[security]>=2.0,<3\n"
           "-r other.txt\n"
           "--index-url https://example.invalid/simple\n"
           "\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == HOLDS


def test_normalized_distribution_names_match(tmp_path):
    """PEP 503: `ruamel.yaml` on disk satisfies `ruamel-yaml` declared."""
    root = str(tmp_path)
    _make_venv(root, packages=["ruamel.yaml"])
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "ruamel-yaml\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == HOLDS


def test_empty_requirements_is_inapplicable(tmp_path):
    root = str(tmp_path)
    _make_venv(root)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "# nothing yet\n")

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == INAPPLICABLE


def test_package_json_missing_node_module_is_violated(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "node_modules"), exist_ok=True)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["package.json"])], root)
    _write(root, "package.json", '{"dependencies": {"react": "^18.0.0"}}')

    ghost.resolve(["2.1"], language="javascript")
    exp = ghost.expectations["pkg:package.json#deps-installed"]
    assert exp.verdict == VIOLATED
    assert "react" in exp.evidence


def test_package_json_without_node_modules_is_unknown(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["package.json"])], root)
    _write(root, "package.json", '{"dependencies": {"react": "^18.0.0"}}')

    ghost.resolve(["2.1"], language="javascript")
    assert ghost.expectations[
        "pkg:package.json#deps-installed"].verdict == UNKNOWN


# ── tautology detection ──────────────────────────────────────────────


def test_inline_only_step_asserts_nothing(tmp_path):
    """A step whose file the plan itself supplies, unchanged, is a tautology.

    EXISTS on an inline-written file is weighted 0 by construction, and
    the file is byte-identical to its pre-state, so TOUCHED is VIOLATED
    rather than contributing. Nothing checkable remains.
    """
    root = str(tmp_path)
    _write(root, "a.py", "x = 1\n")
    step = _step("1.1", target_files=["a.py"], inline_code={"a.py": "x = 1\n"})
    ghost = GhostPlan.build([step], root)

    ghost.resolve(["1.1"], language="python")
    # PARSES still holds, so strength is not zero — but the step is only
    # above the floor because the file happens to be syntactically valid.
    assert ghost.expectations["file:a.py#exists"].weight == 0
    assert ghost.step_strength("1.1") >= MIN_STEP_STRENGTH


def test_unconfirmed_gate_still_counts_as_a_claim(tmp_path):
    """A declared gate is an assertion even when we cannot confirm it.

    Observed on a 20B-model run: a CMD step declared
    `verify: python -c "import pygame; assert pygame.version.verstr
    .startswith('2.6')"` — genuinely falsifiable — but the gate never
    entered the ledger, so it resolved UNKNOWN and the step was reported
    as asserting nothing that could have failed.
    """
    root = str(tmp_path)
    step = PlanStep(
        id="1.1", step_type="CMD", command="pip install pygame==2.6.1",
        verify_cmd='python -c "import pygame; assert pygame.version.verstr"')
    ghost = GhostPlan.build([step], root)

    ghost.resolve(["1.1"], language="python", gate_cmds=[])
    gate = next(e for e in ghost.expectations.values()
                if e.kind == "GATE_PASSED")
    assert gate.verdict == UNKNOWN          # unconfirmed
    assert ghost.step_strength("1.1") == 0  # banked no evidence
    assert ghost.declared_strength("1.1") >= MIN_STEP_STRENGTH
    assert not [g for g in ghost.disagreements(["1.1"])
                if g.kind == "no-checkable-claim"]


def test_step_with_no_declarations_has_zero_strength(tmp_path):
    ghost = GhostPlan.build([_step("1.1")], str(tmp_path))
    ghost.resolve(["1.1"], language="python")
    assert ghost.step_strength("1.1") == 0
    gaps = ghost.disagreements(["1.1"])
    assert any(g.kind == "no-checkable-claim" for g in gaps)


def test_unreadable_file_yields_no_strength(tmp_path):
    """UNKNOWN proves nothing — a step cannot bank evidence it never got."""
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["gone.py"], exports=["X"])],
        str(tmp_path))
    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:gone.py#exports:X"].verdict == UNKNOWN
    assert ghost.step_strength("1.1") == 0


# ── run-level disagreements ──────────────────────────────────────────


def test_failed_but_clean_needs_a_green_acceptance_gate(tmp_path):
    """Structural checks cannot adjudicate a failed run."""
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["a.py"], verify_cmd="pytest -q")], root)
    _write(root, "a.py", "x = 1\n")

    ghost.resolve(["1.1"], language="python", gate_cmds=["pytest -q"])
    gaps = ghost.disagreements(["1.1"], pipeline_success=False)
    assert any(g.kind == "failed-but-clean" for g in gaps)


def test_structurally_perfect_but_behaviourally_broken_stays_silent(tmp_path):
    """Shape is not behaviour, and only a green gate speaks to behaviour.

    Observed on a 20B content-mode run: eight structurally perfect files
    — 41 postconditions green, every plan-declared anchor present — whose
    suite failed with "Ghost out of map bounds at (5, 7)". A real logic
    bug and a correctly-failed run, which this check told the user to
    blame on the harness.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["a.py"], exports=["Game"])], root)
    _write(root, "a.py", "class Game:\n    pass\n")

    ghost.resolve(["1.1"], language="python", gate_cmds=[])
    assert ghost.tally()[VIOLATED] == 0          # nothing structural is wrong
    gaps = ghost.disagreements(["1.1"], pipeline_success=False)
    assert not [g for g in gaps if g.kind == "failed-but-clean"]


def test_successful_clean_run_reports_nothing(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["a.py"], exports=["Board"],
               verify_cmd="pytest -q")], root)
    _write(root, "a.py", "class Board:\n    pass\n")

    ghost.resolve(["1.1"], language="python", gate_cmds=["pytest -q"])
    gaps = ghost.disagreements(["1.1"], tracked_files=["a.py"],
                               pipeline_success=True)
    assert gaps == []


def test_violations_only_reported_for_steps_claimed_done(tmp_path):
    """A step the pipeline never claimed cannot disagree with it."""
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["never.py"])], str(tmp_path))
    ghost.resolve(["1.1"], language="python")
    assert ghost.expectations["file:never.py#exists"].verdict == VIOLATED
    assert ghost.disagreements([]) == []


# ── invariants: read-only, never fatal ───────────────────────────────


def test_resolve_writes_nothing_to_the_project(tmp_path):
    root = str(tmp_path)
    _write(root, "a.py", "x = 1\n")
    before = sorted(os.listdir(root))
    ghost = GhostPlan.build([_step("1.1", target_files=["a.py"])], root)
    ghost.resolve(["1.1"], language="python", gate_cmds=["pytest"])
    ghost.report(["1.1"], tracked_files=["a.py"])
    assert sorted(os.listdir(root)) == before


def test_build_survives_malformed_steps(tmp_path):
    class Junk:
        id = "9.9"
        target_files = None
        exports = None
        imports_from = None
        verify_cmd = None

    ghost = GhostPlan.build([Junk()], str(tmp_path))
    ghost.resolve(["9.9"])
    assert ghost.report(["9.9"]) is not None


def test_resolve_is_idempotent(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("1.1", target_files=["a.py"])], root)
    _write(root, "a.py", "x = 1\n")
    ghost.resolve(["1.1"], language="python")
    first = ghost.tally()
    ghost.resolve(["1.1"], language="python")
    assert ghost.tally() == first


def test_repaired_postcondition_reports_current_state(tmp_path):
    """The verdict describes what shipped; history is kept separately.

    Observed on a real run: the plan ran `python -m venv venv && python
    -m pip install pygame`, which installs into the pipeline's
    interpreter rather than the new venv, so at wave 2 the declared
    dependency genuinely was missing and the shadow said so correctly.
    At wave 6 the agent loop's env self-heal reinstalled it into the
    project venv and the run finished green with the package present.
    A sticky VIOLATED then contradicted a run that was by then entirely
    correct. "Was broken once" and "is broken" are different claims.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("1.1", target_files=["a.py"])], root)

    ghost.resolve(["1.1"], language="python", stage="wave 1")
    assert ghost.expectations["file:a.py#exists"].verdict == VIOLATED

    _write(root, "a.py", "x = 1\n")
    ghost.resolve(["1.1"], language="python", stage="final")

    exp = ghost.expectations["file:a.py#exists"]
    assert exp.verdict == HOLDS          # what shipped
    assert exp.ever_violated is True     # history still recorded
    assert not [g for g in ghost.disagreements(["1.1"])
                if g.kind == "violated-exists"]
    assert any(o.verdict == VIOLATED for o in ghost.journal)


def test_to_dict_is_serializable(tmp_path):
    import json

    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["a.py"], exports=["X"])], root)
    _write(root, "a.py", "X = 1\n")
    ghost.resolve(["1.1"], language="python")
    assert json.loads(json.dumps(ghost.to_dict()))["tally"][HOLDS] >= 1

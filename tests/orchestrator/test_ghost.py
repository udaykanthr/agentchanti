"""Tests for the read-only ghost shadow (orchestrator/ghost.py).

The module exists to REPORT, never to decide, so these tests care about
two things in equal measure: that it names real disagreements, and that
it stays silent (UNKNOWN) whenever the evidence is inconclusive.
"""

import os
import re
import sys

import pytest

from agentchanti.orchestrator.ghost import (
    HOLDS, INAPPLICABLE, UNKNOWN, VIOLATED, GhostPlan, MIN_STEP_STRENGTH,
    degenerate_long_runs,
    ignored_varied_inputs,
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

    The verdict is therefore filesystem-dependent, and both readings are
    correct. Where the name resolves, EXISTS must still HOLD — the file
    genuinely works on this machine, and failing it would be false, so
    the portability defect is reported separately instead of distorting
    that verdict. Where it does not resolve, the declared target really
    is absent, and VIOLATED is the honest answer.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["src/board.py"])], root)
    _write(root, "src/Board.py", "class Board:\n    pass\n")

    ghost.resolve(["2.1"], language="python")
    exists = ghost.expectations["file:src/board.py#exists"]

    if ghost.case_mismatches:            # case-insensitive filesystem
        assert exists.verdict == HOLDS
        assert ghost.case_mismatches["src/board.py"] == "Board.py"
        gaps = ghost.disagreements(["2.1"])
        assert any(g.kind == "filename-case-mismatch" for g in gaps)
    else:                                # case-sensitive filesystem
        assert exists.verdict == VIOLATED


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


def test_violated_export_evidence_names_module_level_symbols(tmp_path):
    """The evidence must show the file's own names, not one class's methods.

    Observed on a real Pac-Man run: `entities.py` declared 14 module-level
    names, and the finding's evidence was an alphabetical head of the
    merged set — eight entries, three of them `Ghost.__init__`-style, with
    `Player` and `GridMover` cut. The verdict was correct and the evidence
    made it look like a false positive.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("3.1", target_files=["entities.py"],
               exports=["position_to_tile"])], root)
    body = ["UP = (0, -1)", "DOWN = (0, 1)", "Player = 1", "GridMover = 2",
            "", "", "class Ghost:"]
    body += [f"    def _m{i}(self):\n        pass\n" for i in range(20)]
    _write(root, "entities.py", "\n".join(body))

    ghost.resolve(["3.1"], language="python")
    exp = ghost.expectations["file:entities.py#exports:position_to_tile"]
    assert exp.verdict == VIOLATED
    # Every module-level name is present, ahead of any class member.
    for name in ("UP", "DOWN", "Player", "GridMover"):
        assert name in exp.evidence
    assert exp.evidence.index("Player") < exp.evidence.index("Ghost._m")
    # A truncated list must say so, or it reads as the complete set.
    assert "more)" in exp.evidence


def test_short_export_list_is_not_marked_truncated(tmp_path):
    """No elision count when nothing was elided."""
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("4.2", target_files=["main.py"], exports=["main"])], root)
    _write(root, "main.py", "def run():\n    return 0\n")

    ghost.resolve(["4.2"], language="python")
    evidence = ghost.expectations["file:main.py#exports:main"].evidence
    assert "run" in evidence
    assert "more)" not in evidence


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
    """Minimal venv skeleton the Executor's detector will accept.

    Built for the running platform. The detector looks for
    ``Scripts/python.exe`` on Windows and ``bin/python`` everywhere else,
    so a fixture that hardcodes either layout is simply invisible on the
    other — the venv goes undetected and every dependency verdict
    collapses to UNKNOWN instead of the HOLDS/VIOLATED under test.
    """
    if os.name == "nt":
        bin_dir = os.path.join(root, "venv", "Scripts")
        exe = "python.exe"
        site = os.path.join(root, "venv", "Lib", "site-packages")
    else:
        bin_dir = os.path.join(root, "venv", "bin")
        exe = "python"
        site = os.path.join(
            root, "venv", "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages")
    os.makedirs(bin_dir, exist_ok=True)
    open(os.path.join(bin_dir, exe), "wb").close()
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


# ── test files the declared runner never collects ────────────────────


UNITTEST_STYLE = (
    "import unittest\n\n\n"
    "class TestThing(unittest.TestCase):\n"
    "    def test_a(self):\n        pass\n"
    "    def test_b(self):\n        pass\n"
)
PYTEST_STYLE = (
    "def test_initial_position(Player, tmp_path):\n    pass\n\n\n"
    "def test_moves(Player):\n    pass\n"
)


def test_pytest_style_file_under_unittest_is_reported(tmp_path):
    """20 of 22 tests were invisible to the project's own command.

    Observed: an agent loop wrote four pytest-style modules while the
    acceptance command was `python -m unittest -v`, which collects only
    TestCase subclasses. unittest reported 2 tests and passed; pytest on
    the same directory reported 22.
    """
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_game.py"],
                 verify_cmd="python -m unittest -v")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_game.py", UNITTEST_STYLE)
    _write(root, "tests/test_player.py", PYTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    gaps = ghost.disagreements(
        ["5.1"], tracked_files=["tests/test_game.py", "tests/test_player.py"])

    bad = [g for g in gaps if g.kind == "tests-never-collected"]
    assert len(bad) == 1
    assert "test_player.py" in bad[0].detail
    assert "pytest-style" in bad[0].detail


def test_unittest_style_file_is_not_reported(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_game.py"],
                 verify_cmd="python -m unittest -v")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_game.py", UNITTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_pytest_runner_collects_both_styles(tmp_path):
    """Under pytest the module-level style is perfectly valid."""
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_player.py"],
                 verify_cmd="pytest -q")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/test_player.py", PYTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_unknown_runner_stays_silent(tmp_path):
    """No identifiable runner means no rule to judge collection by."""
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_player.py"])
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/test_player.py", PYTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_empty_test_file_is_reported(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_empty.py"],
                 verify_cmd="python -m unittest -v")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_empty.py", "import unittest\n")

    ghost.resolve(["5.1"], language="python")
    bad = [g for g in ghost.disagreements(["5.1"])
           if g.kind == "tests-never-collected"]
    assert len(bad) == 1
    assert "no tests" in bad[0].detail


def test_non_test_files_are_never_judged(tmp_path):
    root = str(tmp_path)
    step = _step("2.1", target_files=["game.py"],
                 verify_cmd="python -m unittest -v")
    ghost = GhostPlan.build([step], root)
    _write(root, "game.py", "class Game:\n    pass\n")

    ghost.resolve(["2.1"], language="python")
    assert not [g for g in ghost.disagreements(["2.1"])
                if g.kind == "tests-never-collected"]


def test_unparseable_test_file_is_not_accused(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_bad.py"],
                 verify_cmd="python -m unittest -v")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_bad.py", "def broken(:\n")

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


# ── degenerate long runs ─────────────────────────────────────────────
#
# The blind spot: a suite that satisfies "run 2000+ frames and assert the
# invariants" while simulating fifty, because `update()` early-returns
# once the run ends and every later iteration asserts a frozen state.

_GAME = '''\
PLAYING = "playing"
GAME_OVER = "over"


class Game:
    def __init__(self):
        self.state = PLAYING

    def update(self, dt):
        if self.state is not PLAYING:
            return
        self.tick = dt
'''


def _project(root, test_body, game=_GAME):
    _write(root, "game.py", game)
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, PLAYING, GAME_OVER\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def test_long(self):\n"
           "        game = Game()\n" + test_body)
    return ["game.py", "test_sim.py"]


def test_unguarded_long_run_is_reported(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n")
    found = degenerate_long_runs(root, paths)
    assert len(found) == 1
    assert "2000 times" in found[0][1]
    assert "update()" in found[0][1]


def test_state_pinned_inside_loop_is_silent(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertEqual(game.state, PLAYING)\n")
    assert degenerate_long_runs(root, paths) == []


def test_state_pinned_after_loop_is_silent(tmp_path):
    """A post-loop check still fails when the run ended early."""
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "        self.assertEqual(game.state, PLAYING)\n")
    assert degenerate_long_runs(root, paths) == []


def test_ruling_out_the_terminal_state_is_a_pin(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertNotEqual(game.state, GAME_OVER)\n")
    assert degenerate_long_runs(root, paths) == []


def test_tautological_state_assertion_is_not_a_pin(tmp_path):
    """`assertIn(state, (PLAYING, GAME_OVER))` admits what it should exclude.

    Written verbatim by a real run, in a 2100-iteration loop that
    simulated 246 frames.
    """
    root = str(tmp_path)
    paths = _project(
        root, "        for _ in range(2100):\n"
              "            game.update(0.016)\n"
              "            self.assertIn(game.state, (PLAYING, GAME_OVER))\n")
    assert len(degenerate_long_runs(root, paths)) == 1


def test_deliberately_extended_lifetime_is_silent(tmp_path):
    """A test that disables the thing ending the run thought about it."""
    root = str(tmp_path)
    paths = _project(root, "        game.spawn_protection_timer = 1_000_000.0\n"
                           "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n")
    assert degenerate_long_runs(root, paths) == []


def test_guard_behind_a_validation_prologue_is_still_found(tmp_path):
    """The state guard is not always the first statement.

    A real `update()` opened with `if not math.isfinite(dt): raise`, which
    a `body[0]` reading missed entirely.
    """
    root = str(tmp_path)
    game = ('PLAYING = "playing"\n\n\nclass Game:\n'
            '    def __init__(self):\n        self.state = PLAYING\n\n'
            '    def update(self, dt):\n'
            '        if dt < 0:\n            raise ValueError("dt")\n'
            '        if self.state != PLAYING:\n            return\n'
            '        self.tick = dt\n')
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n", game=game)
    assert len(degenerate_long_runs(root, paths)) == 1


def test_short_loop_is_not_a_long_run(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(10):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n")
    assert degenerate_long_runs(root, paths) == []


def test_loop_that_breaks_out_is_silent(tmp_path):
    """Breaking on termination is not asserting against a frozen state."""
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            if game.state != PLAYING:\n"
                           "                break\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n")
    assert degenerate_long_runs(root, paths) == []


def test_unguarded_advance_method_is_never_accused(tmp_path):
    """No early-return guard means no frames can be silently skipped."""
    root = str(tmp_path)
    game = ('PLAYING = "playing"\n\n\nclass Game:\n'
            '    def __init__(self):\n        self.state = PLAYING\n\n'
            '    def update(self, dt):\n        self.tick = dt\n')
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n", game=game)
    assert degenerate_long_runs(root, paths) == []


def test_unparseable_test_file_is_not_accused_of_degeneracy(tmp_path):
    root = str(tmp_path)
    _write(root, "game.py", _GAME)
    _write(root, "test_sim.py", "def broken(:\n")
    assert degenerate_long_runs(root, ["game.py", "test_sim.py"]) == []


def test_degenerate_long_run_surfaces_as_a_disagreement(tmp_path):
    """End-to-end: the finding reaches the reported disagreement list."""
    root = str(tmp_path)
    _project(root, "        for _ in range(2000):\n"
                   "            game.update(0.016)\n"
                   "            self.assertTrue(game.ok)\n")
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["game.py", "test_sim.py"])], root)
    ghost.resolve(["1.1"], language="python")

    gaps = [g for g in ghost.disagreements(["1.1"])
            if g.kind == "degenerate-long-run"]
    assert len(gaps) == 1
    assert "test_sim.py" in gaps[0].detail


def test_restart_on_termination_is_silent(tmp_path):
    """Noticing the run ended and starting another is honest.

    A real suite wrote exactly this and was live for 2000 of 2000 frames
    across six restarts; the first version of this check reported all
    three of its loops.
    """
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n"
                           "            if game.state is GAME_OVER:\n"
                           "                game.restart()\n")
    assert degenerate_long_runs(root, paths) == []


def test_same_method_name_on_an_unguarded_class_is_not_accused(tmp_path):
    """`Player.update` is not `Game.update`, whatever the name says.

    A real suite drove the player and ghost directly for 800 frames.
    Neither class early-returns, so no frame can be silently skipped, but
    matching on the bare name `update` reported it anyway.
    """
    root = str(tmp_path)
    _write(root, "game.py", _GAME + '''

class Player:
    def __init__(self):
        self.tile = (0, 0)

    def update(self, dt):
        self.tile = (dt, dt)
''')
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, Player, PLAYING, GAME_OVER\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def test_long(self):\n"
           "        player = Player()\n"
           "        for _ in range(800):\n"
           "            player.update(0.016)\n"
           "            self.assertTrue(player.tile)\n")
    assert degenerate_long_runs(root, ["game.py", "test_sim.py"]) == []


def test_guarded_class_is_still_caught_alongside_an_unguarded_twin(tmp_path):
    """Adding an unguarded `Player.update` must not silence `Game.update`."""
    root = str(tmp_path)
    _write(root, "game.py", _GAME + '''

class Player:
    def update(self, dt):
        self.tile = dt
''')
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, Player, PLAYING, GAME_OVER\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def test_long(self):\n"
           "        game = Game()\n"
           "        for _ in range(2000):\n"
           "            game.update(0.016)\n"
           "            self.assertTrue(game.ok)\n")
    found = degenerate_long_runs(root, ["game.py", "test_sim.py"])
    assert len(found) == 1
    assert "Game.update()" in found[0][1]


def test_fixture_receiver_from_setup_is_resolved(tmp_path):
    """`self.game = Game()` in setUp binds the receiver just as well."""
    root = str(tmp_path)
    _write(root, "game.py", _GAME)
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, PLAYING, GAME_OVER\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def setUp(self):\n"
           "        self.game = Game()\n\n"
           "    def test_long(self):\n"
           "        for _ in range(2000):\n"
           "            self.game.update(0.016)\n"
           "            self.assertTrue(self.game.ok)\n")
    assert len(degenerate_long_runs(root, ["game.py", "test_sim.py"])) == 1


def test_guard_with_work_before_the_return_is_found(tmp_path):
    """A guard may do a little work before it leaves.

    Observed: `if self.state != PLAYING or dt == 0.0: self.assert_invariants();
    return` — and re-asserting the invariants is exactly why the frozen
    frames kept passing. Requiring a single-statement body missed it, and
    the suite it belonged to ran 171 of 2001 frames.
    """
    root = str(tmp_path)
    game = ('PLAYING = "playing"\n\n\nclass Game:\n'
            '    def __init__(self):\n        self.state = PLAYING\n\n'
            '    def assert_invariants(self):\n        pass\n\n'
            '    def update(self, dt):\n'
            '        if self.state != PLAYING or dt == 0.0:\n'
            '            self.assert_invariants()\n            return\n'
            '        self.tick = dt\n')
    paths = _project(root, "        for _ in range(2001):\n"
                           "            game.update(0.008)\n"
                           "            self.assertTrue(game.ok)\n", game=game)
    assert len(degenerate_long_runs(root, paths)) == 1


def test_and_guard_is_not_read_as_a_state_guard(tmp_path):
    """`or` fires whenever the state dies; `and` may decline to."""
    root = str(tmp_path)
    game = ('PLAYING = "playing"\n\n\nclass Game:\n'
            '    def __init__(self):\n        self.state = PLAYING\n\n'
            '    def update(self, dt):\n'
            '        if self.state != PLAYING and dt == 0.0:\n'
            '            return\n'
            '        self.tick = dt\n')
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n", game=game)
    assert degenerate_long_runs(root, paths) == []


def test_gating_input_on_the_live_state_is_not_handling_termination(tmp_path):
    """Reading the state to decide whether to send input changes nothing.

    A real suite wrote `if game.state == PLAYING and frame % 11 == 0:
    request_direction(...)` and still ran 44 of its 650 frames.
    """
    root = str(tmp_path)
    paths = _project(
        root, "        for frame in range(650):\n"
              "            if game.state == PLAYING and frame % 11 == 0:\n"
              "                game.request_direction(1)\n"
              "            game.update(0.016)\n"
              "            self.assertTrue(game.ok)\n")
    assert len(degenerate_long_runs(root, paths)) == 1


# ── the guard spelled the other way round ────────────────────────────
#
# Measured on a Pac-Man run whose suite was fully green: `Game.update`
# opened with `if self.state in (WIN, GAME_OVER): return` — the mirror
# image of `is not PLAYING`, naming the states that end the run instead
# of the one that keeps it going. Reading only the first spelling left
# the class unguarded, so nothing was ever a long-run candidate, and the
# suite's `range(700)` acceptance loop simulated 17 frames.

_GAME_DEAD_LISTED = '''\
PLAYING = "playing"
WIN = "win"
GAME_OVER = "over"


class Game:
    def __init__(self):
        self.state = PLAYING

    def update(self, dt):
        if self.state in (WIN, GAME_OVER):
            return
        self.tick = dt
'''


def test_guard_listing_the_terminal_states_is_read(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n",
                     game=_GAME_DEAD_LISTED)
    found = degenerate_long_runs(root, paths)
    assert len(found) == 1
    assert "reaches GAME_OVER/WIN" in found[0][1]


def test_equality_guard_on_one_terminal_state_is_read(tmp_path):
    root = str(tmp_path)
    game = _GAME_DEAD_LISTED.replace("if self.state in (WIN, GAME_OVER):",
                                     "if self.state == GAME_OVER:")
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n",
                     game=game)
    assert len(degenerate_long_runs(root, paths)) == 1


def test_ordering_comparison_is_still_not_a_state_guard(tmp_path):
    """An ordering test says nothing this check can read."""
    root = str(tmp_path)
    game = _GAME_DEAD_LISTED.replace("if self.state in (WIN, GAME_OVER):",
                                     "if self.state < 0:")
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertTrue(game.ok)\n",
                     game=game)
    assert degenerate_long_runs(root, paths) == []


def test_pinning_still_silences_a_dead_listed_guard(tmp_path):
    """The silences must survive the spelling, not just the finding."""
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertEqual(game.state, PLAYING)\n",
                     game=_GAME_DEAD_LISTED)
    assert degenerate_long_runs(root, paths) == []


def test_ruling_out_a_terminal_state_still_silences(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            self.assertNotEqual(game.state, WIN)\n",
                     game=_GAME_DEAD_LISTED)
    assert degenerate_long_runs(root, paths) == []


def test_tautology_is_still_not_a_pin_when_the_guard_lists_the_dead(tmp_path):
    root = str(tmp_path)
    paths = _project(
        root,
        "        for _ in range(2000):\n"
        "            game.update(0.016)\n"
        "            self.assertIn(game.state, (PLAYING, WIN, GAME_OVER))\n",
        game=_GAME_DEAD_LISTED)
    assert len(degenerate_long_runs(root, paths)) == 1


def test_restarting_a_finished_run_still_silences(tmp_path):
    root = str(tmp_path)
    paths = _project(root, "        for _ in range(2000):\n"
                           "            game.update(0.016)\n"
                           "            if game.state == GAME_OVER:\n"
                           "                game.restart()\n",
                     game=_GAME_DEAD_LISTED)
    assert degenerate_long_runs(root, paths) == []


# ── the advance call one level down ──────────────────────────────────
#
# The natural way to write "assert the invariants every frame" is a
# helper that updates and then asserts, with the loop calling only the
# helper. The receiver `self.update_and_check` binds to no class, so the
# loop looked as though it advanced nothing at all.

def _with_helper(root, loop_body, helper_body, game=_GAME):
    _write(root, "game.py", game)
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, PLAYING, GAME_OVER\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def setUp(self):\n"
           "        self.game = Game()\n\n"
           "    def step(self, game, dt):\n" + helper_body + "\n"
           "    def test_long(self):\n" + loop_body)
    return ["game.py", "test_sim.py"]


def test_advance_through_a_helper_is_found(tmp_path):
    root = str(tmp_path)
    paths = _with_helper(
        root,
        "        for _ in range(2000):\n"
        "            self.step(self.game, 0.016)\n",
        "        game.update(dt)\n"
        "        self.assertTrue(game.ok)\n")
    found = degenerate_long_runs(root, paths)
    assert len(found) == 1
    assert "`Game.update()`" in found[0][1]


def test_a_helper_that_pins_the_state_silences_its_callers(tmp_path):
    """The helper runs every iteration, so its pin is the loop's pin."""
    root = str(tmp_path)
    paths = _with_helper(
        root,
        "        for _ in range(2000):\n"
        "            self.step(self.game, 0.016)\n",
        "        game.update(dt)\n"
        "        self.assertEqual(game.state, PLAYING)\n")
    assert degenerate_long_runs(root, paths) == []


def test_a_helper_that_restarts_the_run_silences_its_callers(tmp_path):
    root = str(tmp_path)
    paths = _with_helper(
        root,
        "        for _ in range(2000):\n"
        "            self.step(self.game, 0.016)\n",
        "        game.update(dt)\n"
        "        if game.state is not PLAYING:\n"
        "            game.restart()\n")
    assert degenerate_long_runs(root, paths) == []


def test_helper_driving_an_unguarded_class_is_not_evidence(tmp_path):
    """`Player.update` is not state-guarded, and routing it through a
    helper must not make it look as though it were."""
    root = str(tmp_path)
    game = _GAME + (
        "\n\nclass Player:\n"
        "    def update(self, dt):\n"
        "        self.moved = dt\n")
    _write(root, "game.py", game)
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, Player, PLAYING\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def setUp(self):\n"
           "        self.player = Player()\n\n"
           "    def step(self, player, dt):\n"
           "        player.update(dt)\n\n"
           "    def test_long(self):\n"
           "        for _ in range(2000):\n"
           "            self.step(self.player, 0.016)\n")
    assert degenerate_long_runs(root, ["game.py", "test_sim.py"]) == []


def test_helper_recursion_terminates(tmp_path):
    root = str(tmp_path)
    _write(root, "game.py", _GAME)
    _write(root, "test_sim.py",
           "import unittest\n"
           "from game import Game, PLAYING\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def setUp(self):\n"
           "        self.game = Game()\n\n"
           "    def step(self, game, dt):\n"
           "        self.step(game, dt)\n\n"
           "    def test_long(self):\n"
           "        for _ in range(2000):\n"
           "            self.step(self.game, 0.016)\n")
    assert degenerate_long_runs(root, ["game.py", "test_sim.py"]) == []


# ── a plan that declared no file at all ──────────────────────────────


def test_plan_with_no_file_target_reports_once_not_per_file(tmp_path):
    """Observed: a 20B planner named its files in prose and emitted no
    `target:` line on any of six CODE/TEST steps. Every source file the
    run produced was then reported as an unplanned write — six findings
    that were each true of the plan and useless about the code, burying
    the one write that really was a stray."""
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "venv"), exist_ok=True)
    ghost = GhostPlan.build([_step("1.1", step_type="CMD",
                                   target_files=["venv"])], root)
    found = ghost.disagreements(
        ["1.1"], tracked_files=["game.py", "map.py", "temp.py"])
    kinds = [d.kind for d in found]
    assert kinds.count("plan-declares-no-targets") == 1
    assert "unplanned-write" not in kinds
    detail = [d.detail for d in found
              if d.kind == "plan-declares-no-targets"][0]
    assert "3 file(s)" in detail


def test_one_declared_file_restores_the_per_file_finding(tmp_path):
    """The plan-level finding is about a plan that declared nothing. As
    soon as one step declares a file, a stray write is a stray write."""
    root = str(tmp_path)
    _write(root, "a.py", "A = 1\n")
    ghost = GhostPlan.build([_step("1.1", target_files=["a.py"])], root)
    found = ghost.disagreements(["1.1"], tracked_files=["a.py", "sneaky.py"])
    kinds = [d.kind for d in found]
    assert kinds.count("unplanned-write") == 1
    assert "plan-declares-no-targets" not in kinds


# ── the check is about a shape, not about games ──────────────────────
#
# Every artifact that motivated `degenerate-long-run` happened to be a
# generated game, which is a fact about what got benchmarked, not about
# what the rule reads. What it actually requires is a class whose
# advance method early-returns on a state comparison, and a long literal
# loop that reaches it without pinning that state — a shape shared by
# schedulers, retry loops, workflow engines and stream consumers. These
# tests are written in a domain with no game vocabulary anywhere, and
# fail the moment any is baked into the module.

_INGEST = '''\
RUNNING = "running"
FAILED = "failed"
COMPLETED = "completed"


class Ingest:
    def __init__(self):
        self.status = RUNNING

    def drain(self, batch):
        if self.status in (FAILED, COMPLETED):
            return
        self.rows = batch
'''


def _ingest_project(root, test_body):
    _write(root, "ingest.py", _INGEST)
    _write(root, "test_ingest.py",
           "import unittest\n"
           "from ingest import Ingest, RUNNING, FAILED\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def test_drains_everything(self):\n"
           "        worker = Ingest()\n" + test_body)
    return ["ingest.py", "test_ingest.py"]


def test_a_worker_loop_that_can_stop_draining_is_reported(tmp_path):
    root = str(tmp_path)
    paths = _ingest_project(root, "        for _ in range(5000):\n"
                                  "            worker.drain(10)\n"
                                  "            self.assertTrue(worker.rows)\n")
    found = degenerate_long_runs(root, paths)
    assert len(found) == 1
    assert "`Ingest.drain()`" in found[0][1]
    assert "`status` reaches COMPLETED/FAILED" in found[0][1]


def test_a_worker_loop_that_pins_its_status_is_silent(tmp_path):
    root = str(tmp_path)
    paths = _ingest_project(root, "        for _ in range(5000):\n"
                                  "            worker.drain(10)\n"
                                  "            self.assertEqual(worker.status, RUNNING)\n")
    assert degenerate_long_runs(root, paths) == []


def test_findings_survive_renaming_every_identifier(tmp_path):
    """The proof that nothing about the domain is hardcoded: rename the
    class, the method, the attribute and the states, and the same loop
    is still reported — with the new names, read back out of the source.
    """
    root = str(tmp_path)
    swaps = {"Ingest": "Widget", "drain": "poke", "status": "phase",
             "RUNNING": "ACTIVE", "FAILED": "ABORTED",
             "COMPLETED": "FINISHED", "worker": "gadget", "rows": "blobs"}

    def _rename(text):
        for old, new in swaps.items():
            text = re.sub(rf"\b{old}\b", new, text)
        return text

    _write(root, "ingest.py", _rename(_INGEST))
    _write(root, "test_ingest.py", _rename(
        "import unittest\n"
        "from ingest import Ingest, RUNNING, FAILED\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_drains_everything(self):\n"
        "        worker = Ingest()\n"
        "        for _ in range(5000):\n"
        "            worker.drain(10)\n"
        "            self.assertTrue(worker.rows)\n"))
    found = degenerate_long_runs(root, ["ingest.py", "test_ingest.py"])
    assert len(found) == 1
    assert "`Widget.poke()`" in found[0][1]
    assert "`phase` reaches ABORTED/FINISHED" in found[0][1]


# ── inputs a test varies that the code never reads ───────────────────
#
# Measured: a task demanded ">= 600 frames with dt drawn randomly from
# 0.008..0.05, not a fixed 1/60". The suite looped 600 and then 2000
# times doing exactly that, and passed. `run_frame(self, dt)` never
# mentions `dt` — its docstring said the parameter was "accepted to
# mirror the interface the tests expect". Replaying at dt = 1e-9, 1/60
# and 1e9 gave byte-identical results. Every gate was green.

_SAMPLER = '''\
class Sampler:
    def __init__(self):
        self.seen = 0

    def take(self, amount):
        self.seen += 1
'''


def _sampler_project(root, test_body, source=_SAMPLER):
    _write(root, "sampler.py", source)
    _write(root, "test_sampler.py",
           "import random\n"
           "import unittest\n"
           "from sampler import Sampler\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def setUp(self):\n"
           "        self.sampler = Sampler()\n\n"
           "    def test_varies(self):\n" + test_body)
    return ["sampler.py", "test_sampler.py"]


def test_randomised_argument_the_callee_drops_is_reported(tmp_path):
    root = str(tmp_path)
    paths = _sampler_project(
        root, "        for _ in range(600):\n"
              "            amount = random.uniform(0.008, 0.05)\n"
              "            self.sampler.take(amount)\n")
    found = ignored_varied_inputs(root, paths)
    assert len(found) == 1
    assert "`amount`" in found[0][1]
    assert "`Sampler.take()`" in found[0][1]


def test_argument_passed_inline_is_also_seen(tmp_path):
    """`take(random.uniform(...))` needs no intermediate variable."""
    root = str(tmp_path)
    paths = _sampler_project(
        root, "        for _ in range(600):\n"
              "            self.sampler.take(random.uniform(0.008, 0.05))\n")
    assert len(ignored_varied_inputs(root, paths)) == 1


def test_the_loop_variable_counts_as_variation(tmp_path):
    root = str(tmp_path)
    paths = _sampler_project(
        root, "        for step in range(600):\n"
              "            self.sampler.take(step)\n")
    assert len(ignored_varied_inputs(root, paths)) == 1


def test_a_parameter_the_callee_reads_is_silent(tmp_path):
    """The whole point: this is about the input never arriving."""
    root = str(tmp_path)
    source = _SAMPLER.replace("        self.seen += 1",
                              "        self.seen += amount")
    paths = _sampler_project(
        root, "        for _ in range(600):\n"
              "            amount = random.uniform(0.008, 0.05)\n"
              "            self.sampler.take(amount)\n",
        source=source)
    assert ignored_varied_inputs(root, paths) == []


def test_a_constant_argument_is_not_variation(tmp_path):
    """A fixed-value long run claims endurance, not adversarial input,
    and is `degenerate-long-run`'s business rather than this check's."""
    root = str(tmp_path)
    paths = _sampler_project(
        root, "        for _ in range(2000):\n"
              "            self.sampler.take(0.008)\n")
    assert ignored_varied_inputs(root, paths) == []


def test_var_positional_callee_is_never_accused(tmp_path):
    """`*args` may route the value somewhere this cannot follow."""
    root = str(tmp_path)
    source = ("class Sampler:\n"
              "    def take(self, amount, *args):\n"
              "        self.seen = args\n")
    paths = _sampler_project(
        root, "        for _ in range(600):\n"
              "            self.sampler.take(random.random())\n",
        source=source)
    assert ignored_varied_inputs(root, paths) == []


def test_unresolved_receiver_is_not_evidence(tmp_path):
    """Same rule as everywhere else here: an unbound receiver proves
    nothing, so nothing is claimed."""
    root = str(tmp_path)
    _write(root, "sampler.py", _SAMPLER)
    _write(root, "test_sampler.py",
           "import random\n"
           "import unittest\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def test_varies(self):\n"
           "        for _ in range(600):\n"
           "            self.whatever.take(random.random())\n")
    assert ignored_varied_inputs(
        root, ["sampler.py", "test_sampler.py"]) == []


def test_an_ambiguous_method_name_is_dropped(tmp_path):
    """Two classes with the same qualified name is not a target."""
    root = str(tmp_path)
    source = (_SAMPLER + "\n\nclass Sampler:\n"
              "    def take(self, amount):\n"
              "        self.total = amount\n")
    paths = _sampler_project(
        root, "        for _ in range(600):\n"
              "            self.sampler.take(random.random())\n",
        source=source)
    assert ignored_varied_inputs(root, paths) == []


def test_the_finding_survives_renaming_every_identifier(tmp_path):
    """Nothing about time, frames, dt or games is in the rule."""
    root = str(tmp_path)
    _write(root, "throttle.py",
           "class Throttle:\n"
           "    def admit(self, weight):\n"
           "        self.calls = 1\n")
    _write(root, "test_throttle.py",
           "import random\n"
           "import unittest\n"
           "from throttle import Throttle\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def setUp(self):\n"
           "        self.throttle = Throttle()\n\n"
           "    def test_weights(self):\n"
           "        for _ in range(300):\n"
           "            w = random.choice([1, 5, 9])\n"
           "            self.throttle.admit(w)\n")
    found = ignored_varied_inputs(root, ["throttle.py", "test_throttle.py"])
    assert len(found) == 1
    assert "`weight`" in found[0][1] and "`Throttle.admit()`" in found[0][1]


def test_ignored_input_surfaces_as_a_disagreement(tmp_path):
    root = str(tmp_path)
    _write(root, "sampler.py", _SAMPLER)
    _write(root, "test_sampler.py",
           "import random\n"
           "import unittest\n"
           "from sampler import Sampler\n\n\n"
           "class T(unittest.TestCase):\n"
           "    def test_varies(self):\n"
           "        sampler = Sampler()\n"
           "        for _ in range(600):\n"
           "            sampler.take(random.random())\n")
    ghost = GhostPlan.build(
        [_step("1.1", target_files=["sampler.py", "test_sampler.py"])], root)
    ghost.resolve(["1.1"], language="python")
    found = [d for d in ghost.disagreements(["1.1"])
             if d.kind == "varied-input-ignored"]
    assert len(found) == 1


# ── test files discovery cannot reach ────────────────────────────────
#
# Measured 2026-08-18: a step targeted `tests/test_game.py` with the gate
# `python -m unittest`, and there was no `tests/__init__.py`. Discovery
# recurses only into importable packages, so the six tests it wrote were
# never collected — while the gate passed throughout on the seeded
# acceptance contract at the ROOT. The step spent eight of ten turns
# getting no signal from its own gate about its own work.


def test_test_dir_without_init_is_reported(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_game.py"],
                 verify_cmd="python -m unittest")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/test_game.py", UNITTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    bad = [g for g in ghost.disagreements(["5.1"])
           if g.kind == "tests-never-collected"]
    assert len(bad) == 1
    assert "__init__.py" in bad[0].detail


def test_reachable_test_dir_is_not_reported(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_game.py"],
                 verify_cmd="python -m unittest")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_game.py", UNITTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_root_level_test_file_needs_no_package(tmp_path):
    root = str(tmp_path)
    step = _step("5.1", target_files=["test_game.py"],
                 verify_cmd="python -m unittest")
    ghost = GhostPlan.build([step], root)
    _write(root, "test_game.py", UNITTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_explicit_start_directory_is_not_accused(tmp_path):
    """`discover -s tests` makes tests/ its own top level — it runs fine."""
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_game.py"],
                 verify_cmd="python -m unittest discover -s tests")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/test_game.py", UNITTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_named_test_target_is_not_discovery(tmp_path):
    """`python -m unittest tests.test_game` names its target outright."""
    root = str(tmp_path)
    step = _step("5.1", target_files=["tests/test_game.py"],
                 verify_cmd="python -m unittest tests.test_game")
    ghost = GhostPlan.build([step], root)
    _write(root, "tests/test_game.py", UNITTEST_STYLE)

    ghost.resolve(["5.1"], language="python")
    assert not [g for g in ghost.disagreements(["5.1"])
                if g.kind == "tests-never-collected"]


def test_discovery_detection_reads_a_compound_gate():
    from agentchanti.orchestrator.ghost import discovers_from_project_root

    # The gate the measured run actually recorded.
    assert discovers_from_project_root(
        ["python -m unittest && python main.py --headless --frames 300"])
    assert discovers_from_project_root(["python -m unittest -v"])
    assert not discovers_from_project_root(["pytest -q"])
    assert not discovers_from_project_root(
        ["python -m unittest discover -s tests -t ."])

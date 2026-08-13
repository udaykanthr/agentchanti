"""Tests for deterministic gap repair (orchestrator/ghost_heal.py).

Two obligations carry equal weight here: that a repairable gap actually
gets repaired, and that the healer NEVER fabricates content to make a
check pass. The second is the one that keeps the shadow worth reading.
"""

import os

import pytest

from agentchanti.orchestrator.ghost import (
    HOLDS, VIOLATED, GhostPlan,
)
from agentchanti.orchestrator.ghost_heal import GhostHealer
from agentchanti.orchestrator.plan_step import PlanStep


def _step(sid, **kw):
    kw.setdefault("step_type", "CODE")
    return PlanStep(id=sid, **kw)


def _write(root, rel, text):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _make_venv(root, packages=()):
    scripts = os.path.join(root, "venv", "Scripts")
    os.makedirs(scripts, exist_ok=True)
    open(os.path.join(scripts, "python.exe"), "wb").close()
    site = os.path.join(root, "venv", "Lib", "site-packages")
    os.makedirs(site, exist_ok=True)
    for pkg in packages:
        os.makedirs(os.path.join(site, f"{pkg}-1.0.dist-info"), exist_ok=True)
    return site


class FakeExecutor:
    """Records commands; optionally simulates the install landing."""

    def __init__(self, site_dir=None, install=(), ok=True):
        self.commands = []
        self.site_dir = site_dir
        self.install = list(install)
        self.ok = ok

    def run_command(self, cmd, cwd=None, timeout=None, **kw):
        self.commands.append(cmd)
        if self.ok and self.site_dir:
            for pkg in self.install:
                os.makedirs(os.path.join(self.site_dir,
                                         f"{pkg}-1.0.dist-info"),
                            exist_ok=True)
        return self.ok, "" if self.ok else "ERROR: could not install"


# ── the defect this was built for ────────────────────────────────────


def test_missing_dependency_is_installed_and_verified(tmp_path):
    """The venv/interpreter mismatch that shipped a non-starting app.

    Observed on both benchmark arms: `python -m venv venv && python -m
    pip install pygame` creates the venv but installs into the pipeline's
    interpreter, so the app's own environment never gets the package.
    """
    root = str(tmp_path)
    site = _make_venv(root, packages=[])
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")
    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == VIOLATED

    ex = FakeExecutor(site_dir=site, install=["pygame"])
    healer = GhostHealer(ghost, ex)
    results = healer.heal(["2.1"], language="python")

    assert len(results) == 1 and results[0].ok
    assert "pip install pygame" in ex.commands[0]
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == HOLDS


def test_install_targets_the_project_interpreter(tmp_path):
    """Installing with bare `python` is the bug, not the fix."""
    root = str(tmp_path)
    site = _make_venv(root)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")
    ghost.resolve(["2.1"], language="python")

    ex = FakeExecutor(site_dir=site, install=["pygame"])
    GhostHealer(ghost, ex).heal(["2.1"], language="python")

    cmd = ex.commands[0]
    assert "venv" in cmd and "python.exe" in cmd
    assert not cmd.startswith("python -m pip")


def test_failed_install_leaves_the_verdict_red(tmp_path):
    root = str(tmp_path)
    _make_venv(root)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")
    ghost.resolve(["2.1"], language="python")

    ex = FakeExecutor(ok=False)
    results = GhostHealer(ghost, ex).heal(["2.1"], language="python")

    assert results and not results[0].ok
    assert ghost.expectations[
        "pkg:requirements.txt#deps-installed"].verdict == VIOLATED


def test_each_gap_is_attempted_only_once(tmp_path):
    root = str(tmp_path)
    _make_venv(root)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")
    ghost.resolve(["2.1"], language="python")

    ex = FakeExecutor(ok=False)
    healer = GhostHealer(ghost, ex)
    healer.heal(["2.1"], language="python")
    healer.heal(["2.1"], language="python")
    assert len(ex.commands) == 1


# ── package markers ──────────────────────────────────────────────────


def test_missing_package_marker_is_created(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("5.1", target_files=["tests/__init__.py"])], root)
    ghost.resolve(["5.1"], language="python")
    assert ghost.expectations[
        "file:tests/__init__.py#exists"].verdict == VIOLATED

    GhostHealer(ghost, FakeExecutor()).heal(["5.1"], language="python")

    assert os.path.isfile(os.path.join(root, "tests", "__init__.py"))
    assert ghost.expectations[
        "file:tests/__init__.py#exists"].verdict == HOLDS


def test_missing_file_with_no_plan_body_is_not_invented(tmp_path):
    """With no plan body there is no source to restore from.

    `__init__.py` is healable because empty IS its correct content. A
    file the plan gave no body for needs content nobody specified, and
    writing a stub would make the gap undetectable.
    """
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("2.1", target_files=["game.py"])], root)
    ghost.resolve(["2.1"], language="python")

    results = GhostHealer(ghost, FakeExecutor()).heal(["2.1"],
                                                      language="python")
    assert results == []
    assert not os.path.exists(os.path.join(root, "game.py"))
    assert ghost.expectations["file:game.py#exists"].verdict == VIOLATED


# ── import edges ─────────────────────────────────────────────────────


def _edge_project(root, consumer_body):
    _write(root, "game.py", "class Game:\n    pass\n\n\nclass GameState:\n"
                            "    pass\n")
    _write(root, "tests/__init__.py", "")
    _write(root, "test_game.py", consumer_body)
    step = _step("5.1", target_files=["test_game.py"],
                 imports_from={"game.py": ["Game", "GameState"]})
    return GhostPlan.build([step], root)


def test_missing_import_is_added_when_the_symbol_is_used(tmp_path):
    """The consumer references the names — without the import it crashes."""
    root = str(tmp_path)
    ghost = _edge_project(
        root, "import unittest\n\n\ng = Game()\ns = GameState\n")
    ghost.resolve(["5.1"], language="python")
    assert ghost.expectations["edge:game.py->step:5.1"].verdict == VIOLATED

    results = GhostHealer(ghost, FakeExecutor()).heal(["5.1"],
                                                      language="python")

    assert results and results[0].ok
    text = open(os.path.join(root, "test_game.py"), encoding="utf-8").read()
    assert "from game import Game, GameState" in text
    assert ghost.expectations["edge:game.py->step:5.1"].verdict == HOLDS


def test_unused_declared_import_is_never_written(tmp_path):
    """A plan declaring an import the file does not need is just wrong.

    Observed in a live run: the plan declared `imports: player.py:Direction`
    for the step targeting `main.py`; main.py routes input through
    `game.handle_event` and never references Direction. The healer wrote
    the import anyway and turned the node green — dead code added purely
    to satisfy a mistaken declaration.
    """
    root = str(tmp_path)
    ghost = _edge_project(root, "import unittest\n\n\nx = 1\n")
    ghost.resolve(["5.1"], language="python")

    results = GhostHealer(ghost, FakeExecutor()).heal(["5.1"],
                                                      language="python")
    text = open(os.path.join(root, "test_game.py"), encoding="utf-8").read()

    assert "from game import" not in text
    assert results and not results[0].ok
    assert "dead code" in results[0].detail
    # The mismatch is still REPORTED — refusing to repair is not
    # the same as pretending the plan and the artifact agree.
    assert ghost.expectations["edge:game.py->step:5.1"].verdict == VIOLATED


def test_already_bound_symbol_is_not_reimported(tmp_path):
    """A locally-defined name must not gain a shadowing import."""
    root = str(tmp_path)
    ghost = _edge_project(
        root, "class Game:\n    pass\n\n\nclass GameState:\n    pass\n\n\n"
              "g = Game()\ns = GameState\n")
    ghost.resolve(["5.1"], language="python")

    GhostHealer(ghost, FakeExecutor()).heal(["5.1"], language="python")
    text = open(os.path.join(root, "test_game.py"), encoding="utf-8").read()
    assert "from game import" not in text


def test_only_the_used_symbols_are_imported(tmp_path):
    root = str(tmp_path)
    ghost = _edge_project(root, "import unittest\n\n\ng = Game()\n")
    ghost.resolve(["5.1"], language="python")
    GhostHealer(ghost, FakeExecutor()).heal(["5.1"], language="python")

    text = open(os.path.join(root, "test_game.py"), encoding="utf-8").read()
    assert "from game import Game" in text
    assert "GameState" not in text


def test_import_is_inserted_after_the_existing_imports(tmp_path):
    root = str(tmp_path)
    ghost = _edge_project(
        root, '"""Docstring."""\nimport os\n\n\ng = Game()\ns = GameState\n')
    ghost.resolve(["5.1"], language="python")
    GhostHealer(ghost, FakeExecutor()).heal(["5.1"], language="python")

    lines = open(os.path.join(root, "test_game.py"),
                 encoding="utf-8").read().splitlines()
    assert lines[0] == '"""Docstring."""'
    assert lines.index("from game import Game, GameState") > lines.index(
        "import os")


def test_import_not_added_when_symbol_is_absent(tmp_path):
    """An import of a name that is not there is worse than none.

    It converts a reported gap into an ImportError at runtime.
    """
    root = str(tmp_path)
    _write(root, "game.py", "class Other:\n    pass\n")
    _write(root, "test_game.py", "import unittest\n")
    step = _step("5.1", target_files=["test_game.py"],
                 imports_from={"game.py": ["Game"]})
    ghost = GhostPlan.build([step], root)
    ghost.resolve(["5.1"], language="python")

    results = GhostHealer(ghost, FakeExecutor()).heal(["5.1"],
                                                      language="python")
    text = open(os.path.join(root, "test_game.py"), encoding="utf-8").read()
    assert results == []
    assert "from game import" not in text


def test_import_not_added_across_directories(tmp_path):
    """The module name is ambiguous without knowing sys.path."""
    root = str(tmp_path)
    _write(root, "src/game.py", "class Game:\n    pass\n")
    _write(root, "tests/test_game.py", "import unittest\n")
    step = _step("5.1", target_files=["tests/test_game.py"],
                 imports_from={"src/game.py": ["Game"]})
    ghost = GhostPlan.build([step], root)
    ghost.resolve(["5.1"], language="python")

    results = GhostHealer(ghost, FakeExecutor()).heal(["5.1"],
                                                      language="python")
    assert results == []


def test_source_edits_can_be_disabled(tmp_path):
    """`ghost_heal_source_edits: false` leaves every project file alone."""
    root = str(tmp_path)
    ghost = _edge_project(root, "import unittest\n")
    ghost.resolve(["5.1"], language="python")

    results = GhostHealer(ghost, FakeExecutor(),
                          allow_source_edits=False).heal(["5.1"],
                                                         language="python")
    text = open(os.path.join(root, "test_game.py"), encoding="utf-8").read()
    assert results == []
    assert "from game import" not in text


# ── step drift: the plan was right, the step deviated ────────────────


CSS_PLAN = (".site-header { color: #111; padding: 1rem; }\n"
            ".site-nav { display: flex; }\n")


def test_missing_css_class_is_restored_from_the_plan(tmp_path):
    """The planner specified the rule; the step dropped it.

    Nothing is invented here — the CSS written back is byte-for-byte the
    body the planner put in `inline_code`. This is the small-model
    failure mode: correct plan, drifting step.
    """
    root = str(tmp_path)
    step = _step("3.1", target_files=["style.css"],
                 inline_code={"style.css": CSS_PLAN})
    ghost = GhostPlan.build([step], root)
    _write(root, "style.css", ".site-nav { display: flex; }\n")  # drifted

    ghost.resolve(["3.1"], language="css")
    exp = ghost.expectations["plan:style.css#anchors"]
    assert exp.verdict == VIOLATED
    assert ".site-header" in exp.evidence

    results = GhostHealer(ghost, FakeExecutor()).heal(["3.1"],
                                                      language="css")

    assert results and results[0].ok
    text = open(os.path.join(root, "style.css"), encoding="utf-8").read()
    assert ".site-header" in text and "color: #111" in text
    assert exp.verdict == HOLDS


def test_dropped_class_is_restored_from_plan_not_stubbed(tmp_path):
    """The restored rule carries the planner's declarations, not `{}`."""
    root = str(tmp_path)
    step = _step("3.1", target_files=["style.css"],
                 inline_code={"style.css": CSS_PLAN})
    ghost = GhostPlan.build([step], root)
    _write(root, "style.css", ".site-nav { display: flex; }\n")
    ghost.resolve(["3.1"], language="css")
    GhostHealer(ghost, FakeExecutor()).heal(["3.1"], language="css")

    text = open(os.path.join(root, "style.css"), encoding="utf-8").read()
    assert ".site-header {}" not in text
    assert "padding: 1rem" in text


def test_dropped_python_class_is_restored_from_the_plan(tmp_path):
    root = str(tmp_path)
    body = "class Map:\n    pass\n\n\nclass Tile:\n    pass\n"
    step = _step("2.1", target_files=["map.py"], inline_code={"map.py": body})
    ghost = GhostPlan.build([step], root)
    _write(root, "map.py", "class Tile:\n    pass\n")     # Map dropped

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["plan:map.py#anchors"].verdict == VIOLATED

    results = GhostHealer(ghost, FakeExecutor()).heal(["2.1"],
                                                      language="python")
    text = open(os.path.join(root, "map.py"), encoding="utf-8").read()
    assert results and results[0].ok
    assert "class Map" in text


def test_missing_file_is_restored_from_the_plan_body(tmp_path):
    """A step that produced nothing at all, repaired from the plan."""
    root = str(tmp_path)
    body = "class Board:\n    pass\n"
    step = _step("2.1", target_files=["board.py"],
                 inline_code={"board.py": body})
    ghost = GhostPlan.build([step], root)
    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["file:board.py#exists"].verdict == VIOLATED

    GhostHealer(ghost, FakeExecutor()).heal(["2.1"], language="python")

    assert open(os.path.join(root, "board.py"),
                encoding="utf-8").read() == body


def test_declared_export_restored_when_the_plan_body_has_it(tmp_path):
    """A pure regression: the step dropped a class and added nothing."""
    root = str(tmp_path)
    body = "class Board:\n    pass\n\n\nclass Helper:\n    pass\n"
    step = _step("2.1", target_files=["m.py"], exports=["Board"],
                 inline_code={"m.py": body})
    ghost = GhostPlan.build([step], root)
    _write(root, "m.py", "class Helper:\n    pass\n")      # Board dropped

    ghost.resolve(["2.1"], language="python")
    assert ghost.expectations["file:m.py#exports:Board"].verdict == VIOLATED

    GhostHealer(ghost, FakeExecutor()).heal(["2.1"], language="python")
    assert ghost.expectations["file:m.py#exports:Board"].verdict == HOLDS
    assert "class Board" in open(os.path.join(root, "m.py"),
                                 encoding="utf-8").read()


def test_export_not_restored_when_the_step_added_its_own_work(tmp_path):
    """The conflict guard outranks the export repair."""
    root = str(tmp_path)
    step = _step("2.1", target_files=["m.py"], exports=["Board"],
                 inline_code={"m.py": "class Board:\n    pass\n"})
    ghost = GhostPlan.build([step], root)
    _write(root, "m.py", "class Other:\n    pass\n")

    ghost.resolve(["2.1"], language="python")
    results = GhostHealer(ghost, FakeExecutor()).heal(["2.1"],
                                                      language="python")

    assert results and not any(r.ok for r in results)
    assert "class Other" in open(os.path.join(root, "m.py"),
                                 encoding="utf-8").read()


def test_extra_work_beyond_the_plan_is_never_discarded(tmp_path):
    """A file that diverged in BOTH directions is reported, not clobbered.

    The step dropped `.site-header` but also added `.site-footer` the
    plan never mentioned. Restoring the plan verbatim would delete real
    work, so the conflict is surfaced instead.
    """
    root = str(tmp_path)
    step = _step("3.1", target_files=["style.css"],
                 inline_code={"style.css": CSS_PLAN})
    ghost = GhostPlan.build([step], root)
    drifted = ".site-nav { display: flex; }\n.site-footer { color: #eee; }\n"
    _write(root, "style.css", drifted)

    ghost.resolve(["3.1"], language="css")
    results = GhostHealer(ghost, FakeExecutor()).heal(["3.1"],
                                                      language="css")

    assert results and not results[0].ok
    assert ".site-footer" in results[0].detail
    assert open(os.path.join(root, "style.css"),
                encoding="utf-8").read() == drifted
    assert ghost.expectations["plan:style.css#anchors"].verdict == VIOLATED


def test_intent_mode_plans_have_nothing_to_restore(tmp_path):
    """No inline_code means no plan body — detection only."""
    root = str(tmp_path)
    step = _step("3.1", target_files=["style.css"])       # intent mode
    ghost = GhostPlan.build([step], root)
    _write(root, "style.css", ".site-nav { display: flex; }\n")

    ghost.resolve(["3.1"], language="css")
    assert "plan:style.css#anchors" not in ghost.expectations
    assert GhostHealer(ghost, FakeExecutor()).heal(
        ["3.1"], language="css") == []


# ── the content line ─────────────────────────────────────────────────


def test_no_healer_invents_unspecified_content(tmp_path):
    """PARSES / TOUCHED / GATE_PASSED are never auto-repaired.

    No source specifies what they should contain — not the plan, not the
    filesystem — so repairing them would mean writing the very code
    whose absence is being reported.
    """
    from agentchanti.orchestrator.ghost_heal import _HEALERS
    from agentchanti.orchestrator.ghost import (
        KIND_GATE_PASSED, KIND_PARSES, KIND_TOUCHED,
    )
    for kind in (KIND_PARSES, KIND_TOUCHED, KIND_GATE_PASSED):
        assert kind not in _HEALERS


def test_missing_export_with_no_plan_body_is_only_reported(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["m.py"], exports=["Board"])], root)
    _write(root, "m.py", "class Other:\n    pass\n")
    ghost.resolve(["2.1"], language="python")

    before = open(os.path.join(root, "m.py"), encoding="utf-8").read()
    results = GhostHealer(ghost, FakeExecutor()).heal(["2.1"],
                                                      language="python")
    after = open(os.path.join(root, "m.py"), encoding="utf-8").read()

    assert results == []
    assert before == after
    assert ghost.expectations["file:m.py#exports:Board"].verdict == VIOLATED


def test_broken_syntax_is_reported_not_rewritten(tmp_path):
    root = str(tmp_path)
    ghost = GhostPlan.build([_step("2.1", target_files=["bad.py"])], root)
    _write(root, "bad.py", "def broken(:\n")
    ghost.resolve(["2.1"], language="python")

    before = open(os.path.join(root, "bad.py"), encoding="utf-8").read()
    GhostHealer(ghost, FakeExecutor()).heal(["2.1"], language="python")
    after = open(os.path.join(root, "bad.py"), encoding="utf-8").read()

    assert before == after
    assert ghost.expectations["file:bad.py#parses"].verdict == VIOLATED


def test_healer_never_raises_on_junk(tmp_path):
    class Boom:
        def run_command(self, *a, **kw):
            raise RuntimeError("boom")

    root = str(tmp_path)
    _make_venv(root)
    ghost = GhostPlan.build(
        [_step("2.1", target_files=["requirements.txt"])], root)
    _write(root, "requirements.txt", "pygame\n")
    ghost.resolve(["2.1"], language="python")

    results = GhostHealer(ghost, Boom()).heal(["2.1"], language="python")
    assert all(not r.ok for r in results)

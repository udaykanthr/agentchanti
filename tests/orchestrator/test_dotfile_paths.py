"""A leading dot is part of a file's name, not a prefix to strip.

``str.lstrip("./")`` strips every leading ``.`` and ``/`` CHARACTER, so it
turns ``.gitignore`` into ``gitignore``. Measured 2026-09-13: a plan declared
``target: .gitignore``, the parser recorded ``gitignore``, and the ghost
reported two disagreements for one correct step::

    violated-exists (step 2.3): gitignore — planned target does not exist on disk
    unplanned-write (step -): .gitignore was written but no step declared it

The idiom sat in about fifteen places. The one that mattered most was not
the noisy one: the acceptance-instrument guard stored ``.ci/accept.py`` as
``ci/accept.py`` and compared it against the dotted relpath of the write,
so an instrument under a dot-directory was never protected.
"""
import os

import pytest

from agentchanti.orchestrator.agent_loop import build_step_tools
from agentchanti.orchestrator.ghost import VIOLATED, GhostPlan
from agentchanti.orchestrator.memory import FileMemory
from agentchanti.orchestrator.plan_graph import normalize_path
from agentchanti.orchestrator.plan_step import parse_structured_plan
from agentchanti.paths import norm_rel_path, strip_dot_slash

MEASURED_STEP = """==PLAN==

--STEP 2.3 [CODE] depends:none
Create Git ignore rules for virtual environments and caches.
target: .gitignore
exports: none
imports: none
"""


class TestTheHelper:

    @pytest.mark.parametrize("given, want", [
        (".gitignore", ".gitignore"),
        ("./.env", ".env"),
        ("/.env", ".env"),
        ("././src/app.py", "src/app.py"),
        (".github/workflows/ci.yml", ".github/workflows/ci.yml"),
        ("./src/app.py", "src/app.py"),
        ("src/./app.py", "src/./app.py"),      # a prefix strip, nothing more
        (".", ""),
        ("", ""),
    ])
    def test_prefixes_go_names_stay(self, given, want):
        assert strip_dot_slash(given) == want

    def test_traversal_is_not_rewritten_into_something_else(self):
        """`../x` becoming `x` was a silent rewrite, not a refusal; the
        write paths refuse escapes themselves."""
        assert strip_dot_slash("../x.py") == "../x.py"

    def test_separators_collapse(self):
        assert norm_rel_path(".\\.github\\\\workflows//ci.yml") == \
            ".github/workflows/ci.yml"


class TestThePlanParser:

    def test_the_measured_target_keeps_its_dot(self):
        steps = parse_structured_plan(MEASURED_STEP)
        assert steps[0].target_files == [".gitignore"]

    def test_the_old_idiom_was_the_cause(self):
        """Pins the incident so the rationale cannot rot."""
        assert ".gitignore".lstrip("./") == "gitignore"

    @pytest.mark.parametrize("target", [
        ".env.example", ".github/workflows/ci.yml", ".eslintrc.json"])
    def test_other_dotfiles(self, target):
        plan = MEASURED_STEP.replace("target: .gitignore", f"target: {target}")
        assert parse_structured_plan(plan)[0].target_files == [target]

    def test_the_plan_graph_agrees_with_the_parser(self):
        assert normalize_path("./.env") == ".env"


class TestTheGhostFindingsAreGone:

    def test_the_measured_step_reports_nothing(self, tmp_path):
        root = str(tmp_path)
        steps = parse_structured_plan(MEASURED_STEP)
        ghost = GhostPlan.build(steps, root)
        (tmp_path / ".gitignore").write_text("venv/\n__pycache__/\n")

        ghost.resolve(["2.3"], language="python")

        exists = [e for e in ghost.expectations.values()
                  if e.kind == "EXISTS"]
        assert exists and all(e.verdict != VIOLATED for e in exists), \
            "the planned .gitignore was reported missing"
        assert ghost.unplanned_writes([".gitignore"]) == [], \
            "the planned .gitignore was reported as an unplanned write"


class TestTheAcceptanceGuard:
    """Through `build_step_tools`, because the seam is the wiring."""

    BODY = "# frozen\n"

    def _project(self, tmp_path):
        (tmp_path / ".ci").mkdir()
        (tmp_path / ".ci" / "accept.py").write_text(self.BODY)
        mem = FileMemory()
        mem._acceptance_files = {".ci/accept.py"}
        return build_step_tools(None, mem, project_root=str(tmp_path))

    def test_a_dot_directory_instrument_is_protected(self, tmp_path):
        tools = self._project(tmp_path)
        out = tools._tool_write_file(path=".ci/accept.py",
                                     content="# REWRITTEN\n")
        assert out.startswith("ERROR"), out
        assert (tmp_path / ".ci" / "accept.py").read_text() == self.BODY

    def test_edit_is_refused_too(self, tmp_path):
        tools = self._project(tmp_path)
        out = tools._tool_edit_file(path=".ci/accept.py",
                                    old_text="frozen", new_text="thawed")
        assert out.startswith("ERROR"), out
        assert (tmp_path / ".ci" / "accept.py").read_text() == self.BODY

    def test_an_ordinary_file_beside_it_is_still_writable(self, tmp_path):
        """Or the fix is just a wider refusal."""
        tools = self._project(tmp_path)
        out = tools._tool_write_file(path=".ci/notes.txt", content="ok\n")
        assert not out.startswith("ERROR"), out

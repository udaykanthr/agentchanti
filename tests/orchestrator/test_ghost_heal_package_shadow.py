"""Restoring a module whose name a package has taken breaks the project.

Measured 2026-09-16, benchmark task `django-webapp` on 0.8.3:

    10:36:53  step 6.1 wrote  core/tests.py   (planner's body, 2558 chars)
    10:37:03  the agent DELETED core/tests.py — it had built core/tests/
              as a package (__init__.py, test_views.py, test_forms.py)
    10:37:06  [GhostHeal] EXISTS (core/tests.py) — healed: restore from
              the plan's own body (2558 chars)
    10:37:06  [Ghost] 4 postcondition(s) ... repaired in flight

`manage.py test` then died with `ImportError: 'tests' module incorrectly
imported from ...core\tests. Expected ...core`, and the benchmark graded
the task FAIL. With that one restored file removed the application is
complete: 10 tests OK, `/` -> 200, `/dashboard/` -> 302, `check` clean.

Promoting a module to a package is ordinary, and deleting the old file is
part of doing it — not work being lost. The healer's content was faithful
to the plan; what nothing asked was whether the NAME was still free.
"""
import os

from agentchanti.orchestrator.ghost import HOLDS, VIOLATED, GhostPlan
from agentchanti.orchestrator.ghost_heal import (
    GhostHealer,
    package_shadow_reason,
)
from agentchanti.orchestrator.plan_step import PlanStep

PLANNED_TESTS_PY = (
    "from django.test import TestCase\n\n\n"
    "class CorePageTests(TestCase):\n"
    "    def test_home(self):\n"
    "        self.assertEqual(self.client.get('/').status_code, 200)\n"
)


def _step(sid, **kw):
    kw.setdefault("step_type", "CODE")
    return PlanStep(id=sid, **kw)


def _write(root, rel, text=""):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


class _Exec:
    def run_command(self, cmd, cwd=None, timeout=None, **kw):
        return True, ""


def _django_tree(root):
    """The measured layout: the package exists, the module does not."""
    _write(root, "core/tests/__init__.py")
    _write(root, "core/tests/test_views.py",
           "from django.test import TestCase\n")
    _write(root, "core/tests/test_forms.py",
           "from django.test import TestCase\n")


class TestTheMeasuredIncident:

    def test_the_file_is_not_restored(self, tmp_path):
        root = str(tmp_path)
        _django_tree(root)
        step = _step("6.1", target_files=["core/tests.py"],
                     inline_code={"core/tests.py": PLANNED_TESTS_PY})
        ghost = GhostPlan.build([step], root)
        ghost.resolve(["6.1"], language="python")
        assert ghost.expectations["file:core/tests.py#exists"].verdict \
            == VIOLATED

        results = GhostHealer(ghost, _Exec()).heal(["6.1"], language="python")

        assert not os.path.exists(os.path.join(root, "core", "tests.py")), \
            "restoring it re-creates the ImportError the incident hit"
        assert results and not results[0].ok
        assert "NOT restoring" in results[0].action

    def test_the_refusal_says_what_is_wrong_and_what_to_fix(self, tmp_path):
        root = str(tmp_path)
        _django_tree(root)
        step = _step("6.1", target_files=["core/tests.py"],
                     inline_code={"core/tests.py": PLANNED_TESTS_PY})
        ghost = GhostPlan.build([step], root)
        ghost.resolve(["6.1"], language="python")
        results = GhostHealer(ghost, _Exec()).heal(["6.1"], language="python")

        detail = results[0].describe()
        assert "core/tests/" in detail
        assert "fix the plan, not the tree" in detail

    def test_it_is_not_reported_as_repaired_in_flight(self, tmp_path):
        """The incident's log line said `repaired in flight`. It was not."""
        root = str(tmp_path)
        _django_tree(root)
        step = _step("6.1", target_files=["core/tests.py"],
                     inline_code={"core/tests.py": PLANNED_TESTS_PY})
        ghost = GhostPlan.build([step], root)
        ghost.resolve(["6.1"], language="python")
        GhostHealer(ghost, _Exec()).heal(["6.1"], language="python")

        assert ghost.expectations["file:core/tests.py#exists"].verdict \
            != HOLDS


class TestTheReason:

    def test_the_measured_pair(self, tmp_path):
        root = str(tmp_path)
        _django_tree(root)
        assert package_shadow_reason(root, "core/tests.py")

    def test_windows_separators_are_read_the_same(self, tmp_path):
        root = str(tmp_path)
        _django_tree(root)
        win = "core" + chr(92) + "tests.py"   # a literal backslash, no escaping games
        assert package_shadow_reason(root, win)

    def test_no_package_no_objection(self, tmp_path):
        assert package_shadow_reason(str(tmp_path), "core/tests.py") is None

    def test_a_directory_holding_no_python_is_not_a_package(self, tmp_path):
        """`data/` beside `data.py` claims no module name."""
        root = str(tmp_path)
        _write(root, "data/fixtures.json", "{}")
        assert package_shadow_reason(root, "data.py") is None

    def test_init_py_is_never_shadowed(self, tmp_path):
        """It lives INSIDE the package and names no module of its own."""
        root = str(tmp_path)
        _django_tree(root)
        assert package_shadow_reason(
            root, "core/tests/__init__.py") is None

    def test_a_non_python_target_is_out_of_scope(self, tmp_path):
        root = str(tmp_path)
        _write(root, "docs/index.md", "# hi")
        assert package_shadow_reason(root, "docs.md") is None


class TestOrdinaryRestoresAreUnchanged:

    def test_a_missing_module_with_no_package_is_still_restored(self, tmp_path):
        root = str(tmp_path)
        body = "class Board:\n    pass\n"
        step = _step("2.1", target_files=["board.py"],
                     inline_code={"board.py": body})
        ghost = GhostPlan.build([step], root)
        ghost.resolve(["2.1"], language="python")
        assert ghost.expectations["file:board.py#exists"].verdict == VIOLATED

        results = GhostHealer(ghost, _Exec()).heal(["2.1"], language="python")

        assert results and results[0].ok
        text = open(os.path.join(root, "board.py"), encoding="utf-8").read()
        assert "class Board" in text

    def test_a_missing_package_marker_is_still_created(self, tmp_path):
        root = str(tmp_path)
        _write(root, "core/tests/test_views.py", "x = 1\n")
        step = _step("6.1", target_files=["core/tests/__init__.py"])
        ghost = GhostPlan.build([step], root)
        ghost.resolve(["6.1"], language="python")

        results = GhostHealer(ghost, _Exec()).heal(["6.1"], language="python")

        assert results and results[0].ok
        assert os.path.exists(
            os.path.join(root, "core", "tests", "__init__.py"))


class TestEveryWriteSeam:
    """Three seams can create the collision; all three must refuse it.

    Guarding the healer alone was not enough, and the retest proved it
    within the hour: with GhostHeal refusing, the SAME ImportError came
    back through the plan's own step --

        11:23:03 [PlanStep] Inline test code: wrote 1 file(s) for step
                 8.1: ['spacious_site/core/tests.py']

    written as that step's declared `target:` while another step had
    already built `core/tests/`. A guard is only as strong as the weakest
    thing that can establish its precondition.
    """

    def _write_files(self, root, files):
        from agentchanti.executor import Executor
        return Executor.write_files(files, base_dir=root)

    def _tools(self, root):
        from agentchanti.agent_tools import AgentTools
        return AgentTools(project_root=root, executor=_Exec())

    def test_the_executor_refuses_the_inline_write(self, tmp_path):
        """Seam 2: the plan's own inline CODE/TEST write."""
        root = str(tmp_path)
        _django_tree(root)

        written = self._write_files(root, {"core/tests.py": PLANNED_TESTS_PY})

        assert not os.path.exists(os.path.join(root, "core", "tests.py")), \
            "this is the write the retest caught"
        assert not any("tests.py" in w for w in written)

    def test_the_executor_still_writes_an_ordinary_module(self, tmp_path):
        root = str(tmp_path)
        _django_tree(root)

        written = self._write_files(root, {"core/views.py": "x = 1\n"})

        assert any("views.py" in w for w in written)
        assert os.path.exists(os.path.join(root, "core", "views.py"))

    def test_the_executor_still_writes_into_the_package(self, tmp_path):
        """The correct destination must stay available."""
        root = str(tmp_path)
        _django_tree(root)

        written = self._write_files(
            root, {"core/tests/test_extra.py": "x = 1\n"})

        assert any("test_extra.py" in w for w in written)

    def test_agent_tools_refuses_and_says_where_it_belongs(self, tmp_path):
        """Seam 3: a write the agent loop makes directly."""
        root = str(tmp_path)
        _django_tree(root)

        out = self._tools(root)._tool_write_file(
            "core/tests.py", "import unittest\n")

        assert out.startswith("ERROR")
        assert "core/tests/" in out
        assert "test_<name>.py" in out, "it must name the right destination"
        assert "defect in the PLAN" in out
        assert not os.path.exists(os.path.join(root, "core", "tests.py"))

    def test_agent_tools_still_writes_an_ordinary_file(self, tmp_path):
        root = str(tmp_path)
        _django_tree(root)

        out = self._tools(root)._tool_write_file("core/views.py", "x = 1\n")

        assert out.startswith("OK")

    def test_all_three_seams_ask_one_object(self):
        """One answer per project, as `EMPTY_SUITE_RE` already insists."""
        from agentchanti import agent_tools, paths
        from agentchanti.orchestrator import ghost_heal

        assert ghost_heal.package_shadow_reason is paths.package_shadow_reason
        assert (agent_tools.package_shadow_reason
                is paths.package_shadow_reason)


# The Django stub, verbatim from `manage.py startapp`.
STARTAPP_STUB = "from django.test import TestCase\n\n# Create your tests here.\n"


class TestTheScaffoldLeftBehind:
    """The third direction, and the one a refusal cannot fix.

    Measured 2026-09-16, third consecutive `django-webapp` run:

        13:27:55  manage.py startapp core  -> core/tests.py  (63 bytes)
        13:28:56  Written: core/tests/__init__.py, core/tests/test_forms.py
        13:30:12  Written: core/tests/test_views.py

    Every write went exactly where it belongs, so refusing would block the
    correct behaviour. `manage.py test` died on the leftover stub alone;
    removing those 63 bytes gave `Found 10 test(s) ... OK`.
    """

    def _write_files(self, root, files):
        from agentchanti.executor import Executor
        return Executor.write_files(files, base_dir=root)

    def test_the_measured_stub_is_removed(self, tmp_path):
        from agentchanti.paths import superseded_scaffold_module

        root = str(tmp_path)
        _write(root, "core/tests.py", STARTAPP_STUB)
        _write(root, "core/tests/__init__.py")

        assert superseded_scaffold_module(
            root, "core/tests/test_views.py") == "core/tests.py"

    def test_the_executor_removes_it_on_the_write(self, tmp_path):
        root = str(tmp_path)
        _write(root, "core/tests.py", STARTAPP_STUB)

        self._write_files(root, {
            "core/tests/__init__.py": "",
            "core/tests/test_views.py": "from django.test import TestCase\n",
        })

        assert not os.path.exists(os.path.join(root, "core", "tests.py"))
        assert os.path.exists(
            os.path.join(root, "core", "tests", "test_views.py"))

    def test_agent_tools_removes_it_and_says_so(self, tmp_path):
        from agentchanti.agent_tools import AgentTools

        root = str(tmp_path)
        _write(root, "core/tests.py", STARTAPP_STUB)
        _write(root, "core/tests/__init__.py")
        tools = AgentTools(project_root=root, executor=_Exec())

        out = tools._tool_write_file(
            "core/tests/test_views.py", "from django.test import TestCase\n")

        assert out.startswith("OK")
        assert "core/tests.py" in out, "the model must learn the tree changed"
        assert not os.path.exists(os.path.join(root, "core", "tests.py"))

    def test_a_module_that_declares_anything_is_kept(self, tmp_path):
        """Real content is a conflict to REPORT, never to delete."""
        from agentchanti.paths import superseded_scaffold_module

        root = str(tmp_path)
        _write(root, "core/tests.py",
               "from django.test import TestCase\n\n\n"
               "class OldTests(TestCase):\n    def test_x(self):\n"
               "        self.assertTrue(True)\n")
        _write(root, "core/tests/__init__.py")

        assert superseded_scaffold_module(
            root, "core/tests/test_views.py") is None

    def test_a_single_assignment_counts_as_content(self, tmp_path):
        from agentchanti.paths import superseded_scaffold_module

        root = str(tmp_path)
        _write(root, "core/tests.py", "SKIP_SLOW = True\n")
        _write(root, "core/tests/__init__.py")

        assert superseded_scaffold_module(
            root, "core/tests/test_views.py") is None

    def test_an_unparseable_module_is_kept(self, tmp_path):
        """No opinion means keep the file."""
        from agentchanti.paths import superseded_scaffold_module

        root = str(tmp_path)
        _write(root, "core/tests.py", "def broken( :\n")
        _write(root, "core/tests/__init__.py")

        assert superseded_scaffold_module(
            root, "core/tests/test_views.py") is None

    def test_a_docstring_only_module_is_still_empty(self, tmp_path):
        from agentchanti.paths import superseded_scaffold_module

        root = str(tmp_path)
        _write(root, "core/tests.py", '"""Tests live in tests/ now."""\n')
        _write(root, "core/tests/__init__.py")

        assert superseded_scaffold_module(
            root, "core/tests/test_views.py") == "core/tests.py"

    def test_nothing_is_removed_when_no_sibling_module_exists(self, tmp_path):
        from agentchanti.paths import superseded_scaffold_module

        root = str(tmp_path)
        _write(root, "core/tests/__init__.py")

        assert superseded_scaffold_module(
            root, "core/tests/test_views.py") is None

    def test_an_ordinary_write_removes_nothing(self, tmp_path):
        root = str(tmp_path)
        _write(root, "core/views.py", "x = 1\n")
        _write(root, "core/models.py", "y = 2\n")

        self._write_files(root, {"core/forms.py": "z = 3\n"})

        assert os.path.exists(os.path.join(root, "core", "views.py"))
        assert os.path.exists(os.path.join(root, "core", "models.py"))

    def test_the_end_to_end_layout_is_importable(self, tmp_path):
        """The whole point: exactly one claimant to `core.tests`."""
        root = str(tmp_path)
        _write(root, "core/__init__.py")
        _write(root, "core/tests.py", STARTAPP_STUB)

        self._write_files(root, {
            "core/tests/__init__.py": "",
            "core/tests/test_views.py": "def test_x():\n    assert True\n",
        })

        core = os.path.join(root, "core")
        assert os.path.isdir(os.path.join(core, "tests"))
        assert not os.path.exists(os.path.join(core, "tests.py"))

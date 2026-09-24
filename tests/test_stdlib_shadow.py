"""A gate no code can satisfy is an incentive to replace the standard library.

Measured 2026-09-24, gpt-oss:20b-cloud on "implement a 2d snake game". The
plan gated a step on `tomllib.load(open('pyproject.toml'))` - text mode,
which raises TypeError against ANY project. The agent wrote a 1,955-byte
`tomllib.py` into the project root ("Compatibility shim for the standard
library tomllib", justified by an invented claim that the environment's
tomllib was "a very old stub"). That replaced the real module for pytest
too: the seeded contract ended with 8 errors, and the run failed after 69
turns and 374k tokens over a pyproject.toml that was correct throughout.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from agentchanti.agent_tools import AgentTools, stdlib_shadow
from agentchanti.orchestrator.plan_step import unrunnable_gate_reason

MEASURED_GATE = ('python -c "import tomllib, pathlib; '
                 "data=tomllib.load(open('pyproject.toml')); "
                 "assert data['project']['name']=='snake';\"")


class TestTheGate:

    def test_the_measured_gate_is_refused(self):
        why = unrunnable_gate_reason(MEASURED_GATE)
        assert why and "binary mode" in why

    def test_binary_mode_is_fine(self):
        assert unrunnable_gate_reason(
            'python -c "import tomllib; '
            "d=tomllib.load(open('pyproject.toml','rb')); assert d\"") is None

    def test_loads_on_text_is_fine(self):
        assert unrunnable_gate_reason(
            'python -c "import tomllib; '
            "d=tomllib.loads(open('pyproject.toml').read()); assert d\"") is None

    def test_tomli_too(self):
        assert unrunnable_gate_reason(
            'python -c "import tomli; d=tomli.load(open(\'x.toml\'))"')


class TestTheShadow:

    @pytest.mark.parametrize("path,expected", [
        ("tomllib.py", "tomllib"),
        ("json.py", "json"),
        ("logging/__init__.py", "logging"),   # a package shadows too
        ("Tomllib.py", "tomllib"),            # Windows is case-insensitive
    ])
    def test_top_level_stdlib_names_are_named(self, path, expected):
        assert stdlib_shadow(path) == expected

    @pytest.mark.parametrize("path", [
        "src/tomllib.py",          # not top level - cannot shadow
        "app/utils/json.py",
        "__init__.py",             # names no module of its own
        "snake/board.py",
        "config.py",
        "game.py",
    ])
    def test_ordinary_files_are_untouched(self, path):
        assert stdlib_shadow(path) is None

    def test_a_domain_word_is_not_refused(self):
        """`types.py` and `queue.py` are names a project may legitimately
        own, so the list is narrow by design."""
        assert stdlib_shadow("types.py") is None
        assert stdlib_shadow("queue.py") is None


class TestTheRefusal(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="stdlibshadow_")
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        self.tools = AgentTools(project_root=self.root, executor=executor)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_writing_the_measured_shim_is_refused(self):
        result = self.tools._tool_write_file(
            "tomllib.py", '"""Compatibility shim."""\ndef load(f): ...\n')
        self.assertIn("ERROR", result)
        self.assertIn("tomllib", result)
        self.assertFalse(os.path.exists(os.path.join(self.root, "tomllib.py")))

    def test_the_refusal_names_the_real_defect(self):
        result = self.tools._tool_write_file("json.py", "x = 1\n")
        self.assertIn("defect in the CHECK", result)

    def test_editing_one_is_refused_too(self):
        with open(os.path.join(self.root, "tomllib.py"), "w") as fh:
            fh.write("def load(f):\n    return {}\n")
        result = self.tools._tool_edit_file(
            "tomllib.py", "return {}", "return {'project': {}}")
        self.assertIn("ERROR", result)

    def test_an_ordinary_module_still_writes(self):
        result = self.tools._tool_write_file("board.py", "SIZE = 20\n")
        self.assertNotIn("ERROR", result)
        self.assertTrue(os.path.exists(os.path.join(self.root, "board.py")))

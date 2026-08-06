"""Guards for the three defects seen in the Pac-Man benchmark runs.

Each test names the observed failure it prevents. The "healthy" cases
matter as much as the failing ones: every guard here must be inert on
output from a model that is working correctly.
"""

import os
import tempfile
import unittest

from agentchanti.agent_tools import (
    AgentTools,
    parse_failed_install_targets,
    shadowed_dist,
)
from agentchanti.executor import Executor
from agentchanti.orchestrator.agent_loop import _ENV_CMD_RE


class TestParseFailedInstallTargets(unittest.TestCase):
    def test_plain_install(self):
        self.assertEqual(parse_failed_install_targets("pip install pygame"),
                         {"pygame"})

    def test_pinned_version_is_stripped(self):
        # The exact command from the failing run.
        self.assertEqual(
            parse_failed_install_targets(
                'call venv\\Scripts\\activate && python -m pip install '
                '--upgrade pip && python -m pip install "pygame==2.6.0"'),
            {"pip", "pygame"})

    def test_extras_and_ranges_stripped(self):
        self.assertEqual(
            parse_failed_install_targets("pip install 'pygame[all]>=2,<3'"),
            {"pygame"})

    def test_normalises_separators(self):
        self.assertEqual(
            parse_failed_install_targets("pip install Foo_Bar"), {"foo-bar"})

    def test_flags_and_paths_ignored(self):
        self.assertEqual(
            parse_failed_install_targets(
                "pip install -r requirements.txt --user"), set())

    def test_non_install_command_yields_nothing(self):
        self.assertEqual(parse_failed_install_targets("python -m pytest"),
                         set())
        self.assertEqual(parse_failed_install_targets(""), set())


class TestShadowedDist(unittest.TestCase):
    def test_top_level_package_shadows(self):
        self.assertEqual(shadowed_dist("pygame/__init__.py", {"pygame"}),
                         "pygame")
        self.assertEqual(shadowed_dist("pygame\\draw.py", {"pygame"}),
                         "pygame")

    def test_top_level_module_shadows(self):
        self.assertEqual(shadowed_dist("pygame.py", {"pygame"}), "pygame")

    def test_nested_package_does_not_shadow(self):
        # Legitimate: a project module that happens to share the name but
        # is not importable as the top-level distribution.
        self.assertIsNone(shadowed_dist("src/pygame/draw.py", {"pygame"}))

    def test_unrelated_file_does_not_shadow(self):
        self.assertIsNone(shadowed_dist("game/player.py", {"pygame"}))

    def test_no_failed_installs_never_shadows(self):
        # The healthy path: nothing failed, so the guard cannot fire.
        self.assertIsNone(shadowed_dist("pygame/__init__.py", set()))


class TestWriteFileShadowGuard(unittest.TestCase):
    """End-to-end: the stub that a failing `pip install pygame` produced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_write_allowed_when_no_install_failed(self):
        tools = AgentTools(project_root=self.root)
        out = tools._tool_write_file("pygame/__init__.py", "x = 1\n")
        self.assertTrue(out.startswith("OK:"))
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, "pygame", "__init__.py")))

    def test_write_refused_after_failed_install(self):
        tools = AgentTools(project_root=self.root)
        tools._failed_installs = {"pygame"}
        out = tools._tool_write_file(
            "pygame/__init__.py",
            '"""Stub implementation of pygame for testing environment."""\n')
        self.assertTrue(out.startswith("ERROR:"))
        self.assertIn("shadow", out)
        self.assertFalse(os.path.exists(
            os.path.join(self.root, "pygame", "__init__.py")))

    def test_unrelated_writes_still_allowed_after_failed_install(self):
        tools = AgentTools(project_root=self.root)
        tools._failed_installs = {"pygame"}
        out = tools._tool_write_file("game.py", "import pygame\n")
        self.assertTrue(out.startswith("OK:"))


class TestIsStandaloneModule(unittest.TestCase):
    def test_complete_module_accepted(self):
        # Pattern 5's reason for existing: complete code, no filename.
        src = "import os\n\n\nclass Ghost:\n    def update(self, dt):\n        pass\n"
        self.assertTrue(Executor._is_standalone_module(src, "python"))

    def test_indented_method_fragment_rejected(self):
        # The exact shape that produced "unexpected indent (line 1)" on
        # three files at once.
        src = "    def update(self, dt):\n        self.x += dt\n"
        self.assertFalse(Executor._is_standalone_module(src, "python"))

    def test_syntactically_broken_python_rejected(self):
        src = "def f(:\n    pass\n"
        self.assertFalse(Executor._is_standalone_module(src, "python"))

    def test_empty_rejected(self):
        self.assertFalse(Executor._is_standalone_module("   \n", "python"))

    def test_non_python_indent_check_only(self):
        # No AST for other languages; a col-0 start is enough.
        self.assertTrue(Executor._is_standalone_module(
            "function f() {\n  return 1;\n}\n", "javascript"))
        self.assertFalse(Executor._is_standalone_module(
            "  return 1;\n}\n", "javascript"))

    def test_leading_blank_lines_do_not_confuse_it(self):
        self.assertTrue(Executor._is_standalone_module(
            "\n\nimport os\n", "python"))


class TestEnvCmdDetection(unittest.TestCase):
    def test_install_commands_take_environment_branch(self):
        for cmd in (
            'python -m pip install "pygame==2.6.0"',
            "pip install pygame",
            "npm install",
            "venv\\Scripts\\pip install pygame",
            "python -m venv venv",
        ):
            with self.subTest(cmd=cmd):
                self.assertTrue(_ENV_CMD_RE.search(cmd), cmd)

    def test_code_commands_keep_original_wording(self):
        # These must NOT take the new branch — the original nudge is
        # correct for them, and its wording stays byte-identical.
        for cmd in (
            "python -m pytest",
            "python -m unittest -v",
            "python main.py",
            "go test ./...",
            "npm test",
        ):
            with self.subTest(cmd=cmd):
                self.assertIsNone(_ENV_CMD_RE.search(cmd), cmd)


if __name__ == "__main__":
    unittest.main()

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
from agentchanti.orchestrator.agent_loop import (
    _ENV_CMD_RE,
    _ENV_ERROR_RE,
    get_attempts,
    normalize_command,
    record_attempt,
    reset_attempt_journal,
)


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


class TestEnvErrorDetection(unittest.TestCase):
    """The error text, not the command shape, is the general signal."""

    def test_command_not_found_takes_environment_branch(self):
        # The exact stderr from the cmd-recovery benchmark, where the
        # command itself (`ruff check messy.py`) is NOT an install.
        out = ("exit: FAILED\n'ruff' is not recognized as an internal or "
               "external command,\noperable program or batch file.")
        self.assertIsNone(_ENV_CMD_RE.search("ruff check messy.py"))
        self.assertTrue(_ENV_ERROR_RE.search(out))

    def test_posix_and_import_variants(self):
        for out in (
            "bash: ruff: command not found",
            "ModuleNotFoundError: No module named 'ruff'",
            "The system cannot find the path specified",
            "exec: \"node\": executable file not found in $PATH",
        ):
            with self.subTest(out=out):
                self.assertTrue(_ENV_ERROR_RE.search(out), out)

    def test_ordinary_failures_keep_original_wording(self):
        # A real test failure or traceback must NOT be rerouted — the
        # original "edit the source" nudge is correct for these.
        for out in (
            "exit: FAILED\nE   assert 1 == 2",
            "AssertionError: Walkable tile (15, 7) is unreachable",
            "SyntaxError: invalid syntax",
            "TypeError: unsupported operand type(s)",
            "FAILED test_calc.py::test_add - assert 5 == 6",
        ):
            with self.subTest(out=out):
                self.assertIsNone(_ENV_ERROR_RE.search(out), out)


class TestRepeatStreakPersistsAcrossAttempts(unittest.TestCase):
    """A command re-run once per attempt used to never trip the nudge."""

    def setUp(self):
        reset_attempt_journal()
        self.addCleanup(reset_attempt_journal)

    def _seed(self, step_idx):
        """Mirror the seeding expression used in run_agent_loop()."""
        return {
            n for a in get_attempts(step_idx)
            for c, ok in a.get("commands", [])
            if not ok and (n := normalize_command(c))
        }

    def test_failed_command_from_prior_attempt_is_seeded(self):
        record_attempt(
            3, "loop", "verify-failed", [],
            [("ruff check messy.py", False), ("pip show ruff", True)],
            "could not find ruff")
        seeded = self._seed(3)
        self.assertIn(normalize_command("ruff check messy.py"), seeded)

    def test_successful_commands_are_not_seeded(self):
        record_attempt(3, "loop", "verify-failed", [],
                       [("pip show ruff", True)], "")
        self.assertEqual(self._seed(3), set())

    def test_other_steps_are_unaffected(self):
        record_attempt(3, "loop", "verify-failed", [],
                       [("ruff check messy.py", False)], "")
        self.assertEqual(self._seed(4), set())

    def test_no_prior_attempts_seeds_empty(self):
        # The healthy path: first attempt at a step starts clean, exactly
        # as before this change.
        self.assertEqual(self._seed(9), set())


class TestNudgeRoutingInTheRealLoop(unittest.TestCase):
    """Drives run_agent_loop() itself, not a copy of its logic."""

    RUFF = "ruff check messy.py"
    NOT_FOUND = ("'ruff' is not recognized as an internal or external "
                 "command,\noperable program or batch file.")

    def setUp(self):
        from unittest.mock import MagicMock
        self._mm = MagicMock
        reset_attempt_journal()
        self.addCleanup(reset_attempt_journal)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.executor = MagicMock()

    def _tools(self):
        return AgentTools(project_root=self._tmp.name, executor=self.executor)

    def _llm(self, *responses):
        from agentchanti.llm.chat_types import ChatResponse  # noqa: F401
        llm = self._mm()
        llm.chat.side_effect = list(responses)
        return llm

    @staticmethod
    def _cmd_call(n, command):
        from agentchanti.llm.chat_types import ChatResponse, ToolCall
        return ChatResponse(
            tool_calls=[ToolCall(name="run_command",
                                 arguments={"command": command}, id=f"c{n}")],
            stop_reason="tool_calls")

    @staticmethod
    def _stop():
        from agentchanti.llm.chat_types import ChatResponse
        return ChatResponse(text="giving up", stop_reason="stop")

    def _injected_user_text(self, llm):
        last = llm.chat.call_args_list[-1][0][0]
        return "\n".join(m.content or "" for m in last if m.role == "user")

    def test_command_not_found_routes_to_environment_wording(self):
        from agentchanti.orchestrator.agent_loop import run_agent_loop
        self.executor.run_command.return_value = (False, self.NOT_FOUND)
        llm = self._llm(self._cmd_call(0, self.RUFF),
                        self._cmd_call(1, self.RUFF),
                        self._stop())
        run_agent_loop(llm, self._tools(), "Fix lint", "lint the file",
                       max_turns=3, step_idx=11)
        text = self._injected_user_text(llm)
        self.assertIn("environment or argument problem", text)
        self.assertIn("python -m", text)
        # The wrong advice must NOT appear for this failure.
        self.assertNotIn("the failure is in the code", text)

    def test_ordinary_failure_keeps_edit_the_source_wording(self):
        from agentchanti.orchestrator.agent_loop import run_agent_loop
        self.executor.run_command.return_value = (
            False, "AssertionError: expected 5 got 6")
        llm = self._llm(self._cmd_call(0, "python -m pytest"),
                        self._cmd_call(1, "python -m pytest"),
                        self._stop())
        run_agent_loop(llm, self._tools(), "Fix bug", "fix it",
                       max_turns=3, step_idx=12)
        text = self._injected_user_text(llm)
        self.assertIn("the failure is in the code", text)
        self.assertNotIn("environment or argument problem", text)

    def test_repeat_across_attempts_nudges_on_first_run(self):
        """The cross-attempt case that previously never fired."""
        from agentchanti.orchestrator.agent_loop import run_agent_loop
        record_attempt(13, "loop", "verify-failed", [],
                       [(self.RUFF, False)], "ruff not found")
        self.executor.run_command.return_value = (False, self.NOT_FOUND)
        # ONE run of the command in this attempt — previously not a repeat.
        llm = self._llm(self._cmd_call(0, self.RUFF), self._stop())
        run_agent_loop(llm, self._tools(), "Fix lint", "lint the file",
                       max_turns=2, step_idx=13)
        self.assertIn("already ran", self._injected_user_text(llm))

    def test_first_attempt_at_a_step_is_unchanged(self):
        """Healthy path: no prior attempts, one run, no nudge."""
        from agentchanti.orchestrator.agent_loop import run_agent_loop
        self.executor.run_command.return_value = (False, self.NOT_FOUND)
        llm = self._llm(self._cmd_call(0, self.RUFF), self._stop())
        run_agent_loop(llm, self._tools(), "Fix lint", "lint the file",
                       max_turns=2, step_idx=14)
        self.assertNotIn("already ran", self._injected_user_text(llm))


if __name__ == "__main__":
    unittest.main()

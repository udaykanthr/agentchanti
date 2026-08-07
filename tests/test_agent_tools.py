"""Tests for the agent tool registry (agentchanti/agent_tools.py)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agentchanti.agent_tools import AgentTools
from agentchanti.llm.chat_types import ToolCall


class AgentToolsTestCase(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="agenttools_")
        self.tools = AgentTools(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, rel, content):
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    def _call(self, name, **args):
        return self.tools.execute(ToolCall(name=name, arguments=args))


class TestDefinitions(AgentToolsTestCase):

    def test_six_tools_with_schemas(self):
        defs = self.tools.definitions()
        names = {t.name for t in defs}
        self.assertEqual(names, {"list_files", "read_file", "write_file",
                                 "edit_file", "run_command", "search_code"})
        for t in defs:
            self.assertEqual(t.parameters["type"], "object")
            self.assertTrue(t.description)


class TestReadWrite(AgentToolsTestCase):

    def test_write_then_read(self):
        result = self._call("write_file", path="pkg/mod.py",
                            content="x = 1\ny = 2\n")
        self.assertTrue(result.startswith("OK:"))
        content = self._call("read_file", path="pkg/mod.py")
        self.assertIn("1: x = 1", content)
        self.assertIn("2: y = 2", content)

    def test_read_line_range(self):
        self._write("a.txt", "one\ntwo\nthree\nfour\n")
        content = self._call("read_file", path="a.txt",
                             start_line=2, end_line=3)
        self.assertIn("2: two", content)
        self.assertIn("3: three", content)
        self.assertNotIn("one", content.split("\n", 1)[1])
        self.assertNotIn("four", content)

    def test_read_missing_file(self):
        self.assertTrue(self._call("read_file", path="nope.txt")
                        .startswith("ERROR"))

    def test_write_records_to_memory(self):
        memory = MagicMock()
        tools = AgentTools(project_root=self.root, memory=memory)
        tools.execute(ToolCall(name="write_file",
                               arguments={"path": "m.py", "content": "z = 1"}))
        memory.update.assert_called_once_with({"m.py": "z = 1"},
                                              allow_protected={"m.py"})

    def test_protected_manifest_write_reaches_memory(self):
        """A write that already landed on disk must be tracked.

        FileMemory skips a protected basename that exists on disk, to stop a
        hallucinated manifest clobbering a real one. But _record runs AFTER
        the write, so the guard could not prevent anything — it only made
        memory disagree with the filesystem while logging a WARNING about a
        skip that had not happened. Seen in 4 of 6 benchmark runs: the loop
        created requirements.txt, its gate read 'pygame' back off disk and
        passed, and the content was absent from memory for the rest of the
        run.
        """
        import os

        from agentchanti.orchestrator.memory import FileMemory

        # FileMemory's guard calls os.path.isfile() on the RELATIVE path, so
        # it only fires when the cwd is the project root — which is exactly
        # the condition of a real run, and why this needs the chdir to
        # reproduce at all.
        memory = FileMemory()
        tools = AgentTools(project_root=self.root, memory=memory)
        prev = os.getcwd()
        os.chdir(self.root)
        try:
            tools.execute(ToolCall(name="write_file",
                                   arguments={"path": "requirements.txt",
                                              "content": "pygame"}))
        finally:
            os.chdir(prev)
        self.assertTrue(os.path.isfile(os.path.join(self.root,
                                                    "requirements.txt")),
                        "precondition: the write must have reached disk")
        self.assertEqual(memory.get("requirements.txt"), "pygame")


class TestEditFile(AgentToolsTestCase):

    def test_exact_replace(self):
        self._write("calc.py", "def add(a, b):\n    return a - b\n")
        result = self._call("edit_file", path="calc.py",
                            old_text="return a - b",
                            new_text="return a + b")
        self.assertTrue(result.startswith("OK:"))
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertIn("return a + b", f.read())

    def test_not_found(self):
        self._write("calc.py", "x = 1\n")
        result = self._call("edit_file", path="calc.py",
                            old_text="y = 2", new_text="y = 3")
        self.assertIn("not found", result)

    def test_ambiguous_match_rejected(self):
        self._write("calc.py", "x = 1\nx = 1\n")
        result = self._call("edit_file", path="calc.py",
                            old_text="x = 1", new_text="x = 2")
        self.assertIn("2 locations", result)

    def test_python_syntax_error_rejected_and_file_untouched(self):
        original = "def f():\n    return 1\n"
        self._write("calc.py", original)
        result = self._call("edit_file", path="calc.py",
                            old_text="return 1",
                            new_text="return (1")
        self.assertIn("syntax error", result)
        with open(os.path.join(self.root, "calc.py")) as f:
            self.assertEqual(f.read(), original)


class TestListFiles(AgentToolsTestCase):

    def test_lists_recursively_and_skips_ignored(self):
        self._write("src/app.py", "pass")
        self._write("README.md", "hi")
        self._write("node_modules/pkg/index.js", "junk")
        self._write(".git/config", "junk")
        listing = self._call("list_files")
        self.assertIn("src/app.py", listing)
        self.assertIn("README.md", listing)
        self.assertNotIn("node_modules", listing)
        self.assertNotIn(".git", listing)


class TestSafetyAndDispatch(AgentToolsTestCase):

    def test_path_escape_rejected(self):
        result = self._call("read_file", path="../outside.txt")
        self.assertTrue(result.startswith("ERROR"))
        self.assertIn("outside the project root", result)

    def test_unknown_tool(self):
        result = self._call("does_not_exist")
        self.assertIn("unknown tool", result)
        self.assertIn("read_file", result)  # lists available tools

    def test_bad_arguments(self):
        result = self._call("read_file", wrong_arg="x")
        self.assertTrue(result.startswith("ERROR: bad arguments"))

    def test_execute_all_wraps_tool_messages(self):
        self._write("a.txt", "hello\n")
        msgs = self.tools.execute_all([
            ToolCall(name="read_file", arguments={"path": "a.txt"}, id="c1"),
            ToolCall(name="nope", arguments={}, id="c2"),
        ])
        self.assertEqual([m.role for m in msgs], ["tool", "tool"])
        self.assertEqual(msgs[0].tool_call_id, "c1")
        self.assertEqual(msgs[0].tool_name, "read_file")
        self.assertIn("hello", msgs[0].content)
        self.assertIn("unknown tool", msgs[1].content)


class TestRunCommand(AgentToolsTestCase):

    def test_run_command_success(self):
        executor = MagicMock()
        executor.run_command.return_value = (True, "hi there")
        tools = AgentTools(project_root=self.root, executor=executor)
        result = tools.execute(ToolCall(name="run_command",
                                        arguments={"command": "echo hi"}))
        self.assertIn("exit: success", result)
        self.assertIn("hi there", result)
        executor.run_command.assert_called_once_with(
            "echo hi", timeout=120, cwd=self.root)

    def test_run_command_failure(self):
        executor = MagicMock()
        executor.run_command.return_value = (False, "boom")
        tools = AgentTools(project_root=self.root, executor=executor)
        result = tools.execute(ToolCall(name="run_command",
                                        arguments={"command": "bad"}))
        self.assertIn("exit: FAILED", result)


class TestSearchCode(AgentToolsTestCase):

    def test_without_searcher_degrades_gracefully(self):
        result = self._call("search_code", query="auth")
        self.assertIn("unavailable", result)

    def test_with_searcher_formats_results(self):
        hit = MagicMock()
        hit.file = "src/auth.py"
        hit.line_start, hit.line_end = 10, 20
        hit.symbol_type, hit.symbol_name = "function", "check_auth"
        hit.score = 0.91
        hit.code_snippet = "def check_auth(): ..."
        searcher = MagicMock()
        searcher.search.return_value = [hit]
        tools = AgentTools(project_root=self.root, searcher=searcher)
        result = tools.execute(ToolCall(name="search_code",
                                        arguments={"query": "auth"}))
        self.assertIn("src/auth.py:10-20", result)
        self.assertIn("check_auth", result)
        searcher.search.assert_called_once_with("auth", top_k=5)


class TestHeredocGuard(AgentToolsTestCase):
    """POSIX heredocs cannot work on Windows cmd — the guard must return
    an instructive error instead of a bare exit-1 (observed: an agent
    loop burned a turn on `python - << 'PY'` that failed with 33 chars
    of noise)."""

    HEREDOC_CMD = "python - << 'PY'\nprint('hi')\nPY"

    def test_heredoc_rejected_on_windows(self):
        executor = MagicMock()
        tools = AgentTools(project_root=self.root, executor=executor)
        with patch("agentchanti.agent_tools.os.name", "nt"):
            result = tools.execute(ToolCall(
                name="run_command",
                arguments={"command": self.HEREDOC_CMD}))
        self.assertTrue(result.startswith("ERROR"))
        self.assertIn("write_file", result)
        executor.run_command.assert_not_called()

    def test_heredoc_allowed_on_posix(self):
        executor = MagicMock()
        executor.run_command.return_value = (True, "hi")
        tools = AgentTools(project_root=self.root, executor=executor)
        with patch("agentchanti.agent_tools.os.name", "posix"):
            result = tools.execute(ToolCall(
                name="run_command",
                arguments={"command": self.HEREDOC_CMD}))
        self.assertIn("exit: success", result)
        executor.run_command.assert_called_once()

    def test_plain_command_unaffected_on_windows(self):
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        tools = AgentTools(project_root=self.root, executor=executor)
        with patch("agentchanti.agent_tools.os.name", "nt"):
            result = tools.execute(ToolCall(
                name="run_command",
                arguments={"command": "python -m pytest -q"}))
        self.assertIn("exit: success", result)


if __name__ == "__main__":
    unittest.main()



class TestNoTestsCollected(AgentToolsTestCase):
    """"exit: FAILED" read identically for two different problems.

    Both unittest and pytest exit 5 when the runner COLLECTED NOTHING —
    a discovery problem, with no assertion having run at all. The tool
    result never showed the exit code, so the model could not tell that
    from a failing test and debugged code that was never executed:
    observed a loop spending four consecutive run_command turns
    re-running an empty suite. 19 occurrences across 7 of 8 measured runs.
    """

    def _detect(self, cmd, code, out=""):
        from agentchanti.agent_tools import _no_tests_collected
        return _no_tests_collected(cmd, code, out)

    def test_exit_5_from_a_test_runner_is_a_discovery_problem(self):
        for cmd in ("python -m unittest -v", "python -m pytest -q",
                    "python manage.py test", "tox"):
            with self.subTest(cmd=cmd):
                self.assertTrue(self._detect(cmd, 5))

    def test_a_failing_assertion_is_not(self):
        """The hint must never appear on a real test failure."""
        self.assertFalse(
            self._detect("python -m unittest -v", 1, "FAILED (failures=1)"))

    def test_exit_5_from_an_unrelated_command_is_not(self):
        """5 is an ordinary failure code for other programs."""
        self.assertFalse(self._detect('python -c "import sys; sys.exit(5)"', 5))
        self.assertFalse(self._detect("npm run build", 5))

    def test_output_markers_work_without_an_exit_code(self):
        """Some runners report it in words; the code may be unavailable."""
        for out in ("no tests ran in 0.00s", "Ran 0 tests in 0.000s",
                    "collected 0 items"):
            with self.subTest(out=out):
                self.assertTrue(self._detect("npm test", None, out))

    def test_a_passing_suite_is_not_flagged(self):
        self.assertFalse(self._detect("python -m pytest", 0, "2 passed"))

    def test_the_hint_reaches_the_tool_result(self):
        # Deliberately does NOT assert the "exit: FAILED" prefix. Whether a
        # zero-test run exits non-zero is CPython policy, not this hint's
        # behaviour: unittest gained that exit status in 3.12, so on 3.10
        # and 3.11 the same run reports "exit: success" and the assertion
        # failed on four CI jobs while the hint itself was present and
        # correct. The detector fires on the "Ran 0 tests" output marker,
        # which every version prints, so the hint is what this test is
        # named for and all this test should check. The FAILED prefix stays
        # covered by test_a_real_failure_gets_no_hint_through_the_tool,
        # which runs a genuinely failing assertion and so fails everywhere.
        result = self._call("run_command", command="python -m unittest -v")
        self.assertIn("COLLECTED NO TESTS", result)
        self.assertIn("test_*.py", result)
        self.assertIn("__init__.py", result)

    def test_the_hint_appears_even_when_the_runner_exits_zero(self):
        """The dangerous case, and the one CI caught.

        unittest only started exiting non-zero for a zero-test run in
        3.12. On 3.10/3.11 the same empty project reports "exit: success",
        so gating the hint on failure hid it exactly where a green result
        is backed by zero executed tests.
        """
        from agentchanti.agent_tools import _no_tests_collected
        self.assertTrue(_no_tests_collected(
            "python -m unittest -v", 0, "Ran 0 tests in 0.000s\n\nOK\n"))

    def test_a_real_failure_gets_no_hint_through_the_tool(self):
        self._write("tests/__init__.py", "")
        self._write("tests/test_x.py",
                    "import unittest\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_f(self):\n"
                    "        self.assertEqual(1, 2)\n")
        result = self._call("run_command", command="python -m unittest -v")
        self.assertIn("exit: FAILED", result)
        self.assertNotIn("COLLECTED NO TESTS", result)

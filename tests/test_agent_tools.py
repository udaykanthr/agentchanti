"""Tests for the agent tool registry (agentchanti/agent_tools.py)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

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
        memory.update.assert_called_once_with({"m.py": "z = 1"})


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


if __name__ == "__main__":
    unittest.main()

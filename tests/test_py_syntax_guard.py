"""Python syntax guards must match the interpreter, not just the parser.

``ast.parse`` stops at the parse stage. Future-import placement is only
enforced when the module is compiled, so a file with a second
``from __future__ import annotations`` partway down parses cleanly and
then raises SyntaxError on import.

That gap was real: a chunk edit whose replacement restated the module
header spliced a duplicate header into the middle of src/player.py. Every
write guard passed it, it reached disk, and two diagnosis attempts then
failed against a file that could never import — the pipeline halted.
"""

import os
import tempfile
import unittest

from agentchanti.py_syntax import check_python_syntax, is_valid_python


# Parses fine under ast.parse; rejected by the interpreter.
MID_FILE_FUTURE = (
    "import math\n"
    "\n"
    "from __future__ import annotations\n"
    "\n"
    "X = 1\n"
)

VALID = (
    "from __future__ import annotations\n"
    "\n"
    "import math\n"
    "\n"
    "X = math.pi\n"
)


class CheckPythonSyntaxTest(unittest.TestCase):
    def test_mid_file_future_import_is_rejected(self):
        err = check_python_syntax(MID_FILE_FUTURE, "player.py")
        self.assertIsNotNone(err, "ast.parse's blind spot must be covered")
        self.assertIn("__future__", err)

    def test_ast_parse_really_does_miss_it(self):
        """Documents *why* this module exists — if this ever fails,
        CPython changed and the helper's rationale needs revisiting."""
        import ast
        ast.parse(MID_FILE_FUTURE)          # must NOT raise

    def test_valid_source_passes(self):
        self.assertIsNone(check_python_syntax(VALID, "ok.py"))
        self.assertTrue(is_valid_python(VALID))

    def test_ordinary_syntax_error_still_caught(self):
        err = check_python_syntax("def f(:\n    pass\n", "bad.py")
        self.assertIsNotNone(err)

    def test_indentation_error_caught(self):
        self.assertIsNotNone(check_python_syntax("    x = 1\n", "bad.py"))

    def test_null_bytes_do_not_raise(self):
        """compile() raises ValueError, not SyntaxError, on null bytes."""
        self.assertIsNotNone(check_python_syntax("x = 1\0\n", "bad.py"))

    def test_error_mentions_line_number(self):
        err = check_python_syntax("x = 1\ndef f(:\n", "bad.py")
        self.assertIn("line", err)

    def test_does_not_inherit_callers_future_flags(self):
        """This test module's own __future__ flags must not leak in."""
        self.assertIsNotNone(check_python_syntax(MID_FILE_FUTURE, "x.py"))


class GuardsUseTheStrictCheckTest(unittest.TestCase):
    """The individual write guards must reject the bad splice too."""

    def test_pipeline_content_validation_rejects(self):
        from agentchanti.orchestrator.pipeline import _syntax_gate
        self.assertIsNotNone(_syntax_gate("player.py",
                                                    MID_FILE_FUTURE))
        self.assertIsNone(_syntax_gate("player.py", VALID))

    def test_diagnosis_syntax_check_rejects(self):
        from agentchanti.orchestrator.diagnosis import _check_syntax
        self.assertIsNotNone(_check_syntax("player.py",
                                                  MID_FILE_FUTURE))
        self.assertIsNone(_check_syntax("player.py", VALID))

    def test_diagnosis_check_ignores_non_python(self):
        from agentchanti.orchestrator.diagnosis import _check_syntax
        self.assertIsNone(_check_syntax("notes.txt", MID_FILE_FUTURE))

    def test_agent_tools_edit_file_rejects(self):
        from agentchanti.agent_tools import AgentTools
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "m.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(VALID)
            tools = AgentTools(project_root=root)
            out = tools._tool_edit_file(
                "m.py",
                "import math\n",
                "import math\n\nfrom __future__ import annotations\n")
            self.assertTrue(out.startswith("ERROR"), out)
            # The file on disk must be untouched.
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), VALID)


if __name__ == "__main__":
    unittest.main()

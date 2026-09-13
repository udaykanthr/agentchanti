"""The auto-loader must not promise a whole file it did not read.

Measured 2026-09-10. `_read_full_file` capped at 300 lines and labelled the
result "(full source)" regardless. `pinball/table.py` is 405 lines, so the
IntentAgent received lines 1-300 under a header promising the whole file,
plus a "... (105 more lines)" marker telling it otherwise — and nothing in
its vocabulary could fetch them: KB_SEARCH returns symbol chunks, and
RUN_CMD's allowlist (git/ls/grep/test-runners) has no file reader.

It spent three iterations asking in prose::

    Iteration 1: KB_SEARCH 'table.py full source including complete build_table'
    Iteration 2: KB_SEARCH '...lines 303-454 including rail helper and all...'
    Iteration 3: KB_SEARCH '...lines 330-454... do not truncate'

~28k uncached prompt tokens, then it concluded having never seen the second
half of `build_table` — the exact function the task was about ("changing the
shapes of board").
"""
import textwrap
import unittest

from agentchanti.agents.intent import (
    _AUTOLOAD_MAX_LINES, _READ_FILE_MAX_LINES, IntentAgent,
)


class _Builder:
    """Stands in for the KB context builder — only _project_root is used."""

    def __init__(self, root):
        self._project_root = str(root)


def _write(tmp, rel, lines):
    import os
    path = os.path.join(str(tmp), rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join("line %d\n" % i for i in range(1, lines + 1)))
    return path


class AutoLoadHeaderTellsTheTruth(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.b = _Builder(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_complete_file_says_complete(self):
        _write(self.root, "pkg/small.py", 40)
        out = IntentAgent._read_full_file(self.b, "small.py")
        self.assertIn("(complete, 40 lines)", out)
        self.assertNotIn("TRUNCATED", out)
        self.assertIn("line 40", out)

    def test_a_truncated_file_never_claims_to_be_full(self):
        """The exact defect: 'full source' over a partial read."""
        n = _AUTOLOAD_MAX_LINES + 105          # the measured shape: 405 vs 300
        _write(self.root, "pkg/big.py", n)
        out = IntentAgent._read_full_file(self.b, "big.py")
        self.assertNotIn("full source", out.lower())
        self.assertIn("TRUNCATED", out)
        self.assertIn("of %d" % n, out)

    def test_the_truncation_notice_names_the_tool_that_can_help(self):
        """Otherwise the model can only re-ask KB_SEARCH, which cannot work."""
        n = _AUTOLOAD_MAX_LINES + 105
        _write(self.root, "pkg/big.py", n)
        out = IntentAgent._read_full_file(self.b, "big.py")
        self.assertIn("READ_FILE:", out)
        self.assertIn("%d-%d" % (_AUTOLOAD_MAX_LINES + 1, n), out)
        self.assertIn("KB_SEARCH", out)   # says explicitly that it cannot help


class ReadFileRange(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.b = _Builder(self.root)
        _write(self.root, "pinball/table.py", 405)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_range_the_model_kept_asking_for(self):
        out = IntentAgent._read_file_range(self.b, "pinball/table.py:301-405")
        self.assertIn("lines 301-405 of 405", out)
        self.assertIn("line 301", out)
        self.assertIn("line 405", out)
        self.assertNotIn("line 300\n", out)

    def test_bare_path_reads_from_the_start(self):
        out = IntentAgent._read_file_range(self.b, "pinball/table.py")
        self.assertIn("lines 1-", out)
        self.assertIn("line 1\n", out)

    def test_basename_alone_is_resolved(self):
        """The model names files the way the log does — 'table.py'."""
        out = IntentAgent._read_file_range(self.b, "table.py:400-405")
        self.assertIn("line 405", out)

    def test_a_call_is_bounded(self):
        _write(self.root, "huge.py", _READ_FILE_MAX_LINES + 50)
        out = IntentAgent._read_file_range(self.b, "huge.py:1-%d"
                                           % (_READ_FILE_MAX_LINES + 50))
        self.assertIn("capped at %d lines" % _READ_FILE_MAX_LINES, out)
        self.assertIn("request %d-" % (_READ_FILE_MAX_LINES + 1), out)

    def test_escaping_the_project_root_is_refused(self):
        out = IntentAgent._read_file_range(self.b, "../../../etc/passwd")
        self.assertIn("refused", out.lower())

    def test_a_missing_file_says_so_rather_than_returning_nothing(self):
        out = IntentAgent._read_file_range(self.b, "nope/absent.py")
        self.assertIn("not found", out.lower())

    def test_a_start_past_the_end_is_explained(self):
        out = IntentAgent._read_file_range(self.b, "pinball/table.py:900-950")
        self.assertIn("only 405 lines", out)


class ToolIsAdvertised(unittest.TestCase):

    def test_read_file_is_in_the_action_vocabulary(self):
        """A tool the prompt never mentions is a tool the model cannot use."""
        doc = IntentAgent.__doc__ or ""
        self.assertIn("READ_FILE", doc)


if __name__ == "__main__":
    unittest.main()

"""`--prompt-from-file` must not read a BOM as part of the task.

Measured 2026-09-27 while adding benchmark cases. A task file written by
PowerShell's `-Encoding utf8` — the default for most Windows editors and
Notepad — begins with a UTF-8 BOM. Read as plain utf-8 it survives as
U+FEFF, and `str.strip()` does not remove it because it is not
whitespace, so the run logged:

    Task: ﻿Build a tiny thing in Python.

The invisible character is not cosmetic. The task text is what the
planner and the intent agent read, and it is hashed into the acceptance
seed's task fingerprint — so the same prompt saved by two different
editors produces two different fingerprints, and a seeded contract is
re-seeded when it should have been reused.
"""
import inspect
import re

from agentchanti.orchestrator import cli

TASK = "Build a production-ready todo manager.\n\nIt must persist tasks."


def _read_prompt_file(path):
    """The exact expression cli.py uses, applied to a real file."""
    src = inspect.getsource(cli)
    match = re.search(r'open\(args\.prompt_from_file, "r", '
                      r'encoding="([\w-]+)"\)', src)
    assert match, "the prompt-from-file read changed shape; update this test"
    with open(path, "r", encoding=match.group(1)) as f:
        return f.read().strip()


class TestTheMeasuredCase:

    def test_a_bom_written_file_yields_a_clean_task(self, tmp_path):
        p = tmp_path / "task.txt"
        p.write_bytes(b"\xef\xbb\xbf" + TASK.encode("utf-8"))
        got = _read_prompt_file(p)
        assert got == TASK
        assert not got.startswith("﻿")

    def test_strip_alone_would_not_have_caught_it(self):
        """Why the encoding had to change rather than the stripping."""
        assert ("﻿" + TASK).strip().startswith("﻿")


class TestWhatIsUnchanged:

    def test_a_plain_utf8_file_is_read_identically(self, tmp_path):
        p = tmp_path / "task.txt"
        p.write_text(TASK, encoding="utf-8")
        assert _read_prompt_file(p) == TASK

    def test_non_ascii_content_survives(self, tmp_path):
        """utf-8-sig differs from utf-8 only in the leading BOM."""
        text = "Créer une application — with an em dash and é"
        p = tmp_path / "task.txt"
        p.write_text(text, encoding="utf-8")
        assert _read_prompt_file(p) == text

    def test_surrounding_whitespace_is_still_stripped(self, tmp_path):
        p = tmp_path / "task.txt"
        p.write_bytes(b"\xef\xbb\xbf\n  " + TASK.encode("utf-8") + b"  \n")
        assert _read_prompt_file(p) == TASK

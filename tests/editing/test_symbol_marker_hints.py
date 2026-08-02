"""An [EDIT] marker's parenthetical is a hint, not a range.

A run halted with "Diagnosis produced no actionable fix" — twice — while
holding a correct diagnosis. It had written:

    #### [EDIT]: config.py:MAZE (row 16)

"(row 16)" matched none of the three markers: _EDIT_MARKER wants a
numeric range with a dash, _EDIT_FULL_FILE_MARKER only allows
(new file)/(full file)/(replace), and _EDIT_SYMBOL_MARKER insisted the
parenthetical start with the literal word "lines". The edit was dropped
silently and the pipeline halted.

Parsing it is only half the fix. The content was ONE row of a 21-row
MAZE; resolving the symbol and replacing the whole chunk would splice a
single row over the entire constant — silent corruption, strictly worse
than the silent no-op it replaced.
"""

from __future__ import annotations

import unittest

from agentchanti.editing.chunk_editor import ChunkEditor

SRC = '''WALL = 0
PELLET = 1
PLAYER_SPAWN = 5

MAZE = [
    [WALL, WALL, WALL, WALL, WALL, WALL],
    [WALL, PELLET, PELLET, PELLET, PELLET, WALL],
    [WALL, PELLET, WALL, WALL, PELLET, WALL],
    [WALL, PELLET, WALL, PELLET, PELLET, WALL],
    [WALL, PELLET, PELLET, PELLET, PELLET, PELLET],
    [WALL, WALL, WALL, WALL, WALL, WALL],
]
'''


def _diag(marker, body="    [WALL, PELLET, WALL, WALL, PLAYER_SPAWN, WALL],"):
    return f"2. FIX: add a spawn.\n\n#### [EDIT]: {marker}\n```python\n{body}\n```\n"


class TestMarkerForms(unittest.TestCase):

    def _parsed(self, marker):
        got = ChunkEditor().parse_chunk_response(_diag(marker))
        return 0 if got is None else len(got)

    def test_a_row_hint_is_accepted(self):
        self.assertEqual(self._parsed("config.py:MAZE (row 16)"), 1)

    def test_other_prose_hints_are_accepted_too(self):
        for m in ("config.py:MAZE (row 3)", "config.py:MAZE (the spawn row)",
                  "config.py:MAZE (bottom centre)", "config.py:MAZE"):
            self.assertEqual(self._parsed(m), 1, m)

    def test_a_numeric_range_still_wins(self):
        got = ChunkEditor().parse_chunk_response(
            _diag("config.py:MAZE (lines 5-11)"))
        self.assertEqual(got[0].line_start, 5)
        self.assertEqual(got[0].line_end, 11)

    def test_a_language_tag_is_not_mistaken_for_a_symbol(self):
        """`file.py:python` means full-file, not a symbol named python."""
        got = ChunkEditor().parse_chunk_response(_diag("config.py:python"))
        self.assertTrue(got)
        self.assertNotEqual(got[0].chunk_id, "python")


class TestPartialEditPlacement(unittest.TestCase):

    def _apply(self, marker, body=None):
        ce = ChunkEditor()
        kw = {} if body is None else {"body": body}
        edits = ce.parse_chunk_response(_diag(marker, **kw))
        known = ce.chunk_file("config.py", SRC)
        return ce, ce.apply_chunk_edits(SRC, edits, known_chunks=known)

    def test_one_row_replaces_only_that_row(self):
        ce, out = self._apply("config.py:MAZE (row 3)")
        self.assertFalse(ce.last_apply_rejected)
        self.assertNotEqual(out, SRC)
        self.assertIn("PLAYER_SPAWN, WALL],", out)
        # The other five rows survive — this is the corruption guard.
        self.assertEqual(out.count("    [WALL"), SRC.count("    [WALL"))
        compile(out, "config.py", "exec")

    def test_it_edits_the_row_it_most_resembles(self):
        _, out = self._apply("config.py:MAZE (row 3)")
        rows = [l for l in out.splitlines() if l.startswith("    [WALL")]
        self.assertIn("PLAYER_SPAWN", rows[2])
        self.assertNotIn("PLAYER_SPAWN", rows[1])
        self.assertNotIn("PLAYER_SPAWN", rows[3])

    def test_an_ambiguous_row_is_refused_not_guessed(self):
        """Two equally-good candidates must not be broken by a coin flip —
        editing the wrong row is worse than editing none."""
        src = SRC.replace(
            "    [WALL, PELLET, WALL, PELLET, PELLET, WALL],\n",
            "    [WALL, PELLET, WALL, WALL, PELLET, WALL],\n")
        ce = ChunkEditor()
        edits = ce.parse_chunk_response(_diag("config.py:MAZE (row 3)"))
        known = ce.chunk_file("config.py", src)
        with self.assertRaises(ValueError):
            ce.apply_chunk_edits(src, edits, known_chunks=known)

    def test_a_full_size_symbol_edit_still_replaces_the_chunk(self):
        """The partial path must not hijack a legitimate whole-constant
        rewrite."""
        whole = ("MAZE = [\n"
                 "    [WALL, WALL],\n"
                 "    [WALL, PLAYER_SPAWN],\n"
                 "    [WALL, WALL],\n"
                 "    [WALL, WALL],\n"
                 "    [WALL, WALL],\n"
                 "]")
        ce, out = self._apply("config.py:MAZE", body=whole)
        self.assertFalse(ce.last_apply_rejected)
        self.assertEqual(out.count("MAZE = ["), 1)
        compile(out, "config.py", "exec")


if __name__ == "__main__":
    unittest.main()

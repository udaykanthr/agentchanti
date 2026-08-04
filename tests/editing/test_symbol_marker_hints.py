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


CLASS_SRC = ("import os\n\n\nclass Map:\n"
             + "".join(f"    def m{i}(self):\n        return {i}\n"
                       for i in range(40))
             + "\n\nclass Game:\n    pass\n")


class TestWholeSymbolRewrite(unittest.TestCase):
    """Shape decides whole-vs-partial, not size.

    From a real run: the model rewrote a 179-line `class Map` as 101
    correct lines. The size heuristic read "shorter" as "partial",
    refused to place it, and the run halted — a correct fix discarded.
    Models routinely shorten code when they fix it.
    """

    def _rewrite(self, methods):
        body = "".join(f"    def m{i}(self):\n        return {i}\n"
                       for i in range(methods))
        return f"class Map:\n{body}"

    def _apply(self, new_content, marker="game.py:Map (rewritten)"):
        ce = ChunkEditor()
        diag = (f"FIX:\n#### [EDIT]: {marker}\n```python\n{new_content}```\n")
        edits = ce.parse_chunk_response(diag)
        known = ce.chunk_file("game.py", CLASS_SRC)
        return ce, ce.apply_chunk_edits(CLASS_SRC, edits, known_chunks=known)

    def test_a_shorter_rewrite_replaces_the_whole_class(self):
        ce, out = self._apply(self._rewrite(15))
        self.assertFalse(ce.last_apply_rejected)
        self.assertNotEqual(out, CLASS_SRC)
        self.assertEqual(out.count("class Map:"), 1)
        self.assertEqual(out.count("def m"), 15)
        compile(out, "game.py", "exec")

    def test_neighbouring_code_survives(self):
        _, out = self._apply(self._rewrite(15))
        self.assertIn("class Game:", out)
        self.assertIn("import os", out)

    def test_a_fragment_is_still_treated_as_partial(self):
        """The guard must not be disarmed — a body-only snippet that does
        NOT reopen the declaration stays on the partial path."""
        ce = ChunkEditor()
        diag = ("FIX:\n#### [EDIT]: game.py:Map (one method)\n"
                "```python\n    def m7(self):\n        return 99\n```\n")
        edits = ce.parse_chunk_response(diag)
        known = ce.chunk_file("game.py", CLASS_SRC)
        # Two lines, no declaration: the single-line aligner declines, so
        # this raises rather than overwriting all 40 methods.
        with self.assertRaises(ValueError):
            ce.apply_chunk_edits(CLASS_SRC, edits, known_chunks=known)

    def test_a_renamed_symbol_is_not_a_reopen(self):
        """`class Other:` is not a rewrite of `class Map:`."""
        ce = ChunkEditor()
        diag = ("FIX:\n#### [EDIT]: game.py:Map (rewritten)\n```python\n"
                "class Other:\n    def z(self):\n        return 1\n```\n")
        edits = ce.parse_chunk_response(diag)
        known = ce.chunk_file("game.py", CLASS_SRC)
        with self.assertRaises(ValueError):
            ce.apply_chunk_edits(CLASS_SRC, edits, known_chunks=known)

    def test_a_signature_change_still_counts_as_a_reopen(self):
        """Same symbol, different parameters, is still that symbol."""
        src = "def f(a):\n" + "".join(f"    x{i} = {i}\n" for i in range(20))
        ce = ChunkEditor()
        diag = ("FIX:\n#### [EDIT]: m.py:f (fixed)\n```python\n"
                "def f(a, b=2):\n    return a + b\n```\n")
        edits = ce.parse_chunk_response(diag)
        known = ce.chunk_file("m.py", src)
        out = ce.apply_chunk_edits(src, edits, known_chunks=known)
        self.assertIn("def f(a, b=2):", out)
        self.assertEqual(out.count("def f("), 1)

    def test_a_constant_reopen_is_recognised(self):
        """`MAZE = [` replacing a `MAZE = [` chunk is a whole rewrite."""
        src = "MAZE = [\n" + "".join(f"    [{i}],\n" for i in range(20)) + "]\n"
        ce = ChunkEditor()
        diag = ("FIX:\n#### [EDIT]: c.py:MAZE (rebuilt)\n```python\n"
                "MAZE = [\n    [0],\n    [1],\n]\n```\n")
        edits = ce.parse_chunk_response(diag)
        known = ce.chunk_file("c.py", src)
        out = ce.apply_chunk_edits(src, edits, known_chunks=known)
        self.assertEqual(out.count("MAZE = ["), 1)
        compile(out, "c.py", "exec")


if __name__ == "__main__":
    unittest.main()

"""Module-level constants must be addressable by name.

From a real failed run: the coder wrote a ragged ``DEFAULT_MAZE`` that
raised ``IndexError`` on start-up.  Diagnosis identified the cause exactly
and emitted ``#### [EDIT]: game.py:DEFAULT_MAZE (lines 9-35)`` — but the
constant lived at lines 10-38 and was not a named chunk, so the symbol
could not resolve.  Resolution fell back to the model's own arithmetic,
which was two lines short; the splice left the literal's tail behind:

    ]                                       <- end of the replacement
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWW"      <- orphan, unexpected indent
    ]

The post-splice guard caught the broken syntax and returned the original,
so nothing changed, the same diagnosis ran again, and the pipeline halted
after two attempts having written no fix at all.
"""

from __future__ import annotations

import unittest

from agentchanti.editing.chunk_editor import ChunkEditor, ChunkEditResponse

SOURCE = '''import os

TILE = 24

DEFAULT_MAZE = [
    "WWWW",
    "W..W",
    "W..W",
    "WWWW"
]

OTHER = 1


class Map:
    def __init__(self):
        self.rows = DEFAULT_MAZE
'''


def _chunk(source=SOURCE):
    return ChunkEditor().chunk_file("game.py", source)


class TestConstChunks(unittest.TestCase):

    def _by_id(self, chunks):
        return {c.chunk_id: (c.line_start, c.line_end) for c in chunks}

    def test_a_multiline_constant_spans_exactly_its_literal(self):
        """Not to the next def — a span that is too wide fails the 70%
        full-replacement test just as a short one does."""
        self.assertEqual(self._by_id(_chunk())["const:DEFAULT_MAZE"], (5, 10))

    def test_single_line_constants_are_named_too(self):
        ids = self._by_id(_chunk())
        self.assertEqual(ids["const:TILE"], (3, 3))
        self.assertEqual(ids["const:OTHER"], (12, 12))

    def test_class_bodies_are_not_claimed_as_constants(self):
        """`self.rows = ...` is an attribute, not a module constant, and a
        second chunk over the same lines makes the splice order-dependent."""
        self.assertNotIn("const:self.rows", self._by_id(_chunk()))
        self.assertNotIn("const:rows", self._by_id(_chunk()))

    def test_a_file_that_does_not_parse_yields_no_constants(self):
        """Broken source is when line numbers are least trustworthy;
        guessing spans there is worse than the existing fallback."""
        chunks = _chunk("X = [1,\ndef broken(:\n")
        self.assertEqual([c for c in chunks if c.chunk_type == "const"], [])

    def test_a_constant_never_overlaps_another_chunk(self):
        """Class chunks legitimately contain their method chunks, so this
        is not a global invariant — but a const claiming lines that a def
        also claims would make the splice order-dependent."""
        chunks = _chunk()
        consts = [c for c in chunks if c.chunk_type == "const"]
        self.assertTrue(consts)
        for c in consts:
            span = set(range(c.line_start, c.line_end + 1))
            for other in chunks:
                if other is c:
                    continue
                self.assertFalse(
                    span & set(range(other.line_start, other.line_end + 1)),
                    f"{c.chunk_id} overlaps {other.chunk_id}")


class TestTheOriginalFailure(unittest.TestCase):
    """The end-to-end shape that halted the pipeline."""

    def _edit(self, start, end, body):
        return ChunkEditResponse(file_path="game.py", chunk_id="DEFAULT_MAZE",
                                 line_start=start, line_end=end,
                                 new_content=body)

    def test_a_short_line_range_still_replaces_the_whole_constant(self):
        ed = ChunkEditor()
        # Model asked for 4-8; the literal is really 5-10.
        edit = self._edit(4, 8, 'DEFAULT_MAZE = [\n    "WWWW",\n    "WWWW"\n]\n')
        out = ed.apply_chunk_edits(SOURCE, [edit],
                                   known_chunks=_chunk())
        self.assertFalse(ed.last_apply_rejected)
        self.assertNotEqual(out, SOURCE)
        # No orphan tail left behind by a short splice.
        self.assertEqual(out.count("DEFAULT_MAZE = ["), 1)
        self.assertEqual(out.count('"W..W"'), 0)
        compile(out, "game.py", "exec")

    def test_a_rejected_splice_is_reported_not_silently_swallowed(self):
        """Returning the original unchanged looks exactly like success."""
        ed = ChunkEditor()
        bad = ChunkEditResponse(file_path="game.py", chunk_id="Map",
                                line_start=15, line_end=16,
                                new_content="    if True:\n")
        out = ed.apply_chunk_edits(SOURCE, [bad], known_chunks=_chunk())
        self.assertEqual(out, SOURCE)
        self.assertTrue(ed.last_apply_rejected)

    def test_the_flag_resets_between_calls(self):
        ed = ChunkEditor()
        ed.last_apply_rejected = True
        good = ChunkEditResponse(file_path="game.py", chunk_id="TILE",
                                 line_start=3, line_end=3,
                                 new_content="TILE = 32\n")
        out = ed.apply_chunk_edits(SOURCE, [good], known_chunks=_chunk())
        self.assertIn("TILE = 32", out)
        self.assertFalse(ed.last_apply_rejected)


if __name__ == "__main__":
    unittest.main()

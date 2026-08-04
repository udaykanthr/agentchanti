"""A fix that rewrites `self.X = ...` lines inside a method must land.

A diagnosis fix inside a method body is usually a few attribute
assignments with fresh values AND fresh comments, so nothing in it
matches the original textually and the content aligner has no anchor.
The edit was refused and the fix thrown away: observed on a Pac-Man run
where three consecutive attempts offered the same 7-line rewrite of
`Map.__init__`'s spawn coordinates and all three were dropped with "no
unambiguous match", ending the run.

The assignment TARGETS say unambiguously what the fragment replaces.
"""

import unittest

from agentchanti.editing.chunk_editor import (
    ChunkEditor, ChunkEditResponse, FileChunk)


ORIGINAL = '''\
from typing import List, Tuple

Tile = Tuple[int, int]


class Map:
    def __init__(self):
        self.width = 19
        self.height = 21
        # Spawn points.
        self.player_spawn: Tile = (1, 1)
        self.ghost_spawns: List[Tile] = [(2, 2), (3, 3)]
        self.pellets = set()
        self._validate()

    def _validate(self):
        return True
'''

FRAGMENT = '''\
        # Spawn points are guaranteed walkable.
        self.player_spawn: Tile = (9, 11)

        # All ghost spawns must sit on walkable corridor tiles.
        self.ghost_spawns: List[Tile] = [(9, 4), (9, 5), (9, 6), (9, 7)]
'''


def _chunk():
    lines = ORIGINAL.splitlines()
    start = next(i for i, l in enumerate(lines, 1) if "def __init__" in l)
    end = next(i for i, l in enumerate(lines, 1) if "self._validate()" in l)
    return FileChunk(file_path="map.py", chunk_id="Map.__init__",
                     content="\n".join(lines[start - 1:end]),
                     line_start=start, line_end=end,
                     chunk_type="method",
                     signature="    def __init__(self):")


def _edit(content=FRAGMENT):
    return ChunkEditResponse(file_path="map.py", chunk_id="Map.__init__",
                             line_start=0, line_end=0, new_content=content)


class AssignmentRealignmentTest(unittest.TestCase):
    def setUp(self):
        self.lines = ORIGINAL.splitlines(True)
        self.chunk = _chunk()

    def _resolve(self, content=FRAGMENT):
        return ChunkEditor._assignment_realignment(
            _edit(content), self.chunk, self.lines)

    def test_span_is_found(self):
        self.assertIsNotNone(self._resolve())

    def test_applied_edit_updates_values_and_keeps_the_rest(self):
        editor = ChunkEditor()
        result = editor.apply_chunk_edits(ORIGINAL, [_edit()], [self.chunk])
        self.assertIn("(9, 11)", result)
        self.assertNotIn("(1, 1)", result)
        # Everything around the replaced span must survive.
        for kept in ("self.width = 19", "self.pellets = set()",
                     "self._validate()", "def _validate"):
            self.assertIn(kept, result, f"{kept} must survive")

    def test_result_compiles(self):
        from agentchanti.py_syntax import is_valid_python
        editor = ChunkEditor()
        result = editor.apply_chunk_edits(ORIGINAL, [_edit()], [self.chunk])
        self.assertTrue(is_valid_python(result))

    def test_old_comment_is_replaced_not_stacked(self):
        editor = ChunkEditor()
        result = editor.apply_chunk_edits(ORIGINAL, [_edit()], [self.chunk])
        self.assertNotIn("# Spawn points.\n", result)
        self.assertIn("guaranteed walkable", result)

    # ── refusals ──────────────────────────────────────────────────────
    def test_refuses_when_an_attribute_is_unknown(self):
        """A fragment adding a NEW attribute is additive, not a
        replacement — placing it here would be a guess."""
        self.assertIsNone(self._resolve(
            "        self.player_spawn: Tile = (9, 11)\n"
            "        self.brand_new = 3\n"))

    def test_refuses_without_assignments(self):
        self.assertIsNone(self._resolve("        return None\n"))

    def test_refuses_when_span_holds_other_statements(self):
        """Replacing across a real statement would delete it."""
        original = ORIGINAL.replace(
            "        self.ghost_spawns: List[Tile] = [(2, 2), (3, 3)]\n",
            "        self.compute()\n"
            "        self.ghost_spawns: List[Tile] = [(2, 2), (3, 3)]\n")
        lines = original.splitlines()
        start = next(i for i, l in enumerate(lines, 1) if "def __init__" in l)
        end = next(i for i, l in enumerate(lines, 1) if "self._validate()" in l)
        chunk = FileChunk(file_path="map.py", chunk_id="Map.__init__",
                          content="\n".join(lines[start - 1:end]),
                          line_start=start, line_end=end,
                          chunk_type="method",
                          signature="    def __init__(self):")
        self.assertIsNone(ChunkEditor._assignment_realignment(
            _edit(), chunk, original.splitlines(True)))

    def test_equality_is_not_an_assignment(self):
        self.assertIsNone(self._resolve("        if self.width == 19:\n"))


if __name__ == "__main__":
    unittest.main()

"""A `#### [EDIT]:` marker with a free-text parenthetical must still name
its file — and must not be allowed to truncate that file.

The prompt template shows "(lines start-end)", so models generalise the
parentheses into a description of the change:

    #### [EDIT]: `src/map.py` (export contract / layout alias)

Every stricter marker pattern failed on that line, the block went
anonymous, and the fix was discarded. Observed twice in one run:
diagnosis had correctly found that the module exported ``BUILTIN_MAZE``
while the gate imported ``MAZE_LAYOUT``, wrote the right fix, and the
pipeline halted anyway.

Recovering the path is only half of it. Such a marker claims the whole
file, and the block is often just one section — applying a section as a
full-file replacement deletes the rest and still leaves valid Python, so
no syntax gate downstream can catch it.
"""

import unittest

from agentchanti.editing.chunk_editor import (
    DESCRIPTIVE_CHUNK_ID, ChunkEditor, _drops_module_symbols)


ORIGINAL = '''\
"""Maze module."""

BUILTIN_MAZE = [[1, 1], [1, 0]]

TILE_SIZE = 24


class Map:
    def __init__(self, layout):
        self.layout = layout

    def is_walkable(self, x, y):
        return self.layout[y][x] == 0
'''

# A section only — no `Map`. Applying this whole would delete the class.
FRAGMENT = '''\
BUILTIN_MAZE = [[1, 1], [1, 0]]
MAZE_LAYOUT = BUILTIN_MAZE
TILE_SIZE = 24
'''

# A genuine whole-file rewrite: keeps every top-level name.
WHOLE_FILE = '''\
"""Maze module."""

BUILTIN_MAZE = [[1, 1], [1, 0]]
MAZE_LAYOUT = BUILTIN_MAZE

TILE_SIZE = 24


class Map:
    def __init__(self, layout):
        self.layout = layout

    def is_walkable(self, x, y):
        return self.layout[y][x] == 0
'''


def _response(marker: str, body: str) -> str:
    return f"{marker}\n```python\n{body}```\n"


class DropsModuleSymbolsTest(unittest.TestCase):
    def test_fragment_drops_the_class(self):
        self.assertEqual(_drops_module_symbols(ORIGINAL, FRAGMENT), {"Map"})

    def test_whole_file_drops_nothing(self):
        self.assertEqual(_drops_module_symbols(ORIGINAL, WHOLE_FILE), set())

    def test_unparseable_replacement_is_treated_as_dropping(self):
        self.assertTrue(_drops_module_symbols(ORIGINAL, "def f(:\n"))


class DescriptiveMarkerParseTest(unittest.TestCase):
    def setUp(self):
        self.editor = ChunkEditor()

    def _parse(self, marker):
        return self.editor.parse_chunk_response(
            _response(marker, WHOLE_FILE))

    def test_backticked_path_with_description(self):
        edits = self._parse(
            "#### [EDIT]: `src/map.py` (export contract / layout alias)")
        self.assertIsNotNone(edits, "marker must not go anonymous")
        self.assertEqual(edits[0].file_path, "src/map.py")
        self.assertEqual(edits[0].chunk_id, DESCRIPTIVE_CHUNK_ID)

    def test_plain_path_with_description(self):
        edits = self._parse(
            "#### [EDIT]: src/map.py (export contract / layout alias)")
        self.assertIsNotNone(edits)
        self.assertEqual(edits[0].file_path, "src/map.py")

    def test_quoted_path_with_description(self):
        edits = self._parse('#### [EDIT]: "src/map.py" (adds the alias)')
        self.assertIsNotNone(edits)
        self.assertEqual(edits[0].file_path, "src/map.py")

    def test_existing_markers_still_take_precedence(self):
        """A numeric range must keep its precise meaning."""
        edits = self.editor.parse_chunk_response(
            _response("#### [EDIT]: src/map.py:Map (lines 8-14)", WHOLE_FILE))
        self.assertIsNotNone(edits)
        self.assertNotEqual(edits[0].chunk_id, DESCRIPTIVE_CHUNK_ID)


class DescriptiveMarkerApplyTest(unittest.TestCase):
    def setUp(self):
        self.editor = ChunkEditor()

    def _apply(self, body):
        edits = self.editor.parse_chunk_response(_response(
            "#### [EDIT]: `src/map.py` (export contract / layout alias)",
            body))
        self.assertIsNotNone(edits)
        return self.editor.apply_chunk_edits(ORIGINAL, edits)

    def test_whole_file_rewrite_is_applied(self):
        result = self._apply(WHOLE_FILE)
        self.assertIn("MAZE_LAYOUT", result)
        self.assertIn("class Map", result)
        self.assertFalse(self.editor.last_apply_rejected)

    def test_fragment_is_refused_not_applied(self):
        """The whole point: never silently delete `Map`."""
        result = self._apply(FRAGMENT)
        self.assertEqual(result, ORIGINAL)
        self.assertIn("class Map", result)
        self.assertTrue(self.editor.last_apply_rejected)


if __name__ == "__main__":
    unittest.main()

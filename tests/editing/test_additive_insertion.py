"""An addition replaces nothing, so it must be inserted, not matched.

Diagnosis answered a missing attribute with two new `@property` methods:

    #### [EDIT]: player.py:Player (tile properties section)
        @property
        def tile(self) -> Tile: ...
        @property
        def tile_pos(self) -> Tile: ...

Content alignment declined it (multi-line), nothing else could place it,
and the edit was discarded — twice — before the run halted. Alignment was
never going to work: an addition has no counterpart in the original to
align against.

The discriminator is NAMES, not similarity. A fragment defining only
symbols the chunk lacks can be appended. The moment it redefines an
existing member it is a replacement, and appending would leave two
definitions with the last silently winning — worse than refusing.
"""

from __future__ import annotations

import unittest

from agentchanti.editing.chunk_editor import ChunkEditor

SRC = '''from typing import Tuple

Tile = Tuple[int, int]


class Player:
    def __init__(self, game_map):
        self.map = game_map
        self.tile_col = 0
        self.tile_row = 0

    def update(self, dt, keys):
        self.tile_col += 1

    def reset(self):
        self.tile_col = 0


class Ghost:
    pass
'''

ADDITION = ("    @property\n"
            "    def tile(self) -> Tile:\n"
            "        return (self.tile_col, self.tile_row)\n"
            "\n"
            "    @property\n"
            "    def tile_pos(self) -> Tile:\n"
            "        return self.tile\n")


def _apply(body, marker="player.py:Player (tile properties section)",
           src=SRC):
    ce = ChunkEditor()
    diag = f"2. FIX:\n\n#### [EDIT]: {marker}\n```python\n{body}```\n"
    edits = ce.parse_chunk_response(diag)
    known = ce.chunk_file("player.py", src)
    return ce, ce.apply_chunk_edits(src, edits, known_chunks=known)


class TestAdditiveInsertion(unittest.TestCase):

    def test_new_members_are_inserted(self):
        ce, out = _apply(ADDITION)
        self.assertFalse(ce.last_apply_rejected)
        self.assertNotEqual(out, SRC)
        self.assertIn("def tile_pos", out)
        compile(out, "player.py", "exec")

    def test_nothing_is_deleted(self):
        """An insertion must not consume the lines it lands after."""
        _, out = _apply(ADDITION)
        for kept in ("def __init__", "def update", "def reset",
                     "class Ghost", "from typing import Tuple"):
            self.assertIn(kept, out, kept)

    def test_it_lands_inside_the_class(self):
        ns: dict = {}
        _, out = _apply(ADDITION)
        exec(compile(out, "player.py", "exec"), ns)
        p = ns["Player"](None)
        self.assertEqual(p.tile, (0, 0))
        self.assertEqual(p.tile_pos, (0, 0))

    def test_a_blank_line_separates_it_from_the_previous_member(self):
        """Valid Python either way, but butting a def against the previous
        body fails every style gate."""
        _, out = _apply(ADDITION)
        lines = out.splitlines()
        i = next(n for n, l in enumerate(lines) if "@property" in l)
        self.assertEqual(lines[i - 1].strip(), "")

    def test_the_insertion_does_not_disturb_following_code(self):
        _, out = _apply(ADDITION)
        self.assertLess(out.index("def tile_pos"), out.index("class Ghost"))


class TestRefusals(unittest.TestCase):
    """It must never turn a replacement into a duplicate."""

    def test_redefining_an_existing_member_is_refused(self):
        """Appending a second `def reset` would leave two definitions with
        the last one silently winning."""
        body = "    def reset(self):\n        self.tile_col = 99\n"
        with self.assertRaises(ValueError):
            _apply(body)

    def test_a_fragment_defining_nothing_is_refused(self):
        """Bare statements appended to a class body corrupt it."""
        body = "    self.tile_col = 5\n    self.tile_row = 6\n"
        with self.assertRaises(ValueError):
            _apply(body)

    def test_a_sibling_definition_is_refused(self):
        """`class Other:` at indent 0 is not a MEMBER of `class Player:` —
        appending it would graft an unrelated class into the file under
        the guise of editing Player."""
        body = "class Other:\n    def z(self):\n        return 1\n"
        with self.assertRaises(ValueError):
            _apply(body)

    def test_a_partial_redefinition_is_refused_wholesale(self):
        """One new name plus one existing name is still a replacement —
        placing half of it would be worse than placing none."""
        body = ("    def brand_new(self):\n        return 1\n"
                "\n"
                "    def update(self, dt, keys):\n        return 2\n")
        with self.assertRaises(ValueError):
            _apply(body)


class TestExistingBehaviourIntact(unittest.TestCase):

    def test_a_whole_class_rewrite_still_replaces(self):
        body = ("class Player:\n"
                "    def __init__(self, game_map):\n"
                "        self.map = game_map\n"
                "        self.tile_col = 0\n"
                "        self.tile_row = 0\n")
        ce, out = _apply(body, marker="player.py:Player (rewritten)")
        self.assertFalse(ce.last_apply_rejected)
        self.assertEqual(out.count("class Player:"), 1)
        self.assertNotIn("def update", out)
        self.assertIn("class Ghost", out)
        compile(out, "player.py", "exec")

    def test_a_single_line_fragment_still_aligns(self):
        src = ("MAZE = [\n"
               + "".join(f"    [{i}, 0, 1, 2],\n" for i in range(8))
               + "]\n")
        ce = ChunkEditor()
        diag = ("2. FIX:\n\n#### [EDIT]: c.py:MAZE (row 3)\n"
                "```python\n    [3, 9, 9, 2],\n```\n")
        edits = ce.parse_chunk_response(diag)
        out = ce.apply_chunk_edits(src, edits,
                                   known_chunks=ce.chunk_file("c.py", src))
        self.assertEqual(out.count("    ["), 8)   # replaced, not appended
        self.assertIn("[3, 9, 9, 2],", out)


if __name__ == "__main__":
    unittest.main()

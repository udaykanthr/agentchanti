"""Tests for Executor._looks_like_code — the guard that stops a coder's
natural-language reply being written to disk as a source file.

The guard used to count any capital-initial line as a sentence. An
idiomatic Python constants module is *entirely* capital-initial lines
(``TILE_SIZE = 24``), so every generated constants.py was rejected as
prose; the CODE step then died with "No files parsed from coder
response" and no retry could ever win, because the code was correct. The
classic (non-agent-loop) path halted the whole pipeline on it.
"""

import unittest

from agentchanti.executor import Executor


CONSTANTS_PY = """\
TILE_SIZE = 24
MAZE_COLS = 28
MAZE_ROWS = 31
SCREEN_WIDTH = TILE_SIZE * MAZE_COLS
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
PLAYER_SPEED = 80.0
POWER_PELLET_DURATION = 6.0
START, PLAYING, WIN, GAME_OVER = range(4)
"""

PROSE = """\
The game is organised into four classes.
This module defines the maze layout and its helpers.
Here is a short explanation of each one.
Please review the description carefully before continuing.
"""


class LooksLikeCodeTest(unittest.TestCase):
    def assert_code(self, content, msg=""):
        self.assertTrue(Executor._looks_like_code(content), msg)

    def assert_prose(self, content, msg=""):
        self.assertFalse(Executor._looks_like_code(content), msg)

    # ── the regression ────────────────────────────────────────────────
    def test_constants_module_is_code(self):
        self.assert_code(CONSTANTS_PY, "constants.py must not read as prose")

    def test_constants_survive_parse_code_blocks(self):
        """End to end: the block must reach the returned file map."""
        text = ("#### [FILE]: constants.py\n```python\n"
                + CONSTANTS_PY + "```")
        self.assertIn("constants.py", Executor.parse_code_blocks(text))

    def test_all_caps_enum_block_is_code(self):
        self.assert_code("RED = 1\nGREEN = 2\nBLUE = 3\nALPHA = 4\n")

    # ── the guard still guards ────────────────────────────────────────
    def test_prose_is_still_rejected(self):
        self.assert_prose(PROSE)

    def test_apology_reply_is_still_rejected(self):
        self.assert_prose(
            "Sorry, I cannot complete this request as written.\n"
            "Could you clarify which module you want changed.\n"
            "Thanks for your patience while I look into it.\n")

    def test_prose_is_rejected_by_parse_code_blocks(self):
        text = "#### [FILE]: notes.py\n```python\n" + PROSE + "```"
        self.assertNotIn("notes.py", Executor.parse_code_blocks(text))

    def test_very_long_lines_still_rejected(self):
        self.assert_prose("\n".join(["word " * 40] * 4))

    def test_empty_is_not_code(self):
        self.assert_prose("")
        self.assert_prose("   \n  \n")

    # ── other languages must not regress ──────────────────────────────
    def test_typescript_is_code(self):
        self.assert_code(
            "export const TILE = 24;\n"
            "export interface Point { x: number; y: number }\n"
            "export function move(p: Point): Point {\n"
            "  return p;\n}\n")

    def test_python_class_is_code(self):
        self.assert_code(
            "class Ghost:\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "    def update(self, dt):\n"
            "        self.x += dt\n")

    def test_capitalised_yaml_mapping_is_code(self):
        self.assert_code(
            "Name: build\nOn: push\nJobs:\n  test:\n"
            "    runs-on: ubuntu-latest\n")


if __name__ == "__main__":
    unittest.main()

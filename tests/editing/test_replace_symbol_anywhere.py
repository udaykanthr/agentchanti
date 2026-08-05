"""A named symbol does not need the line range that failed to match.

Observed live (classic mode, Pac-Man task, 2026-08-05). Step 3's diagnosis
produced a fix for ``maze.py:_generate_grid`` twice, and both times the
chunk editor refused to place it:

  attempt 1/3  Cannot place partial edit for maze.py:_generate_grid --
               13 line(s) into a 58-line chunk with no unambiguous match;
               refusing to overwrite it
  attempt 2/3  applied, did not fix it
  attempt 3/3  Cannot place partial edit ... 11 line(s) into a 59-line chunk
  -> Failed after 3 diagnosis attempts. Halting pipeline.

Refusing an ambiguous splice is right. Throwing the fix away and spending
one of three attempts is not: a single editor limitation consumed the whole
retry budget while a usable fix sat in hand. The chunk id names one symbol,
so it can be found by name instead of by line range.
"""

from __future__ import annotations

from agentchanti.editing.symbol_merge import replace_symbol_anywhere


MODULE = '''\
import random


class Maze:
    """A tile maze."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _generate_grid(self):
        return [[0] * self.width for _ in range(self.height)]

    def is_wall(self, x, y):
        return self.grid[y][x] == 1


def helper():
    return 1
'''


def test_replaces_a_method_inside_a_class():
    fragment = (
        "def _generate_grid(self):\n"
        "    grid = [[1] * self.width for _ in range(self.height)]\n"
        "    grid[1][1] = 0\n"
        "    return grid\n"
    )
    out = replace_symbol_anywhere(MODULE, fragment)
    assert out is not None
    assert "grid[1][1] = 0" in out
    # The replacement is indented back into the class body.
    assert "    def _generate_grid(self):" in out
    assert "        grid = [[1]" in out
    # Everything the fragment did not mention survives.
    assert "def is_wall(self, x, y):" in out
    assert "def helper():" in out
    assert "import random" in out
    compile(out, "<merged>", "exec")


def test_replaces_a_module_level_function():
    out = replace_symbol_anywhere(MODULE, "def helper():\n    return 42\n")
    assert out is not None
    assert "return 42" in out
    assert "class Maze:" in out
    compile(out, "<merged>", "exec")


def test_an_already_indented_fragment_is_handled():
    fragment = (
        "    def is_wall(self, x, y):\n"
        "        return bool(self.grid[y][x])\n"
    )
    out = replace_symbol_anywhere(MODULE, fragment)
    assert out is not None
    assert "return bool(self.grid[y][x])" in out
    compile(out, "<merged>", "exec")


# ── refusals: a guess is worse than no fix ────────────────────────────

def test_refuses_an_unknown_symbol():
    assert replace_symbol_anywhere(
        MODULE, "def not_in_the_file():\n    return 1\n") is None


def test_refuses_when_the_fragment_defines_several_symbols():
    # Which owner each belongs to would be a guess; merge_class_members
    # and merge_module_symbols handle that case with more context.
    fragment = ("def helper():\n    return 2\n\n"
                "def other():\n    return 3\n")
    assert replace_symbol_anywhere(MODULE, fragment) is None


def test_refuses_an_ambiguous_name_in_two_classes():
    src = ('class A:\n    def run(self):\n        return 1\n\n'
           'class B:\n    def run(self):\n        return 2\n')
    assert replace_symbol_anywhere(src, "def run(self):\n    return 9\n") is None


def test_refuses_unparseable_input():
    assert replace_symbol_anywhere("def broken(:\n", "def helper():\n    x=1\n") is None
    assert replace_symbol_anywhere(MODULE, "def broken(:\n") is None


def test_refuses_when_the_result_would_not_compile():
    # A fragment that parses alone but breaks the file once indented into
    # the class body must not be applied.
    bad = "def _generate_grid(self):\n    return (\n"
    assert replace_symbol_anywhere(MODULE, bad) is None


def test_a_module_level_name_wins_over_a_method_of_the_same_name():
    src = ('def run():\n    return 1\n\n'
           'class A:\n    def run(self):\n        return 2\n')
    out = replace_symbol_anywhere(src, "def run():\n    return 9\n")
    assert out is not None
    assert "return 9" in out
    # The method is untouched.
    assert "        return 2" in out
    compile(out, "<merged>", "exec")

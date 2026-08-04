"""A partial rewrite labelled as a whole file must be merged, not dropped.

Diagnosis answers "add is_walkable" with `#### [FILE]: src/map.py` and a
body holding only the definitions it touched. The structural guard sees
25 of 30 definitions about to vanish and refuses — correct, but the step
then has no fix and the run halts. The replacement is a set of
definitions, not a file; merging by name applies what the model changed
and keeps what it never mentioned.
"""

import unittest

from agentchanti.editing.symbol_merge import merge_module_symbols
from agentchanti.py_syntax import is_valid_python


ORIGINAL = '''\
"""Maze module."""

import math

TILE = 24


def helper(x):
    return x * 2


class Map:
    def __init__(self, layout):
        self.layout = layout

    def is_walkable_tile(self, x, y):
        return self.layout[y][x] == 0


class Other:
    pass
'''

# Only Map — everything else omitted.
PARTIAL = '''\
class Map:
    def __init__(self, layout):
        self.layout = layout

    def is_walkable(self, x, y):
        return self.layout[y][x] == 0
'''


class MergeTest(unittest.TestCase):
    def test_partial_rewrite_keeps_untouched_symbols(self):
        merged = merge_module_symbols(ORIGINAL, PARTIAL)
        self.assertIsNotNone(merged)
        for kept in ("def helper", "class Other", "TILE = 24", "import math"):
            self.assertIn(kept, merged, f"{kept} must survive the merge")

    def test_partial_rewrite_applies_the_change(self):
        merged = merge_module_symbols(ORIGINAL, PARTIAL)
        self.assertIn("def is_walkable(self", merged)

    def test_unmentioned_methods_survive(self):
        """The bug this exists for: most "deleted definitions" the
        structural guard counts are METHODS. Replacing a class wholesale
        loses every method the model did not restate, so a same-named
        class is merged member by member."""
        merged = merge_module_symbols(ORIGINAL, PARTIAL)
        self.assertIn("is_walkable_tile", merged)
        self.assertIn("def is_walkable(self", merged)

    def test_redefined_method_is_replaced_not_duplicated(self):
        repl = ("class Map:\n"
                "    def is_walkable_tile(self, x, y):\n"
                "        return True  # rewritten\n")
        merged = merge_module_symbols(ORIGINAL, repl)
        self.assertIn("rewritten", merged)
        self.assertEqual(merged.count("def is_walkable_tile"), 1)
        self.assertIn("def __init__", merged)

    def test_result_compiles(self):
        self.assertTrue(is_valid_python(merge_module_symbols(ORIGINAL,
                                                             PARTIAL)))

    def test_new_symbols_are_appended(self):
        merged = merge_module_symbols(
            ORIGINAL, PARTIAL + "\n\ndef brand_new():\n    return 1\n")
        self.assertIn("def brand_new", merged)
        self.assertIn("class Other", merged)
        self.assertTrue(is_valid_python(merged))

    def test_new_imports_are_added(self):
        merged = merge_module_symbols(
            ORIGINAL, "from collections import deque\n\n" + PARTIAL)
        self.assertIn("from collections import deque", merged)
        self.assertIn("import math", merged)
        self.assertTrue(is_valid_python(merged))

    def test_existing_import_not_duplicated(self):
        merged = merge_module_symbols(ORIGINAL, "import math\n\n" + PARTIAL)
        self.assertEqual(merged.count("import math"), 1)

    def test_decorated_definition_replaced_whole(self):
        original = ("import functools\n\n\n"
                    "@functools.cache\n"
                    "def f(x):\n    return 1\n\n\n"
                    "def g():\n    return 2\n")
        merged = merge_module_symbols(
            original, "@functools.cache\ndef f(x):\n    return 99\n")
        self.assertIn("return 99", merged)
        self.assertNotIn("return 1", merged)
        self.assertEqual(merged.count("@functools.cache"), 1)
        self.assertIn("def g", merged)

    # ── refusals ──────────────────────────────────────────────────────
    def test_unparseable_replacement_refused(self):
        self.assertIsNone(merge_module_symbols(ORIGINAL, "def f(:\n"))

    def test_unparseable_original_refused(self):
        self.assertIsNone(merge_module_symbols("def f(:\n", PARTIAL))

    def test_unrelated_module_refused(self):
        """No shared names means this is not the same module."""
        self.assertIsNone(merge_module_symbols(
            ORIGINAL, "def totally_unrelated():\n    return 0\n"))


if __name__ == "__main__":
    unittest.main()

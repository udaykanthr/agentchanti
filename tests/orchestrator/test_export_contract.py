"""A step's declared exports are a contract its verify command depends on.

A plan step declares ``exports:`` and its ``verify:`` imports exactly
those names. When the coder invents its own naming the gate CANNOT pass,
however many times it is retried — and the diagnosis sees only
"ImportError: cannot import name X", with nothing pointing at the
contract.

Observed on a real run: the step promised ``tile_to_pixel_center`` and
``is_at_tile_center``; the file defined ``pixel_center_for_tile`` and
``is_aligned_to_tile_center``. Same behaviour, different words. Two
diagnosis rounds ran and the pipeline halted without either of them
being told what was actually wrong.
"""

from __future__ import annotations

import unittest

from agentchanti.orchestrator.memory import FileMemory
from agentchanti.orchestrator.step_handlers import _broken_export_promise

SRC = '''TILE_SIZE = 24


def pixel_to_tile(x, y):
    return (int(x // TILE_SIZE), int(y // TILE_SIZE))


def pixel_center_for_tile(c, r):
    return (c * TILE_SIZE, r * TILE_SIZE)


def snap_to_tile_center(x, y):
    return (x, y)


def is_aligned_to_tile_center(x, y):
    return True
'''

IMPORT_ERR = ("ImportError: cannot import name 'tile_to_pixel_center' "
              "from 'tile_utils'. Did you mean: 'snap_to_tile_center'?")


class _Step:
    language = "python"
    target_files = ["tile_utils.py"]

    def __init__(self, exports):
        self.exports = exports


def _mem(src=SRC, path="tile_utils.py"):
    m = FileMemory()
    m.update({path: src})
    return m


DECLARED = ["pixel_to_tile", "tile_to_pixel_center",
            "snap_to_tile_center", "is_at_tile_center"]


class TestDetection(unittest.TestCase):

    def test_the_broken_promise_is_named(self):
        msg = _broken_export_promise(_Step(DECLARED), _mem(), IMPORT_ERR)
        self.assertIn("BROKEN EXPORT CONTRACT", msg)
        self.assertIn("missing: tile_to_pixel_center", msg)

    def test_every_missing_export_is_reported_at_once(self):
        """Python raises on the first bad name only; fixing them one per
        round trip burns a diagnosis attempt each time."""
        msg = _broken_export_promise(_Step(DECLARED), _mem(), IMPORT_ERR)
        self.assertIn("missing: is_at_tile_center", msg)

    def test_what_the_file_does_define_is_included(self):
        msg = _broken_export_promise(_Step(DECLARED), _mem(), IMPORT_ERR)
        self.assertIn("pixel_center_for_tile", msg)
        self.assertIn("is_aligned_to_tile_center", msg)

    def test_no_mapping_is_asserted(self):
        """difflib matched tile_to_pixel_center -> is_aligned_to_tile_center.
        A confident wrong hint renames the wrong function, so the message
        states the two lists and leaves the mapping to the model."""
        msg = _broken_export_promise(_Step(DECLARED), _mem(), IMPORT_ERR)
        self.assertNotIn("closest", msg.lower())

    def test_exports_that_ARE_defined_are_not_listed(self):
        msg = _broken_export_promise(_Step(DECLARED), _mem(), IMPORT_ERR)
        self.assertNotIn("missing: pixel_to_tile\n", msg)
        self.assertNotIn("missing: snap_to_tile_center", msg)


class TestSilence(unittest.TestCase):
    """It must never mislabel a failure it cannot explain."""

    def test_an_unrelated_gate_failure_says_nothing(self):
        self.assertEqual(
            _broken_export_promise(_Step(DECLARED), _mem(),
                                   "AssertionError: pellet count 0"), "")

    def test_a_step_declaring_no_exports_says_nothing(self):
        self.assertEqual(
            _broken_export_promise(_Step([]), _mem(), IMPORT_ERR), "")

    def test_a_name_error_about_a_defined_symbol_says_nothing(self):
        """The symbol exists, so the contract is intact and the real cause
        is elsewhere."""
        self.assertEqual(
            _broken_export_promise(
                _Step(["snap_to_tile_center"]), _mem(),
                "cannot import name 'snap_to_tile_center' from 'x'"), "")

    def test_an_unreadable_file_says_nothing(self):
        """No evidence beats bad evidence: an extractor that saw nothing
        must not be read as 'the file defines nothing'."""
        self.assertEqual(
            _broken_export_promise(_Step(DECLARED), FileMemory(),
                                   IMPORT_ERR), "")

    def test_an_error_naming_an_undeclared_symbol_says_nothing(self):
        self.assertEqual(
            _broken_export_promise(
                _Step(DECLARED), _mem(),
                "cannot import name 'something_else' from 'x'"), "")


class TestOtherErrorShapes(unittest.TestCase):

    def test_attribute_error_is_recognised(self):
        msg = _broken_export_promise(
            _Step(DECLARED), _mem(),
            "AttributeError: module 'tile_utils' has no attribute "
            "'is_at_tile_center'")
        self.assertIn("missing: is_at_tile_center", msg)

    def test_name_error_is_recognised(self):
        msg = _broken_export_promise(
            _Step(DECLARED), _mem(),
            "NameError: name 'tile_to_pixel_center' is not defined")
        self.assertIn("missing: tile_to_pixel_center", msg)


if __name__ == "__main__":
    unittest.main()

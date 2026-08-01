"""Relevance-ordered package grounding.

The installed-packages prompt block used to be `sorted(versions)[:40]` —
alphabetical, so on any environment past ~40 packages the cap dropped
everything after the letter 'e'. Measured on a 178-package interpreter it
kept up to `email-validator` and discarded pygame, pytest, requests and
rich: a block headed "write code against these EXACT versions" that
omitted the library the project actually imports.
"""
from __future__ import annotations

import unittest

from agentchanti.orchestrator.api_grounding import (
    _imported_top_levels, grounding_packages,
)


def _many(n: int) -> dict:
    """A synthetic environment big enough to trigger the cap."""
    v = {f"aaa-filler-{i:03d}": "1.0" for i in range(n)}
    v.update({"pygame": "2.6.1", "pytest": "9.0.2", "requests": "2.32.5"})
    return v


class TestImportedTopLevels(unittest.TestCase):

    def test_plain_and_from_imports(self):
        self.assertEqual(
            _imported_top_levels(["import pygame\nfrom flask import Flask\n"]),
            {"pygame", "flask"})

    def test_multi_import_on_one_line(self):
        """`import a, b` used to yield only `a`, silently dropping b."""
        self.assertEqual(
            _imported_top_levels(["import unittest, pytest\n"]),
            {"unittest", "pytest"})

    def test_aliases_and_dotted_paths(self):
        self.assertEqual(
            _imported_top_levels(["import numpy as np, pandas\n"
                                  "from os.path import join\n"]),
            {"numpy", "pandas", "os"})

    def test_unparseable_source_is_survivable(self):
        self.assertEqual(_imported_top_levels(["def f(:\n"]), set())
        self.assertEqual(_imported_top_levels([None, ""]), set())


class TestGroundingPackages(unittest.TestCase):

    def test_imported_packages_survive_the_cap(self):
        mem = {"main.py": "import pygame\n",
               "tests/t.py": "import pytest\nimport requests\n"}
        out = grounding_packages(_many(200), mem, limit=40)
        self.assertEqual(len(out), 40)
        for want in ("pygame==2.6.1", "pytest==9.0.2", "requests==2.32.5"):
            self.assertIn(want, out)

    def test_relevant_packages_come_first(self):
        out = grounding_packages(_many(200), {"main.py": "import pygame\n"})
        self.assertTrue(out[0].startswith("pygame=="), out[:3])

    def test_a_project_local_module_is_not_a_package(self):
        """`from src.map import Map` must not promote a PyPI 'src'."""
        versions = dict(_many(5))
        versions["src"] = "9.9.9"
        out = grounding_packages(
            versions, {"src/map.py": "class Map: pass\n",
                       "main.py": "from src.map import Map\n"})
        self.assertNotIn("src==9.9.9", out)

    def test_build_tooling_is_excluded(self):
        out = grounding_packages(
            {"pip": "25.1", "setuptools": "80.0", "wheel": "0.45",
             "pygame": "2.6.1"}, None)
        self.assertEqual(out, ["pygame==2.6.1"])

    def test_small_environment_is_unchanged(self):
        out = grounding_packages({"pygame": "2.6.1", "flask": "3.1.2"}, None)
        self.assertEqual(sorted(out), ["flask==3.1.2", "pygame==2.6.1"])

    def test_accepts_a_filememory_like_object(self):
        class Mem:
            def all_files(self):
                return {"main.py": "import pygame\n"}

        out = grounding_packages(_many(200), Mem())
        self.assertIn("pygame==2.6.1", out)

    def test_a_broken_memory_degrades_instead_of_emptying_the_block(self):
        """Relevance is a bonus; losing it must not lose the block."""
        class Broken:
            def as_dict(self):
                raise RuntimeError("gone")

        out = grounding_packages(_many(50), Broken())
        self.assertEqual(len(out), 40)

    def test_empty_inputs(self):
        self.assertEqual(grounding_packages({}, None), [])
        self.assertEqual(grounding_packages(None, None), [])


if __name__ == "__main__":
    unittest.main()

"""Tests for the plan's declared-component graph (orchestrator/plan_graph.py)."""

from __future__ import annotations

import unittest

from agentchanti.orchestrator.plan_graph import PlanGraph, module_key
from agentchanti.orchestrator.plan_step import PlanStep


def _map_player_ghost():
    return [
        PlanStep(id="2.1", step_type="CODE", index=0,
                 target_files=["src/map.py"], exports=["Map", "TILE_SIZE"]),
        PlanStep(id="2.2", step_type="CODE", index=1,
                 target_files=["src/player.py"], exports=["Player"]),
        PlanStep(id="2.3", step_type="CODE", index=2,
                 target_files=["src/ghost.py"], exports=["Ghost"]),
    ]


class TestModuleKey(unittest.TestCase):

    def test_every_spelling_reduces_to_one_identity(self):
        for spec in ("src/map.py", "src\\map.py", "src.map", "src.map.py",
                     "./src/map.py", "src//map.py", "src/map"):
            with self.subTest(spec=spec):
                self.assertEqual(module_key(spec), "src/map")

    def test_dots_in_a_path_are_not_package_separators(self):
        """`src/my.utils.py` is a filename, not a package path."""
        self.assertEqual(module_key("src/my.utils.py"), "src/my.utils")


class TestResolution(unittest.TestCase):
    """Every import spelling observed across the benchmark runs."""

    def setUp(self):
        self.graph = PlanGraph(_map_player_ghost())

    def test_resolves_all_observed_spellings(self):
        for spec in ("src/map.py", "src\\map.py", "src.map", "src.map.py",
                     "map.py", "./src/map.py", "src/map"):
            with self.subTest(spec=spec):
                self.assertEqual(self.graph.producer_of(spec, ["Map"]), "2.1")

    def test_symbol_rescues_a_spelling_no_path_logic_can(self):
        """`src.map.Map` glues the symbol onto the module path.

        Resolution by exported symbol is notation-independent, which is
        the whole reason the graph keys on symbols at all.
        """
        self.assertEqual(self.graph.producer_of("src.map.Map", ["Map"]), "2.1")

    def test_symbol_alone_resolves_an_unrecognisable_path(self):
        self.assertEqual(
            self.graph.producer_of("totally/unknown/spelling", ["Ghost"]),
            "2.3")

    def test_unknown_spec_resolves_to_nothing(self):
        self.assertIsNone(self.graph.producer_of("numpy", ["array"]))
        self.assertIsNone(self.graph.producer_of("", []))

    def test_ambiguous_symbol_is_not_used(self):
        steps = [
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["a/util.py"], exports=["helper"]),
            PlanStep(id="2.2", step_type="CODE", index=1,
                     target_files=["b/util.py"], exports=["helper"]),
        ]
        self.assertIsNone(PlanGraph(steps).producer_of("mystery", ["helper"]))

    def test_explicit_directory_never_falls_back_to_basename(self):
        """`src/public/index.js` must not bind to `src/admin/index.js`."""
        steps = [
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["src/admin/index.js"]),
            PlanStep(id="2.2", step_type="CODE", index=1,
                     target_files=["src/public/page.js"]),
        ]
        graph = PlanGraph(steps)
        self.assertIsNone(graph.producer_of("src/public/index.js", []))

    def test_ambiguous_basename_is_not_used(self):
        steps = [
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["a/util.py"]),
            PlanStep(id="2.2", step_type="CODE", index=1,
                     target_files=["b/util.py"]),
        ]
        self.assertIsNone(PlanGraph(steps).producer_of("util.py", []))

    def test_placeholder_targets_are_ignored(self):
        steps = [PlanStep(id="4.2", step_type="CMD", index=0,
                          target_files=["none"])]
        self.assertEqual(PlanGraph(steps).nodes, [])


class TestLifecycle(unittest.TestCase):
    """Nodes are intent first, fact later."""

    def setUp(self):
        self.graph = PlanGraph(_map_player_ghost())

    def test_nodes_start_planned(self):
        self.assertTrue(all(n.status == "planned" for n in self.graph.nodes))
        self.assertEqual(len(self.graph.pending_paths()), 3)

    def test_building_then_built(self):
        self.graph.mark_building("2.1")
        self.assertEqual(self.graph.nodes[0].status, "building")
        self.graph.mark_built("2.1", ["Map", "TILE_SIZE"])
        self.assertEqual(self.graph.nodes[0].status, "built")
        self.assertNotIn("src/map.py", self.graph.pending_paths())

    def test_reconcile_reports_a_broken_export_promise(self):
        self.graph.mark_built("2.1", ["Map"])          # TILE_SIZE missing
        self.assertEqual(self.graph.reconcile("2.1"), ["TILE_SIZE"])

    def test_reconcile_clean_when_promises_kept(self):
        self.graph.mark_built("2.1", ["Map", "TILE_SIZE", "extra"])
        self.assertEqual(self.graph.reconcile("2.1"), [])

    def test_no_actual_exports_is_not_evidence_of_absence(self):
        """An unreadable or unparseable file must not be reported as a
        step that dropped every export it declared."""
        self.graph.mark_built("2.1", [])
        self.assertEqual(self.graph.reconcile("2.1"), [])

    def test_reconcile_ignores_steps_that_have_not_been_built(self):
        self.assertEqual(self.graph.reconcile("2.1"), [])


class TestDefaultExportVocabulary(unittest.TestCase):
    """`exports: default Footer` vs extracted `['Footer', 'default']`.

    Planners spell a default export as `default Foo`; the JS extractor
    reports the flag `default` alongside the name. Comparing the two
    literally warned that `default Footer` was missing from a file that
    exported precisely that — on six-plus consecutive runs, never once
    correctly. A warning that is always wrong is worse than none: it
    trains the reader to skip the line that will one day be right.
    """

    def _satisfied(self, spec, actual):
        from agentchanti.orchestrator.plan_graph import _export_satisfied
        return _export_satisfied(spec, set(actual))

    def test_the_warnings_seen_in_real_runs_are_silenced(self):
        for spec, actual in (
            ("default Footer", ["Footer", "default"]),
            ("default App", ["App", "default"]),
            ("default HomePage", ["HomePage", "default"]),
            ("App as default", ["default"]),
        ):
            with self.subTest(spec=spec):
                self.assertTrue(self._satisfied(spec, actual))

    def test_a_bare_name_matches_a_file_that_only_default_exports_it(self):
        # `function App() {}; export default App` — the extractor keeps
        # the flag and loses the name.
        self.assertTrue(self._satisfied("App", ["default"]))

    def test_plain_named_exports_still_match(self):
        self.assertTrue(self._satisfied("Footer", ["Footer", "default"]))

    def test_a_genuinely_missing_export_is_still_reported(self):
        """The teeth must survive: this is the case the check exists for."""
        self.assertFalse(self._satisfied("TILE_SIZE", ["Map"]))
        self.assertFalse(self._satisfied("Missing", ["Footer", "default"]))

    def test_a_named_export_absent_from_a_multi_export_file_still_warns(self):
        # `default` is present, but so are other names — so the file is
        # not the "only a default export" shape, and a missing name is
        # a real finding rather than a vocabulary artefact.
        self.assertFalse(self._satisfied("Sidebar", ["Footer", "default"]))

    def test_module_level_constants_count_as_exports(self):
        """The extractor used to see only class/def, so a step declaring
        TILE_SIZE, START, WIN … was reported as defining none of them —
        while the acceptance gate importing exactly those symbols passed."""
        from agentchanti.language_backend import get_backend
        source = (
            "import pygame\n"
            "TILE_SIZE = 24\n"
            "TILE_WALL = 0\n"
            "TILE_PELLET: int = 2\n"
            "START = 'start'\n"
            "class Map:\n"
            "    INNER = 1\n"
            "    def __init__(self):\n"
            "        local = 2\n"
            "def helper():\n"
            "    other = 3\n"
            "_private = 4\n"
        )
        found = set(get_backend("python").extract_exports(source))
        for name in ("TILE_SIZE", "TILE_WALL", "TILE_PELLET", "START",
                     "Map", "helper"):
            self.assertIn(name, found)
        # Class-body and function-local names are not module attributes.
        for name in ("INNER", "local", "other", "_private"):
            self.assertNotIn(name, found)

    def test_reconcile_is_clean_for_a_constants_module(self):
        graph = PlanGraph([
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["config.py"],
                     exports=["TILE_SIZE", "SCREEN_WIDTH"]),
        ])
        from agentchanti.language_backend import get_backend
        actual = get_backend("python").extract_exports(
            "TILE_SIZE = 24\nSCREEN_WIDTH = 640\n")
        graph.mark_built("2.1", actual)
        self.assertEqual(graph.reconcile("2.1"), [])


class TestUnresolvedImports(unittest.TestCase):

    def test_lists_imports_no_step_produces(self):
        steps = _map_player_ghost()
        steps[1].imports_from = {"src.map.py": ["Map"], "pygame": ["Rect"]}
        graph = PlanGraph(steps)
        gaps = graph.unresolved_imports(steps)
        self.assertEqual(gaps, [("2.2", "pygame")])


if __name__ == "__main__":
    unittest.main()

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


class TestUnresolvedImports(unittest.TestCase):

    def test_lists_imports_no_step_produces(self):
        steps = _map_player_ghost()
        steps[1].imports_from = {"src.map.py": ["Map"], "pygame": ["Rect"]}
        graph = PlanGraph(steps)
        gaps = graph.unresolved_imports(steps)
        self.assertEqual(gaps, [("2.2", "pygame")])


if __name__ == "__main__":
    unittest.main()

"""Tests for checkpoint save/load with PlanStep support."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from agentchanti.checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint


class TestCheckpointBasic(unittest.TestCase):
    """Existing checkpoint functionality should work unchanged."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test_checkpoint.json")

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_save_and_load(self):
        save_checkpoint(self.filepath, task="build app", steps=["step 1", "step 2"],
                        completed_step=0, file_memory_dict={"a.py": "code"},
                        step_results={0: "done"}, language="python")
        state = load_checkpoint(self.filepath)
        self.assertIsNotNone(state)
        self.assertEqual(state["task"], "build app")
        self.assertEqual(state["steps"], ["step 1", "step 2"])
        self.assertEqual(state["completed_step"], 0)
        self.assertEqual(state["step_results"], {0: "done"})

    def test_load_missing_file(self):
        self.assertIsNone(load_checkpoint("/nonexistent/path.json"))

    def test_clear(self):
        save_checkpoint(self.filepath, task="t", steps=[], completed_step=0,
                        file_memory_dict={}, step_results={}, language="python")
        self.assertTrue(os.path.exists(self.filepath))
        clear_checkpoint(self.filepath)
        self.assertFalse(os.path.exists(self.filepath))

    def test_dependencies_saved(self):
        deps = {1: {0}, 2: {0, 1}}
        save_checkpoint(self.filepath, task="t", steps=["a", "b", "c"],
                        completed_step=1, file_memory_dict={},
                        step_results={0: "done", 1: "done"}, language="js",
                        dependencies=deps)
        state = load_checkpoint(self.filepath)
        self.assertIn("dependencies", state)
        self.assertEqual(state["dependencies"]["1"], [0])


class TestCheckpointPlanSteps(unittest.TestCase):
    """PlanStep save/load through checkpoint."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test_checkpoint.json")

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def _make_step(self, **kwargs):
        from agentchanti.orchestrator.plan_step import PlanStep
        defaults = {
            "id": "1.1", "step_type": "CODE", "description": "",
            "depends_on": [], "index": 0,
        }
        defaults.update(kwargs)
        return PlanStep(**defaults)

    def test_plan_steps_saved_as_dicts(self):
        """PlanStep objects should be serialized via to_dict()."""
        steps = [
            self._make_step(id="1.1", step_type="CMD", description="Install deps",
                            command="npm install express"),
            self._make_step(id="2.1", step_type="CODE", description="Create server",
                            target_files=["src/server.js"],
                            exports=["app", "startServer"],
                            imports_from={"express": ["express"]},
                            depends_on=["1.1"], index=1),
        ]
        save_checkpoint(self.filepath, task="build api",
                        steps=["Install deps", "Create server"],
                        completed_step=0, file_memory_dict={},
                        step_results={0: "done"}, language="javascript",
                        plan_steps=steps)

        # Verify raw JSON structure
        with open(self.filepath, "r") as f:
            raw = json.load(f)
        self.assertIn("plan_steps", raw)
        self.assertEqual(len(raw["plan_steps"]), 2)
        self.assertEqual(raw["plan_steps"][0]["step_type"], "CMD")
        self.assertEqual(raw["plan_steps"][0]["command"], "npm install express")
        self.assertEqual(raw["plan_steps"][1]["exports"], ["app", "startServer"])

    def test_plan_steps_round_trip(self):
        """Save PlanStep → load → from_dict should preserve all fields."""
        from agentchanti.orchestrator.plan_step import PlanStep

        original = self._make_step(
            id="2.1", step_type="CODE", description="Create utils",
            target_files=["src/utils.ts"],
            exports=["formatDate", "parseUrl"],
            imports_from={"lodash": ["debounce"]},
            status="completed",
            actual_exports=["formatDate", "parseUrl", "slugify"],
            depends_on=["1.1"], index=1,
        )
        save_checkpoint(self.filepath, task="t",
                        steps=["Install", "Create utils"],
                        completed_step=1, file_memory_dict={},
                        step_results={0: "done", 1: "done"},
                        language="typescript",
                        plan_steps=[self._make_step(), original])

        state = load_checkpoint(self.filepath)
        restored = PlanStep.from_dict(state["plan_steps"][1])

        self.assertEqual(restored.id, "2.1")
        self.assertEqual(restored.step_type, "CODE")
        self.assertEqual(restored.description, "Create utils")
        self.assertEqual(restored.target_files, ["src/utils.ts"])
        self.assertEqual(restored.exports, ["formatDate", "parseUrl"])
        self.assertEqual(restored.imports_from, {"lodash": ["debounce"]})
        self.assertEqual(restored.status, "completed")
        self.assertEqual(restored.actual_exports, ["formatDate", "parseUrl", "slugify"])
        self.assertEqual(restored.depends_on, ["1.1"])
        self.assertEqual(restored.index, 1)

    def test_no_plan_steps_backward_compat(self):
        """Checkpoints without plan_steps should load fine (legacy compat)."""
        save_checkpoint(self.filepath, task="t", steps=["a"],
                        completed_step=0, file_memory_dict={},
                        step_results={0: "done"}, language="python")
        state = load_checkpoint(self.filepath)
        self.assertNotIn("plan_steps", state)

    def test_status_preserved_on_resume(self):
        """Step status (completed/pending) should survive checkpoint round-trip."""
        from agentchanti.orchestrator.plan_step import PlanStep

        steps = [
            self._make_step(id="1.1", step_type="CMD", status="completed", index=0),
            self._make_step(id="2.1", step_type="CODE", status="in_progress", index=1),
            self._make_step(id="3.1", step_type="TEST", status="pending", index=2),
        ]
        save_checkpoint(self.filepath, task="t", steps=["a", "b", "c"],
                        completed_step=0, file_memory_dict={},
                        step_results={0: "done"}, language="python",
                        plan_steps=steps)

        state = load_checkpoint(self.filepath)
        restored = [PlanStep.from_dict(d) for d in state["plan_steps"]]
        self.assertEqual(restored[0].status, "completed")
        self.assertEqual(restored[1].status, "in_progress")
        self.assertEqual(restored[2].status, "pending")


if __name__ == "__main__":
    unittest.main()

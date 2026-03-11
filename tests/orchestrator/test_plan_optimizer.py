"""
Tests for the PlanOptimizer — step merging, pruning, and dependency handling.
"""

from __future__ import annotations

import unittest


class TestOptimizePlan(unittest.TestCase):
    """Tests for the main optimize_plan function."""

    def test_empty_plan(self):
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps, deps = optimize_plan([])
        self.assertEqual(steps, [])
        self.assertEqual(deps, {})

    def test_removes_noop_steps(self):
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Analyze the project structure",
            "Create `src/app.js` with Express server setup",
            "Review the code for errors",
        ]
        result, _ = optimize_plan(steps)
        self.assertEqual(len(result), 1)
        self.assertIn("src/app.js", result[0])

    def test_merges_install_steps(self):
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Install express with `npm install express`",
            "Install cors with `npm install cors`",
            "Create `src/server.js` with Express setup",
        ]
        result, _ = optimize_plan(steps)
        # Should merge the two npm install steps
        install_steps = [s for s in result if "npm install" in s.lower()]
        self.assertEqual(len(install_steps), 1)
        self.assertIn("express", install_steps[0])
        self.assertIn("cors", install_steps[0])

    def test_preserves_dependencies(self):
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Install express with `npm install express`",
            "Create `src/server.js` with Express setup",
        ]
        deps = {1: {0}}
        result, new_deps = optimize_plan(steps, dependencies=deps)
        self.assertEqual(len(result), 2)


class TestHasFrameworkConflict(unittest.TestCase):
    """Tests for framework conflict detection."""

    def test_no_conflict_same_framework(self):
        from multi_agent_coder.orchestrator.plan_optimizer import has_framework_conflict
        self.assertFalse(has_framework_conflict({"react"}, {"react"}))

    def test_conflict_different_frameworks(self):
        from multi_agent_coder.orchestrator.plan_optimizer import has_framework_conflict
        self.assertTrue(has_framework_conflict({"react"}, {"angular"}))

    def test_no_conflict_unrelated(self):
        from multi_agent_coder.orchestrator.plan_optimizer import has_framework_conflict
        self.assertFalse(has_framework_conflict({"react"}, {"django"}))


if __name__ == "__main__":
    unittest.main()

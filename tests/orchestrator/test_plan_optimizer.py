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


class TestReorderTestInfra(unittest.TestCase):
    """Tests for test infrastructure reordering."""

    def test_moves_late_infra_before_test_writing(self):
        """Setup file and config steps after test-writing should be moved before it."""
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Create React components in `src/components/`",
            "Create unit and integration tests in `__tests__/` using vitest",
            "Add test setup file `src/setupTests.js` to import jest-dom",
            "Update `vitest.config.js` to configure Vitest with jsdom environment",
            "Add test scripts to `package.json`: `\"test\": \"vitest\"`",
        ]
        result, _ = optimize_plan(steps)

        # Find indices of key steps in result
        test_write_idx = None
        setup_idx = None
        config_idx = None
        scripts_idx = None
        for i, s in enumerate(result):
            if "unit and integration tests" in s.lower():
                test_write_idx = i
            if "setupTests" in s or "setup file" in s.lower():
                setup_idx = i
            if "vitest.config" in s.lower() and "jsdom" in s.lower():
                config_idx = i
            if "test scripts" in s.lower() and "package.json" in s.lower():
                scripts_idx = i

        self.assertIsNotNone(test_write_idx, "test-writing step not found")
        self.assertIsNotNone(setup_idx, "setup step not found")
        self.assertIsNotNone(config_idx, "config step not found")
        self.assertIsNotNone(scripts_idx, "scripts step not found")

        # All infra steps should come before the test-writing step
        self.assertLess(setup_idx, test_write_idx,
                        "setup file should come before test writing")
        self.assertLess(config_idx, test_write_idx,
                        "vitest config should come before test writing")
        self.assertLess(scripts_idx, test_write_idx,
                        "test scripts should come before test writing")

    def test_no_reorder_when_infra_already_first(self):
        """No reordering needed when infra is already before test writing."""
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Create `vitest.config.js` with jsdom environment and globals",
            "Add test setup file `vitest.setup.js`",
            "Create tests in `__tests__/` using vitest and testing-library",
        ]
        result, _ = optimize_plan(steps)
        # Order should be preserved
        self.assertEqual(len(result), 3)
        self.assertIn("vitest.config", result[0])
        self.assertIn("setup", result[1].lower())
        self.assertIn("tests", result[2].lower())

    def test_no_reorder_when_no_test_steps(self):
        """No reordering when there are no test steps."""
        from multi_agent_coder.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Create `src/App.jsx` with routing",
            "Create `src/components/Header.jsx`",
        ]
        result, _ = optimize_plan(steps)
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

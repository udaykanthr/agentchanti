"""
Tests for the PlanOptimizer — step merging, pruning, and dependency handling.
"""

from __future__ import annotations

import unittest


class TestOptimizePlan(unittest.TestCase):
    """Tests for the main optimize_plan function."""

    def test_empty_plan(self):
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
        steps, deps = optimize_plan([])
        self.assertEqual(steps, [])
        self.assertEqual(deps, {})

    def test_removes_noop_steps(self):
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Analyze the project structure",
            "Create `src/app.js` with Express server setup",
            "Review the code for errors",
        ]
        result, _ = optimize_plan(steps)
        self.assertEqual(len(result), 1)
        self.assertIn("src/app.js", result[0])

    def test_merges_install_steps(self):
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
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
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
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
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
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
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
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
        from agentchanti.orchestrator.plan_optimizer import optimize_plan
        steps = [
            "Create `src/App.jsx` with routing",
            "Create `src/components/Header.jsx`",
        ]
        result, _ = optimize_plan(steps)
        self.assertEqual(len(result), 2)


class TestOptimizeStructuredPlan(unittest.TestCase):
    """Tests for the structured PlanStep optimizer."""

    def _make_step(self, **kwargs):
        from agentchanti.orchestrator.plan_step import PlanStep
        defaults = {
            "id": "1.1", "step_type": "CODE", "description": "",
            "depends_on": [], "index": 0,
        }
        defaults.update(kwargs)
        return PlanStep(**defaults)

    def test_empty_plan(self):
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        result = optimize_structured_plan([])
        self.assertEqual(result, [])

    def test_preserves_step_type(self):
        """Step types must survive optimization (unlike legacy path)."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="1.1", step_type="CMD", description="Install deps",
                            command="npm install express"),
            self._make_step(id="2.1", step_type="CODE", description="Create server",
                            target_files=["src/server.js"],
                            exports=["app"], depends_on=["1.1"], index=1),
        ]
        result = optimize_structured_plan(steps)
        self.assertEqual(result[0].step_type, "CMD")
        self.assertEqual(result[1].step_type, "CODE")

    def test_preserves_exports_imports(self):
        """Exports and imports_from must survive optimization."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="1.1", step_type="CODE",
                            description="Create utils",
                            target_files=["src/utils.ts"],
                            exports=["formatDate", "parseUrl"]),
            self._make_step(id="2.1", step_type="CODE",
                            description="Create App",
                            target_files=["src/App.tsx"],
                            imports_from={"src/utils.ts": ["formatDate"]},
                            depends_on=["1.1"], index=1),
        ]
        result = optimize_structured_plan(steps)
        self.assertEqual(result[0].exports, ["formatDate", "parseUrl"])
        self.assertEqual(result[1].imports_from, {"src/utils.ts": ["formatDate"]})

    def test_merges_install_cmd_steps(self):
        """Multiple npm install CMD steps should merge into one."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="1.1", step_type="CMD",
                            description="Install express",
                            command="npm install express"),
            self._make_step(id="1.2", step_type="CMD",
                            description="Install cors",
                            command="npm install cors",
                            depends_on=["1.1"], index=1),
            self._make_step(id="2.1", step_type="CODE",
                            description="Create server",
                            depends_on=["1.2"], index=2),
        ]
        result = optimize_structured_plan(steps)
        # Should merge to 2 steps: 1 merged install + 1 code
        cmd_steps = [s for s in result if s.step_type == "CMD"]
        self.assertEqual(len(cmd_steps), 1)
        self.assertIn("express", cmd_steps[0].command)
        self.assertIn("cors", cmd_steps[0].command)

    def test_merge_rewires_dependencies(self):
        """After merging installs, dependents point to the merged step."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="1.1", step_type="CMD",
                            description="Install A",
                            command="npm install express"),
            self._make_step(id="1.2", step_type="CMD",
                            description="Install B",
                            command="npm install cors", index=1),
            self._make_step(id="2.1", step_type="CODE",
                            description="Use both",
                            depends_on=["1.2"], index=2),
        ]
        result = optimize_structured_plan(steps)
        code_step = [s for s in result if s.step_type == "CODE"][0]
        # Should depend on the merged install step (1.1)
        self.assertIn("1.1", code_step.depends_on)

    def test_merges_same_file_code_steps(self):
        """CODE steps targeting the same file should merge."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="2.1", step_type="CODE",
                            description="Add Header component",
                            target_files=["src/App.tsx"],
                            exports=["Header"]),
            self._make_step(id="2.2", step_type="CODE",
                            description="Add Footer component",
                            target_files=["src/App.tsx"],
                            exports=["Footer"], index=1),
        ]
        result = optimize_structured_plan(steps)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].target_files, ["src/App.tsx"])
        self.assertIn("Header", result[0].exports)
        self.assertIn("Footer", result[0].exports)
        self.assertIn("AND", result[0].description)

    def test_no_noop_removal(self):
        """Structured optimizer does NOT remove steps by description patterns.

        Unlike legacy optimizer, structured plans have explicit types.
        IGNORE steps are already typed by the LLM.
        """
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="1.1", step_type="CODE",
                            description="Analyze the project and create config",
                            target_files=["tsconfig.json"]),
        ]
        result = optimize_structured_plan(steps)
        # Should NOT be removed even though description starts with "Analyze"
        self.assertEqual(len(result), 1)

    def test_reindex_fixes_indices(self):
        """After optimization, indices should be 0..N-1."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan
        steps = [
            self._make_step(id="1.1", step_type="CMD",
                            description="Install", command="npm install express",
                            index=0),
            self._make_step(id="2.1", step_type="CODE",
                            description="Create A",
                            target_files=["src/a.ts"],
                            depends_on=["1.1"], index=5),
            self._make_step(id="3.1", step_type="TEST",
                            description="Test A",
                            depends_on=["2.1"], index=10),
        ]
        result = optimize_structured_plan(steps)
        for i, step in enumerate(result):
            self.assertEqual(step.index, i)

    def test_skip_redundant_installs_with_kb(self):
        """Packages already in KB should be removed from install commands."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan

        class MockKB:
            def is_package_installed(self, name):
                return name == "express"

        steps = [
            self._make_step(id="1.1", step_type="CMD",
                            description="Install deps",
                            command="npm install express cors"),
            self._make_step(id="2.1", step_type="CODE",
                            description="Create server",
                            depends_on=["1.1"], index=1),
        ]
        result = optimize_structured_plan(steps, knowledge_base=MockKB())
        cmd_step = [s for s in result if s.step_type == "CMD"][0]
        self.assertIn("cors", cmd_step.command)
        self.assertNotIn("express", cmd_step.command)

    def test_skip_all_installed_removes_step(self):
        """If all packages in a CMD step are installed, remove the step entirely."""
        from agentchanti.orchestrator.plan_optimizer import optimize_structured_plan

        class MockKB:
            def is_package_installed(self, name):
                return True

        steps = [
            self._make_step(id="1.1", step_type="CMD",
                            description="Install deps",
                            command="npm install express cors"),
            self._make_step(id="2.1", step_type="CODE",
                            description="Create server",
                            depends_on=["1.1"], index=1),
        ]
        result = optimize_structured_plan(steps, knowledge_base=MockKB())
        # Install step should be removed
        cmd_steps = [s for s in result if s.step_type == "CMD"]
        self.assertEqual(len(cmd_steps), 0)
        # Code step should have no dependencies (install was removed)
        code_step = [s for s in result if s.step_type == "CODE"][0]
        self.assertEqual(code_step.depends_on, [])


class TestHasFrameworkConflict(unittest.TestCase):
    """Tests for framework conflict detection."""

    def test_no_conflict_same_framework(self):
        from agentchanti.orchestrator.plan_optimizer import has_framework_conflict
        self.assertFalse(has_framework_conflict({"react"}, {"react"}))

    def test_conflict_different_frameworks(self):
        from agentchanti.orchestrator.plan_optimizer import has_framework_conflict
        self.assertTrue(has_framework_conflict({"react"}, {"angular"}))

    def test_no_conflict_unrelated(self):
        from agentchanti.orchestrator.plan_optimizer import has_framework_conflict
        self.assertFalse(has_framework_conflict({"react"}, {"django"}))


class TestNormalizeTechKeywords(unittest.TestCase):
    """Tests for normalize_tech_keywords alias resolution."""

    def test_tailwindcss_normalizes(self):
        from agentchanti.orchestrator.plan_optimizer import normalize_tech_keywords
        self.assertEqual(normalize_tech_keywords({"tailwindcss"}), {"tailwind"})

    def test_reactjs_normalizes(self):
        from agentchanti.orchestrator.plan_optimizer import normalize_tech_keywords
        self.assertEqual(normalize_tech_keywords({"reactjs"}), {"react"})

    def test_mixed_normalization(self):
        from agentchanti.orchestrator.plan_optimizer import normalize_tech_keywords
        result = normalize_tech_keywords({"tailwindcss", "react", "vitejs"})
        self.assertEqual(result, {"tailwind", "react", "vite"})

    def test_no_alias_passthrough(self):
        from agentchanti.orchestrator.plan_optimizer import normalize_tech_keywords
        self.assertEqual(normalize_tech_keywords({"django", "flask"}), {"django", "flask"})

    def test_tailwindcss_matches_tailwind_doc(self):
        """Task with 'tailwindcss' should overlap with doc tagged 'tailwind'."""
        from agentchanti.orchestrator.plan_optimizer import (
            _TECH_KEYWORDS, normalize_tech_keywords,
        )
        task_techs = normalize_tech_keywords(set(
            w.lower() for w in _TECH_KEYWORDS.findall("React, Tailwindcss, Vite")
        ))
        doc_techs = normalize_tech_keywords(set(
            w.lower() for w in _TECH_KEYWORDS.findall("tailwind css v4 setup")
        ))
        self.assertIn("tailwind", task_techs)
        self.assertIn("tailwind", doc_techs)
        self.assertTrue(bool(task_techs & doc_techs))


if __name__ == "__main__":
    unittest.main()

"""
Tests for execution wave calculation — verifies that build_step_waves
correctly groups steps into parallel waves based on dependency markers
parsed by parse_step_dependencies.
"""

import unittest

from agentchanti.executor import Executor
from agentchanti.orchestrator.pipeline import build_step_waves


class TestBuildStepWaves(unittest.TestCase):
    """Verify build_step_waves produces parallel waves from dependency info."""

    def test_parallel_waves_with_combined_markers(self):
        """LLM format: (CMD, depends: 1): should be parsed correctly."""
        raw_steps = [
            "Initialize project (CMD):",
            "Install dependency A (CMD, depends: 1):",
            "Install dependency B (CMD, depends: 1):",
            "Install dependency C (CMD, depends: 1):",
            "Configure tooling (CODE, depends: 2):",
            "Update CSS (CODE, depends: 1, 3):",
            "Update config file (CODE, depends: 1, 4):",
        ]
        steps, deps = Executor.parse_step_dependencies(raw_steps)

        waves = build_step_waves(steps, deps)

        # Step 0 has no deps → wave 0
        self.assertIn(0, waves[0])
        # Steps 1,2,3 all depend only on step 0 → same wave
        parallel_wave = waves[1]
        self.assertIn(1, parallel_wave)
        self.assertIn(2, parallel_wave)
        self.assertIn(3, parallel_wave)
        # Total waves should be < total steps (parallel grouping)
        self.assertLess(len(waves), len(steps))

    def test_standalone_depends_marker(self):
        """Standalone (depends: N) format should also work."""
        raw_steps = [
            "Step one",
            "Step two (depends: 1)",
            "Step three (depends: 1)",
        ]
        steps, deps = Executor.parse_step_dependencies(raw_steps)

        waves = build_step_waves(steps, deps)

        # Steps 1 and 2 should be in the same wave (both depend only on 0)
        self.assertEqual(waves[0], [0])
        self.assertIn(1, waves[1])
        self.assertIn(2, waves[1])

    def test_no_markers_falls_back_sequential(self):
        """Steps without any dependency markers should execute sequentially."""
        raw_steps = [
            "Step one",
            "Step two",
            "Step three",
        ]
        steps, deps = Executor.parse_step_dependencies(raw_steps)

        waves = build_step_waves(steps, deps)

        # Should be fully sequential: [[0], [1], [2]]
        self.assertEqual(waves, [[0], [1], [2]])

    def test_cleaned_steps_strip_combined_marker(self):
        """parse_step_dependencies should strip the full (CMD, depends: N): group."""
        raw_steps = [
            "Init project (CMD):",
            "Install deps (CMD, depends: 1):",
        ]
        steps, deps = Executor.parse_step_dependencies(raw_steps)

        # The full (CMD, depends: 1): should be stripped
        self.assertNotIn("depends", steps[1])
        self.assertNotIn("CMD", steps[1])
        self.assertEqual(deps[1], {0})

    def test_real_world_plan_format(self):
        """Test with the exact LLM output format from the bug report."""
        raw_steps = [
            "Initialize Vite React project in subfolder `vite-react-tailwind-spa` using the React template (CMD):",
            "Install React Router dependency (CMD, depends: 1):",
            "Install Tailwind CSS v4 and PostCSS dependencies (CMD, depends: 1):",
            "Install Vitest and React Testing Library dependencies (CMD, depends: 1):",
            "Configure PostCSS (CODE, depends: 3):",
            "Update the main stylesheet (CODE, depends: 1, 3):",
            "Configure Vite to support Vitest (CODE, depends: 1, 4):",
            "Create the Vitest setup file (CODE, depends: 4, 7):",
            "Add npm test scripts (CMD, depends: 4, 7, 8):",
            "Implement page layout and routing (CODE, depends: 2, 6):",
            "Implement main SPA structure (CODE, depends: 6, 10):",
            "Create UI components (CODE, depends: 6, 11):",
            "Add unit tests (CODE, depends: 8, 11, 12):",
            "Run the test suite (CMD, depends: 13):",
        ]
        steps, deps = Executor.parse_step_dependencies(raw_steps)

        waves = build_step_waves(steps, deps)

        # Must NOT be all-sequential (which was the bug)
        self.assertLess(len(waves), len(steps),
                        f"Waves should be grouped, got sequential: {waves}")

        # Steps 1,2,3 should be in the same wave (all depend only on 0)
        # Find the wave containing step 1
        wave_for_1 = [w for w in waves if 1 in w][0]
        self.assertIn(2, wave_for_1)
        self.assertIn(3, wave_for_1)


if __name__ == "__main__":
    unittest.main()

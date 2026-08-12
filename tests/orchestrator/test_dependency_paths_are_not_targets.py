"""An installed dependency is not a module the plan builds.

Observed twice in one benchmark session. A CMD step declares where pip
put its package:

    --STEP 1.2 [CMD] depends:1.1
    produces: venv\\Lib\\site-packages\\pygame

`PlanGraph.prefix_for` matched any planned target ending in `/pygame`,
so a later gate importing pygame was reported as

    imports `pygame`, but the plan targets that module at
    `venv/Lib/site-packages/pygame.py`. The gate runs from the project
    root, where this import fails

which is false — pygame resolves fine from the root. The cost is not a
warning: the gate had already been repaired in place, and this discarded
that repair and forced a full re-plan. It hit the first run of the
session, and again the most expensive run of it (253,706 tokens, the
only re-plan in fifteen runs).
"""

import unittest

from agentchanti.orchestrator.plan_graph import (
    PlanGraph,
    _is_dependency_path,
)
from agentchanti.orchestrator.plan_step import PlanStep, check_gate_consistency


class DependencyPathTest(unittest.TestCase):

    def test_the_path_from_the_run(self):
        self.assertTrue(_is_dependency_path("venv/Lib/site-packages/pygame"))

    def test_other_installed_trees(self):
        for key in (".venv/lib/python3.13/site-packages/pygame",
                    "venv/Lib/site-packages/numpy/core",
                    "node_modules/react/index",
                    "frontend/node_modules/lodash",
                    "vendor/autoload",
                    "__pypackages__/3.13/lib/x",
                    "virtualenv/Lib/site-packages/x",
                    "server/dist-packages/y"):
            with self.subTest(key=key):
                self.assertTrue(_is_dependency_path(key), key)

    def test_real_project_paths_are_not_dependencies(self):
        """The check must not start ignoring the plan's own targets."""
        for key in ("src/config", "pacman_clone/src/config", "map",
                    "game/entities", "app/models/user", "lib/util",
                    "environment/config", "env_setup/config"):
            with self.subTest(key=key):
                self.assertFalse(_is_dependency_path(key), key)

    def test_windows_separators(self):
        self.assertTrue(_is_dependency_path(r"venv\Lib\site-packages\pygame"))


class PrefixForTest(unittest.TestCase):

    def _graph(self, *targets):
        steps = []
        for i, t in enumerate(targets, start=1):
            s = PlanStep(id=f"1.{i}", step_type="CODE", description="x")
            s.target_files = [t]
            steps.append(s)
        return PlanGraph(steps)

    def test_a_dependency_never_supplies_a_prefix(self):
        graph = self._graph("venv/Lib/site-packages/pygame.py", "map.py")
        self.assertIsNone(graph.prefix_for("pygame"))

    def test_a_real_subdirectory_target_still_does(self):
        """The check this function exists for must keep working: a plan
        targeting pacman_clone/src/config.py has a gate that cannot
        import src.config from the repo root."""
        graph = self._graph("pacman_clone/src/config.py")
        self.assertEqual(graph.prefix_for("src/config"), "pacman_clone")


class GateConsistencyTest(unittest.TestCase):

    def _steps(self):
        install = PlanStep(id="1.2", step_type="CMD", description="install")
        install.target_files = [r"venv\Lib\site-packages\pygame"]
        code = PlanStep(id="5.1", step_type="CODE", description="entry point")
        code.target_files = ["main.py"]
        code.verify_cmd = ('python -c "import pygame; from main import main; '
                           'assert callable(main)"')
        return [install, code]

    def test_the_verbatim_false_positive_is_gone(self):
        self.assertEqual(check_gate_consistency(self._steps()), [])

    def test_a_genuine_cwd_mismatch_is_still_reported(self):
        sub = PlanStep(id="2.1", step_type="CODE", description="config")
        sub.target_files = ["pacman_clone/src/config.py"]
        gate = PlanStep(id="3.1", step_type="CODE", description="uses it")
        gate.target_files = ["pacman_clone/src/game.py"]
        gate.verify_cmd = 'python -c "from src.config import TILE_SIZE"'
        issues = dict(check_gate_consistency([sub, gate]))
        self.assertIn("3.1", issues)
        self.assertIn("pacman_clone", issues["3.1"])


if __name__ == "__main__":
    unittest.main()

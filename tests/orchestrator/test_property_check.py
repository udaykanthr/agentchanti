"""Tests for the adversarial property check (orchestrator/property_check.py).

The stage exists because a Pac-Man run shipped with every ghost able to
walk through walls while every layer of verification was green — the step
gate only constructed Game(), the generated tests used fixed dt of 0.05 /
0.1 / 0.2, and the smoke test never simulated at all. The defect lived
purely in the gap between a fixed and a variable timestep:

    uniform dt = 1/60   wall-frames =   0
    jittery dt          wall-frames = 129
"""

from __future__ import annotations

import unittest
import unittest.mock
from unittest.mock import MagicMock

from agentchanti.orchestrator.property_check import (
    build_property_step,
    run_property_check,
    simulation_files,
)


class _Memory:
    def __init__(self, files: dict):
        self._files = files

    def as_dict(self):
        return dict(self._files)

    def all_files(self):
        return list(self._files)


_GAME = (
    "import pygame\n"
    "class Ghost:\n"
    "    def update(self, delta_time: float, game_map, now: float) -> None:\n"
    "        self.position.x += self.direction[0] * delta_time\n"
    "class Game:\n"
    "    def run(self):\n"
    "        while self.running:\n"
    "            dt = self.clock.tick(60) / 1000\n"
    "            self.update(dt)\n"
    "            pygame.display.flip()\n"
)


class TestSimulationDetection(unittest.TestCase):
    """Narrow on purpose — a CRUD project must pay nothing."""

    def test_detects_an_update_dt_loop(self):
        self.assertEqual(
            simulation_files(_Memory({"main.py": _GAME})), ["main.py"])

    def test_accepts_common_delta_parameter_names(self):
        for param in ("dt", "delta", "delta_time", "elapsed", "tick",
                      "timestep", "time_step"):
            with self.subTest(param=param):
                src = (f"class E:\n    def update(self, {param}):\n"
                       f"        pass\n"
                       "def loop():\n    clock.tick(60)\n")
                self.assertEqual(
                    simulation_files(_Memory({"e.py": src})), ["e.py"])

    def test_ignores_a_crud_update_method(self):
        """`Order.update(**fields)` is not a simulation step."""
        crud = ("from django.db import models\n"
                "class Order(models.Model):\n"
                "    def update(self, **fields):\n"
                "        self.save()\n"
                "def view(request):\n"
                "    clock.tick(60)\n")     # even with a decoy marker
        self.assertEqual(simulation_files(_Memory({"models.py": crud})), [])

    def test_requires_a_frame_loop_not_just_the_signature(self):
        lone = "class T:\n    def update(self, dt):\n        self.x += dt\n"
        self.assertEqual(simulation_files(_Memory({"t.py": lone})), [])

    def test_skips_command_output_and_non_python(self):
        mem = _Memory({"_cmd_output/step_1.txt": _GAME,
                       "notes.md": _GAME, "main.py": _GAME})
        self.assertEqual(simulation_files(mem), ["main.py"])

    def test_empty_or_broken_memory_is_not_a_simulation(self):
        self.assertEqual(simulation_files(_Memory({})), [])
        broken = MagicMock()
        broken.as_dict.side_effect = RuntimeError("gone")
        self.assertEqual(simulation_files(broken), [])


class TestPropertyStepText(unittest.TestCase):
    """The protocol is fixed by the harness, not left to the model.

    A model asked for "smooth animation" writes fixed-dt tests every
    time — that is exactly how the wall bug survived.
    """

    def _text(self):
        return build_property_step(["main.py"], "Build a Pac-Man clone")

    def test_mandates_randomised_delta_time(self):
        text = self._text().lower()
        self.assertIn("randomly", text)
        self.assertIn("0.008", text)
        self.assertIn("seeded", text)

    def test_mandates_a_long_run_and_per_iteration_assertions(self):
        text = self._text().lower()
        self.assertIn("600", text)
        self.assertIn("every iteration", text)

    def test_keeps_a_fixed_dt_control(self):
        self.assertIn("1/60", self._text())

    def test_forbids_weakening_the_assertion(self):
        """Without this the loop 'fixes' the test instead of the source."""
        text = self._text().lower()
        self.assertIn("do not weaken", text)
        self.assertIn("fix the source", text)

    def test_always_includes_the_baseline_invariants(self):
        text = self._text().lower()
        self.assertIn("impassable", text)
        self.assertIn("nan", text)

    def test_names_the_simulation_files(self):
        self.assertIn("engine.py", build_property_step(["engine.py"], "t"))


class TestRunPropertyCheckSkips(unittest.TestCase):
    """A skip must never fail a run that has nothing to check."""

    def _args(self, **over):
        base = dict(memory=_Memory({"main.py": _GAME}), executor=MagicMock(),
                    coder=MagicMock(), display=None, task="t",
                    language="python", cfg=MagicMock())
        base.update(over)
        return base

    def test_disabled_by_config(self):
        cfg = MagicMock()
        cfg.PROPERTY_CHECK_ENABLED = False
        self.assertEqual(run_property_check(**self._args(cfg=cfg)), (True, ""))

    def test_non_python_project(self):
        self.assertEqual(
            run_property_check(**self._args(language="javascript")),
            (True, ""))

    def test_not_a_simulation(self):
        mem = _Memory({"views.py": "def index(request):\n    return None\n"})
        self.assertEqual(
            run_property_check(**self._args(memory=mem)), (True, ""))

    def test_a_raising_stage_does_not_fail_the_run(self):
        """Never take a pipeline down over the property stage.

        The guard must cover the whole stage — an earlier cut left
        build_step_tools outside the try and a raise there escaped.
        """
        cfg = MagicMock()
        cfg.PROPERTY_CHECK_ENABLED = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        with unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop.agent_loop_enabled",
                return_value=True), \
             unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop.build_step_tools",
                side_effect=RuntimeError("boom")):
            ok, err = run_property_check(**self._args(cfg=cfg))
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_a_violated_invariant_fails_the_run(self):
        cfg = MagicMock()
        cfg.PROPERTY_CHECK_ENABLED = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        with unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop.agent_loop_enabled",
                return_value=True), \
             unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop.build_step_tools",
                return_value=MagicMock()), \
             unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop."
                "run_agent_loop_with_escalation",
                return_value=(False, "ghost entered wall at iteration 42")):
            ok, err = run_property_check(**self._args(cfg=cfg))
        self.assertFalse(ok)
        self.assertIn("iteration 42", err)


if __name__ == "__main__":
    unittest.main()

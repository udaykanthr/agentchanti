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

    def _run_with_loop_result(self, result, *, write_test_file, cfg=None):
        """Run the stage in a temp cwd, optionally with the test file present."""
        import os
        import tempfile
        cfg = cfg or MagicMock()
        cfg.PROPERTY_CHECK_ENABLED = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        prev = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            if write_test_file:
                with open("test_properties.py", "w", encoding="utf-8") as fh:
                    fh.write("import unittest\n")
            with unittest.mock.patch(
                    "agentchanti.orchestrator.agent_loop.agent_loop_enabled",
                    return_value=True), \
                 unittest.mock.patch(
                    "agentchanti.orchestrator.agent_loop.build_step_tools",
                    return_value=MagicMock()), \
                 unittest.mock.patch(
                    "agentchanti.orchestrator.agent_loop."
                    "run_agent_loop_with_escalation", return_value=result):
                # executor.project_root must not be a MagicMock path here.
                ex = MagicMock()
                ex.project_root = tmp
                return run_property_check(**self._args(cfg=cfg, executor=ex))
        finally:
            os.chdir(prev)

    def test_a_violated_invariant_fails_the_run(self):
        ok, err = self._run_with_loop_result(
            (False, "ghost entered wall at iteration 42"),
            write_test_file=True)
        self.assertFalse(ok)
        self.assertIn("iteration 42", err)

    def test_a_test_file_that_was_never_written_is_a_skip(self):
        """The check that never ran must not fail the run.

        Observed: the loop spent all 8 turns chasing a pre-existing crash
        in the project's own suite and never authored test_properties.py.
        The verify reported `unittest.loader._FailedTest ... ERROR` and the
        pipeline was failed for "invariants violated" it had never checked.
        """
        ok, err = self._run_with_loop_result(
            (False, "Verification still failing after 8 turns:\nexit: FAILED\n"
                    "test_properties (unittest.loader._FailedTest"
                    ".test_properties) ... ERROR"),
            write_test_file=False)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_an_unimportable_test_file_is_a_skip(self):
        """A broken harness says nothing about the source under test."""
        ok, err = self._run_with_loop_result(
            (False, "ERROR: test_properties (unittest.loader._FailedTest)\n"
                    "ModuleNotFoundError: No module named 'entities'"),
            write_test_file=True)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_a_real_assertion_failure_still_fails_the_run(self):
        """The skip paths must not swallow a genuine violation."""
        ok, err = self._run_with_loop_result(
            (False, "AssertionError: ghost at (4, 5) is a wall tile "
                    "on iteration 517 with dt=0.0431"),
            write_test_file=True)
        self.assertFalse(ok)
        self.assertIn("iteration 517", err)


class TestPropertyCheckWriteScope(unittest.TestCase):
    """The stage may author its own file and fix the source it tests.

    It must not touch another step's deliverables. Observed: the loop
    rewrote the plan's own tests/test_game.py, discarding the 600-frame
    adversarial suite verified in an earlier wave, then edited main.py --
    and still never produced test_properties.py, so the stage reported
    "skipped" while its edits stayed on disk and reached the final commit.
    """

    def test_scope_is_the_property_file_plus_the_simulation_sources(self):
        captured = {}

        def _fake_build(executor, memory, kb_context_builder=None,
                        project_root=".", write_scope=None):
            captured["scope"] = write_scope
            return MagicMock()

        cfg = MagicMock()
        cfg.PROPERTY_CHECK_ENABLED = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        with unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop.agent_loop_enabled",
                return_value=True),              unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop.build_step_tools",
                side_effect=_fake_build),              unittest.mock.patch(
                "agentchanti.orchestrator.agent_loop."
                "run_agent_loop_with_escalation", return_value=(True, "ok")):
            run_property_check(
                memory=_Memory({"main.py": _GAME}), executor=MagicMock(),
                coder=MagicMock(), display=None, task="t",
                language="python", cfg=cfg)

        scope = captured.get("scope")
        self.assertIsNotNone(scope, "build_step_tools got no write_scope")
        self.assertIn("test_properties.py", scope)
        self.assertIn("main.py", scope,
                      "the source under test must stay writable")

    def test_other_paths_are_not_in_scope(self):
        from agentchanti.agent_tools import AgentTools
        tools = AgentTools(project_root=".",
                           write_scope=["test_properties.py", "main.py"])
        self.assertIsNone(tools._write_denied("test_properties.py"))
        self.assertIsNone(tools._write_denied("main.py"))
        self.assertIsNotNone(tools._write_denied("tests/test_game.py"))
        self.assertIsNotNone(tools._write_denied("README.md"))


if __name__ == "__main__":
    unittest.main()

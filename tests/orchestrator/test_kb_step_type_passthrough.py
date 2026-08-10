"""The per-step KB call must tell the builder what kind of step it is.

`ContextBuilder.build_context` guards both global-doc injection paths with
``step_type != "CMD"`` — "Skip for CMD steps, install commands don't need
them" (context_builder.py:507). The parameter defaults to None, and
``None != "CMD"`` is True, so a caller that omits it gets the OPPOSITE of
the intended behaviour.

Every other caller passes it (diagnosis.py, and the three TEST-step sites).
`_execute_step` — the one path that runs for EVERY step — did not, so the
guard was dead code there. Observed on a hello-world run: a
`python -m venv venv && ... pip install none` step was handed 1,392 tokens
of "Python Test Generation Instructions" plus an async-patterns doc.
"""

import unittest
from unittest.mock import MagicMock, patch

from agentchanti.orchestrator.pipeline import _execute_step
from agentchanti.orchestrator.plan_step import PlanStep


class KBStepTypePassthroughTest(unittest.TestCase):
    def _run_step(self, plan_step):
        """Drive _execute_step and return the kwargs the KB builder saw."""
        builder = MagicMock()
        builder.build_context.return_value = MagicMock(
            kb_available=False, behavioral_instructions=[], global_patterns=[],
            token_count=0, sources_used=[], local_symbols=[], error_fixes=[])

        with patch("agentchanti.orchestrator.pipeline._handle_cmd_step",
                   return_value=(True, "")), \
             patch("agentchanti.orchestrator.pipeline._handle_code_step",
                   return_value=(True, "")):
            _execute_step(
                0, "Install Python venv and required packages",
                steps=["Install Python venv and required packages"],
                llm_client=MagicMock(),
                executor=MagicMock(),
                coder=MagicMock(), reviewer=MagicMock(), tester=MagicMock(),
                task="write a program to print hello world in python",
                memory=MagicMock(), display=MagicMock(),
                language="python",
                plan_step=plan_step, all_plan_steps=[plan_step],
                kb_context_builder=builder,
            )
        if not builder.build_context.called:
            self.skipTest("KB builder not reached on this path")
        return builder.build_context.call_args.kwargs

    def test_a_cmd_step_is_announced_as_cmd(self):
        """The whole point: the builder can then skip its doc injection."""
        step = PlanStep(id="1.1", step_type="CMD",
                        description="Install deps\n> python -m venv venv")
        self.assertEqual(self._run_step(step).get("step_type"), "CMD")

    def test_a_code_step_is_announced_as_code(self):
        step = PlanStep(id="2.1", step_type="CODE",
                        description="Create hello_world.py")
        self.assertEqual(self._run_step(step).get("step_type"), "CODE")

    def test_no_plan_step_degrades_to_none_rather_than_raising(self):
        """Heuristic (unstructured) plans have no PlanStep to ask."""
        kwargs = self._run_step(None)
        self.assertIsNone(kwargs.get("step_type"))


class GuardSemanticsTest(unittest.TestCase):
    """Why omitting the argument silently inverted the rule."""

    def test_none_reads_as_not_a_cmd_step(self):
        self.assertTrue(None != "CMD")      # noqa: E711 — the bug, spelled out
        self.assertFalse("CMD" != "CMD")


if __name__ == "__main__":
    unittest.main()

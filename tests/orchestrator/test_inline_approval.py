"""Regression tests for the inline-code approval gate.

The "inline fast path" in pipeline._execute_step writes planner-supplied code
directly to disk without invoking the Coder agent. Before the fix, this path
also bypassed the diff approval gate — meaning code was written even when the
user did not pass --auto. These tests pin the gate so it can never silently
regress.

See: bugfix branch — pipeline.py:1238 user-approval gate insertion.
"""
import unittest
from unittest.mock import MagicMock, patch

# Mock noisy module-level loggers before importing pipeline.
with patch('agentchanti.orchestrator.pipeline.log'), \
     patch('agentchanti.orchestrator.pipeline._logger'):
    from agentchanti.orchestrator.pipeline import _execute_step

from agentchanti.orchestrator.plan_step import PlanStep


class TestInlineApprovalGate(unittest.TestCase):
    """The inline fast path must respect prompt_diff_approval / --auto."""

    def setUp(self):
        self.task = "Build a snake game"
        self.step_text = (
            "Create the complete Snake game implementation\n"
            "target: snake_game.py"
        )
        self.steps = [self.step_text]

        self.llm_client = MagicMock()
        self.executor = MagicMock()
        self.executor.write_files = MagicMock()
        self.memory = MagicMock()
        self.memory.all_files.return_value = {}
        self.memory.base_dir = "."
        # _kb_context is read by some downstream code paths.
        self.memory._kb_context = ""
        self.memory._content_fixes = None
        self.memory._scaffolded_subproject = None

        self.display = MagicMock()
        self.display.steps = {0: {"type": "CODE"}}
        self.coder = MagicMock()
        self.tester = MagicMock()
        self.reviewer = MagicMock()

    def _make_inline_step(self):
        """A CODE PlanStep with planner-supplied inline code."""
        step = PlanStep(
            id="2.1",
            step_type="CODE",
            description=self.step_text,
            target_files=["snake_game.py"],
            inline_code={"snake_game.py": "import pygame\n# game code\n"},
        )
        step.index = 0
        return step

    @patch('agentchanti.orchestrator.pipeline._handle_code_step')
    @patch('agentchanti.diff_display.prompt_diff_approval')
    def test_inline_rejection_falls_back_to_coder(
            self, mock_approval, mock_handle_code):
        """Reject in the diff editor → no write, Coder runs instead."""
        mock_approval.return_value = False  # user rejects
        mock_handle_code.return_value = (True, "")
        plan_step = self._make_inline_step()

        _execute_step(
            0, self.step_text,
            steps=self.steps,
            llm_client=self.llm_client,
            executor=self.executor,
            coder=self.coder,
            reviewer=self.reviewer,
            tester=self.tester,
            task=self.task,
            memory=self.memory,
            display=self.display,
            language="python",
            auto=False,  # ← interactive mode, gate must fire
            plan_step=plan_step,
            all_plan_steps=[plan_step],
        )

        # Approval was prompted with the inline file.
        mock_approval.assert_called_once()
        called_files = mock_approval.call_args[0][0]
        self.assertIn("snake_game.py", called_files)

        # The rejected inline file was NEVER written to disk.
        for call in self.executor.write_files.call_args_list:
            written = call.args[0] if call.args else call.kwargs.get("files", {})
            self.assertNotIn(
                "snake_game.py", written,
                "Rejected inline code must not reach disk",
            )

        # inline_code was cleared so the Coder regenerates from scratch.
        self.assertEqual(plan_step.inline_code, {})

        # Coder was invoked as the fall-back.
        mock_handle_code.assert_called_once()

    @patch('agentchanti.orchestrator.pipeline._handle_code_step')
    @patch('agentchanti.diff_display.prompt_diff_approval')
    def test_inline_approval_writes_files(
            self, mock_approval, mock_handle_code):
        """Approve in the diff editor → file is written, Coder is skipped."""
        mock_approval.return_value = True  # user approves
        plan_step = self._make_inline_step()

        _execute_step(
            0, self.step_text,
            steps=self.steps,
            llm_client=self.llm_client,
            executor=self.executor,
            coder=self.coder,
            reviewer=self.reviewer,
            tester=self.tester,
            task=self.task,
            memory=self.memory,
            display=self.display,
            language="python",
            auto=False,
            plan_step=plan_step,
            all_plan_steps=[plan_step],
        )

        # Approval gate fired.
        mock_approval.assert_called_once()

        # Inline file was written.
        all_written: dict = {}
        for call in self.executor.write_files.call_args_list:
            written = call.args[0] if call.args else call.kwargs.get("files", {})
            all_written.update(written)
        self.assertIn("snake_game.py", all_written)

        # Coder fall-back was NOT invoked — the inline fast path handled it.
        mock_handle_code.assert_not_called()

    @patch('agentchanti.orchestrator.pipeline._handle_code_step')
    @patch('agentchanti.diff_display.prompt_diff_approval')
    def test_auto_mode_skips_approval_prompt(
            self, mock_approval, mock_handle_code):
        """--auto skips the gate entirely (no interactive prompt)."""
        plan_step = self._make_inline_step()

        _execute_step(
            0, self.step_text,
            steps=self.steps,
            llm_client=self.llm_client,
            executor=self.executor,
            coder=self.coder,
            reviewer=self.reviewer,
            tester=self.tester,
            task=self.task,
            memory=self.memory,
            display=self.display,
            language="python",
            auto=True,  # ← non-interactive
            plan_step=plan_step,
            all_plan_steps=[plan_step],
        )

        # Gate must NOT prompt the user in --auto mode.
        mock_approval.assert_not_called()

        # Inline file was still written.
        all_written: dict = {}
        for call in self.executor.write_files.call_args_list:
            written = call.args[0] if call.args else call.kwargs.get("files", {})
            all_written.update(written)
        self.assertIn("snake_game.py", all_written)

        mock_handle_code.assert_not_called()


if __name__ == '__main__':
    unittest.main()

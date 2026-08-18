"""`models.escalation` must reach the classic diagnosis loop, not just the
agent loop.

Measured 2026-08-15 on the `pacman-strict` task with ``agent_loop: false``:
the run logged ``Escalation model configured: gpt-5.6-sol`` at startup, then
step 4 spent all three diagnosis attempts on the base model and halted the
pipeline. gpt-5.6-sol was never called — ``diagnosis.py`` had no reference
to escalation at all, so the setting was dead config on that path. The
failure then read as "even the strong model could not fix it" when the
strong model was never asked.

Only the LAST attempt escalates: the earlier ones are cheap and often
right (in that same run, attempt 1 of step 2 root-caused a typo'd
attribute correctly), so escalating from the start would pay the premium
on every failure.
"""

import unittest
from unittest.mock import MagicMock, patch

from agentchanti.orchestrator.pipeline import (
    MAX_DIAGNOSIS_RETRIES, _run_diagnosis_loop,
)


NO_FIX = (False, False, False, [], [])


def _kwargs(coder):
    """A classic-path diagnosis loop whose fixes never apply.

    ``_apply_fix`` returning nothing makes the loop run every attempt and
    fall out, which is exactly the measured shape and lets the test see
    which client each attempt used.
    """
    cfg = MagicMock()
    cfg.AGENT_LOOP = False          # classic path
    display = MagicMock()
    display.steps = [{"type": "CODE"}]
    memory = MagicMock()
    memory.all_files.return_value = {}
    return dict(
        steps=["step"], llm_client=MagicMock(), executor=MagicMock(),
        coder=coder, reviewer=MagicMock(), tester=MagicMock(),
        task="task", memory=memory, display=display,
        language="python", cfg=cfg)


class TestClassicDiagnosisEscalation(unittest.TestCase):

    @patch("agentchanti.orchestrator.pipeline._apply_fix", return_value=NO_FIX)
    @patch("agentchanti.orchestrator.pipeline._diagnose_failure",
           return_value="1. ROOT CAUSE: unknown")
    def test_only_the_final_attempt_escalates(self, mock_diag, _fix):
        coder = MagicMock()
        coder.escalation_client = MagicMock()
        coder.escalation_client.model = "gpt-5.6-sol"
        kwargs = _kwargs(coder)

        _run_diagnosis_loop(0, "step text",
                            "AssertionError: 'game_over' != 'win'", **kwargs)

        self.assertEqual(len(mock_diag.call_args_list), MAX_DIAGNOSIS_RETRIES)
        # positional arg 4 of _diagnose_failure is the llm client
        used = [call.args[4] for call in mock_diag.call_args_list]
        for client in used[:-1]:
            self.assertIs(client, kwargs["llm_client"])
        self.assertIs(used[-1], coder.escalation_client)

    @patch("agentchanti.orchestrator.pipeline._apply_fix", return_value=NO_FIX)
    @patch("agentchanti.orchestrator.pipeline._diagnose_failure",
           return_value="1. ROOT CAUSE: unknown")
    def test_no_escalation_configured_keeps_the_base_client(self, mock_diag, _f):
        coder = MagicMock()
        coder.escalation_client = None
        kwargs = _kwargs(coder)
        # A real client has no such attribute at all unless cli.py sets it;
        # a bare MagicMock would auto-create a truthy one.
        kwargs["llm_client"].escalation_client = None

        _run_diagnosis_loop(0, "step text",
                            "AssertionError: 'game_over' != 'win'", **kwargs)

        used = [call.args[4] for call in mock_diag.call_args_list]
        self.assertEqual(len(used), MAX_DIAGNOSIS_RETRIES)
        self.assertTrue(all(c is kwargs["llm_client"] for c in used))

    @patch("agentchanti.orchestrator.pipeline._apply_fix", return_value=NO_FIX)
    @patch("agentchanti.orchestrator.pipeline._diagnose_failure",
           return_value="1. ROOT CAUSE: unknown")
    def test_the_raw_client_can_carry_the_escalation_client(self, mock_diag, _f):
        """CMD-step failures arrive holding the raw client, not an agent.

        ``cli.py`` attaches the escalation client to ``llm_client`` too for
        exactly that case, so the loop must look there when the coder has
        none.
        """
        coder = MagicMock()
        coder.escalation_client = None
        kwargs = _kwargs(coder)
        kwargs["llm_client"].escalation_client = MagicMock()
        kwargs["llm_client"].escalation_client.model = "gpt-5.6-sol"

        _run_diagnosis_loop(0, "step text",
                            "AssertionError: 'game_over' != 'win'", **kwargs)

        used = [call.args[4] for call in mock_diag.call_args_list]
        self.assertIs(used[-1], kwargs["llm_client"].escalation_client)


if __name__ == "__main__":
    unittest.main()

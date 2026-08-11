"""Withholding a tool has to mean the tool does not run.

The loop already narrows the offered tool list when a model spends turn
after turn inspecting instead of editing (`_WITHHOLD_READONLY_AT`). But the
offer is only a request, and `execute_all` ran whatever came back, so a
model that ignored the list kept being served.

Observed on gpt-oss:120b-cloud, step 9 of a Pac-Man run::

    turn 3/8: read_file
    step 9: nudge ignored — withholding read-only tools
    Chat: ... 3 tools          <- read-only removed from the offer
    turn 4/8: read_file        <- executed anyway
    turn 5/8: read_file
    turn 6/8: read_file
    turn 7/8: read_file
    stats: step=9 turns=8 outcome=verify-failed tools={'read_file': 7}
    step 9: loop failed — escalating to stronger model

Seven reads, zero writes, the whole turn budget gone, then an escalation to
a far more expensive model — from an intervention that silently did nothing.
"""

import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import AgentTools
from agentchanti.llm.chat_types import ToolCall

ACTING = frozenset({"write_file", "edit_file", "run_command"})


class WithheldToolIsRefusedTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.tools = AgentTools(project_root=self.root, executor=MagicMock())
        with open(f"{self.root}/a.py", "w") as fh:
            fh.write("print(1)\n")

    def _read(self):
        return ToolCall(name="read_file", arguments={"path": "a.py"}, id="1")

    def test_a_withheld_tool_does_not_run(self):
        out = self.tools.execute(self._read(), allowed=ACTING)
        self.assertTrue(out.startswith("ERROR:"))
        self.assertIn("disabled", out)

    def test_the_refusal_names_what_the_model_may_use(self):
        """It is fed back as a tool result, so it has to be actionable."""
        out = self.tools.execute(self._read(), allowed=ACTING)
        for name in ACTING:
            self.assertIn(name, out)

    def test_an_allowed_tool_still_runs_while_others_are_withheld(self):
        call = ToolCall(name="write_file",
                        arguments={"path": "b.py", "content": "x = 1\n"},
                        id="2")
        self.tools.execute(call, allowed=ACTING)
        with open(f"{self.root}/b.py") as fh:
            self.assertEqual(fh.read(), "x = 1\n")

    def test_no_restriction_is_the_default(self):
        """Every turn that withholds nothing must behave exactly as before."""
        self.assertNotIn("ERROR", self.tools.execute(self._read()))
        self.assertNotIn("ERROR", self.tools.execute(self._read(), allowed=None))

    def test_unknown_outranks_withheld(self):
        """A name that never existed is a different mistake from a
        deliberately removed one — 'disabled' would send the model hunting
        for a way to turn it back on."""
        out = self.tools.execute(
            ToolCall(name="no_such_tool", arguments={}, id="3"), allowed=ACTING)
        self.assertIn("unknown tool", out)

    def test_execute_all_applies_the_restriction(self):
        msgs = self.tools.execute_all([self._read()], allowed=ACTING)
        self.assertEqual(len(msgs), 1)
        self.assertIn("disabled", msgs[0].content)
        self.assertEqual(msgs[0].tool_name, "read_file")

    def test_execute_all_without_a_restriction_is_unchanged(self):
        msgs = self.tools.execute_all([self._read()])
        self.assertNotIn("ERROR", msgs[0].content)


class LoopWiringTest(unittest.TestCase):
    """The refusal only helps if the loop hands its offer to the executor."""

    def test_the_loop_passes_the_offered_tools_to_execute_all(self):
        import inspect

        from agentchanti.orchestrator import agent_loop
        src = inspect.getsource(agent_loop.run_agent_loop)
        self.assertIn("allowed=_allowed", src,
                      "execute_all is no longer told what was offered")
        self.assertIn("tools_for_turn", src)

    def test_the_final_turn_is_deliberately_unenforced(self):
        """Tools are absent from the last offer to prod a text summary, but a
        model that edits anyway may have just fixed the step — the gate runs
        after, so refusing would discard a working fix."""
        import inspect

        from agentchanti.orchestrator import agent_loop
        src = inspect.getsource(agent_loop.run_agent_loop)
        self.assertIn("if tools_for_turn is not None else None", src)


if __name__ == "__main__":
    unittest.main()

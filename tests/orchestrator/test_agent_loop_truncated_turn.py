"""A loop turn cut at the output cap must not be carried forward whole.

Measured 2026-09-22, glm-5.3-flash rewriting page.tsx: two consecutive
turns hit completion=16,384 with no tool call, each was appended to the
conversation in full, and the prompt snowballed 6k -> 23k -> 35k before the
model finally called write_file — ~130k tokens, 45% of the run.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import AgentTools
from agentchanti.llm.chat_types import ChatResponse, ToolCall
from agentchanti.orchestrator.agent_loop import run_agent_loop

HUGE = "export default function Home() {\n" + ("  <section>x</section>\n" * 4000)


def _truncated():
    return ChatResponse(text=HUGE, stop_reason="length")


def _write():
    return ChatResponse(tool_calls=[ToolCall(
        name="write_file", arguments={"path": "page.tsx", "content": "ok\n"},
        id="w1")], stop_reason="tool_calls")


class TestTruncatedTurn(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="agentloop_trunc_")
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        self.tools = AgentTools(project_root=self.root, executor=executor)
        self.sent = []

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _llm(self, *responses):
        llm = MagicMock()
        queue = list(responses)

        def chat(messages, tools=None):
            self.sent.append(sum(len(m.content or "") for m in messages))
            return queue.pop(0)

        llm.chat.side_effect = chat
        return llm

    def test_the_truncated_reply_is_not_resent(self):
        llm = self._llm(_truncated(), _truncated(), _write(),
                        ChatResponse(text="Wrote page.tsx.", stop_reason="stop"))

        success, _ = run_agent_loop(llm, self.tools, "rewrite page.tsx",
                                    "task", max_turns=8)

        self.assertTrue(success)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "page.tsx")))
        # Every prompt after a truncated turn stays far below the size of
        # one truncated reply; before, each added the whole of it.
        self.assertTrue(all(n < len(HUGE) / 4 for n in self.sent), self.sent)

    def test_the_model_is_told_to_call_the_tool(self):
        llm = self._llm(_truncated(), _write(),
                        ChatResponse(text="done", stop_reason="stop"))
        run_agent_loop(llm, self.tools, "rewrite page.tsx", "task", max_turns=8)
        second = llm.chat.call_args_list[1][0][0]
        nudge = [m for m in second if m.role == "user"][-1].content
        self.assertIn("write_file", nudge)
        self.assertIn("cut off", nudge)

    def test_a_truncated_turn_is_not_a_done_claim(self):
        """Before, it fell through to the 'model stopped calling tools'
        exit, which with no tool used ended the step as `no-tools`."""
        llm = self._llm(_truncated(), _write(),
                        ChatResponse(text="done", stop_reason="stop"))
        success, _ = run_agent_loop(llm, self.tools, "rewrite page.tsx",
                                    "task", max_turns=8)
        self.assertTrue(success)

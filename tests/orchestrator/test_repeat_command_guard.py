"""Re-running a failing command unchanged cannot change its answer.

Observed live (loop mode, Pac-Man task, 2026-08-05). Step 5's gate was

    python -c "from pacman_game.map import Map; m = Map(); ..."

and it failed because the generated maze had an unreachable pellet — which
the module's own ``_validate_reachability`` was reporting by name. The loop
spent turns 4, 5, 6 and 7 of its 8-turn budget re-running that identical
command, varying only the working directory:

    turn 4/8  cd /d "C:\\...\\Temp\\tmpnnz83y8b" && python -c "<gate>"
    turn 5/8  python -c "<gate>"
    turn 6/8  python -c "<gate>"
    turn 7/8  cd . && python -c "<gate>"

~38k sent tokens, the maze never touched, then escalation to the stronger
model — which fixed it in two turns. The system prompt already says "do not
retry the same command unchanged"; this makes it a mechanism.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import AgentTools
from agentchanti.llm.chat_types import ChatResponse, ToolCall
from agentchanti.orchestrator.agent_loop import run_agent_loop

GATE = 'python -c "from pacman_game.map import Map; Map()"'


def _tool_response(*calls):
    return ChatResponse(tool_calls=list(calls), stop_reason="tool_calls")


def _run_gate(n):
    return _tool_response(ToolCall(name="run_command",
                                   arguments={"command": GATE}, id=f"c{n}"))


class RepeatedCommandGuard(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="repeatcmd_")
        self.executor = MagicMock()
        # Every command fails, exactly as the real gate did.
        self.executor.run_command.return_value = (False, "AssertionError: "
                                                         "Pellet unreachable")
        self.tools = AgentTools(project_root=self.root,
                                executor=self.executor)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _llm(self, *responses):
        llm = MagicMock()
        llm.chat.side_effect = list(responses)
        return llm

    def _user_texts(self, llm):
        """Every injected user message across the whole conversation."""
        last = llm.chat.call_args_list[-1][0][0]
        return "\n".join(m.content or "" for m in last if m.role == "user")

    def test_second_identical_failure_earns_a_nudge(self):
        llm = self._llm(*[_run_gate(i) for i in range(4)],
                        ChatResponse(text="giving up", stop_reason="stop"))
        run_agent_loop(llm, self.tools, "Create Map", "build the game",
                       max_turns=5)
        text = self._user_texts(llm)
        self.assertIn("already ran", text)
        self.assertIn("cannot change the result", text)

    def test_third_identical_failure_withholds_run_command(self):
        llm = self._llm(*[_run_gate(i) for i in range(4)],
                        ChatResponse(text="giving up", stop_reason="stop"))
        run_agent_loop(llm, self.tools, "Create Map", "build the game",
                       max_turns=5)

        # Find the first call whose offered tool set drops run_command.
        offered = [
            {t.name for t in (call[1].get("tools") or [])}
            for call in llm.chat.call_args_list
            if call[1].get("tools")
        ]
        self.assertTrue(any("run_command" not in s for s in offered),
                        f"run_command never withheld; offered={offered}")
        # Editing tools stay available — the fix has to be made somehow.
        withheld = next(s for s in offered if "run_command" not in s)
        self.assertIn("edit_file", withheld)
        self.assertIn("write_file", withheld)

    def test_an_edit_between_runs_resets_the_streak(self):
        """Re-running after a change is legitimate — that is verification."""
        llm = self._llm(
            _run_gate(0),
            _tool_response(ToolCall(name="write_file",
                                    arguments={"path": "m.py",
                                               "content": "x = 1\n"},
                                    id="w")),
            _run_gate(1),
            ChatResponse(text="done", stop_reason="stop"),
        )
        run_agent_loop(llm, self.tools, "Create Map", "build the game",
                       max_turns=5)
        self.assertNotIn("already ran", self._user_texts(llm))
        for call in llm.chat.call_args_list:
            names = {t.name for t in (call[1].get("tools") or [])}
            if names:
                self.assertIn("run_command", names)

    def test_a_different_command_is_not_a_repeat(self):
        llm = self._llm(
            _run_gate(0),
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "python -m pytest"},
                                    id="c9")),
            ChatResponse(text="done", stop_reason="stop"),
        )
        run_agent_loop(llm, self.tools, "Create Map", "build the game",
                       max_turns=5)
        self.assertNotIn("already ran", self._user_texts(llm))

    def test_a_passing_command_is_never_treated_as_a_repeat(self):
        self.executor.run_command.return_value = (True, "OK")
        llm = self._llm(_run_gate(0), _run_gate(1),
                        ChatResponse(text="done", stop_reason="stop"))
        run_agent_loop(llm, self.tools, "Create Map", "build the game",
                       max_turns=5)
        self.assertNotIn("already ran", self._user_texts(llm))


if __name__ == "__main__":
    unittest.main()

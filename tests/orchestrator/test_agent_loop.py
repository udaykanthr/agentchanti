"""Tests for the bounded agent micro-loop (orchestrator/agent_loop.py)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agentchanti.agent_tools import AgentTools
from agentchanti.config import Config
from agentchanti.llm.chat_types import ChatResponse, ToolCall
from agentchanti.orchestrator.agent_loop import (
    AGENT_LOOP_SYSTEM_PROMPT,
    RECOVERY_FAILED_MARKER,
    agent_loop_enabled,
    build_step_tools,
    run_agent_loop,
    run_recovery_loop,
    verify_cmd_for_language,
)


def _tool_response(*calls):
    return ChatResponse(tool_calls=list(calls), stop_reason="tool_calls")


def _final(text="Done. Tests pass."):
    return ChatResponse(text=text, stop_reason="stop")


class AgentLoopTestCase(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="agentloop_")
        self.executor = MagicMock()
        self.executor.run_command.return_value = (True, "ok")
        self.tools = AgentTools(project_root=self.root, executor=self.executor)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _llm(self, *responses):
        llm = MagicMock()
        llm.chat.side_effect = list(responses)
        return llm


class TestRunAgentLoop(AgentLoopTestCase):

    def test_happy_path_writes_file_and_succeeds(self):
        llm = self._llm(
            _tool_response(ToolCall(name="write_file",
                                    arguments={"path": "app.py",
                                               "content": "x = 1\n"},
                                    id="c1")),
            _final("Wrote app.py."),
        )
        success, info = run_agent_loop(
            llm, self.tools, "Create app.py", "build the app", max_turns=5)
        self.assertTrue(success)
        self.assertEqual(info, "Wrote app.py.")
        self.assertTrue(os.path.isfile(os.path.join(self.root, "app.py")))
        # System prompt is the stable prefix on every call
        first_messages = llm.chat.call_args_list[0][0][0]
        self.assertEqual(first_messages[0].role, "system")
        self.assertEqual(first_messages[0].content, AGENT_LOOP_SYSTEM_PROMPT)

    def test_tool_results_fed_back_to_model(self):
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "echo hi"},
                                    id="c1")),
            _final(),
        )
        run_agent_loop(llm, self.tools, "step", "task", max_turns=5)
        second_messages = llm.chat.call_args_list[1][0][0]
        roles = [m.role for m in second_messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        self.assertIn("ok", second_messages[-1].content)

    def test_no_tool_calls_is_failure(self):
        llm = self._llm(_final("All good, trust me."))
        success, info = run_agent_loop(
            llm, self.tools, "Fix bug", "task", max_turns=5)
        self.assertFalse(success)
        self.assertIn("no tool calls", info)

    def test_max_turns_exhaustion_is_failure(self):
        endless = _tool_response(ToolCall(name="list_files", arguments={},
                                          id="c"))
        llm = self._llm(endless, endless, endless)
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=3)
        self.assertFalse(success)
        self.assertIn("exhausted 3 turns", info)
        self.assertEqual(llm.chat.call_count, 3)

    def test_final_turn_withholds_tools_and_nudges(self):
        endless = _tool_response(ToolCall(name="list_files", arguments={},
                                          id="c"))
        llm = self._llm(endless, endless, _final("wrapped up"))
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=3)
        self.assertTrue(success)
        self.assertEqual(info, "wrapped up")
        # Non-final turns get tool definitions; the final turn gets none
        self.assertTrue(llm.chat.call_args_list[0][1]["tools"])
        self.assertIsNone(llm.chat.call_args_list[2][1]["tools"])
        # And the model was told to summarize
        final_messages = llm.chat.call_args_list[2][0][0]
        self.assertEqual(final_messages[-1].role, "user")
        self.assertIn("no longer available", final_messages[-1].content)

    def test_exhaustion_accepted_when_verify_passes(self):
        endless = _tool_response(ToolCall(name="list_files", arguments={},
                                          id="c"))
        llm = self._llm(endless, endless, endless)
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=3,
            verify_cmd="python -m pytest -q")
        self.assertTrue(success)
        self.assertIn("verified complete", info)

    def test_final_turn_verify_failure_is_terminal(self):
        self.executor.run_command.side_effect = [
            (True, "did something"),   # model tool call
            (False, "1 failed"),       # final-turn verify → fail, no retry
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("done (claim)"),
        )
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=2,
            verify_cmd="python -m pytest -q")
        self.assertFalse(success)
        self.assertIn("Verification still failing", info)

    def test_verify_cmd_failure_feeds_back_then_passes(self):
        self.executor.run_command.side_effect = [
            (True, "edited"),        # model's own tool call
            (False, "1 failed"),     # verify attempt 1 → fail
            (True, "2 passed"),      # verify attempt 2 → pass
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "apply fix"},
                                    id="c1")),
            _final("done (claim 1)"),
            _final("done (claim 2)"),
        )
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=6,
            verify_cmd="python -m pytest -q")
        self.assertTrue(success)
        self.assertEqual(info, "done (claim 2)")
        # After the failed verification the model saw the failure output
        third_messages = llm.chat.call_args_list[2][0][0]
        self.assertIn("Verification command failed", third_messages[-1].content)
        self.assertIn("1 failed", third_messages[-1].content)

    def test_user_message_contains_step_task_and_context(self):
        llm = self._llm(
            _tool_response(ToolCall(name="list_files", arguments={}, id="c")),
            _final(),
        )
        run_agent_loop(llm, self.tools, "the step", "the task",
                       language="python", context="3 files tracked")
        user_msg = llm.chat.call_args_list[0][0][0][1]
        self.assertIn("Overall task: the task", user_msg.content)
        self.assertIn("Current step: the step", user_msg.content)
        self.assertIn("python", user_msg.content)
        self.assertIn("3 files tracked", user_msg.content)


class TestGating(unittest.TestCase):

    def _cfg(self, enabled=True):
        cfg = MagicMock()
        cfg.AGENT_LOOP = enabled
        cfg.AGENT_LOOP_MAX_TURNS = 8
        return cfg

    def _client(self, tools_ok=True):
        client = MagicMock()
        client.supports_tools.return_value = tools_ok
        return client

    def test_enabled_when_flag_and_native_tools(self):
        self.assertTrue(agent_loop_enabled(self._cfg(True), self._client(True)))

    def test_disabled_without_flag(self):
        self.assertFalse(agent_loop_enabled(self._cfg(False), self._client(True)))

    def test_disabled_without_native_tools(self):
        self.assertFalse(agent_loop_enabled(self._cfg(True), self._client(False)))

    def test_disabled_with_missing_cfg_or_client(self):
        self.assertFalse(agent_loop_enabled(None, self._client(True)))
        self.assertFalse(agent_loop_enabled(self._cfg(True), None))

    def test_build_step_tools_picks_up_kb_searcher(self):
        kb = MagicMock()
        kb._searcher = MagicMock()
        tools = build_step_tools(MagicMock(), MagicMock(),
                                 kb_context_builder=kb, project_root=".")
        self.assertIs(tools._searcher, kb._searcher)


class TestConfigFlags(unittest.TestCase):

    def test_defaults_off(self):
        cfg = Config({})
        self.assertFalse(cfg.AGENT_LOOP)
        self.assertEqual(cfg.AGENT_LOOP_MAX_TURNS, 8)

    def test_yaml_opt_in(self):
        cfg = Config({"agent_loop": True, "agent_loop_max_turns": 5})
        self.assertTrue(cfg.AGENT_LOOP)
        self.assertEqual(cfg.AGENT_LOOP_MAX_TURNS, 5)


class TestVerifyCmdForLanguage(unittest.TestCase):

    def test_python_default(self):
        self.assertEqual(verify_cmd_for_language("python"),
                         "python -m pytest -q")
        self.assertEqual(verify_cmd_for_language(None),
                         "python -m pytest -q")

    def test_go(self):
        self.assertEqual(verify_cmd_for_language("go"), "go test ./...")

    def test_js_without_package_json(self):
        root = tempfile.mkdtemp(prefix="vcl_")
        try:
            self.assertIsNone(verify_cmd_for_language("javascript", root))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_js_with_test_script(self):
        root = tempfile.mkdtemp(prefix="vcl_")
        try:
            with open(f"{root}/package.json", "w") as f:
                f.write('{"scripts": {"test": "vitest run"}}')
            self.assertEqual(verify_cmd_for_language("typescript", root),
                             "npm test --silent")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_js_without_test_script(self):
        root = tempfile.mkdtemp(prefix="vcl_")
        try:
            with open(f"{root}/package.json", "w") as f:
                f.write('{"scripts": {"build": "tsc"}}')
            self.assertIsNone(verify_cmd_for_language("javascript", root))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_unknown_language(self):
        self.assertIsNone(verify_cmd_for_language("rust"))


class TestRunRecoveryLoop(AgentLoopTestCase):

    def test_error_context_reaches_model(self):
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "fix"}, id="c")),
            _final("recovered"),
        )
        ok, info = run_recovery_loop(
            llm, self.tools, "run the tests", "overall task",
            "Command `g++ x.cpp` failed.\nOutput:\n'g++' is not recognized")
        self.assertTrue(ok)
        user_msg = llm.chat.call_args_list[0][0][0][1]
        self.assertIn("previous attempt at this step FAILED", user_msg.content)
        self.assertIn("g++", user_msg.content)


class TestDiagnosisLoopRecovery(unittest.TestCase):
    """_run_diagnosis_loop delegates to the recovery loop when enabled."""

    def _kwargs(self, cfg):
        display = MagicMock()
        display.steps = [{"type": "CODE"}]
        return dict(
            steps=["step"], llm_client=MagicMock(), executor=MagicMock(),
            coder=MagicMock(), reviewer=MagicMock(), tester=MagicMock(),
            task="task", memory=MagicMock(), display=display,
            language="python", cfg=cfg)

    def _cfg(self, enabled=True):
        cfg = MagicMock()
        cfg.AGENT_LOOP = enabled
        cfg.AGENT_LOOP_MAX_TURNS = 8
        return cfg

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_recovery_success_returns_true(self, mock_rec):
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        kwargs = self._kwargs(self._cfg())
        kwargs["llm_client"].supports_tools.return_value = True
        result = _run_diagnosis_loop(0, "step text", "assertion failed",
                                     **kwargs)
        self.assertTrue(result)
        mock_rec.assert_called_once()

    @patch("agentchanti.orchestrator.pipeline._diagnose_failure")
    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(False, "could not fix"))
    def test_recovery_failure_skips_classic_diagnosis(self, mock_rec,
                                                      mock_diag):
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        kwargs = self._kwargs(self._cfg())
        kwargs["llm_client"].supports_tools.return_value = True
        result = _run_diagnosis_loop(0, "step text", "assertion failed",
                                     **kwargs)
        self.assertFalse(result)
        mock_rec.assert_called_once()
        mock_diag.assert_not_called()

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop")
    def test_marker_prevents_second_recovery(self, mock_rec):
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        kwargs = self._kwargs(self._cfg())
        kwargs["llm_client"].supports_tools.return_value = True
        result = _run_diagnosis_loop(
            0, "step text",
            f"{RECOVERY_FAILED_MARKER} Command `x` failed.", **kwargs)
        self.assertFalse(result)
        mock_rec.assert_not_called()


class TestStepHandlerIntegration(unittest.TestCase):
    """The handlers delegate to the loop when the flag is on."""

    def _common(self):
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        agent = MagicMock()
        agent.llm_client.supports_tools.return_value = True
        memory = MagicMock()
        memory.summary.return_value = "files: none"
        return cfg, agent, memory, MagicMock()  # display

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop ran"))
    def test_code_step_delegates(self, mock_loop):
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        cfg, coder, memory, display = self._common()
        result = _handle_code_step(
            "write code", coder, MagicMock(), MagicMock(), "task",
            memory, display, 0, cfg=cfg)
        self.assertEqual(result, (True, "loop ran"))
        mock_loop.assert_called_once()
        self.assertEqual(mock_loop.call_args[1]["max_turns"], 8)

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop ran"))
    def test_test_step_delegates_with_verify_cmd(self, mock_loop):
        from agentchanti.orchestrator.step_handlers import _handle_test_step
        cfg, tester, memory, display = self._common()
        result = _handle_test_step(
            "write tests", tester, MagicMock(), MagicMock(), MagicMock(),
            "task", memory, display, 0, language="python", cfg=cfg)
        self.assertEqual(result, (True, "loop ran"))
        self.assertEqual(mock_loop.call_args[1]["verify_cmd"],
                         "python -m pytest -q")


if __name__ == "__main__":
    unittest.main()

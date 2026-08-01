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
    RECOVERY_BLOCKED_MARKER,
    RECOVERY_FAILED_MARKER,
    agent_loop_enabled,
    attempt_env_self_heal,
    attempt_digest,
    build_step_tools,
    get_loop_stats,
    record_attempt,
    reset_attempt_journal,
    loop_stats_summary,
    reset_loop_stats,
    reverifiable_cmd,
    run_agent_loop,
    run_recovery_loop,
    truncate_middle,
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

    def test_read_only_streak_triggers_act_nudge(self):
        read = _tool_response(ToolCall(name="read_file",
                                       arguments={"path": "a.txt"}, id="c"))
        llm = self._llm(read, read, read, read, _final("done"))
        self._write_helper()
        success, _ = run_agent_loop(
            llm, self.tools, "fix the tests", "task", max_turns=8)
        self.assertTrue(success)
        # The conversation contains exactly one act-now nudge, placed
        # right after the 2nd read-only turn's tool result (threshold
        # lowered 3 → 2). (call_args holds a reference to the mutated
        # list — inspect the final state.)
        messages = llm.chat.call_args_list[-1][0][0]
        nudges = [i for i, m in enumerate(messages)
                  if m.role == "user" and "ACT now" in m.content]
        self.assertEqual(len(nudges), 1)
        self.assertEqual(messages[nudges[0] - 1].role, "tool")

    def test_acting_resets_read_only_streak(self):
        read = _tool_response(ToolCall(name="read_file",
                                       arguments={"path": "a.txt"}, id="c"))
        write = _tool_response(ToolCall(
            name="write_file",
            arguments={"path": "b.txt", "content": "x"}, id="w"))
        # read, write, read: the write resets the streak so it never
        # reaches the (now 2) nudge threshold.
        llm = self._llm(read, write, read, _final("done"))
        self._write_helper()
        run_agent_loop(llm, self.tools, "step", "task", max_turns=8)
        # No nudge anywhere: streak never reached 2
        for call in llm.chat.call_args_list:
            for msg in call[0][0]:
                if msg.role == "user":
                    self.assertNotIn("ACT now", msg.content)

    def test_nudge_ignored_withholds_read_only_tools(self):
        # 3 read-only turns: nudge fires after the 2nd, is ignored on the
        # 3rd → the 4th call must offer only acting tools (thresholds
        # lowered 3/4 → 2/3).
        read = _tool_response(ToolCall(name="read_file",
                                       arguments={"path": "a.txt"}, id="c"))
        write = _tool_response(ToolCall(
            name="write_file",
            arguments={"path": "b.txt", "content": "x"}, id="w"))
        llm = self._llm(read, read, read, write, _final("done"))
        self._write_helper()
        success, _ = run_agent_loop(
            llm, self.tools, "fix the tests", "task", max_turns=8)
        self.assertTrue(success)
        fourth_tools = llm.chat.call_args_list[3][1]["tools"]
        self.assertEqual({t.name for t in fourth_tools},
                         {"write_file", "edit_file", "run_command"})
        # Escalation message present exactly once
        messages = llm.chat.call_args_list[-1][0][0]
        escalations = [m for m in messages if m.role == "user"
                       and "Inspection tools are now disabled" in m.content]
        self.assertEqual(len(escalations), 1)
        # Acting restores the full toolset on the following call
        fifth_tools = llm.chat.call_args_list[4][1]["tools"]
        self.assertIn("read_file", {t.name for t in fifth_tools})

    def _write_helper(self):
        with open(os.path.join(self.root, "a.txt"), "w") as f:
            f.write("data")

    def test_preload_injects_existing_target_file_into_opening_message(self):
        with open(os.path.join(self.root, "app.py"), "w") as f:
            f.write("def existing():\n    return 42\n")
        llm = self._llm(
            _tool_response(ToolCall(name="write_file",
                                    arguments={"path": "app.py",
                                               "content": "x\n"}, id="c")),
            _final(),
        )
        run_agent_loop(llm, self.tools, "edit app.py", "task",
                       preload_files=["app.py"])
        opening = llm.chat.call_args_list[0][0][0][1]
        # Existing content is handed over up front so the model needn't
        # spend a turn on read_file, and it's marked do-not-re-read.
        self.assertIn("do NOT call read_file", opening.content)
        self.assertIn("def existing():", opening.content)

    def test_preload_skips_missing_and_empty_files(self):
        # A file the step will create doesn't exist yet; an empty file has
        # nothing to hand over. Neither should produce a preload block.
        open(os.path.join(self.root, "empty.py"), "w").close()
        llm = self._llm(
            _tool_response(ToolCall(name="write_file",
                                    arguments={"path": "new.py",
                                               "content": "x\n"}, id="c")),
            _final(),
        )
        run_agent_loop(llm, self.tools, "create files", "task",
                       preload_files=["new.py", "empty.py"])
        opening = llm.chat.call_args_list[0][0][0][1]
        self.assertNotIn("already read for you", opening.content)

    def test_preload_none_leaves_message_unchanged(self):
        llm = self._llm(
            _tool_response(ToolCall(name="list_files", arguments={}, id="c")),
            _final(),
        )
        run_agent_loop(llm, self.tools, "step", "task", preload_files=None)
        opening = llm.chat.call_args_list[0][0][0][1]
        self.assertNotIn("already read for you", opening.content)

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

    def test_windows_platform_note_in_user_message(self):
        # On Windows the user message must warn off POSIX text tools
        # (observed: a loop burned a turn on `sed` before finding
        # read_file); the system prompt must stay byte-identical.
        llm = self._llm(
            _tool_response(ToolCall(name="list_files", arguments={}, id="c")),
            _final(),
        )
        with patch("os.name", "nt"):
            run_agent_loop(llm, self.tools, "the step", "the task")
        messages = llm.chat.call_args_list[0][0][0]
        self.assertEqual(messages[0].content, AGENT_LOOP_SYSTEM_PROMPT)
        self.assertIn("sed", messages[1].content)
        self.assertIn("Windows", messages[1].content)

    def test_no_platform_note_on_posix(self):
        llm = self._llm(
            _tool_response(ToolCall(name="list_files", arguments={}, id="c")),
            _final(),
        )
        with patch("os.name", "posix"):
            run_agent_loop(llm, self.tools, "the step", "the task")
        user_msg = llm.chat.call_args_list[0][0][0][1]
        self.assertNotIn("Windows", user_msg.content)


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

    def test_defaults_on(self):
        cfg = Config({})
        self.assertTrue(cfg.AGENT_LOOP)
        self.assertEqual(cfg.AGENT_LOOP_MAX_TURNS, 8)

    def test_yaml_opt_out(self):
        cfg = Config({"agent_loop": False, "agent_loop_max_turns": 5})
        self.assertFalse(cfg.AGENT_LOOP)
        self.assertEqual(cfg.AGENT_LOOP_MAX_TURNS, 5)


class TestLoopTelemetry(AgentLoopTestCase):

    def setUp(self):
        super().setUp()
        reset_loop_stats()

    def test_happy_path_recorded(self):
        llm = self._llm(
            _tool_response(ToolCall(name="write_file",
                                    arguments={"path": "a.py",
                                               "content": "x = 1\n"},
                                    id="c")),
            _final(),
        )
        run_agent_loop(llm, self.tools, "step", "task", max_turns=5)
        stats = get_loop_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["outcome"], "done")
        self.assertEqual(stats[0]["turns"], 2)
        self.assertEqual(stats[0]["tool_calls"], {"write_file": 1})
        self.assertFalse(stats[0]["recovery"])

    def test_recovery_flagged_and_summary_line(self):
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final(),
        )
        run_recovery_loop(llm, self.tools, "step", "task", "err")
        stats = get_loop_stats()
        self.assertTrue(stats[0]["recovery"])
        summary = loop_stats_summary()
        self.assertIn("1 loop run(s)", summary)
        self.assertIn("1 recovery run(s)", summary)

    def test_no_runs_means_no_summary(self):
        self.assertIsNone(loop_stats_summary())

    def test_exhaustion_outcome_recorded(self):
        endless = _tool_response(ToolCall(name="list_files", arguments={},
                                          id="c"))
        llm = self._llm(endless, endless, endless)
        run_agent_loop(llm, self.tools, "step", "task", max_turns=3)
        self.assertEqual(get_loop_stats()[0]["outcome"], "exhausted")


class TestLoopPreloadPaths(unittest.TestCase):
    """The loop should not spend a turn reading what the plan already named.

    Only target_files were preloaded, so a step creating a NEW file had
    nothing to preload and burned turn 1 on read_file calls for its own
    declared imports — observed as `turn 1/8: read_file, read_file,
    read_file, list_files`, and the act-now nudge that follows.
    """

    def setUp(self):
        from agentchanti.orchestrator import memory as _m
        _m.clear_plan_context_files()
        self.addCleanup(_m.clear_plan_context_files)

    def _step(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        return PlanStep(
            id="3.1", step_type="CODE", index=0,
            target_files=["pkg/entities.py"],
            imports_from={"pkg/map.py": ["Map"],
                          "pkg/constants.py": ["TILE_SIZE"]})

    def test_includes_declared_imports_not_just_targets(self):
        from agentchanti.orchestrator.memory import set_plan_context_files
        from agentchanti.orchestrator.step_handlers import _loop_preload_paths
        set_plan_context_files({"pkg/map.py": "m", "pkg/constants.py": "c"})
        self.assertEqual(
            _loop_preload_paths(self._step()),
            ["pkg/entities.py", "pkg/map.py", "pkg/constants.py"])

    def test_target_comes_first(self):
        """For a step EDITING a file, that file matters most in the budget."""
        from agentchanti.orchestrator.memory import set_plan_context_files
        from agentchanti.orchestrator.step_handlers import _loop_preload_paths
        set_plan_context_files({"pkg/map.py": "m"})
        self.assertEqual(_loop_preload_paths(self._step())[0],
                         "pkg/entities.py")

    def test_deduplicates_across_targets_and_context(self):
        from agentchanti.orchestrator.memory import set_plan_context_files
        from agentchanti.orchestrator.step_handlers import _loop_preload_paths
        set_plan_context_files({"pkg/entities.py": "e", "pkg/map.py": "m"})
        self.assertEqual(_loop_preload_paths(self._step()),
                         ["pkg/entities.py", "pkg/map.py"])

    def test_works_without_plan_context(self):
        from agentchanti.orchestrator.step_handlers import _loop_preload_paths
        self.assertEqual(_loop_preload_paths(self._step()),
                         ["pkg/entities.py"])

    def test_step_without_targets_or_context_is_empty(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _loop_preload_paths
        self.assertEqual(
            _loop_preload_paths(PlanStep(id="1", step_type="CODE", index=0)),
            [])


class TestEnvSelfHeal(AgentLoopTestCase):
    """Missing-dependency verify failures are healed with one install
    instead of being fed to the model (the observed run burned 16 turns
    while the fix was `pip install pytest`)."""

    def test_the_step_s_own_target_is_never_pip_installed(self):
        """A TEST step's gate fails before the test file exists.

        `python -m unittest -v test_main` reports "No module named
        test_main" on the first run; memory has no such file yet, so the
        heal fired `pip install test_main` — a pointless call, and exactly
        the dependency-confusion hazard the local-module guard exists to
        prevent. The plan already declares the file as this step's target.
        """
        healed: set[str] = set()
        fired = attempt_env_self_heal(
            self.tools, "ModuleNotFoundError: No module named 'test_main'",
            "python", healed, verify_cmd="python -m unittest -v test_main",
            planned_files=["test_main.py"])
        self.assertFalse(fired)
        self.executor.run_command.assert_not_called()

    def test_a_planned_package_directory_is_never_pip_installed(self):
        """Same hazard, one level up: the module is a PACKAGE the step makes.

        Observed: a step targeting `tests/__init__.py, tests/test_map.py`
        failed its gate with "No module named 'tests'" before either file
        existed, and the pre-loop heal fired
        `pip install tests` — a real name on PyPI.
        """
        healed: set[str] = set()
        fired = attempt_env_self_heal(
            self.tools, "ModuleNotFoundError: No module named 'tests'",
            "python", healed, verify_cmd="python -m unittest -v tests.test_map",
            planned_files=["tests/__init__.py", "tests/test_map.py"])
        self.assertFalse(fired)
        self.executor.run_command.assert_not_called()

    def test_test_step_preloop_heal_passes_the_planned_targets(self):
        """The guard only works if the call site actually supplies them.

        The in-loop `_run_verify` heal passed `planned_files`; the pre-loop
        heal in `_handle_test_step` did not, which is how `pip install
        tests` still reached the network.
        """
        import inspect

        from agentchanti.orchestrator import step_handlers
        src = inspect.getsource(step_handlers)
        idx = src.find("while not _pre_ok and attempt_env_self_heal(")
        self.assertGreater(idx, 0, "pre-loop heal call not found")
        self.assertIn("planned_files=", src[idx:idx + 300],
                      "the pre-loop heal must pass the step's planned files")

    def test_a_real_missing_package_still_heals(self):
        self.executor.run_command.return_value = (True, "installed")
        healed: set[str] = set()
        self.assertTrue(attempt_env_self_heal(
            self.tools, "ModuleNotFoundError: No module named 'pytest'",
            "python", healed, verify_cmd="python -m pytest",
            planned_files=["test_main.py"]))

    def test_missing_python_module_installed_and_reverified(self):
        self.executor.run_command.side_effect = [
            (True, "edited"),                                # model tool call
            (False, "ModuleNotFoundError: No module named 'pytest'"),  # verify
            (True, "Successfully installed pytest"),         # self-heal
            (True, "12 passed"),                             # re-verify
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "apply fix"},
                                    id="c1")),
            _final("done"),
        )
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=6,
            verify_cmd="python manage.py test --noinput")
        self.assertTrue(success)
        self.assertEqual(info, "done")
        install_cmd = self.executor.run_command.call_args_list[2][0][0]
        self.assertEqual(install_cmd, "python -m pip install pytest")

    def test_heal_keeps_cd_prefix_of_verify_cmd(self):
        self.executor.run_command.side_effect = [
            (True, "edited"),
            (False, "No module named 'pytest'"),
            (True, "installed"),
            (True, "ok"),
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("done"),
        )
        success, _ = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=6,
            verify_cmd="cd app && python manage.py test --noinput")
        self.assertTrue(success)
        install_cmd = self.executor.run_command.call_args_list[2][0][0]
        self.assertEqual(install_cmd, "cd app && python -m pip install pytest")

    def test_module_healed_only_once_per_loop(self):
        # Install "succeeds" but the error persists → the second verify
        # failure must NOT trigger another install of the same module.
        self.executor.run_command.side_effect = [
            (True, "edited"),
            (False, "No module named 'pytest'"),   # verify 1
            (True, "installed"),                   # heal (once)
            (False, "No module named 'pytest'"),   # re-verify → feedback
            (False, "No module named 'pytest'"),   # final-turn verify
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("claim 1"),
            _final("claim 2"),
        )
        success, info = run_agent_loop(
            llm, self.tools, "step", "task", max_turns=3,
            verify_cmd="python -m pytest -q")
        self.assertFalse(success)
        installs = [c[0][0] for c in self.executor.run_command.call_args_list
                    if "pip install" in c[0][0]]
        self.assertEqual(installs, ["python -m pip install pytest"])

    def test_missing_npm_package_installed(self):
        self.executor.run_command.side_effect = [
            (True, "edited"),
            (False, "Cannot find package '@testing-library/react' "
                    "imported from App.test.jsx"),
            (True, "added 1 package"),
            (True, "3 passed"),
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("done"),
        )
        success, _ = run_agent_loop(
            llm, self.tools, "step", "task", language="javascript",
            max_turns=6, verify_cmd="npm test --silent")
        self.assertTrue(success)
        install_cmd = self.executor.run_command.call_args_list[2][0][0]
        self.assertEqual(install_cmd,
                         "npm install -D @testing-library/react")

    def test_local_module_not_pip_installed(self):
        # `No module named 'main'` where main/ is a project package is a
        # code problem — installing "main" from PyPI would be wrong (and
        # a supply-chain risk). The failure goes to the model instead.
        memory = MagicMock()
        memory.all_files.return_value = {"main/views.py": "", "manage.py": ""}
        tools = AgentTools(project_root=self.root, executor=self.executor,
                           memory=memory)
        self.executor.run_command.side_effect = [
            (True, "edited"),
            (False, "ModuleNotFoundError: No module named 'main'"),  # verify
            (False, "ModuleNotFoundError: No module named 'main'"),  # final
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("claim 1"),
            _final("claim 2"),
        )
        success, _ = run_agent_loop(
            llm, tools, "step", "task", max_turns=3,
            verify_cmd="python -m pytest -q")
        self.assertFalse(success)
        installs = [c[0][0] for c in self.executor.run_command.call_args_list
                    if "pip install" in c[0][0]]
        self.assertEqual(installs, [])


class TestVerifyCmdForLanguage(unittest.TestCase):

    def test_python_default(self):
        root = tempfile.mkdtemp(prefix="vcl_")
        try:
            self.assertEqual(verify_cmd_for_language("python", root),
                             "python -m pytest -q")
            self.assertEqual(verify_cmd_for_language(None, root),
                             "python -m pytest -q")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_python_django_project_uses_manage_py(self):
        root = tempfile.mkdtemp(prefix="vcl_")
        try:
            with open(f"{root}/manage.py", "w") as f:
                f.write("#!/usr/bin/env python\n")
            self.assertEqual(verify_cmd_for_language("python", root),
                             "python manage.py test --noinput")
        finally:
            shutil.rmtree(root, ignore_errors=True)

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


class TestTruncateMiddle(unittest.TestCase):

    def test_short_text_unchanged(self):
        self.assertEqual(truncate_middle("abc", 100), "abc")

    def test_keeps_head_and_tail_with_marker(self):
        text = ("Command `npm run build` failed\n"
                + "frame line\n" * 500
                + "TypeError: x is not a function")
        out = truncate_middle(text, 400)
        self.assertLess(len(out), len(text))
        self.assertIn("Command `npm run build` failed", out)
        self.assertIn("TypeError: x is not a function", out)
        self.assertIn("chars truncated", out)

    def test_traceback_exception_survives_the_observed_case(self):
        # The observed failure: a ~5000-char Django probe output whose
        # LAST line names the exception, sliced to 4000. A head slice
        # dropped the exception entirely.
        text = ("Internal Server Error: /\n" + "x" * 4900
                + "\nNoReverseMatch: Reverse for 'home' not found.")
        self.assertIn("NoReverseMatch", truncate_middle(text, 4000))


class TestReverifiableCmd(unittest.TestCase):

    def test_idempotent_commands_pass_through(self):
        for cmd in ("npm run build:css",
                    "pip install django",
                    "python -m pytest -q",
                    "cd site && npm test --silent"):
            self.assertEqual(reverifiable_cmd(cmd), cmd)

    def test_one_shot_scaffold_commands_excluded(self):
        for cmd in (
            "mkdir site && cd site && python -m venv venv && pip install x",
            "django-admin startproject config .",
            "python manage.py startapp core",
            "npm create vite@latest my-app",
            "npx create-react-app app",
            "git init",
            "cargo new hello",
        ):
            self.assertIsNone(reverifiable_cmd(cmd), cmd)

    def test_empty_and_none(self):
        self.assertIsNone(reverifiable_cmd(None))
        self.assertIsNone(reverifiable_cmd(""))

    def test_venv_activation_is_not_scaffolding(self):
        # Regression: a bare `venv` alternative matched the activation
        # path and silently dropped every plan-declared verify gate.
        for cmd in (
            r"cd site && call venv\Scripts\activate && python manage.py check",
            "source venv/bin/activate && pytest -q",
            r".venv\Scripts\python.exe -m pytest",
        ):
            self.assertEqual(reverifiable_cmd(cmd), cmd, cmd)

    def test_venv_creation_still_excluded(self):
        for cmd in (
            "python -m venv venv && pip install django",
            "python3 -m venv .venv",
            "py -m venv env",
            "virtualenv venv",
        ):
            self.assertIsNone(reverifiable_cmd(cmd), cmd)


class TestRunRecoveryLoop(AgentLoopTestCase):

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_recovery_loop_escalates_on_failure(self, mock_loop):
        # Recovery loops are where turn budgets die — they get the same
        # one-shot stronger-model retry as first-attempt loops.
        mock_loop.side_effect = [(False, "still failing at turn 8"),
                                 (True, "fixed by stronger model")]
        escalation = MagicMock()
        escalation.supports_tools.return_value = True
        ok, info = run_recovery_loop(
            MagicMock(name="primary"), MagicMock(), "step", "task", "err",
            escalation_client=escalation)
        self.assertTrue(ok)
        self.assertEqual(mock_loop.call_count, 2)
        self.assertIs(mock_loop.call_args_list[1][0][0], escalation)
        self.assertTrue(mock_loop.call_args_list[1][1]["_recovery"])

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_blocked_admission_escalates(self, mock_loop):
        # Regression: a self-reported "done" whose summary admits the
        # blocker was flipped to failure AFTER the escalation wrapper
        # returned — the stronger model never got its shot.
        mock_loop.side_effect = [
            (True, f"cannot install tailwind. {RECOVERY_BLOCKED_MARKER}"),
            (True, "fixed: wrote configs manually, build passes"),
        ]
        escalation = MagicMock()
        escalation.supports_tools.return_value = True
        ok, info = run_recovery_loop(
            MagicMock(name="primary"), MagicMock(), "step", "task", "err",
            escalation_client=escalation)
        self.assertTrue(ok)
        self.assertEqual(mock_loop.call_count, 2)
        self.assertIs(mock_loop.call_args_list[1][0][0], escalation)
        # Escalated context carries the blocked attempt's admission
        self.assertIn("FAILED", mock_loop.call_args_list[1][1]["context"])

    def test_recovery_prefers_rerunning_commands(self):
        # Regression: a scaffold-command recovery hand-wrote the 13 files
        # startproject would have generated. The context must steer the
        # model toward correcting and re-running the command instead.
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "fix"}, id="c")),
            _final("recovered"),
        )
        run_recovery_loop(llm, self.tools, "scaffold the project", "task",
                          "cd nope && django-admin startproject failed")
        user_msg = llm.chat.call_args_list[0][0][0][1]
        self.assertIn("re-running that command", user_msg.content)
        self.assertIn("do NOT hand-write the files", user_msg.content)

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

    def test_error_tail_survives_truncation(self):
        # Long tracebacks name the exception on the LAST line; the model
        # must see it or it spends the whole budget hunting for the error.
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "fix"}, id="c")),
            _final("recovered"),
        )
        error = ("Internal Server Error: /\n"
                 + "  File django/template/base.py, in render\n" * 300
                 + "NoReverseMatch: Reverse for 'home' not found.")
        run_recovery_loop(llm, self.tools, "step", "task", error)
        user_msg = llm.chat.call_args_list[0][0][0][1]
        self.assertIn("NoReverseMatch", user_msg.content)
        self.assertIn("Internal Server Error", user_msg.content)

    def test_blocked_admission_is_not_a_recovery(self):
        # Without a verify_cmd the exit rests on the summary — one that
        # admits the blocker must not count as recovered.
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("The required tool cannot be installed.\n"
                   f"{RECOVERY_BLOCKED_MARKER} — tailwindcss CLI missing"),
        )
        ok, info = run_recovery_loop(llm, self.tools, "step", "task", "err")
        self.assertFalse(ok)
        self.assertIn("blocked", info)

    def test_passing_verify_outranks_blocked_admission(self):
        self.executor.run_command.side_effect = [
            (True, "did something"),   # model's tool call
            (True, "built fine"),      # deterministic verify → pass
        ]
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final(f"{RECOVERY_BLOCKED_MARKER} (model is wrong)"),
        )
        ok, _ = run_recovery_loop(
            llm, self.tools, "step", "task", "err",
            verify_cmd="npm run build:css")
        self.assertTrue(ok)

    def test_verify_criterion_grounded_in_context(self):
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "x"}, id="c")),
            _final("done"),
        )
        run_recovery_loop(
            llm, self.tools, "step", "task", "err",
            verify_cmd="npm run build:css")
        user_msg = llm.chat.call_args_list[0][0][0][1]
        self.assertIn("complete ONLY when", user_msg.content)
        self.assertIn("npm run build:css", user_msg.content)


class TestCmdStepRecoveryVerify(unittest.TestCase):
    """_handle_cmd_step hands the failed command to the recovery loop as
    its deterministic verify gate — when the command is safe to re-run."""

    def _run_failed_cmd(self, command):
        from agentchanti.orchestrator.step_handlers import _handle_cmd_step
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        llm = MagicMock()
        llm.supports_tools.return_value = True
        memory = MagicMock()
        memory.summary.return_value = ""
        memory._scaffolded_subproject = None
        memory.all_files.return_value = {}
        executor = MagicMock()
        executor.run_command.return_value = (False, "boom")
        plan_step = MagicMock()
        plan_step.command = command
        return _handle_cmd_step(
            f"Run: {command}", executor, llm, memory, MagicMock(), 0,
            plan_step=plan_step, cfg=cfg)

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_failed_cmd_becomes_verify_cmd(self, mock_rec):
        ok, _ = self._run_failed_cmd("npm run build:css")
        self.assertTrue(ok)
        self.assertEqual(mock_rec.call_args[1]["verify_cmd"],
                         "npm run build:css")

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_scaffold_cmd_gets_no_verify(self, mock_rec):
        self._run_failed_cmd("django-admin startproject config .")
        self.assertIsNone(mock_rec.call_args[1]["verify_cmd"])


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

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_code_step_recovery_is_gated_on_the_declared_verify(self, mock_rec):
        """A CODE step's recovery used to run with verify_cmd=None.

        The exit then rested entirely on the model's summary, and an honest
        "the verification is still failing" that happens not to end with the
        RECOVERY: blocked marker counted as success. Observed: a ghost step
        whose declared gate asserted the ghost moves was marked recovered
        while the ghost stayed stationary, and the run reported Finished.
        """
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        from agentchanti.orchestrator.plan_step import PlanStep
        gate = ('python -c "from ghost import Ghost; g=Ghost(); '
                'p0=g.pixel_pos(); g.update(0.2); assert g.pixel_pos()!=p0"')
        step = PlanStep(id="2.3", step_type="CODE", index=0, verify_cmd=gate)
        kwargs = self._kwargs(self._cfg())
        kwargs["llm_client"].supports_tools.return_value = True
        kwargs["memory"].as_dict.return_value = {}
        _run_diagnosis_loop(0, "step text", "assertion failed",
                            plan_step=step, **kwargs)
        self.assertEqual(mock_rec.call_args[1]["verify_cmd"], gate)

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_test_step_recovery_keeps_the_declared_gate(self, mock_rec):
        """A TEST step's recovery used to get the LANGUAGE DEFAULT instead.

        Recovery runs only after the main loop already failed the declared
        gate, so handing recovery a different command does not recover the
        step — it redefines success. Observed: a step declaring
        `python -m unittest -v` (the task's own stated acceptance
        criterion) failed on a native crash, recovery was gated on
        `python -m pytest -q`, pip-installed pytest, went green, and the
        ledger recorded the SUBSTITUTE. `unittest` was never checked again
        and the run reported Finished.
        """
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        from agentchanti.orchestrator.plan_step import PlanStep
        gate = "python -m unittest -v"
        step = PlanStep(id="11.1", step_type="TEST", index=0, verify_cmd=gate)
        kwargs = self._kwargs(self._cfg())
        kwargs["display"].steps = [{"type": "TEST"}]
        kwargs["llm_client"].supports_tools.return_value = True
        kwargs["memory"].as_dict.return_value = {}
        _run_diagnosis_loop(0, "step text", "suite failed",
                            plan_step=step, **kwargs)
        self.assertEqual(mock_rec.call_args[1]["verify_cmd"], gate)

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_test_step_recovery_falls_back_when_nothing_declared(
            self, mock_rec):
        """With no declared gate, a TEST recovery still gets a real one."""
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        from agentchanti.orchestrator.plan_step import PlanStep
        step = PlanStep(id="11.1", step_type="TEST", index=0, verify_cmd=None)
        kwargs = self._kwargs(self._cfg())
        kwargs["display"].steps = [{"type": "TEST"}]
        kwargs["llm_client"].supports_tools.return_value = True
        kwargs["memory"].as_dict.return_value = {}
        _run_diagnosis_loop(0, "step text", "suite failed",
                            plan_step=step, **kwargs)
        self.assertEqual(mock_rec.call_args[1]["verify_cmd"],
                         "python -m pytest -q")

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed"))
    def test_recovered_step_records_its_gate(self, mock_rec):
        """Otherwise nothing rechecks that gate for the rest of the run."""
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger
        get_gate_ledger().reset()
        gate = 'python -c "import g; assert g.ok()"'
        step = PlanStep(id="2.3", step_type="CODE", index=0, verify_cmd=gate)
        kwargs = self._kwargs(self._cfg())
        kwargs["llm_client"].supports_tools.return_value = True
        kwargs["memory"].as_dict.return_value = {}
        self.assertTrue(_run_diagnosis_loop(
            0, "step text", "assertion failed", plan_step=step, **kwargs))
        self.assertIn(gate, get_gate_ledger().gates())
        get_gate_ledger().reset()

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(False, "still broken"))
    def test_failed_recovery_records_no_gate(self, mock_rec):
        from agentchanti.orchestrator.pipeline import _run_diagnosis_loop
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger
        get_gate_ledger().reset()
        gate = 'python -c "import g; assert g.ok()"'
        step = PlanStep(id="2.3", step_type="CODE", index=0, verify_cmd=gate)
        kwargs = self._kwargs(self._cfg())
        kwargs["llm_client"].supports_tools.return_value = True
        kwargs["memory"].as_dict.return_value = {}
        self.assertFalse(_run_diagnosis_loop(
            0, "step text", "assertion failed", plan_step=step, **kwargs))
        self.assertEqual(get_gate_ledger().gates(), {})

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
        # Keep subproject detection out of play for the gate tests
        memory._scaffolded_subproject = None
        memory.all_files.return_value = {}
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
        executor = MagicMock()
        executor.run_command.return_value = (False, "1 failed: assert x")
        result = _handle_test_step(
            "write tests", tester, MagicMock(), MagicMock(), executor,
            "task", memory, display, 0, language="python", cfg=cfg)
        self.assertEqual(result, (True, "loop ran"))
        self.assertEqual(mock_loop.call_args[1]["verify_cmd"],
                         "python -m pytest -q")
        # The loop is grounded in the pre-run verify output + exit criterion
        ctx = mock_loop.call_args[1]["context"]
        self.assertIn("Current test status (FAILING)", ctx)
        self.assertIn("1 failed: assert x", ctx)
        self.assertIn("complete ONLY when", ctx)

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop ran"))
    def test_plan_declared_verify_reaches_code_loop(self, mock_loop):
        # A CODE step with a plan-declared verify: gains a deterministic
        # exit gate (previously the CODE loop path had none at all).
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        cfg, coder, memory, display = self._common()
        ps = PlanStep(id="1.1", step_type="CODE",
                      verify_cmd="python manage.py check")
        _handle_code_step(
            "write code", coder, MagicMock(), MagicMock(), "task",
            memory, display, 0, cfg=cfg, plan_step=ps)
        self.assertEqual(mock_loop.call_args[1]["verify_cmd"],
                         "python manage.py check")
        self.assertIn("complete ONLY when",
                      mock_loop.call_args[1]["context"])

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop ran"))
    def test_passed_gate_recorded_in_ledger(self, mock_loop):
        # A successful step's declared verify lands in the monotonic
        # ledger so later fix rounds can be checked for regressions.
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger
        get_gate_ledger().reset()
        try:
            cfg, coder, memory, display = self._common()
            ps = PlanStep(id="3.1", step_type="CODE",
                          verify_cmd="python manage.py check")
            _handle_code_step(
                "write code", coder, MagicMock(), MagicMock(), "task",
                memory, display, 0, cfg=cfg, plan_step=ps)
            self.assertEqual(get_gate_ledger().gates(),
                             {"python manage.py check": "3.1"})
        finally:
            get_gate_ledger().reset()

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(False, "loop failed"))
    def test_failed_step_not_recorded_in_ledger(self, mock_loop):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger
        get_gate_ledger().reset()
        try:
            cfg, coder, memory, display = self._common()
            ps = PlanStep(id="3.1", step_type="CODE",
                          verify_cmd="python manage.py check")
            _handle_code_step(
                "write code", coder, MagicMock(), MagicMock(), "task",
                memory, display, 0, cfg=cfg, plan_step=ps)
            self.assertEqual(get_gate_ledger().gates(), {})
        finally:
            get_gate_ledger().reset()

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop ran"))
    def test_plan_declared_verify_beats_language_default(self, mock_loop):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _handle_test_step
        cfg, tester, memory, display = self._common()
        executor = MagicMock()
        executor.run_command.return_value = (True, "OK")
        ps = PlanStep(id="5.1", step_type="TEST",
                      verify_cmd="python manage.py test main --noinput")
        _handle_test_step(
            "run tests", tester, MagicMock(), MagicMock(), executor,
            "task", memory, display, 0, language="python", cfg=cfg,
            plan_step=ps)
        self.assertEqual(mock_loop.call_args[1]["verify_cmd"],
                         "python manage.py test main --noinput")


class TestLoopEscalation(unittest.TestCase):
    """One retry with a stronger model when the first loop fails."""

    def _clients(self, esc_supports_tools=True):
        primary = MagicMock(name="primary")
        escalation = MagicMock(name="escalation")
        escalation.supports_tools.return_value = esc_supports_tools
        return primary, escalation

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_success_skips_escalation(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            run_agent_loop_with_escalation,
        )
        mock_loop.return_value = (True, "done")
        primary, escalation = self._clients()
        result = run_agent_loop_with_escalation(
            primary, MagicMock(), "step", "task",
            escalation_client=escalation)
        self.assertEqual(result, (True, "done"))
        mock_loop.assert_called_once()
        self.assertIs(mock_loop.call_args[0][0], primary)

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_failure_without_escalation_client(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            run_agent_loop_with_escalation,
        )
        mock_loop.return_value = (False, "verify failed")
        primary, _ = self._clients()
        result = run_agent_loop_with_escalation(
            primary, MagicMock(), "step", "task")
        self.assertEqual(result, (False, "verify failed"))
        mock_loop.assert_called_once()

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_failure_escalates_with_error_in_context(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            run_agent_loop_with_escalation,
        )
        mock_loop.side_effect = [
            (False, "NoReverseMatch: 'dashboard' not found"),
            (True, "fixed by stronger model"),
        ]
        primary, escalation = self._clients()
        result = run_agent_loop_with_escalation(
            primary, MagicMock(), "step", "task",
            escalation_client=escalation, context="base context")
        self.assertEqual(result, (True, "fixed by stronger model"))
        self.assertEqual(mock_loop.call_count, 2)
        # Second run uses the escalation client with the failure in context
        second = mock_loop.call_args_list[1]
        self.assertIs(second[0][0], escalation)
        self.assertIn("base context", second[1]["context"])
        self.assertIn("NoReverseMatch", second[1]["context"])

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_no_escalation_without_tool_support(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            run_agent_loop_with_escalation,
        )
        mock_loop.return_value = (False, "failed")
        primary, escalation = self._clients(esc_supports_tools=False)
        result = run_agent_loop_with_escalation(
            primary, MagicMock(), "step", "task",
            escalation_client=escalation)
        self.assertEqual(result, (False, "failed"))
        mock_loop.assert_called_once()

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_same_client_not_escalated(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            run_agent_loop_with_escalation,
        )
        mock_loop.return_value = (False, "failed")
        primary, _ = self._clients()
        result = run_agent_loop_with_escalation(
            primary, MagicMock(), "step", "task",
            escalation_client=primary)
        self.assertEqual(result, (False, "failed"))
        mock_loop.assert_called_once()


class TestAttemptJournal(AgentLoopTestCase):
    """Cross-attempt memory for the retry ladder.

    Modelled on the run that spent 54 turns and 497k tokens across four
    blind attempts, each re-editing the same three files with conflicting
    fixes, none of them finding a two-line bug.
    """

    def _record_pacman_ladder(self):
        record_attempt(0, "first attempt", "verify-failed",
                       ["tests/test_gameplay.py"] * 6,
                       [("python -m unittest discover -s tests -v", False)],
                       "Adjusted the ghost direction assertions.")
        record_attempt(0, "escalation (stronger model)", "verify-failed",
                       ["src/ghost.py", "src/game.py", "src/player.py"],
                       [("python -m unittest discover -s tests -v", False)],
                       "Rewrote ghost movement; tests still fail.")

    def test_digest_is_empty_before_any_attempt(self):
        self.assertEqual(attempt_digest(0), "")

    def test_digest_names_files_commands_and_verdicts(self):
        self._record_pacman_ladder()
        digest = attempt_digest(0)
        self.assertIn("attempt 1", digest)
        self.assertIn("attempt 2", digest)
        self.assertIn("tests/test_gameplay.py (x6)", digest)
        self.assertIn("src/ghost.py", digest)
        self.assertIn("-> FAILED", digest)
        self.assertIn("Rewrote ghost movement", digest)

    def test_digest_flags_files_churned_across_attempts(self):
        """The signal that matters: same files, still red, wrong cause."""
        record_attempt(0, "first attempt", "verify-failed",
                       ["src/ghost.py"], [], "no luck")
        record_attempt(0, "escalation", "verify-failed",
                       ["src/ghost.py"], [], "still no luck")
        digest = attempt_digest(0)
        self.assertIn("NOTE:", digest)
        self.assertIn("src/ghost.py", digest.split("NOTE:")[1])

    def test_no_churn_note_for_a_single_attempt(self):
        record_attempt(0, "first attempt", "verify-failed",
                       ["src/ghost.py", "src/ghost.py"], [], "x")
        self.assertNotIn("NOTE:", attempt_digest(0))

    def test_journal_is_per_step(self):
        record_attempt(0, "first attempt", "verify-failed", ["a.py"], [], "x")
        self.assertEqual(attempt_digest(1), "")

    def test_reset_clears_it(self):
        record_attempt(0, "first attempt", "verify-failed", ["a.py"], [], "x")
        reset_attempt_journal()
        self.assertEqual(attempt_digest(0), "")

    def test_digest_is_bounded(self):
        for i in range(12):
            record_attempt(0, f"attempt-{i}", "verify-failed",
                           [f"f{j}.py" for j in range(30)],
                           [(f"cmd-{j}", False) for j in range(20)],
                           "x" * 4000)
        digest = attempt_digest(0)
        # 4 most recent attempts only, and the oldest are dropped.
        self.assertIn("attempt-11", digest)
        self.assertNotIn("attempt-0 ", digest)
        self.assertIn("more", digest)   # file list truncated
        self.assertLess(len(digest), 6000)

    def test_loop_records_edits_and_commands(self):
        """The journal is populated from real tool traffic, not hand-fed."""
        llm = self._llm(
            _tool_response(ToolCall(
                name="write_file",
                arguments={"path": "app.py", "content": "x = 1\n"},
                id="1")),
            _tool_response(ToolCall(
                name="run_command",
                arguments={"command": "false-cmd"}, id="2")),
            _final("could not finish"),
        )
        self.executor.run_command.return_value = (False, "boom")
        ok, _ = run_agent_loop(llm, self.tools, "step", "task",
                               max_turns=3, verify_cmd="check")
        self.assertFalse(ok)
        digest = attempt_digest(0)
        self.assertIn("app.py", digest)
        self.assertIn("false-cmd", digest)
        self.assertIn("-> FAILED", digest)

    def test_failed_edit_is_not_recorded_as_an_edit(self):
        """AgentTools returns errors as strings; a rejected edit must not
        look like a change the next attempt should avoid repeating."""
        llm = self._llm(
            _tool_response(ToolCall(
                name="edit_file",
                arguments={"path": "missing.py", "old": "a", "new": "b"},
                id="1")),
            _final("gave up"),
        )
        run_agent_loop(llm, self.tools, "step", "task", max_turns=2)
        self.assertNotIn("missing.py", attempt_digest(0))

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_escalation_context_carries_digest_and_full_error(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            run_agent_loop_with_escalation,
        )
        self._record_pacman_ladder()
        mock_loop.side_effect = [
            (False, "AssertionError: (2, 2) == (2, 2)"),
            (True, "ok"),
        ]
        primary = MagicMock()
        escalation = MagicMock()
        escalation.supports_tools.return_value = True
        run_agent_loop_with_escalation(
            primary, MagicMock(), "step", "task",
            escalation_client=escalation, step_idx=0)
        ctx = mock_loop.call_args_list[1][1]["context"]
        self.assertIn("Previous attempts", ctx)          # narrative
        self.assertIn("src/ghost.py", ctx)
        self.assertIn("AssertionError: (2, 2)", ctx)     # full latest error
        self.assertEqual(
            mock_loop.call_args_list[1][1]["attempt_label"],
            "escalation (stronger model)")


class TestPlanStepBrief(unittest.TestCase):
    """Plan metadata reaches the loop — in intent mode it is all the
    loop knows about WHAT to build."""

    def test_brief_contents(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _plan_step_brief
        ps = PlanStep(id="2.1", step_type="CODE",
                      target_files=["main/views.py"],
                      exports=["home", "dashboard"],
                      imports_from={"main/urls.py": ["urlpatterns"]})
        brief = _plan_step_brief(ps)
        self.assertIn("Target files: main/views.py", brief)
        self.assertIn("Must export: home, dashboard", brief)
        self.assertIn("main/urls.py: urlpatterns", brief)
        self.assertEqual(_plan_step_brief(None), "")

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "ok"))
    def test_brief_reaches_code_loop_context(self, mock_loop):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        coder = MagicMock()
        coder.llm_client.supports_tools.return_value = True
        coder.escalation_client = None
        memory = MagicMock()
        memory.summary.return_value = "files: none"
        memory._scaffolded_subproject = None
        memory.all_files.return_value = {}
        ps = PlanStep(id="2.1", step_type="CODE",
                      target_files=["main/views.py"], exports=["home"])
        _handle_code_step(
            "create views", coder, MagicMock(), MagicMock(), "task",
            memory, MagicMock(), 0, cfg=cfg, plan_step=ps)
        ctx = mock_loop.call_args[1]["context"]
        self.assertIn("Target files: main/views.py", ctx)
        self.assertIn("Must export: home", ctx)


class TestVariantGateEscape(AgentLoopTestCase):
    """Unpassable-gate escape: a recovery gate that re-runs a malformed
    original command accepts a flag-variant success the loop produced.

    Replay: `pip install --yes pygame` (invalid flag, exit 2 forever) —
    both recovery loops installed pygame correctly and the run still
    failed on the gate."""

    _GATE = r"cd app && call venv\Scripts\activate && pip install --yes pygame"
    _FIXED = r"cd app && call venv\Scripts\activate && pip install pygame"

    def _executor_rejecting_yes(self):
        def _run(cmd, **kw):
            if "--yes" in cmd:
                return (False, "no such option: --yes")
            return (True, "Successfully installed pygame")
        self.executor.run_command.side_effect = _run

    def test_recovery_accepts_flag_variant_success(self):
        self._executor_rejecting_yes()
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": self._FIXED},
                                    id="c1")),
            _final("installed pygame"),
        )
        success, info = run_agent_loop(
            llm, self.tools, "install pygame", "task", max_turns=4,
            verify_cmd=self._GATE, _recovery=True)
        self.assertTrue(success)
        self.assertEqual(info, "installed pygame")

    def test_non_recovery_keeps_strict_gate(self):
        # CODE/TEST loops keep strict verify semantics — no escape.
        self._executor_rejecting_yes()
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": self._FIXED},
                                    id="c1")),
            _final("done"), _final("done"), _final("done"),
        )
        success, _ = run_agent_loop(
            llm, self.tools, "install pygame", "task", max_turns=4,
            verify_cmd=self._GATE, _recovery=False)
        self.assertFalse(success)

    def test_unrelated_success_not_accepted(self):
        # A successful but different command is not evidence for the gate.
        self._executor_rejecting_yes()
        llm = self._llm(
            _tool_response(ToolCall(name="run_command",
                                    arguments={"command": "echo hello"},
                                    id="c1")),
            _final("done"), _final("done"), _final("done"),
        )
        success, _ = run_agent_loop(
            llm, self.tools, "install pygame", "task", max_turns=4,
            verify_cmd=self._GATE, _recovery=True)
        self.assertFalse(success)

    def test_equivalence_helper(self):
        from agentchanti.orchestrator.agent_loop import (
            commands_equivalent_modulo_flags,
        )
        self.assertTrue(commands_equivalent_modulo_flags(
            self._FIXED, self._GATE))
        self.assertTrue(commands_equivalent_modulo_flags(
            "pip install pygame", "pip install --yes -q pygame"))
        # Identical strings prove nothing new — the gate ran it itself.
        self.assertFalse(commands_equivalent_modulo_flags(
            self._GATE, self._GATE))
        # Different non-flag args are different commands.
        self.assertFalse(commands_equivalent_modulo_flags(
            "pip install requests", "pip install --yes pygame"))
        self.assertFalse(commands_equivalent_modulo_flags(None, self._GATE))
        self.assertFalse(commands_equivalent_modulo_flags("", ""))


class TestDeclaredVerifyGate(unittest.TestCase):
    """The classic-path acceptance gate and its command resolution."""

    def _memory(self, sub=None):
        memory = MagicMock()
        memory._scaffolded_subproject = sub
        memory.all_files.return_value = (
            {f"{sub}/manage.py": ""} if sub else {})
        return memory

    def test_declared_cmd_passthrough(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        ps = PlanStep(id="1.1", step_type="CODE", verify_cmd="pytest -q")
        self.assertEqual(_declared_verify_cmd(ps, self._memory()), "pytest -q")

    def test_none_without_plan_step_or_verify(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        self.assertIsNone(_declared_verify_cmd(None, self._memory()))
        ps = PlanStep(id="1.1", step_type="CODE")
        self.assertIsNone(_declared_verify_cmd(ps, self._memory()))

    def test_scaffold_command_rejected(self):
        # A one-shot scaffold command is not a re-runnable gate
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        ps = PlanStep(id="1.1", step_type="CODE",
                      verify_cmd="django-admin startproject config .")
        self.assertIsNone(_declared_verify_cmd(ps, self._memory()))

    def test_gate_failure_flips_success(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import (
            _gate_on_declared_verify,
        )
        ps = PlanStep(id="1.1", step_type="CODE",
                      verify_cmd="python manage.py test")
        executor = MagicMock()
        executor.run_command.return_value = (False, "FAILED (errors=2)")
        ok, err = _gate_on_declared_verify(
            True, "handler said done", ps, executor, self._memory(),
            MagicMock(), 0)
        self.assertFalse(ok)
        self.assertIn("python manage.py test", err)
        self.assertIn("FAILED (errors=2)", err)

    def test_activation_and_placeholder_sanitised(self):
        # The observed plan: every verify wrapped in cd <placeholder> +
        # venv activation. The gate must survive as the bare command.
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        ps = PlanStep(
            id="2.1", step_type="CODE",
            verify_cmd=(r"cd <project_name> && call venv\Scripts\activate "
                        r"&& python manage.py check"))
        cmd = _declared_verify_cmd(ps, self._memory(),
                                   task="create a django application")
        self.assertEqual(cmd, "python manage.py check")

    def test_existing_cd_kept(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        ps = PlanStep(id="2.1", step_type="CODE",
                      verify_cmd="cd . && python manage.py check")
        cmd = _declared_verify_cmd(ps, self._memory())
        self.assertEqual(cmd, "cd . && python manage.py check")

    def test_heredoc_verify_rejected(self):
        # A planner emitted `verify: python - <<PY` (multi-line script);
        # the parser keeps only the opener and cmd.exe has no heredocs.
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        ps = PlanStep(id="2.1", step_type="CODE",
                      verify_cmd="python - <<PY")
        self.assertIsNone(_declared_verify_cmd(ps, self._memory()))

    def test_activation_only_command_is_none(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        ps = PlanStep(id="2.1", step_type="CODE",
                      verify_cmd=r"call venv\Scripts\activate")
        self.assertIsNone(_declared_verify_cmd(ps, self._memory()))

    def test_no_cd_prefix_when_cmd_references_subproject_path(self):
        # Regression (pygame run): the plan's root-relative gate
        # `unittest discover -s game` was prefixed to `cd game && ... -s
        # game`, which looks for game/game/ — unpassable by correct code.
        # The loop burned 8 turns and the escalation model "fixed" it by
        # creating a duplicate nested game/game/ package.
        from unittest.mock import patch
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        cases = [
            'python -m unittest discover -s game -p "test_*.py" -v',
            "pytest game/tests -q",
            "python -m game.main",
            r"python game\test_main.py",
        ]
        with patch("agentchanti.orchestrator.step_handlers."
                   "_detect_subproject_root", return_value="game"):
            for verify in cases:
                ps = PlanStep(id="3.1", step_type="TEST", verify_cmd=verify)
                cmd = _declared_verify_cmd(ps, self._memory())
                self.assertEqual(cmd, verify, verify)  # unprefixed, as written

    def test_cd_prefix_still_added_for_inside_style_commands(self):
        # Commands that don't reference the subproject are written to run
        # inside it — the prefix must survive this fix.
        from unittest.mock import patch
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        with patch("agentchanti.orchestrator.step_handlers."
                   "_detect_subproject_root", return_value="app"):
            ps = PlanStep(id="3.1", step_type="TEST", verify_cmd="npm test")
            self.assertEqual(_declared_verify_cmd(ps, self._memory()),
                             "cd app && npm test")
            # Substring of another word is not a reference — still prefixed.
            ps2 = PlanStep(id="3.2", step_type="TEST",
                           verify_cmd="pytest webapp/tests -q")
            with patch("agentchanti.orchestrator.step_handlers."
                       "_detect_subproject_root", return_value="web"):
                self.assertEqual(_declared_verify_cmd(ps2, self._memory()),
                                 "cd web && pytest webapp/tests -q")

    def test_gate_pass_and_noop_cases(self):
        from agentchanti.orchestrator.plan_step import PlanStep
        from agentchanti.orchestrator.step_handlers import (
            _gate_on_declared_verify,
        )
        ps = PlanStep(id="1.1", step_type="CODE", verify_cmd="pytest -q")
        executor = MagicMock()
        executor.run_command.return_value = (True, "3 passed")
        self.assertEqual(
            _gate_on_declared_verify(True, "done", ps, executor,
                                     self._memory(), MagicMock(), 0),
            (True, "done"))
        # Already-failed results pass through without running the gate
        executor.reset_mock()
        self.assertEqual(
            _gate_on_declared_verify(False, "broke", ps, executor,
                                     self._memory(), MagicMock(), 0),
            (False, "broke"))
        executor.run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestPreloadBudget(unittest.TestCase):
    """One oversized file must not empty the whole preload bundle.

    The char-budget check used to `break`, so a single file larger than
    _PRELOAD_MAX_CHARS discarded every block — including the smaller files
    behind it. Generated modules routinely run 20-40 KB, so in practice
    nothing was ever preloaded: `[PlanStep] Injected 3 plan-context files`
    logged while the loop's opening message stayed under 1.1k tokens and
    the model still spent turn 1 on read_file.
    """

    def setUp(self):
        import os
        import tempfile

        from agentchanti.executor import Executor
        from agentchanti.orchestrator.agent_loop import build_step_tools
        from agentchanti.orchestrator.memory import FileMemory

        self._prev = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        os.makedirs("src", exist_ok=True)
        # Sizes matching a real run: a 40 KB module and a 3 KB one.
        with open("src/game.py", "w", encoding="utf-8") as fh:
            fh.write("class Game:\n" + "    x = 1\n" * 4000)
        with open("src/map.py", "w", encoding="utf-8") as fh:
            fh.write("class Map:\n" + "    y = 2\n" * 300)
        self.tools = build_step_tools(Executor(), FileMemory())

    def tearDown(self):
        import os
        os.chdir(self._prev)

    def _preload(self, paths):
        from agentchanti.orchestrator.agent_loop import _preload_target_files
        return _preload_target_files(self.tools, paths)

    def test_an_oversized_file_still_preloads_truncated(self):
        from agentchanti.orchestrator.agent_loop import _PRELOAD_MAX_CHARS
        blob = self._preload(["src/game.py"])
        self.assertGreater(len(blob), _PRELOAD_MAX_CHARS // 2,
                           "an oversized file preloaded nothing at all")
        self.assertIn("truncated at", blob,
                      "a truncated preload must say so, like read_file does")

    def test_a_big_file_first_does_not_discard_the_bundle(self):
        blob = self._preload(["main.py", "src/game.py", "src/map.py"])
        self.assertIn("src/game.py", blob)

    def test_smaller_files_behind_a_big_one_are_reached(self):
        """The budget is spent, not abandoned, when a file does not fit."""
        from agentchanti.orchestrator.agent_loop import _PRELOAD_MAX_CHARS
        blob = self._preload(["src/map.py", "src/game.py"])
        self.assertIn("src/map.py", blob)
        self.assertIn("src/game.py", blob)
        self.assertLessEqual(len(blob), _PRELOAD_MAX_CHARS + 500,
                             "the char budget must still bound the bundle")

    def test_nonexistent_paths_cost_nothing(self):
        self.assertEqual(self._preload(["main.py", "nope.py"]), "")

    def test_empty_input_is_empty(self):
        self.assertEqual(self._preload([]), "")
        self.assertEqual(self._preload(None), "")


class TestPreloadListing(unittest.TestCase):
    """Orientation is the loop's reflex first move — hand it over for free.

    Measured on a 7-step run, every single step opened with `list_files`:
    a whole turn out of eight spent learning a layout the harness already
    knows. The answer also rides along in every later turn once fetched,
    so the round trip buys nothing.
    """

    def setUp(self):
        import os
        import tempfile

        from agentchanti.executor import Executor
        from agentchanti.orchestrator.agent_loop import build_step_tools
        from agentchanti.orchestrator.memory import FileMemory

        self._prev = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        os.makedirs("src", exist_ok=True)
        with open("src/map.py", "w", encoding="utf-8") as fh:
            fh.write("class Map:\n    pass\n")
        with open("main.py", "w", encoding="utf-8") as fh:
            fh.write("def main():\n    pass\n")
        self.tools = build_step_tools(Executor(), FileMemory())

    def tearDown(self):
        import os
        os.chdir(self._prev)

    def test_lists_the_tree_and_tells_the_model_not_to_repeat_it(self):
        from agentchanti.orchestrator.agent_loop import _preload_listing
        out = _preload_listing(self.tools)
        self.assertIn("main.py", out)
        self.assertIn("src/map.py", out)
        self.assertIn("do NOT call list_files", out)

    def test_a_large_tree_is_skipped_not_truncated(self):
        """A half-listing is worse than none — it reads as complete."""
        import os

        from agentchanti.orchestrator.agent_loop import _preload_listing
        for i in range(400):
            with open(f"f{i}_with_a_longish_name.py", "w",
                      encoding="utf-8") as fh:
                fh.write("x = 1\n")
        self.assertEqual(_preload_listing(self.tools), "")

    def test_a_broken_tools_object_is_survivable(self):
        from unittest.mock import MagicMock

        from agentchanti.orchestrator.agent_loop import _preload_listing
        broken = MagicMock()
        broken._tool_list_files.side_effect = RuntimeError("boom")
        self.assertEqual(_preload_listing(broken), "")

    def test_it_reaches_the_opening_message(self):
        from agentchanti.orchestrator.agent_loop import _build_user_message
        msg = _build_user_message("step", "task", "python", "",
                                  preloaded="PRELOADED",
                                  listing="LISTING")
        self.assertIn("LISTING", msg)
        self.assertIn("PRELOADED", msg)


class TestMonotonicEscalationLadder(unittest.TestCase):
    """Never step back down to the weaker model mid-ladder.

    The ladder was loop(weak) -> loop(strong) -> recovery(weak) ->
    recovery(strong). Once the stronger model had failed, the next rung
    went back to the weaker one — the least likely attempt to succeed, at
    a full turn budget to find out. Observed on a Pac-Man run: step 7 took
    32 turns across those four attempts, 47% of the whole run's turns,
    and only the final strong attempt worked.
    """

    def _clients(self):
        weak, strong = MagicMock(), MagicMock()
        weak.supports_tools.return_value = True
        strong.supports_tools.return_value = True
        return weak, strong

    def test_detects_a_failed_escalation_in_the_journal(self):
        from agentchanti.orchestrator.agent_loop import (
            ESCALATION_ATTEMPT_LABEL, escalation_already_failed,
            record_attempt,
        )
        self.assertFalse(escalation_already_failed(3))
        record_attempt(3, "first attempt", "verify-failed", [], [], "")
        self.assertFalse(escalation_already_failed(3))
        record_attempt(3, ESCALATION_ATTEMPT_LABEL, "verify-failed", [], [], "")
        self.assertTrue(escalation_already_failed(3))

    def test_a_successful_escalation_does_not_count(self):
        from agentchanti.orchestrator.agent_loop import (
            ESCALATION_ATTEMPT_LABEL, escalation_already_failed,
            record_attempt,
        )
        record_attempt(4, ESCALATION_ATTEMPT_LABEL, "verified", [], [], "")
        self.assertFalse(escalation_already_failed(4))

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_recovery_starts_strong_after_a_failed_escalation(self, mock_loop):
        from agentchanti.orchestrator.agent_loop import (
            ESCALATION_ATTEMPT_LABEL, record_attempt, run_recovery_loop,
        )
        mock_loop.return_value = (True, "fixed")
        weak, strong = self._clients()
        record_attempt(7, ESCALATION_ATTEMPT_LABEL, "verify-failed", [], [], "")
        ok, _ = run_recovery_loop(weak, MagicMock(), "step", "task", "err",
                                  step_idx=7, escalation_client=strong)
        self.assertTrue(ok)
        self.assertEqual(mock_loop.call_count, 1,
                         "the weaker model must not get a recovery run")
        self.assertIs(mock_loop.call_args[0][0], strong)

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop")
    def test_without_a_failed_escalation_recovery_starts_weak(self, mock_loop):
        """The normal ladder is unchanged — the weak model still gets a go."""
        from agentchanti.orchestrator.agent_loop import run_recovery_loop
        mock_loop.return_value = (True, "fixed")
        weak, strong = self._clients()
        ok, _ = run_recovery_loop(weak, MagicMock(), "step", "task", "err",
                                  step_idx=8, escalation_client=strong)
        self.assertTrue(ok)
        self.assertIs(mock_loop.call_args[0][0], weak)

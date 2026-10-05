"""A `verify:` that is a tool call rather than a shell command.

Measured 2026-10-06, the first agentchanti run to change a live external
application. Three ordering fixes landed first, after which the plan was
entirely tool-based and the model did the work CORRECTLY — a cube at the
origin with Z rotation keyed 0 deg at frame 1 and 360 deg at frame 60,
confirmed by evaluating the scene independently of the run. The verdict was
`Pipeline failed`, because the plan's gates were:

    verify: blender__get_scene_info
    verify: blender__get_object_info

`_merged_gate` conjoined two of them and the executor ran
`blender__get_scene_info && blender__get_object_info` through cmd.exe. No
shell command observes a running Blender session, so the gate could not pass
over any artifact.

The tests are shaped around the two ways this could go wrong, which matter
more than the parsing:

**Reading a shell gate as a tool gate** would send a real command somewhere
that cannot run it. So `parse` returns None for anything it is not certain
about, including a gate that mixes a tool call with a command, and a gate
naming a tool nobody offered.

**Turning an impossible gate into a tautological one** would trade one false
verdict for another, which is the mistake `empty_suite_reason` exists to
prevent. A read-only call that asserts nothing passes whenever the server is
up — including over the state before the step ran — so it is reported.
"""
import json

import pytest

from agentchanti.orchestrator import tool_gates


class _Bridge:
    """Stands in for `MCPBridge`: all `parse` needs is `owns`."""

    def __init__(self, *names):
        self.names = set(names)

    def owns(self, name):
        return name in self.names


class _Tools:
    """Stands in for `AgentTools`, recording what the gate asked for."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def execute_all(self, calls):
        from agentchanti.llm.chat_types import Message
        out = []
        for call in calls:
            self.calls.append((call.name, call.arguments))
            body = self.results.pop(0) if self.results else "ok"
            out.append(Message(role="tool", content=body))
        return out


def _bridge():
    return _Bridge("blender__get_scene_info", "blender__get_object_info",
                   "blender__execute_code")


class TestWhatCountsAsAToolGate:
    def test_a_bare_qualified_tool_name(self):
        """What the planner actually wrote, both times it was measured."""
        gates = tool_gates.parse("blender__get_scene_info", _bridge())
        assert gates is not None
        assert [g.tool for g in gates] == ["blender__get_scene_info"]
        assert gates[0].arguments == {}

    def test_the_merged_gate_from_the_incident(self):
        """`_merged_gate` conjoins the gates of merged steps, and a
        tool-only plan declares no file target for it to key on, so this is
        the shape that actually reached the executor."""
        gates = tool_gates.parse(
            "blender__get_scene_info && blender__get_object_info", _bridge())
        assert [g.tool for g in gates] == ["blender__get_scene_info",
                                           "blender__get_object_info"]

    def test_the_explicit_prefix_with_arguments(self):
        gate = 'mcp:blender__execute_code {"code": "assert 1 == 1"}'
        gates = tool_gates.parse(gate, _bridge())
        assert gates[0].tool == "blender__execute_code"
        assert gates[0].arguments == {"code": "assert 1 == 1"}

    def test_arguments_may_contain_spaces_and_braces(self):
        """The name/payload split is at the first brace, not the first
        space, because a code payload is full of both."""
        code = "import bpy\nd = {'a': 1}\nassert d['a'] == 1"
        gate = "mcp:blender__execute_code " + json.dumps({"code": code})
        gates = tool_gates.parse(gate, _bridge())
        assert gates[0].arguments["code"] == code

    @pytest.mark.parametrize("gate", [
        "python -m pytest -q",
        "npm test",
        'node -e "require(\'./x\')"',
    ])
    def test_an_ordinary_shell_gate_is_left_alone(self, gate):
        assert tool_gates.parse(gate, _bridge()) is None

    def test_a_mixed_gate_is_left_to_the_shell(self):
        """There is no single executor for half a tool call and half a
        command, and honouring one half would silently drop the other."""
        assert tool_gates.parse(
            "blender__get_scene_info && python -m pytest", _bridge()) is None

    def test_a_tool_nobody_offered_is_not_a_tool_gate(self):
        """An ordinary plan mistake. Routing it here would fail with a less
        useful message than the shell's."""
        assert tool_gates.parse("blender__export_scene", _bridge()) is None

    def test_no_bridge_means_no_tool_gates(self):
        assert tool_gates.parse("blender__get_scene_info", None) is None

    @pytest.mark.parametrize("payload", ["{not json}", "[1, 2]", "{"])
    def test_a_payload_that_is_not_a_json_object_is_refused(self, payload):
        """Dropping the arguments and calling the tool bare would measure
        something the plan never asked for, and PASSING that way is the
        worse of the two outcomes."""
        assert tool_gates.parse(
            f"mcp:blender__execute_code {payload}", _bridge()) is None

    def test_is_tool_gate_agrees_with_parse(self):
        b = _bridge()
        assert tool_gates.is_tool_gate("blender__get_scene_info", b) is True
        assert tool_gates.is_tool_gate("python -m pytest", b) is False


class TestTheVerdictSpeaksTheExecutorsLanguage:
    """`verify_passed`, `GateLedger.record` and `observe_gate_verdict` all
    read `exit: success`. A tool gate answers in the same vocabulary so none
    of them needs to know this module exists."""

    def test_a_working_call_passes(self):
        gates = tool_gates.parse("blender__get_scene_info", _bridge())
        out = tool_gates.run(gates, _Tools('{"scene": "Scene"}'))
        assert out.startswith("exit: success")
        from agentchanti.orchestrator.agent_loop import verify_passed
        assert verify_passed(out) is True

    @pytest.mark.parametrize("err", [
        "ERROR from the MCP tool: AssertionError",
        "ERROR: unknown MCP tool 'blender__nope'",
    ])
    def test_an_error_result_fails(self, err):
        gates = tool_gates.parse("blender__get_scene_info", _bridge())
        out = tool_gates.run(gates, _Tools(err))
        assert out.startswith("exit: failure")
        from agentchanti.orchestrator.agent_loop import verify_passed
        assert verify_passed(out) is False

    def test_it_stops_at_the_first_failure_like_the_shell_does(self):
        gates = tool_gates.parse(
            "blender__get_scene_info && blender__get_object_info", _bridge())
        tools = _Tools("ERROR: first one failed", "never reached")
        out = tool_gates.run(gates, tools)
        assert out.startswith("exit: failure")
        assert len(tools.calls) == 1, "the second segment ran after a failure"

    def test_the_arguments_reach_the_tool(self):
        gate = 'mcp:blender__get_object_info {"name": "Cube"}'
        tools = _Tools("{}")
        tool_gates.run(tool_gates.parse(gate, _bridge()), tools)
        assert tools.calls == [("blender__get_object_info", {"name": "Cube"})]

    def test_a_raising_dispatch_is_a_failure_not_a_pass(self):
        """`AgentTools.execute` is documented never to raise, but if the
        dispatch around it does, the step must not go green carrying it."""
        class _Boom:
            def execute_all(self, calls):
                raise RuntimeError("transport died")
        out = tool_gates.run(
            tool_gates.parse("blender__get_scene_info", _bridge()), _Boom())
        assert out.startswith("exit: failure")
        assert "transport died" in out

    def test_the_gate_text_is_echoed_so_a_reader_can_see_it(self):
        """`GateLedger` keeps each gate's last output, and a verdict with no
        record of what was measured is what `_green_suites_contradicting`
        had to be fixed for."""
        out = tool_gates.run(
            tool_gates.parse("blender__get_scene_info", _bridge()),
            _Tools("{}"))
        assert "blender__get_scene_info" in out


class TestAToolGateThatCannotFailIsStillShallow:
    """Measured: `verify: blender__get_scene_info` passes whenever the
    server is reachable, including over the scene as it was before the step
    ran. Making that gate RUN without saying so would replace a verdict
    that was always red with one that is always green."""

    def test_a_read_only_call_with_no_assertion_is_reported(self):
        gates = tool_gates.parse("blender__get_scene_info", _bridge())
        reason = tool_gates.shallow_tool_gate_reason(gates)
        assert reason and "asserts nothing" in reason

    def test_the_reason_names_the_fix(self):
        gates = tool_gates.parse("blender__get_scene_info", _bridge())
        reason = tool_gates.shallow_tool_gate_reason(gates)
        assert "execute_code" in reason and "assert" in reason

    def test_an_assertion_in_the_payload_clears_it(self):
        gate = ("mcp:blender__execute_code "
                + json.dumps({"code": "import bpy; assert bpy.data.objects"}))
        assert tool_gates.shallow_tool_gate_reason(
            tool_gates.parse(gate, _bridge())) is None

    def test_a_comparison_counts_as_an_assertion(self):
        """Not every language spells it `assert`."""
        gate = ("mcp:blender__execute_code "
                + json.dumps({"code": "if x !== 60 { throw 1 }"}))
        assert tool_gates.shallow_tool_gate_reason(
            tool_gates.parse(gate, _bridge())) is None

    def test_one_strong_segment_is_enough(self):
        """A gate with any segment that can fail IS a measurement."""
        gate = ("blender__get_scene_info && mcp:blender__execute_code "
                + json.dumps({"code": "assert 1"}))
        gates = tool_gates.parse(gate, _bridge())
        assert gates is not None and len(gates) == 2
        assert tool_gates.shallow_tool_gate_reason(gates) is None

    def test_no_gates_is_not_a_complaint(self):
        assert tool_gates.shallow_tool_gate_reason([]) is None


class TestItIsActuallyWiredIntoTheLoop:
    """`protect_acceptance_files` existed, was called only from a test, and
    was INERT in production for as long as that was true."""

    def test_verify_once_consults_tool_gates(self):
        import inspect
        from agentchanti.orchestrator import agent_loop
        src = inspect.getsource(agent_loop.run_agent_loop)
        assert "tool_gates.parse(" in src, (
            "the loop's gate runner never asks whether the gate is a tool "
            "call, so it goes to the shell and can only fail")
        assert "tool_gates.run(" in src
        # and the weakness is surfaced rather than silently accepted
        assert "shallow_tool_gate_reason" in src

    def test_the_shell_path_is_still_the_default(self):
        """A run with no MCP server must behave exactly as before."""
        import inspect
        from agentchanti.orchestrator import agent_loop
        src = inspect.getsource(agent_loop.run_agent_loop)
        assert "_verify_call(_cmd)" in src, (
            "the ordinary shell gate path was removed")

    def test_the_planner_is_told_the_syntax(self):
        """The guard is the backstop; a gate that can fail from the start
        costs nothing."""
        from agentchanti.mcp_bridge import (MCPBridge, MCPServerSpec,
                                            planner_summary, tool_defs_for)

        class _T:
            def __init__(self, n):
                self.name = n
                self.description = ""
                self.inputSchema = {"type": "object", "properties": {}}

        spec = MCPServerSpec("blender", command="s",
                             allow=["execute_code"])
        bridge = MCPBridge([spec])
        offered, _ = tool_defs_for("blender", [_T("execute_code")], spec)
        bridge._defs.extend(offered)
        out = planner_summary(bridge)
        assert "mcp:" in out
        assert "must be able to FAIL" in out

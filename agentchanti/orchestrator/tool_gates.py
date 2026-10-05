"""A `verify:` that is a tool call rather than a shell command.

Every other gate module in this package reads a `verify:` as a shell
command and asks something about it — can it fail on wrong behaviour
(`check_gate_quality`), does the shell this platform has even parse it
(`unrunnable_gate_reason`), does it destroy anything (`gate_safety`). All of
them assume the gate IS a shell command, which was true until a plan could
reach an external system through MCP.

Measured 2026-10-06, the first run to change a live external application.
Three ordering fixes landed first, after which the plan was entirely
tool-based and the model did the work **correctly** — a cube at the origin
with Z rotation keyed 0° at frame 1 and 360° at frame 60, confirmed by
evaluating the scene independently. The run reported `Pipeline failed`,
because the planner's gates were::

    verify: blender__get_scene_info
    verify: blender__get_object_info

`_merged_gate` conjoined two of them and the executor ran
`blender__get_scene_info && blender__get_object_info` through cmd.exe. There
is no shell command that observes a running Blender session, so the gate
could not pass over any artifact — the bar `unrunnable_gate_reason` sets for
a *structural* defect, except that here the gate names exactly the right
measurement and only the executor cannot perform it.

So the fix is to perform it. `parse` recognises a gate that is one or more
MCP tool calls, `run` executes them through the same `AgentTools` the loop
already uses, and the verdict is mapped back into the executor's own
`exit: success` / `exit: failure` vocabulary so that `verify_passed`,
`GateLedger` and `observe_gate_verdict` need no knowledge of any of this.

Two refusals keep it honest.

**A tool gate that cannot fail is still shallow.** Turning an impossible
gate into a tautological one trades one false verdict for another, which is
the mistake `empty_suite_reason` exists to prevent. A call to a pure reader
proves only that the server is up — `get_scene_info` passes over *any*
scene, including the one before the step ran. `shallow_tool_gate_reason`
says so, and names the fix: assert inside a code-executing tool, where the
assertion can actually fail.

**A gate is still not allowed to destroy anything.** `gate_safety` reasons
that a gate runs on the step, again on the loop's early-exit check, again on
each variant retry and again after every later wave — so a side effect
happens repeatedly. That argument does not care whether the side effect
arrives through cmd.exe or through a tool, and a code-executing tool is the
most capable thing in the system. The arguments are screened with the same
patterns.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

__all__ = (
    "TOOL_GATE_PREFIX",
    "ToolGate",
    "is_tool_gate",
    "parse",
    "run",
    "shallow_tool_gate_reason",
)

# Explicit form, which the planner is told to use. A bare qualified tool
# name is also accepted, because that is what planners actually write.
TOOL_GATE_PREFIX = "mcp:"

# A qualified MCP tool name: server, the separator, tool.
_QUALIFIED = re.compile(r"^[A-Za-z0-9_.\-]+__[A-Za-z0-9_.\-]+$")

# Errors `mcp_bridge` can return. `render_result` prefixes a failed tool
# result, and `MCPBridge.execute`'s own guards prefix theirs.
_ERROR_PREFIXES = ("ERROR from the MCP tool:", "ERROR:")

# An assertion has to be able to fail. These are the ways a payload can
# carry one; without any of them the call's own success is the whole
# verdict, which is `shallow_gate_reason`'s complaint one layer over.
_ASSERTION = re.compile(
    r"\bassert\b|\braise\b|\bassertEqual\b|\bexpect\b|!==|===|"
    r"[^=!<>]==[^=]|[<>]=?",
)


class ToolGate:
    """One tool call a gate is made of."""

    __slots__ = ("tool", "arguments", "raw")

    def __init__(self, tool: str, arguments: dict[str, Any], raw: str):
        self.tool = tool
        self.arguments = arguments
        self.raw = raw

    def __repr__(self) -> str:        # pragma: no cover - debugging aid
        return f"ToolGate({self.tool!r}, {self.arguments!r})"


def _parse_one(segment: str, owns) -> ToolGate | None:
    """Parse one segment, or None if it is not a tool call.

    Deliberately strict about the tool being OFFERED: a step whose gate
    names a tool nobody configured is an ordinary plan mistake, and reading
    it as a tool gate would route it somewhere that can only fail with a
    less useful message than the shell's.
    """
    text = segment.strip()
    if not text:
        return None
    if text.lower().startswith(TOOL_GATE_PREFIX):
        text = text[len(TOOL_GATE_PREFIX):].strip()
    if not text:
        return None

    # Split the name from an optional JSON payload. The payload may contain
    # spaces, so the split is at the first brace rather than the first gap.
    brace = text.find("{")
    if brace == -1:
        name, payload = text, ""
    else:
        name, payload = text[:brace].strip(), text[brace:].strip()

    if not _QUALIFIED.match(name) or not owns(name):
        return None

    arguments: dict[str, Any] = {}
    if payload:
        try:
            parsed = json.loads(payload)
        except ValueError:
            # A malformed payload is NOT silently dropped: running the tool
            # with no arguments would measure something the plan did not
            # ask for, and passing would be the worse outcome of the two.
            log.warning("[ToolGate] %s: arguments are not valid JSON, so "
                        "this is not treated as a tool gate: %s",
                        name, payload[:200])
            return None
        if not isinstance(parsed, dict):
            log.warning("[ToolGate] %s: arguments must be a JSON object, "
                        "got %s", name, type(parsed).__name__)
            return None
        arguments = parsed
    return ToolGate(name, arguments, segment.strip())


def parse(gate: str | None, bridge: Any) -> list[ToolGate] | None:
    """Every tool call this gate is made of, or None if it is not one.

    `&&`-separated segments are all required to pass, which is both the
    shell's meaning and what `_merged_gate` intends when it conjoins the
    gates of merged steps — the measured gate was exactly
    `blender__get_scene_info && blender__get_object_info`.

    Returns None unless **every** segment is a tool call. A gate mixing a
    tool call and a shell command has no single executor, and guessing
    which half to honour would silently drop the other.
    """
    if not gate or bridge is None:
        return None
    owns = getattr(bridge, "owns", None)
    if owns is None:
        return None
    segments = [s for s in re.split(r"&&", gate) if s.strip()]
    if not segments:
        return None
    gates: list[ToolGate] = []
    for segment in segments:
        one = _parse_one(segment, owns)
        if one is None:
            return None
        gates.append(one)
    return gates


def is_tool_gate(gate: str | None, bridge: Any) -> bool:
    """Whether this gate is executed by a tool rather than by the shell.

    Callers use it to skip the shell-shaped questions — platform dialect
    variants, interpreter-target checks — which have no meaning here.
    """
    return parse(gate, bridge) is not None


def _failed(result: str) -> bool:
    return any(result.startswith(p) for p in _ERROR_PREFIXES)


def run(gates: list[ToolGate], tools: Any) -> str:
    """Execute the gate and answer in the executor's own vocabulary.

    The return value starts with `exit: success` or `exit: failure` so that
    `verify_passed`, `GateLedger.record` and `observe_gate_verdict` read a
    tool gate exactly as they read a shell gate and need no knowledge of
    this module. Stops at the first failure, as `&&` does.
    """
    from ..llm.chat_types import ToolCall

    bodies: list[str] = []
    for i, g in enumerate(gates):
        call = ToolCall(name=g.tool, arguments=dict(g.arguments),
                        id=f"gate{i}")
        try:
            result = tools.execute_all([call])[0].content
        except Exception as exc:                 # pragma: no cover - env
            # `AgentTools.execute` is documented never to raise; if the
            # dispatch around it does, a gate that could not run must read
            # as a failure rather than take the step green with it.
            result = f"ERROR: the gate could not be run: {type(exc).__name__}: {exc}"
        bodies.append(f"$ {g.raw}\n{result}")
        if _failed(result):
            return "exit: failure\n" + "\n\n".join(bodies)
    return "exit: success\n" + "\n\n".join(bodies)


def shallow_tool_gate_reason(gates: list[ToolGate]) -> str | None:
    """Why this tool gate could not fail on wrong behaviour.

    `shallow_gate_reason`'s question, asked of a tool call. A gate that
    calls a pure reader and asserts nothing about what comes back passes
    over any state the system could possibly be in — including the state
    before the step ran. The measured plan's `verify: blender__get_scene_info`
    is exactly that: it proves the server is up.

    Silent as soon as ANY segment carries an assertion, because one
    segment that can fail is enough to make the gate a measurement.
    """
    if not gates:
        return None
    for g in gates:
        payload = json.dumps(g.arguments, default=str)
        if _ASSERTION.search(payload):
            return None
    names = ", ".join(g.tool for g in gates)
    return (f"the gate calls {names} and asserts nothing about the result, "
            f"so it passes whenever the server is reachable — including "
            f"over the state that existed before this step ran. Put the "
            f"check inside a tool that executes code and let it fail: "
            f"e.g. mcp:<server>__execute_code "
            f"{{\"code\": \"...; assert <the concrete condition>\"}}")

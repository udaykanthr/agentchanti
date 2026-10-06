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

import ast
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

# Existence, type and constant checks hold over anything the system could
# possibly be, so an assertion built only from them cannot fail. Named here
# rather than reusing `seed_strength._substantive_assertions`, which is
# calibrated for unittest SUITES: measured against the real gates, it scores
# the tautology `assert scene is not None; assert isinstance(objs, list)` as
# 2 (passing it) and a single correct `assert abs(z - 360.0) < 0.01` as 1
# (failing it at that module's two-assertion bar). Borrowing it would have
# been a regression in both directions.
_EXISTENCE_CALLS = frozenset({"hasattr", "isinstance", "callable", "len"})


def _is_constant(node: ast.AST) -> bool:
    return isinstance(node, (ast.Constant, ast.Tuple, ast.List, ast.Set))


def _can_fail(test: ast.AST) -> bool:
    """Whether this assertion's subject could make it false.

    Three shapes cannot, and all three were observed in real gates:
    `assert x is not None` (existence), `assert isinstance(x, T)` (type),
    and `assert <const> == <const>` (tautology). Everything else compares
    something read from the system against something, which is a claim.
    """
    # `assert x` / `assert x.y` — truthiness of a thing that exists.
    if isinstance(test, (ast.Name, ast.Attribute, ast.Subscript)):
        return False
    if _is_constant(test):
        return False
    if isinstance(test, ast.Call):
        func = test.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else "")
        # `assert isinstance(...)` on its own says nothing about behaviour;
        # `assert len(x) == 3` is a comparison and reaches the branch below.
        return name not in _EXISTENCE_CALLS
    if isinstance(test, ast.Compare):
        left, comparators = test.left, test.comparators
        # `is None` / `is not None` with nothing else is an existence check.
        if all(isinstance(op, (ast.Is, ast.IsNot)) for op in test.ops) and all(
                isinstance(c, ast.Constant) and c.value is None
                for c in comparators):
            return False
        # Two constants compared to each other assert nothing.
        if _is_constant(left) and all(_is_constant(c) for c in comparators):
            return False
        return True
    if isinstance(test, ast.BoolOp):
        # `A and B` can fail if either side can; `A or B` needs both.
        parts = [_can_fail(v) for v in test.values]
        return any(parts) if isinstance(test.op, ast.And) else all(parts)
    if isinstance(test, ast.UnaryOp):
        return _can_fail(test.operand)
    return True


def _falsifiable_assertions(code: str) -> int | None:
    """How many assertions in this payload could actually fail.

    None when the payload is not parseable Python — a JavaScript or shell
    payload is a different question and must not be accused on the strength
    of a Python parse failing.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and _can_fail(node.test):
            count += 1
        elif isinstance(node, ast.Raise):
            # `if x != 60: raise AssertionError(...)` is the same claim
            # spelled differently, and real gates spell it both ways.
            count += 1
    return count


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
        # Python payloads are parsed, because the shapes that cannot fail are
        # decidable and were all observed in real gates: `assert scene is not
        # None`, `assert isinstance(objs, list)`, `assert 1 == 1`. A regex
        # that only looks for the word `assert` passes every one of them.
        for value in g.arguments.values():
            if not isinstance(value, str):
                continue
            falsifiable = _falsifiable_assertions(value)
            if falsifiable is None:
                continue        # not Python; the text check below decides
            if falsifiable:
                return None
        payload = json.dumps(g.arguments, default=str)
        if _ASSERTION.search(payload) and not any(
                isinstance(v, str) and _falsifiable_assertions(v) == 0
                for v in g.arguments.values()):
            return None
    names = ", ".join(g.tool for g in gates)
    return (f"the gate calls {names} and asserts nothing about the result, "
            f"so it passes whenever the server is reachable — including "
            f"over the state that existed before this step ran. Put the "
            f"check inside a tool that executes code and let it fail: "
            f"e.g. mcp:<server>__execute_code "
            f"{{\"code\": \"...; assert <the concrete condition>\"}}")

"""External tools from MCP servers, offered to the agent loop.

WHY THIS IS NOT CALLED `agentchanti/mcp/`

A package of that name would be a second claimant to the module name `mcp`
for anything doing a relative-looking import, which is the exact hazard
`package_shadow_reason` exists to refuse one directory up. One module, an
unambiguous name.

WHAT IT IS FOR

`AgentTools` is already a tool registry: `definitions()` says what the model
may call and `execute()` runs one call, returning a string even on failure.
An MCP server is another source of the same thing, so the bridge's whole job
is to produce `ToolDef`s and run calls under the same contract.

THE RISK THIS MODULE IS SHAPED AROUND

`_tool_write_file` carries SEVEN refusals — the acceptance-instrument guard,
`phantom_root_manifest_reason`, the seeded-contract guard,
`package_shadow_reason`, `stdlib_shadow`, `shadowed_dist` and
`toolchain_shim` — and every one of them lives inside that method. An
external tool that writes files reaches NONE of them.

The sharpest consequence is evidence integrity. The whole of `evidence.py`
rests on one sentence: the model neither wrote nor can edit the acceptance
instrument. That is enforced only by `_acceptance_refusal` on
`AgentTools.write_file`. Hand the loop an MCP server that can write, and a
run could rewrite its own acceptance contract while
`require_independent_evidence` still reported `independent` — not a new hole
but the destruction of the guarantee six shipped language contracts exist to
provide.

So this release admits READ-ONLY servers only, and says so out loud rather
than hoping. A server is offered to the model when the operator marks it
`read_only: true` or names an explicit `allow:` list; anything else is
withheld WITH ITS REASON, because a tool silently missing is
indistinguishable from a server that failed to start. Write-capable servers
wait for the guards to move below the tool boundary, where `AgentTools`,
`Executor` and this module can share them.

WHAT "UNAVAILABLE" MUST NEVER DO

`mcp` is an optional dependency (`pip install agentchanti[mcp]`). Absent
package, unreachable server, a server that dies mid-run — every one of them
yields no tools and one warning, never an exception into the pipeline. An
absent instrument must not convict the code, which is the rule
`_INCONCLUSIVE_MARKERS` and the toolchain skips in every shipped contract
already follow.

ONE EVENT LOOP, OWNED BY ONE THREAD

The MCP client is async while `AgentTools.execute` is sync, and an MCP
session is stateful — so a loop created and destroyed per call would drop
the session every time. A dedicated thread owns one loop for the bridge's
lifetime and calls are handed to it with `run_coroutine_threadsafe`. The
gateway prototype this borrows its catalog shape from puts the reason
bluntly: a dead event loop is a subtle, miserable class of bug.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .llm.chat_types import ToolDef

log = logging.getLogger(__name__)

# A tool name the model sees is `server__tool`. Namespacing is not cosmetic:
# a server exposing `read_file` would otherwise shadow the guarded built-in
# of that name, and `AgentTools.execute` dispatches by name.
NAME_SEPARATOR = "__"

# Bounds, matching what `run_command` already enforces. A server is another
# process doing who-knows-what, and the dev-server incident is the
# precedent: an unbounded call hung a pipeline for four hours.
DEFAULT_CALL_TIMEOUT = 60.0
MAX_RESULT_CHARS = 20_000


@dataclass(frozen=True)
class WithheldTool:
    """A tool that exists and was not offered, and why.

    Carried beside the offered set rather than dropped, because a tool
    missing without explanation reads exactly like a server that failed to
    start — and the operator would debug the wrong thing. Borrowed from the
    gateway prototype's `WithheldTool`, which keeps the rule id for the
    same reason.
    """

    server: str
    tool: str
    reason: str

    @property
    def qualified(self) -> str:
        return f"{self.server}{NAME_SEPARATOR}{self.tool}"


@dataclass(frozen=True)
class MCPServerSpec:
    """One configured server.

    `read_only` and `allow` are the operator's assertion, never ours: the
    MCP protocol's `readOnlyHint` is optional and advisory, so a server can
    simply not set it, and trusting its absence either way would be
    guessing about something that writes files.
    """

    name: str
    transport: str = "stdio"          # "stdio" | "http"
    command: str | None = None        # stdio
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None            # http
    headers: dict[str, str] = field(default_factory=dict)
    read_only: bool = False
    allow: tuple[str, ...] = ()
    timeout: float = DEFAULT_CALL_TIMEOUT

    def problem(self) -> str | None:
        """Why this spec cannot be used, or None."""
        if not self.name:
            return "a server entry has no name"
        if NAME_SEPARATOR in self.name:
            return (f"server name {self.name!r} contains "
                    f"{NAME_SEPARATOR!r}, which separates the server from "
                    f"the tool in a qualified name")
        if self.transport == "stdio":
            if not self.command:
                return f"server {self.name!r} is stdio but names no command"
        elif self.transport == "http":
            if not self.url:
                return f"server {self.name!r} is http but names no url"
        else:
            return (f"server {self.name!r} has unknown transport "
                    f"{self.transport!r} (expected 'stdio' or 'http')")
        if not self.read_only and not self.allow:
            # The Phase-1 fence, stated as a configuration error so the
            # operator sees it at startup rather than wondering where the
            # tools went.
            return (f"server {self.name!r} is neither marked "
                    f"`read_only: true` nor given an `allow:` list. This "
                    f"release offers read-only MCP tools only, because a "
                    f"tool that writes files would bypass every guard on "
                    f"AgentTools.write_file — including the one that keeps "
                    f"the acceptance instrument unwritable")
        return None


def load_specs(raw: Any) -> tuple[list[MCPServerSpec], list[str]]:
    """Parse the `mcp:` config section. Returns (specs, problems).

    Never raises: a malformed entry becomes a problem string and the rest
    are still loaded, because one bad server should not cost a run its
    other tools.
    """
    specs: list[MCPServerSpec] = []
    problems: list[str] = []
    if not raw:
        return specs, problems
    entries = raw.get("servers") if isinstance(raw, dict) else raw
    if isinstance(entries, dict):
        # Mapping form: {name: {...}} — the name is the key.
        entries = [{**v, "name": k} for k, v in entries.items()
                   if isinstance(v, dict)]
    if not isinstance(entries, list):
        return specs, [f"mcp config should be a list or mapping of servers, "
                       f"got {type(entries).__name__}"]
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"mcp server entry is not a mapping: {entry!r}")
            continue
        try:
            spec = MCPServerSpec(
                name=str(entry.get("name") or "").strip(),
                transport=str(entry.get("transport") or "stdio").lower(),
                command=entry.get("command"),
                args=tuple(str(a) for a in (entry.get("args") or ())),
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                url=entry.get("url"),
                headers={str(k): str(v)
                         for k, v in (entry.get("headers") or {}).items()},
                read_only=bool(entry.get("read_only")),
                allow=tuple(str(a) for a in (entry.get("allow") or ())),
                timeout=float(entry.get("timeout") or DEFAULT_CALL_TIMEOUT),
            )
        except (TypeError, ValueError) as exc:
            problems.append(f"mcp server entry {entry!r} is malformed: {exc}")
            continue
        problem = spec.problem()
        if problem:
            problems.append(problem)
            continue
        if spec.name in seen:
            problems.append(f"duplicate mcp server name {spec.name!r} — "
                            f"names qualify tool names and must be unique")
            continue
        seen.add(spec.name)
        specs.append(spec)
    return specs, problems


def mcp_available() -> tuple[bool, str]:
    """Whether the optional `mcp` package can be imported, and why not."""
    try:
        import mcp  # noqa: F401
    except Exception as exc:                       # pragma: no cover - env
        return False, (f"the `mcp` package is not importable ({exc.__class__.__name__}"
                       f"); install it with `pip install agentchanti[mcp]`")
    return True, ""


def split_qualified(name: str) -> tuple[str, str] | None:
    """`server__tool` -> ("server", "tool"), or None if not qualified."""
    if NAME_SEPARATOR not in name:
        return None
    server, _, tool = name.partition(NAME_SEPARATOR)
    if not server or not tool:
        return None
    return server, tool


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Cap a result, keeping BOTH ends.

    A head-only slice hands the model everything except the conclusion,
    which `truncate_middle` was written for after a recovery loop spent its
    budget hunting an error the slice had cut off.
    """
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return (text[:head] + f"\n... [{len(text) - limit} characters elided] ...\n"
            + text[-tail:])


def tool_defs_for(server: str, tools: list[Any],
                  spec: MCPServerSpec) -> tuple[list[ToolDef],
                                                list[WithheldTool]]:
    """Translate one server's `tools/list` into our own ToolDefs.

    Translated rather than passed through, because agentchanti is
    provider-agnostic: the same tools have to reach Ollama, LM Studio,
    OpenAI, Gemini and Anthropic through `chat(messages, tools)`. Routing
    MCP through any one provider's built-in support would make these tools
    work on that provider and silently vanish on the other four.
    """
    offered: list[ToolDef] = []
    withheld: list[WithheldTool] = []
    for tool in tools:
        name = getattr(tool, "name", None) or (
            tool.get("name") if isinstance(tool, dict) else None)
        if not name:
            continue
        if spec.allow and name not in spec.allow:
            withheld.append(WithheldTool(
                server, name,
                "not in this server's `allow:` list"))
            continue
        desc = (getattr(tool, "description", None)
                or (tool.get("description") if isinstance(tool, dict) else None)
                or f"{name} (from MCP server {server})")
        schema = (getattr(tool, "inputSchema", None)
                  or (tool.get("inputSchema") if isinstance(tool, dict) else None)
                  or {"type": "object", "properties": {}})
        offered.append(ToolDef(
            name=f"{server}{NAME_SEPARATOR}{name}",
            description=f"[{server}] {desc}",
            parameters=schema if isinstance(schema, dict) else
            {"type": "object", "properties": {}},
        ))
    return offered, withheld


class MCPBridge:
    """Tools from configured MCP servers, under `AgentTools`' contract.

    Deliberately inert until `start()` succeeds, and safe to use when it
    does not: `definitions()` returns an empty list and `execute()` returns
    an error string. Nothing here raises into the pipeline.
    """

    def __init__(self, specs: list[MCPServerSpec] | None = None):
        self._specs = {s.name: s for s in (specs or [])}
        self._defs: list[ToolDef] = []
        self._withheld: list[WithheldTool] = []
        self._sessions: dict[str, Any] = {}
        self._problems: list[str] = []
        self._loop = None
        self._thread: threading.Thread | None = None
        self._stack: Any = None
        self._started = False

    def start(self) -> bool:
        """Connect every configured server; True if any tool is offered.

        Returns rather than raises on every failure path — an absent or
        broken MCP server must cost the run its tools and nothing else.
        """
        try:
            return _bridge_start(self)
        except Exception as exc:                    # pragma: no cover - env
            self._problems.append(
                f"the MCP bridge could not start: {type(exc).__name__}: {exc}")
            self._started = False
            return False

    def stop(self) -> None:
        """Close sessions and stop the loop. Safe to call more than once."""
        try:
            _bridge_stop(self)
        except Exception as exc:                    # pragma: no cover - env
            log.debug("[MCP] stop raised, continuing: %s", exc)

    def __enter__(self) -> "MCPBridge":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # ---------------------------------------------------------------- state
    @property
    def problems(self) -> list[str]:
        """Everything that went wrong, for one warning at startup."""
        return list(self._problems)

    @property
    def withheld(self) -> list[WithheldTool]:
        return list(self._withheld)

    def definitions(self) -> list[ToolDef]:
        return list(self._defs)

    def owns(self, name: str) -> bool:
        """Whether *name* is one of this bridge's qualified tool names."""
        return any(d.name == name for d in self._defs)

    # --------------------------------------------------------------- running
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one qualified tool call, returning a string either way.

        Mirrors `AgentTools.execute`: an error is a result the model reads,
        never an exception, because the loop has to be able to carry on and
        say what went wrong.
        """
        if not self._started:
            return (f"ERROR: MCP tool '{name}' is unavailable — no MCP "
                    f"server session is running.")
        parts = split_qualified(name)
        if parts is None:
            return (f"ERROR: '{name}' is not a qualified MCP tool name "
                    f"(expected server{NAME_SEPARATOR}tool).")
        server, tool = parts
        if not self.owns(name):
            offered = ", ".join(d.name for d in self._defs) or "none"
            return (f"ERROR: unknown MCP tool '{name}'. Offered: {offered}")
        session = self._sessions.get(server)
        if session is None:
            return (f"ERROR: MCP server '{server}' has no live session; its "
                    f"tools are unavailable for the rest of this run.")
        spec = self._specs.get(server)
        timeout = spec.timeout if spec else DEFAULT_CALL_TIMEOUT
        try:
            result = self._call(session, tool, arguments, timeout)
        except TimeoutError:
            return (f"ERROR: MCP tool '{name}' did not return within "
                    f"{timeout:.0f}s and was abandoned.")
        except Exception as exc:                    # pragma: no cover - env
            return f"ERROR: MCP tool '{name}' failed: {type(exc).__name__}: {exc}"
        return _truncate(result)

    def _call(self, session: Any, tool: str, arguments: dict[str, Any],
              timeout: float) -> str:
        """Hand the coroutine to the bridge's own loop and wait."""
        import asyncio
        coro = session.call_tool(tool, arguments or {})
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        response = future.result(timeout=timeout)
        return render_result(response)


def render_result(response: Any) -> str:
    """Flatten an MCP tool result into text.

    An error result is returned as text rather than raised, for the reason
    `AgentTools.execute` never raises: the model has to be able to read what
    went wrong and act on it.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    parts: list[str] = []
    for item in content or ():
        text = (getattr(item, "text", None)
                or (item.get("text") if isinstance(item, dict) else None))
        if text:
            parts.append(str(text))
            continue
        kind = (getattr(item, "type", None)
                or (item.get("type") if isinstance(item, dict) else None)
                or "content")
        # Binary and resource blocks are named rather than dumped: a
        # base64 image in the conversation is tokens the model cannot use.
        parts.append(f"[{kind} block, not rendered as text]")
    body = "\n".join(parts) if parts else "(the tool returned no content)"
    is_error = bool(getattr(response, "isError", None)
                    or (isinstance(response, dict) and response.get("isError")))
    return f"ERROR from the MCP tool: {body}" if is_error else body


# --------------------------------------------------------------------------
# Connecting. Everything below imports `mcp` lazily, so this module is
# importable — and every function above it usable — on an install without
# the optional dependency.
# --------------------------------------------------------------------------

def _start_loop() -> tuple[Any, threading.Thread]:
    """A dedicated event loop on its own daemon thread.

    One loop for the bridge's lifetime, because an MCP session is stateful:
    a loop created per call would drop the session every time, and
    `asyncio.run` inside a method called from an already-running loop
    raises outright.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    # Daemon: a server that will not shut down must not keep the CLI alive.
    # The orphaned `next dev` that hung a pipeline for four hours is the
    # precedent for not trusting a child process to exit.
    thread = threading.Thread(target=_run, name="agentchanti-mcp",
                              daemon=True)
    thread.start()
    ready.wait(timeout=10)
    return loop, thread


async def _open_session(spec: MCPServerSpec, stack: Any) -> Any:
    """Open one server's session inside *stack*, returning it initialised."""
    from mcp import ClientSession
    if spec.transport == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=spec.command or "",
                                       args=list(spec.args),
                                       env=dict(spec.env) or None)
        read, write = await stack.enter_async_context(stdio_client(params))
    else:
        from mcp.client.streamable_http import streamablehttp_client
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(spec.url or "", headers=dict(spec.headers)))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


class _BridgeRuntime:
    """What `start()` builds: the loop, the exit stack and the sessions."""

    def __init__(self) -> None:
        self.stack: Any = None


def _bridge_start(bridge: "MCPBridge") -> bool:
    """Connect every configured server. Returns whether anything is offered.

    Each server is opened independently and a failure is recorded rather
    than raised: one unreachable server must not cost a run the tools of
    the others, and it must not cost the run at all.
    """
    import asyncio
    from contextlib import AsyncExitStack

    # "Nothing configured" comes FIRST. Asked the other way round, every
    # user who never wanted MCP got told to install it — a warning about an
    # absent dependency nobody asked for is pure noise, and noise in the
    # startup output is what trains a reader to skip it.
    if not bridge._specs:
        return False
    ok, why = mcp_available()
    if not ok:
        bridge._problems.append(why)
        return False

    bridge._loop, bridge._thread = _start_loop()
    bridge._started = True
    stack = AsyncExitStack()
    bridge._stack = stack

    async def _enter() -> None:
        await stack.__aenter__()

    asyncio.run_coroutine_threadsafe(_enter(), bridge._loop).result(timeout=30)

    for name, spec in bridge._specs.items():
        async def _connect(spec=spec):
            session = await _open_session(spec, stack)
            listed = await session.list_tools()
            return session, listed

        try:
            future = asyncio.run_coroutine_threadsafe(_connect(), bridge._loop)
            session, listed = future.result(timeout=max(spec.timeout, 30.0))
        except Exception as exc:
            # Named, not swallowed: a server missing from the tool list with
            # no explanation reads exactly like one the operator forgot to
            # configure.
            bridge._problems.append(
                f"MCP server {name!r} did not start: "
                f"{type(exc).__name__}: {exc}")
            continue
        tools = getattr(listed, "tools", None) or []
        defs, withheld = tool_defs_for(name, list(tools), spec)
        bridge._sessions[name] = session
        bridge._defs.extend(defs)
        bridge._withheld.extend(withheld)
        log.info("[MCP] %s: %d tool(s) offered%s", name, len(defs),
                 f", {len(withheld)} withheld" if withheld else "")
    return bool(bridge._defs)


def _bridge_stop(bridge: "MCPBridge") -> None:
    """Close every session and stop the loop. Never raises."""
    import asyncio

    stack = getattr(bridge, "_stack", None)
    if stack is not None and bridge._loop is not None:
        async def _exit() -> None:
            await stack.__aexit__(None, None, None)
        try:
            asyncio.run_coroutine_threadsafe(
                _exit(), bridge._loop).result(timeout=20)
        except Exception as exc:
            log.debug("[MCP] shutdown raised, continuing: %s", exc)
    if bridge._loop is not None:
        try:
            bridge._loop.call_soon_threadsafe(bridge._loop.stop)
        except Exception:
            pass
    bridge._sessions.clear()
    bridge._defs.clear()
    bridge._started = False

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

WHAT CAN AND CANNOT CONNECT

Verified live against mcp 2.3.0: **stdio** (a child process) and
**streamable HTTP** (including a configured header reaching the server,
which is how a gateway carries an agent's bearer token). The SDK renamed the
HTTP opener and changed how headers are passed between majors, so both
spellings are handled — `pyproject` declares `mcp>=1.2,<3` and the range
spans the rename.

Four limits, and the first is the one to read twice:

1. **`read_only` is an operator's ASSERTION, not an enforcement.** It gates
   whether this bridge *offers* a server's tools; it does not constrain what
   the server does. Mark a filesystem server `read_only: true` and its write
   tools are hidden from the model — but nothing here prevents that server
   writing if some other path reaches it. The fence buys intent, not safety,
   and the guards still have to move below the tool boundary before a
   write-capable server is trustworthy. Do not read the config key as a
   sandbox.
2. **Tools only.** `tools/list` and `tools/call`. MCP also carries
   resources, prompts, sampling, roots and elicitation, and a server whose
   value is in those contributes nothing here.
3. **Static headers only — no OAuth.** Many hosted servers require OAuth
   2.1 and cannot connect at all. A gateway that mints its own bearer token
   works, which is the case the `headers:` key exists for.
4. **No `tools/list_changed`.** The catalog is read once at startup, so a
   server that changes its tool list mid-session is not noticed.

A stdio server also needs its command present: `uvx mcp-server-fetch` is
only connectable on a machine that has `uvx`.

ONE EVENT LOOP, OWNED BY ONE THREAD

The MCP client is async while `AgentTools.execute` is sync, and an MCP
session is stateful — so a loop created and destroyed per call would drop
the session every time. A dedicated thread owns one loop for the bridge's
lifetime and calls are handed to it with `run_coroutine_threadsafe`. The
gateway prototype this borrows its catalog shape from puts the reason
bluntly: a dead event loop is a subtle, miserable class of bug.
"""
from __future__ import annotations

import json
import atexit
import logging
import re
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
    # How to snapshot and put back this system's state, in `mcp:` tool-call
    # syntax with `{path}` substituted. Declared by the OPERATOR, never
    # guessed: saving a .blend, dumping a database and exporting a browser
    # profile share no vocabulary, and inventing one is how a backstop
    # silently captures the wrong thing. See orchestrator/external_state.py.
    snapshot: dict = field(default_factory=dict)

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
                snapshot={str(k): str(v) for k, v in
                          (entry.get("snapshot") or {}).items()},
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


def _sdk_field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field from an SDK object under any of its spellings.

    The SDK's models are pydantic with **snake_case attributes and
    camelCase wire aliases** — `input_schema`/`inputSchema`,
    `is_error`/`isError`, `structured_content`/`structuredContent` — and in
    1.x some of those attributes were camelCase outright. Reading one
    spelling is a silent failure every time, because a missing attribute
    yields the default rather than an error.

    Three measured instances before this existed, each found only by
    running against a real server:

    * `is_error` — a FAILED tool call was reported to the model as a
      success, which is the wrong direction for a verification system.
    * `input_schema` — every tool was advertised as taking NO arguments, so
      the model would call it with `{}` and get "'url' is a required
      property" forever. Tools that appeared to work were unusable.
    * `structured_content` — a result carrying its whole payload there read
      as "the tool returned no content".

    Fixing those one at a time would leave the fourth. Dict access is
    supported too, for a server or a test speaking raw JSON.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    if isinstance(obj, dict):
        for name in names:
            if obj.get(name) is not None:
                return obj[name]
    return default


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
    # Sorted by name, because the serialised tool list is part of the
    # provider's CACHED PREFIX. Measured 2026-10-04 against gpt-5.6-terra:
    # two identical calls reported 1,339 of 1,342 prompt tokens cached (99%),
    # and the same messages with a DIFFERENT tool list reported 0. So any
    # change to the list — including a reordering that changes nothing
    # semantically — invalidates the whole prefix, the byte-identical system
    # prompt with it.
    #
    # `tools/list` order is the server's choice and nothing in the protocol
    # pins it, so a server backed by a dict could reorder between runs and
    # every turn would pay full price for the entire prompt. Sorting costs
    # nothing and removes the possibility.
    for tool in sorted(tools, key=lambda t: str(_sdk_field(t, "name") or "")):
        name = _sdk_field(tool, "name")
        if not name:
            continue
        if spec.allow and name not in spec.allow:
            withheld.append(WithheldTool(
                server, name,
                "not in this server's `allow:` list"))
            continue
        desc = (_sdk_field(tool, "description")
                or f"{name} (from MCP server {server})")
        # BOTH spellings: 2.x names the attribute `input_schema` and keeps
        # `inputSchema` as the wire alias. Reading only the alias advertised
        # every tool as taking no arguments.
        schema = _sdk_field(tool, "input_schema", "inputSchema",
                            default={"type": "object", "properties": {}})
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
    content = _sdk_field(response, "content")
    # 2.x added a structured payload that may carry the whole result while
    # `content` is empty. Without this the tool would read as having
    # returned nothing.
    if not content:
        structured = _sdk_field(response, "structured_content",
                                "structuredContent")
        if structured:
            return json.dumps(structured, default=str)[:MAX_RESULT_CHARS]
    parts: list[str] = []
    for item in content or ():
        text = _sdk_field(item, "text")
        if text:
            parts.append(str(text))
            continue
        kind = _sdk_field(item, "type", default="content")
        # Binary and resource blocks are named rather than dumped: a
        # base64 image in the conversation is tokens the model cannot use.
        parts.append(f"[{kind} block, not rendered as text]")
    body = "\n".join(parts) if parts else "(the tool returned no content)"
    # BOTH spellings. In mcp 2.x the Python attribute is `is_error` and
    # `isError` is only the wire alias; in 1.x the attribute itself was
    # `isError`. Checking one name meant a FAILED tool call was reported to
    # the model as a success — the wrong direction for a system whose whole
    # question is whether something actually worked.
    #
    # Found by running against a real server. The 52 unit tests missed it
    # because the fake result object used `isError`, so they validated the
    # assumption rather than the SDK: a fake built from what you believe
    # tests your belief.
    is_error = bool(_sdk_field(response, "is_error", "isError"))
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
        # The SDK renamed this between majors and changed how headers are
        # passed, and `pyproject` declares `mcp>=1.2,<3` — so the range this
        # package supports spans the rename and both have to work.
        #
        #   1.x  streamablehttp_client(url, headers={...})
        #   2.x  streamable_http_client(url, http_client=AsyncClient(...))
        #
        # Found by installing the package: the first cut used the 1.x name
        # and the 1.x `headers=` keyword, so the HTTP transport would have
        # raised ImportError on 2.x and TypeError on the version actually
        # released. 52 unit tests passed throughout, because every one of
        # them uses a fake session.
        import mcp.client.streamable_http as _sh
        opener = getattr(_sh, "streamable_http_client", None)
        if opener is not None:                      # 2.x
            http_client = _sh.create_mcp_http_client(
                headers=dict(spec.headers) or None)
            await stack.enter_async_context(http_client)
            streams = await stack.enter_async_context(
                opener(spec.url or "", http_client=http_client))
        else:                                       # 1.x
            streams = await stack.enter_async_context(
                _sh.streamablehttp_client(spec.url or "",
                                          headers=dict(spec.headers)))
        # 1.x yields (read, write, get_session_id); 2.x yields two. Taking
        # the first two rather than unpacking a fixed arity, so a third
        # element appearing or vanishing is not a crash.
        read, write = streams[0], streams[1]
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
        tools = _sdk_field(listed, "tools", default=[])
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


# --------------------------------------------------------------------------
# Run-scoped lifecycle. One bridge per run, attached to FileMemory so
# `build_step_tools` can reach it, and stopped on EVERY exit path.
# --------------------------------------------------------------------------

_ACTIVE: MCPBridge | None = None


def ensure_started(cfg: Any) -> MCPBridge | None:
    """Start the configured servers once. Returns the bridge, or None.

    Separate from `attach_to` because the servers' lifetime has nothing to do
    with FileMemory's. `attach_to` was the only way in, and on `cli.py`'s
    fresh path FileMemory is not created until AFTER the plan has been made —
    so asking `getattr(memory, "_mcp_bridge")` while planning raised
    `UnboundLocalError: cannot access local variable 'memory'` and crashed the
    run before its first step. The planner needs the tool list and the plan
    comes first, so the start has to be answerable without that object.

    Idempotent, because `cli.py` reaches it on more than one path and starting
    a second set of sessions would double every server's handshake and leave
    the first set orphaned.

    Returns None when nothing is configured — the ordinary case, which must
    stay silent. Never raises: every failure here costs the run its external
    tools and nothing else.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    try:
        specs, problems = load_specs(getattr(cfg, "MCP", None))
    except Exception as exc:                        # pragma: no cover - env
        log.warning("[MCP] configuration could not be read: %s", exc)
        return None
    for problem in problems:
        # A rejected server is a WARNING, not a silent omission: its tools
        # are missing either way, and only one of those two outcomes tells
        # the operator which line to fix.
        log.warning("[MCP] %s", problem)
    if not specs:
        return None
    bridge = MCPBridge(specs)
    bridge.start()
    for problem in bridge.problems:
        log.warning("[MCP] %s", problem)
    for withheld in bridge.withheld:
        log.info("[MCP] withheld %s — %s", withheld.qualified, withheld.reason)
    offered = bridge.definitions()
    if offered:
        log.info("[MCP] %d external tool(s) offered to the loop: %s",
                 len(offered), ", ".join(d.name for d in offered))
    _ACTIVE = bridge
    # `cli.py` stops the bridge in `main()`'s finally; `api.py` has no such
    # wrapper, and a library caller that never stops it would leave every
    # stdio server running as an orphaned child process. Registered on the
    # one path that actually creates a bridge, so a run with no servers
    # configured registers nothing. Double-stopping is safe: `stop_active`
    # clears `_ACTIVE` first.
    atexit.register(stop_active)
    return bridge


def attach_to(memory: Any, cfg: Any) -> MCPBridge | None:
    """Start the configured servers and hang the bridge on *memory*.

    The binding half of `ensure_started`: `build_step_tools` reads
    `memory._mcp_bridge`, which is how a step's `AgentTools` gets the
    external tools.
    """
    bridge = ensure_started(cfg)
    if bridge is not None:
        memory._mcp_bridge = bridge
    return bridge


def stop_active() -> None:
    """Close the run's bridge. Safe to call when there is none.

    Called from `main()`'s finally so it runs on an exception and on
    KeyboardInterrupt too. A stdio server is a CHILD PROCESS, and the
    orphaned `next dev` that held a pipe open for four hours is why this
    does not rely on the loop thread being a daemon.
    """
    global _ACTIVE
    bridge, _ACTIVE = _ACTIVE, None
    if bridge is not None:
        bridge.stop()


# Deliberately small. The planner never CALLS these, so it does not need
# parameter schemas — and schemas are the expensive part: a real
# third-party tool measured 1,209 characters, against 48 for its name and
# first line. 25 tools summarised this way cost about what one of them
# costs in full.
PLANNER_SUMMARY_MAX_TOOLS = 40
PLANNER_SUMMARY_DESC_CHARS = 110



# A command that only prints. `echo` is the one the planner reaches for,
# because a CMD step must carry a shell line and there is no shell line that
# calls a tool. Redirection and chaining are excluded deliberately:
# `echo x > f` writes a file and `echo x && npm i` installs, so neither is a
# no-op, and reading them as one would skip real work.
_NOOP_CMD_RE = re.compile(r"^\s*(?:echo|rem|::|:|true)\b", re.IGNORECASE)
_HAS_EFFECT_RE = re.compile(r"[>|&;]")


def hollow_tool_command(cmd: str, step_text: str,
                        bridge: "MCPBridge | None") -> str | None:
    """Why this CMD step cannot do what it says it does.

    Measured 2026-10-05, the run after the planner was first told its tools
    exist. The planner adopted them — and the plan vocabulary is CMD / CODE /
    TEST, so a tool call is none of the three. It put the intent in the only
    slot a CMD step has, a shell line::

        > echo Configure the active Blender scene through
          blender__execute_blender_code: retain/add a cube at (0,0,0), ...

    `echo` exits 0, so the step reported success having done nothing, and
    because the step classified CMD the agent loop never ran — which is the
    only place `chat(messages, tools=...)` is called and therefore the only
    place a tool can be invoked. Zero tool calls in the whole run.

    Awareness without an execution path is worse than no awareness: a step
    that proposes a tool and then no-ops is a step that cannot fail.

    Returns None unless the command is **provably** inert and the step names
    a tool that is actually available, so an ordinary `echo` step and a step
    naming a tool nobody configured are both left alone.
    """
    if bridge is None:
        return None
    names = [d.name for d in bridge.definitions()]
    if not names:
        return None
    if not cmd or not _NOOP_CMD_RE.match(cmd) or _HAS_EFFECT_RE.search(cmd):
        return None
    # Qualified names only. A bare `export_scene` is an ordinary English
    # phrase in a plan description; `blender__export_scene` is not.
    haystack = f"{cmd}\n{step_text or ''}"
    named = sorted({n for n in names if n in haystack})
    if not named:
        return None
    return (f"the step's command only prints and cannot call "
            f"{', '.join(named)}, which it names — a shell command cannot "
            f"invoke an external tool, so this step must run as a "
            f"tool-calling conversation")


def planner_summary(bridge: "MCPBridge | None") -> str:
    """A short description of the external tools, for the PLANNER.

    The planner has never seen tool definitions: `chat(messages, tools=...)`
    is called in exactly one place, the agent loop. So a plan is formed
    without any knowledge that external tools exist, and the loop then
    executes steps that were decided before the tools were visible.

    Measured 2026-10-05 against a live Blender MCP server: five tools were
    offered to the loop, `execute_blender_code` among them, and the run
    called `run_command`, `read_file`, `write_file` and `edit_file` — never
    one of the five. The plan said "write a Python script and run Blender
    headless", which was settled before the tools entered the picture. The
    model was not declining to use them; it was executing a strategy chosen
    without them.

    Returns "" when there is nothing to say, so an ordinary run's planner
    prompt is byte-for-byte what it was.
    """
    if bridge is None:
        return ""
    defs = bridge.definitions()
    if not defs:
        return ""
    lines = ["EXTERNAL TOOLS AVAILABLE TO STEPS",
             # What the tools REACH, before how to call them. Measured
             # 2026-10-05: told only that the tools existed and how to
             # invoke them, the planner went back to `blender --background
             # --python script.py` — correct work in a fresh process that
             # could never touch the session the task was about. A planner
             # has no reason to prefer a tool over a script it knows how to
             # write, unless it is told what the tool can reach that the
             # script cannot.
             "Each of these acts on a LIVE EXTERNAL SYSTEM that already "
             "exists and that this run did not create — the application, "
             "service or session the server is attached to. Nothing else "
             "in the pipeline can reach it: a script you write runs in a "
             "fresh process of its own, so it cannot observe or change "
             "that system's state.",
             "So when the task concerns that system — what it currently "
             "holds, or a change someone expects to see IN it — call the "
             "tool. A script that reproduces the work in a separate "
             "process is a DIFFERENT RESULT, however correct the script "
             "is. Use ordinary files and commands for everything else.",
             "A step that uses one must be a CODE or TEST step: those "
             "run as a tool-calling conversation, which is the only "
             "place a tool can be called. A CMD step runs one shell "
             "command, and no shell command can invoke a tool — do "
             "not write `echo` to stand in for calling one.",
             # Measured 2026-10-06: the plan declared `verify:
             # blender__get_scene_info`, the executor ran it through
             # cmd.exe, and a correct step failed. A bare tool name is also
             # a gate that cannot FAIL -- it passes whenever the server is
             # up, including over the state before the step ran.
             "A step's `verify:` may call a tool instead of the shell, "
             "written as `mcp:<tool> {\"arg\": value}` on one line. It "
             "must be able to FAIL: calling a read-only tool and asserting "
             "nothing passes whenever the server is reachable, which "
             "proves nothing about this step. Put the check where it can "
             "fail — in a tool that executes code, with a real assertion:",
             "  verify: mcp:<server>__execute_code {\"code\": \"<read the "
             "live state>; assert <the concrete condition this step "
             "establishes>\"}",
             ""]
    shown = defs[:PLANNER_SUMMARY_MAX_TOOLS]
    for d in shown:
        # First line only: a description's later paragraphs are usage detail
        # the planner cannot act on.
        first = (d.description or "").strip().splitlines()[0] if d.description else ""
        if len(first) > PLANNER_SUMMARY_DESC_CHARS:
            first = first[:PLANNER_SUMMARY_DESC_CHARS - 1].rstrip() + "…"
        lines.append(f"  {d.name} — {first}" if first else f"  {d.name}")
    if len(defs) > len(shown):
        lines.append(f"  … and {len(defs) - len(shown)} more")
    return "\n".join(lines)

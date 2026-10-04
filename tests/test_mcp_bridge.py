"""MCP servers as a source of tools, under `AgentTools`' contract.

Two things these tests are shaped around.

**The optional dependency.** `mcp` is an extra, and CI does not install it,
so every test here runs without it. The connection path is exercised with a
fake session rather than a real server: a test that silently skips when a
package is missing is a test that never runs, which is the
`_INCONCLUSIVE_MARKERS` mistake pointed at the suite instead of the code.

**The read-only fence.** `_tool_write_file` carries seven refusals and all
of them live inside that method, so an external tool that writes files
reaches none. The sharpest consequence is that `_acceptance_refusal` is the
only thing keeping a run from rewriting its own acceptance contract — so a
write-capable MCP server would let a run author the instrument that
certifies it, while `require_independent_evidence` still said `independent`.
Until the guards move below the tool boundary, a server must be declared
read-only or given an explicit allow list.
"""
import pytest

from agentchanti.llm.chat_types import ToolDef
from agentchanti.mcp_bridge import (NAME_SEPARATOR, MAX_RESULT_CHARS,
                                    MCPBridge, MCPServerSpec, WithheldTool,
                                    load_specs, mcp_available, render_result,
                                    split_qualified, tool_defs_for)


class _Tool:
    """What `tools/list` hands back, in the shape the SDK uses."""

    def __init__(self, name, description="", schema=None):
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {}}


class _Block:
    def __init__(self, text=None, type="text"):
        if text is not None:
            self.text = text
        self.type = type


class _Result:
    """A tool result, in whichever spelling the SDK major uses.

    `which` exists because a fake built from what you BELIEVE tests your
    belief: the first version of this class only had `isError`, which is the
    1.x attribute, so the tests passed while `render_result` could not see a
    2.x error result at all — and reported a failed call as a success.
    """

    def __init__(self, content, isError=False, which="both",
                 structured=None):
        self.content = content
        if which in ("both", "1.x"):
            self.isError = isError
        if which in ("both", "2.x"):
            self.is_error = isError
        if structured is not None:
            self.structured_content = structured


class TestTheReadOnlyFence:
    """The whole reason this release does not offer write-capable servers."""

    def test_a_server_with_no_assertion_is_refused(self):
        specs, problems = load_specs([{"name": "x", "command": "srv"}])
        assert specs == []
        assert len(problems) == 1
        # The refusal has to say WHY, or an operator removes the guard
        # rather than the risk.
        assert "read_only" in problems[0]
        assert "acceptance instrument" in problems[0]

    def test_read_only_is_accepted(self):
        specs, problems = load_specs(
            [{"name": "fetch", "command": "srv", "read_only": True}])
        assert problems == []
        assert [s.name for s in specs] == ["fetch"]

    def test_an_explicit_allow_list_is_accepted(self):
        specs, problems = load_specs(
            [{"name": "gh", "command": "srv", "allow": ["list_prs"]}])
        assert problems == []
        assert specs[0].allow == ("list_prs",)

    def test_an_allow_list_withholds_everything_else(self):
        spec = MCPServerSpec(name="gh", command="srv", allow=("list_prs",))
        offered, withheld = tool_defs_for(
            "gh", [_Tool("list_prs"), _Tool("merge_pr"), _Tool("delete_repo")],
            spec)
        assert [d.name for d in offered] == ["gh__list_prs"]
        assert {w.tool for w in withheld} == {"merge_pr", "delete_repo"}
        assert all("allow" in w.reason for w in withheld)

    def test_withheld_tools_are_reported_not_dropped(self):
        """A tool missing with no explanation reads exactly like a server
        that failed to start, and the operator debugs the wrong thing."""
        w = WithheldTool("gh", "merge_pr", "not in this server's allow list")
        assert w.qualified == f"gh{NAME_SEPARATOR}merge_pr"
        assert w.reason


class TestConfigParsing:

    def test_no_config_is_not_an_error(self):
        for empty in (None, {}, [], ""):
            assert load_specs(empty) == ([], [])

    def test_one_bad_entry_does_not_cost_the_others(self):
        specs, problems = load_specs([
            {"name": "good", "command": "srv", "read_only": True},
            {"name": "", "command": "srv", "read_only": True},
            {"name": "also-good", "transport": "http",
             "url": "http://x/mcp", "read_only": True},
        ])
        assert [s.name for s in specs] == ["good", "also-good"]
        assert len(problems) == 1

    def test_the_mapping_form_takes_the_name_from_the_key(self):
        specs, problems = load_specs(
            {"servers": {"fetch": {"command": "srv", "read_only": True}}})
        assert problems == []
        assert specs[0].name == "fetch"

    def test_duplicate_names_are_refused(self):
        _, problems = load_specs([
            {"name": "a", "command": "s", "read_only": True},
            {"name": "a", "command": "s", "read_only": True},
        ])
        assert any("duplicate" in p for p in problems)

    def test_a_name_containing_the_separator_is_refused(self):
        """It would make a qualified name ambiguous to split."""
        _, problems = load_specs(
            [{"name": f"a{NAME_SEPARATOR}b", "command": "s",
              "read_only": True}])
        assert any(NAME_SEPARATOR in p for p in problems)

    @pytest.mark.parametrize("entry,fragment", [
        ({"name": "a", "read_only": True}, "names no command"),
        ({"name": "a", "transport": "http", "read_only": True}, "names no url"),
        ({"name": "a", "transport": "carrier-pigeon", "command": "s",
          "read_only": True}, "unknown transport"),
    ])
    def test_an_unusable_spec_says_what_is_missing(self, entry, fragment):
        specs, problems = load_specs([entry])
        assert specs == []
        assert fragment in problems[0]


class TestNamespacing:
    """A server exposing `read_file` must not shadow the guarded built-in of
    that name — `AgentTools.execute` dispatches by name, so a collision
    would silently replace a tool carrying seven refusals with one carrying
    none. This is `package_shadow_reason` one layer up."""

    def test_tools_are_qualified_by_server(self):
        offered, _ = tool_defs_for(
            "fs", [_Tool("read_file")], MCPServerSpec("fs", read_only=True))
        assert offered[0].name == f"fs{NAME_SEPARATOR}read_file"
        assert offered[0].name != "read_file"

    def test_a_qualified_name_splits_back(self):
        assert split_qualified("fs__read_file") == ("fs", "read_file")

    @pytest.mark.parametrize("name", ["read_file", "", "__x", "x__", "__"])
    def test_an_unqualified_name_is_not_claimed(self, name):
        assert split_qualified(name) is None

    def test_the_description_names_the_server(self):
        """The model has to be able to tell two servers' tools apart when
        both expose something called `search`."""
        offered, _ = tool_defs_for(
            "docs", [_Tool("search", "find things")],
            MCPServerSpec("docs", read_only=True))
        assert offered[0].description.startswith("[docs]")

    def test_a_tool_with_no_name_is_skipped_not_crashed_on(self):
        offered, _ = tool_defs_for("s", [_Tool(""), {"nope": 1}],
                                   MCPServerSpec("s", read_only=True))
        assert offered == []

    def test_definitions_are_our_own_tooldefs(self):
        """Translated rather than passed through: agentchanti is
        provider-agnostic, so routing MCP through any one provider's
        built-in support would make these tools work there and vanish on the
        other four."""
        offered, _ = tool_defs_for("s", [_Tool("x")],
                                   MCPServerSpec("s", read_only=True))
        assert isinstance(offered[0], ToolDef)


class TestResultRendering:

    def test_text_blocks_are_joined(self):
        assert render_result(_Result([_Block("one"), _Block("two")])) == "one\ntwo"

    def test_an_error_result_is_text_not_an_exception(self):
        """Same contract as `AgentTools.execute`: the model must be able to
        read what went wrong and act on it."""
        out = render_result(_Result([_Block("bad path")], isError=True))
        assert out.startswith("ERROR from the MCP tool:")
        assert "bad path" in out

    def test_a_binary_block_is_named_not_dumped(self):
        """A base64 image in the conversation is tokens the model cannot
        use, paid for on every later turn."""
        out = render_result(_Result([_Block(type="image")]))
        assert "image" in out and "not rendered" in out

    def test_no_content_says_so(self):
        assert "no content" in render_result(_Result([]))

    def test_a_dict_response_works_too(self):
        assert render_result({"content": [{"type": "text", "text": "hi"}]}) == "hi"


class TestItIsInertWithoutAServer:
    """`mcp` is an optional extra and a server can die mid-run. Neither may
    raise into the pipeline: an absent instrument must not convict the
    code."""

    def test_an_unstarted_bridge_offers_nothing(self):
        b = MCPBridge([MCPServerSpec("fetch", command="s", read_only=True)])
        assert b.definitions() == []
        assert not b.owns("fetch__get")

    def test_executing_without_a_session_returns_an_error_string(self):
        b = MCPBridge([MCPServerSpec("fetch", command="s", read_only=True)])
        out = b.execute("fetch__get", {})
        assert out.startswith("ERROR")
        assert "unavailable" in out

    def test_start_without_the_package_is_false_and_explains(self):
        ok, _ = mcp_available()
        if ok:                      # pragma: no cover - env dependent
            pytest.skip("mcp is installed here; the absent path is covered "
                        "by test_the_absent_package_is_reported")
        b = MCPBridge([MCPServerSpec("fetch", command="s", read_only=True)])
        assert b.start() is False
        assert any("mcp" in p for p in b.problems)
        assert any("pip install" in p for p in b.problems)

    def test_stop_is_safe_before_start_and_twice(self):
        b = MCPBridge([])
        b.stop()
        b.stop()

    def test_the_context_manager_never_raises(self):
        with MCPBridge([MCPServerSpec("f", command="s",
                                      read_only=True)]) as b:
            assert b.definitions() == []

    def test_no_servers_configured_is_not_a_problem(self):
        b = MCPBridge([])
        assert b.start() is False
        assert b.problems == []


class TestABridgeWithAFakeSession:
    """The execute path, without a real server. The fake stands in for the
    session object the SDK returns, which is the only thing `execute`
    touches."""

    def _bridge(self, result, *, tool="get", timeout=5.0):
        spec = MCPServerSpec("fetch", command="s", read_only=True,
                             timeout=timeout)
        b = MCPBridge([spec])
        offered, _ = tool_defs_for("fetch", [_Tool(tool)], spec)
        b._defs.extend(offered)
        b._sessions["fetch"] = object()
        b._started = True
        b._call = lambda *a, **k: result() if callable(result) else result
        return b

    def test_a_successful_call_returns_the_text(self):
        b = self._bridge("the page body")
        assert b.execute("fetch__get", {"url": "x"}) == "the page body"

    def test_an_unknown_tool_lists_what_is_offered(self):
        b = self._bridge("x")
        out = b.execute("fetch__nope", {})
        assert "unknown MCP tool" in out
        assert "fetch__get" in out

    def test_an_unqualified_name_is_refused(self):
        b = self._bridge("x")
        assert "not a qualified MCP tool name" in b.execute("get", {})

    def test_a_timeout_is_reported_not_raised(self):
        def boom():
            raise TimeoutError
        out = self._bridge(boom).execute("fetch__get", {})
        assert out.startswith("ERROR")
        assert "did not return within" in out

    def test_any_other_failure_is_reported_not_raised(self):
        def boom():
            raise RuntimeError("connection reset")
        out = self._bridge(boom).execute("fetch__get", {})
        assert out.startswith("ERROR")
        assert "connection reset" in out

    def test_a_huge_result_is_capped_keeping_both_ends(self):
        """A head-only slice hands the model everything except the
        conclusion, which is what `truncate_middle` exists for."""
        body = "HEAD" + ("x" * (MAX_RESULT_CHARS * 2)) + "TAIL"
        out = self._bridge(body).execute("fetch__get", {})
        assert len(out) < len(body)
        assert out.startswith("HEAD")
        assert out.endswith("TAIL")
        assert "elided" in out

    def test_a_dead_session_says_the_server_is_gone(self):
        b = self._bridge("x")
        b._sessions.clear()
        out = b.execute("fetch__get", {})
        assert "has no live session" in out


class _FakeMemory:
    def summary(self):
        return ""


def _live_bridge(tool="get", result="PAGE BODY"):
    """A bridge with one offered tool and a fake session."""
    spec = MCPServerSpec("fetch", command="s", read_only=True)
    bridge = MCPBridge([spec])
    offered, _ = tool_defs_for("fetch", [_Tool(tool)], spec)
    bridge._defs.extend(offered)
    bridge._sessions["fetch"] = object()
    bridge._started = True
    bridge._call = lambda *a, **k: result
    return bridge


class TestItIsActuallyWired:
    """`protect_acceptance_files` existed, was called only from a test, and
    the guard it implemented was INERT in production for as long as that
    was true. These go through `build_step_tools` and `attach_to` rather
    than setting the attribute by hand, because the wiring is the seam."""

    def test_build_step_tools_transfers_the_bridge(self):
        from agentchanti.orchestrator.agent_loop import build_step_tools
        bridge = _live_bridge()
        memory = _FakeMemory()
        memory._mcp_bridge = bridge
        tools = build_step_tools(executor=None, memory=memory,
                                 project_root=".")
        assert tools._mcp is bridge

    def test_no_bridge_on_memory_leaves_tools_unchanged(self):
        from agentchanti.orchestrator.agent_loop import build_step_tools
        tools = build_step_tools(executor=None, memory=_FakeMemory(),
                                 project_root=".")
        assert tools._mcp is None
        assert all(NAME_SEPARATOR not in d.name for d in tools.definitions())

    def test_the_tools_appear_in_definitions(self):
        from agentchanti.orchestrator.agent_loop import build_step_tools
        memory = _FakeMemory()
        memory._mcp_bridge = _live_bridge()
        tools = build_step_tools(executor=None, memory=memory,
                                 project_root=".")
        names = [d.name for d in tools.definitions()]
        assert "fetch__get" in names
        # And the six built-ins are still all there.
        for builtin in ("list_files", "read_file", "write_file", "edit_file",
                        "run_command", "search_code"):
            assert builtin in names

    def test_a_call_reaches_the_server(self):
        from agentchanti.llm.chat_types import ToolCall
        from agentchanti.orchestrator.agent_loop import build_step_tools
        memory = _FakeMemory()
        memory._mcp_bridge = _live_bridge(result="the page")
        tools = build_step_tools(executor=None, memory=memory,
                                 project_root=".")
        out = tools.execute(ToolCall(id="1", name="fetch__get",
                                     arguments={"url": "x"}))
        assert out == "the page"

    def test_an_external_tool_obeys_the_per_turn_offer(self):
        """The loop withholds tools on its final turn and when a model
        spends turn after turn reading. An MCP tool escaping that would
        defeat the control for exactly the tools whose behaviour is least
        known."""
        from agentchanti.llm.chat_types import ToolCall
        from agentchanti.orchestrator.agent_loop import build_step_tools
        memory = _FakeMemory()
        memory._mcp_bridge = _live_bridge()
        tools = build_step_tools(executor=None, memory=memory,
                                 project_root=".")
        out = tools.execute(ToolCall(id="1", name="fetch__get", arguments={}),
                            allowed={"write_file"})
        assert "disabled for this turn" in out

    def test_unknown_still_outranks_withheld(self):
        """A name that never existed is a different mistake from one taken
        away this turn, and "disabled" would send the model hunting for a
        way to re-enable it."""
        from agentchanti.llm.chat_types import ToolCall
        from agentchanti.orchestrator.agent_loop import build_step_tools
        memory = _FakeMemory()
        memory._mcp_bridge = _live_bridge()
        tools = build_step_tools(executor=None, memory=memory,
                                 project_root=".")
        out = tools.execute(ToolCall(id="1", name="fetch__nope", arguments={}),
                            allowed={"write_file"})
        assert "unknown tool" in out


class TestAttachAndStop:

    def test_nothing_configured_attaches_nothing_and_says_nothing(self, caplog):
        from agentchanti import mcp_bridge as mb

        class _Cfg:
            MCP = None
        memory = _FakeMemory()
        with caplog.at_level("WARNING"):
            assert mb.attach_to(memory, _Cfg()) is None
        assert not hasattr(memory, "_mcp_bridge")
        assert "[MCP]" not in caplog.text

    def test_a_rejected_server_is_warned_about_not_silently_dropped(self, caplog):
        """Its tools are missing either way; only one of those outcomes
        tells the operator which line to fix."""
        from agentchanti import mcp_bridge as mb

        class _Cfg:
            MCP = {"servers": [{"name": "w", "command": "s"}]}
        with caplog.at_level("WARNING"):
            mb.attach_to(_FakeMemory(), _Cfg())
        assert "read_only" in caplog.text

    def test_attach_is_idempotent(self):
        """`cli.py` builds FileMemory on more than one path, and a second
        set of sessions would double every handshake and orphan the first."""
        from agentchanti import mcp_bridge as mb

        class _Cfg:
            MCP = {"servers": [{"name": "f", "command": "s",
                                "read_only": True}]}
        mb.stop_active()
        try:
            a = mb.attach_to(_FakeMemory(), _Cfg())
            b = mb.attach_to(_FakeMemory(), _Cfg())
            # Without mcp installed both are None; with it, the same object.
            assert a is b
        finally:
            mb.stop_active()

    def test_stop_active_is_safe_with_no_bridge(self):
        from agentchanti import mcp_bridge as mb
        mb.stop_active()
        mb.stop_active()

    def test_main_stops_the_bridge_on_every_exit_path(self):
        """Including the exception and KeyboardInterrupt branches — a stdio
        server is a child process, and an orphan holding a pipe open is what
        hung a pipeline for four hours."""
        import inspect

        from agentchanti.orchestrator import cli
        src = inspect.getsource(cli.main)
        assert "finally:" in src
        assert "stop_active()" in src
        # In the finally, i.e. after the handlers rather than inside one.
        assert src.index("KeyboardInterrupt") < src.index("stop_active")


class TestBothSdkSpellings:
    """`pyproject` declares `mcp>=1.2,<3`, and the SDK renamed things inside
    that range. Verified against a real mcp 2.3.0 install, which is the only
    way either of these was found — 52 unit tests passed over both.
    """

    @pytest.mark.parametrize("which", ["1.x", "2.x", "both"])
    def test_an_error_result_is_seen_in_either_spelling(self, which):
        """2.x names the attribute `is_error` and keeps `isError` only as a
        wire alias. Reading one name reported a FAILED call as a success,
        which is the wrong direction for a verification system."""
        out = render_result(_Result([_Block("bad path")], isError=True,
                                    which=which))
        assert out.startswith("ERROR from the MCP tool:"), which

    @pytest.mark.parametrize("which", ["1.x", "2.x", "both"])
    def test_a_success_is_not_mistaken_for_an_error(self, which):
        out = render_result(_Result([_Block("fine")], isError=False,
                                    which=which))
        assert not out.startswith("ERROR"), which

    def test_a_structured_payload_is_not_read_as_empty(self):
        """2.x may carry the whole result in `structured_content` with
        `content` empty; without this the tool reads as returning nothing."""
        out = render_result(_Result([], structured={"rows": [1, 2]}))
        assert "rows" in out
        assert "no content" not in out

    def test_the_http_transport_handles_both_apis(self):
        """1.x: `streamablehttp_client(url, headers=...)`.
        2.x: `streamable_http_client(url, http_client=...)`.
        The first cut used the 1.x name AND keyword, so the HTTP transport
        would have raised ImportError on the version actually released."""
        import inspect

        from agentchanti import mcp_bridge
        src = inspect.getsource(mcp_bridge._open_session)
        assert "streamable_http_client" in src          # 2.x
        assert "streamablehttp_client" in src           # 1.x
        assert "create_mcp_http_client" in src          # 2.x headers path
        # Arity-tolerant: 1.x yields three streams, 2.x two.
        assert "streams[0], streams[1]" in src


class TestTheDocumentedLimits:
    """`read_only` is the one most likely to be misread, so it is pinned in
    the module's own documentation as well as in behaviour.

    Verified live against mcp 2.3.0: stdio and streamable HTTP both work,
    including a configured header reaching the server. What is NOT supported
    is written down rather than left for someone to discover.
    """

    def _doc(self):
        """Whitespace-normalised, because the prose is hard-wrapped and a
        phrase that happens to straddle a newline is not a missing phrase.
        The first version of this test asserted on the raw string and failed
        on "as a\\n   sandbox"."""
        from agentchanti import mcp_bridge
        return " ".join((mcp_bridge.__doc__ or "").split())

    def test_read_only_is_documented_as_an_assertion_not_a_sandbox(self):
        doc = self._doc()
        assert "ASSERTION, not an enforcement" in doc
        assert "as a sandbox" in doc

    def test_the_unsupported_surface_is_named(self):
        doc = self._doc()
        for unsupported in ("resources", "prompts", "OAuth",
                            "tools/list_changed"):
            assert unsupported in doc, unsupported

    def test_only_tools_are_requested_from_a_server(self):
        """If this grows a `list_resources` call, the limit above is stale
        and the documentation has to move with it."""
        import inspect

        from agentchanti import mcp_bridge
        src = inspect.getsource(mcp_bridge)
        assert "list_tools" in src
        assert "call_tool" in src
        assert "list_resources" not in src
        assert "list_prompts" not in src


class TestEverySdkFieldSpelling:
    """Three defects of one shape, each found only against a real server.

    The SDK's models are pydantic with snake_case attributes and camelCase
    wire aliases, and a missing attribute yields a default rather than an
    error — so reading one spelling fails silently every time. Fixing them
    one at a time would have left the fourth, so all SDK field access goes
    through `_sdk_field`.
    """

    def _f(self):
        from agentchanti.mcp_bridge import _sdk_field
        return _sdk_field

    def test_either_spelling_is_read(self):
        f = self._f()

        class Snake:
            input_schema = {"type": "object", "properties": {"url": {}}}

        class Camel:
            inputSchema = {"type": "object", "properties": {"url": {}}}

        for obj in (Snake(), Camel()):
            got = f(obj, "input_schema", "inputSchema")
            assert "url" in got["properties"], obj

    def test_dicts_work_too(self):
        f = self._f()
        assert f({"isError": True}, "is_error", "isError") is True

    def test_the_default_is_returned_when_absent(self):
        f = self._f()
        sentinel = {"type": "object", "properties": {}}
        assert f(object(), "input_schema", "inputSchema",
                 default=sentinel) is sentinel

    def test_a_tool_schema_survives_the_snake_case_spelling(self):
        """THE defect: reading only `inputSchema` advertised every tool as
        taking no arguments, so the model would call it with `{}` and get
        "'url' is a required property" forever. Measured against the real
        mcp-server-fetch, whose schema arrived empty."""
        from agentchanti.mcp_bridge import MCPServerSpec, tool_defs_for

        class RealWorldTool:
            name = "fetch"
            description = "Fetches a URL."
            input_schema = {"type": "object",
                            "properties": {"url": {"type": "string"},
                                           "max_length": {"type": "integer"}},
                            "required": ["url"]}

        offered, _ = tool_defs_for("fetch", [RealWorldTool()],
                                   MCPServerSpec("fetch", read_only=True))
        params = offered[0].parameters
        assert list(params["properties"]) == ["url", "max_length"]
        assert params["required"] == ["url"]

    def test_no_direct_getattr_on_sdk_objects_remains(self):
        """If one creeps back, the class of bug is back with it."""
        import inspect

        from agentchanti import mcp_bridge
        src = inspect.getsource(mcp_bridge)
        for leak in ('getattr(tool,', 'getattr(response,', 'getattr(item,',
                     'getattr(listed,'):
            assert leak not in src, leak


class TestTheToolListIsDeterministic:
    """The serialised tool list is part of the provider's CACHED PREFIX.

    Measured 2026-10-04 against gpt-5.6-terra with three calls: two identical
    ones reported 1,339 of 1,342 prompt tokens cached (99%), and the same
    messages with a DIFFERENT tool list reported **0**. So a change to the
    list — including a reordering that means nothing — invalidates the entire
    prefix, taking the byte-identical system prompt with it.

    `tools/list` order is the server's choice and the protocol does not pin
    it, so a dict-backed server could reorder between runs and every turn
    would pay full price for the whole prompt.
    """

    def _offer(self, names):
        from agentchanti.mcp_bridge import MCPServerSpec, tool_defs_for
        offered, _ = tool_defs_for(
            "s", [_Tool(n) for n in names],
            MCPServerSpec("s", read_only=True))
        return [d.name for d in offered]

    def test_order_does_not_depend_on_the_servers_order(self):
        forward = self._offer(["alpha", "beta", "gamma"])
        shuffled = self._offer(["gamma", "alpha", "beta"])
        assert forward == shuffled, "a reordered tools/list changed our list"

    def test_the_order_is_by_name(self):
        assert self._offer(["zeta", "alpha"]) == ["s__alpha", "s__zeta"]

    def test_a_nameless_tool_does_not_break_sorting(self):
        """Sorting has to tolerate what the filter later drops."""
        assert self._offer(["beta", "", "alpha"]) == ["s__alpha", "s__beta"]

    def test_the_same_config_yields_the_same_list_twice(self):
        """The whole point: byte-identical across startups, or the cache is
        lost on every run."""
        first = self._offer(["fetch", "head", "resolve"])
        second = self._offer(["fetch", "head", "resolve"])
        assert first == second


class TestThePlannerIsToldTheToolsExist:
    """Measured 2026-10-05 against a live Blender MCP server: five tools
    were offered to the loop, `execute_blender_code` among them, and the run
    called `run_command`, `read_file`, `write_file` and `edit_file` — never
    one of the five. `chat(messages, tools=...)` is called in exactly one
    place, the agent loop, so the plan was formed without any knowledge that
    external tools existed and the loop then executed a strategy chosen
    before they were visible.

    The silence cases matter as much as the content: an ordinary run must
    pay nothing and its planner prompt must be byte-identical.
    """

    def test_no_bridge_says_nothing(self):
        from agentchanti.mcp_bridge import planner_summary
        assert planner_summary(None) == ""

    def test_a_bridge_with_no_tools_says_nothing(self):
        """A configured server whose every tool was withheld is the same
        case as no server at all, as far as the planner is concerned."""
        from agentchanti.mcp_bridge import planner_summary
        assert planner_summary(MCPBridge([])) == ""

    def test_the_tools_are_named(self):
        from agentchanti.mcp_bridge import planner_summary
        out = planner_summary(_live_bridge(tool="get"))
        assert "fetch__get" in out
        assert "EXTERNAL TOOLS AVAILABLE TO STEPS" in out

    def test_only_the_first_line_of_a_description_is_carried(self):
        """Schemas and usage paragraphs are the expensive part, and the
        planner never CALLS these tools, so it cannot act on either."""
        from agentchanti.mcp_bridge import planner_summary, tool_defs_for
        spec = MCPServerSpec("fetch", command="s", read_only=True)
        bridge = MCPBridge([spec])
        offered, _ = tool_defs_for(
            "fetch",
            [_Tool("get", "Fetch a URL.\nPass `raw` to skip conversion.\n"
                          "Returns markdown.")],
            spec)
        bridge._defs.extend(offered)
        out = planner_summary(bridge)
        assert "Fetch a URL." in out
        assert "raw" not in out, "a usage paragraph reached the planner"

    def test_a_long_description_is_truncated(self):
        from agentchanti.mcp_bridge import (PLANNER_SUMMARY_DESC_CHARS,
                                            planner_summary, tool_defs_for)
        spec = MCPServerSpec("fetch", command="s", read_only=True)
        bridge = MCPBridge([spec])
        offered, _ = tool_defs_for("fetch", [_Tool("get", "x" * 500)], spec)
        bridge._defs.extend(offered)
        line = [ln for ln in planner_summary(bridge).splitlines()
                if "fetch__get" in ln][0]
        assert len(line) < PLANNER_SUMMARY_DESC_CHARS + 40
        assert line.endswith("…")

    def test_a_tool_with_no_description_is_still_named(self):
        """It is named with no ` — ` separator, which is why the log line
        counts `bridge.definitions()` rather than counting separators."""
        from agentchanti.mcp_bridge import planner_summary
        out = planner_summary(_live_bridge(tool="get"))
        assert "fetch__get" in out

    def test_the_summary_is_byte_identical_across_calls(self):
        """It rides in the planner prompt, so a reordering would cost the
        cached prefix — the same requirement `tool_defs_for` carries."""
        from agentchanti.mcp_bridge import planner_summary
        assert planner_summary(_live_bridge()) == planner_summary(_live_bridge())


class TestBothEntryPointsTellThePlanner:
    """`cli.py` and `api.py` run the same pipeline, and every fix in this
    codebase that landed on one of them only has had to be found again on
    the other. These read the source because the defect being pinned is a
    missing CALL, which no behavioural test of the function can see.
    """

    def _source(self, rel):
        import pathlib
        import agentchanti
        return (pathlib.Path(agentchanti.__file__).parent / rel
                ).read_text(encoding="utf-8")

    @pytest.mark.parametrize("rel", ["orchestrator/cli.py", "api.py"])
    def test_the_entry_point_attaches_and_tells_the_planner(self, rel):
        src = self._source(rel)
        assert "attach_to(memory, cfg)" in src, f"{rel} starts no servers"
        assert "planner_summary(" in src, f"{rel} plans without the tools"

    @pytest.mark.parametrize("rel", ["orchestrator/cli.py", "api.py"])
    def test_the_summary_is_appended_exactly_once(self, rel):
        """`planner.process` is called inside loops on both paths — the
        retry loop and the interactive approval loop — so appending at the
        call site would repeat the whole summary on every re-plan."""
        src = self._source(rel)
        assert src.count("planner_summary(") == 1, (
            f"{rel} appends the summary more than once; a re-plan would "
            f"carry it twice")

    def test_the_summary_is_appended_before_the_planner_runs(self):
        """Ordering is the whole fix: appended after the first
        `planner.process`, the first plan — the one that usually ships — is
        still formed without the tools."""
        src = self._source("orchestrator/cli.py")
        assert src.index("planner_summary(") < src.index("planner.process(")

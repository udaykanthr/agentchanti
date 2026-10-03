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
    def __init__(self, content, isError=False):
        self.content = content
        self.isError = isError


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

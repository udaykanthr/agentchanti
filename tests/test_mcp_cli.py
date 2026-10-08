"""`agentchanti mcp list` / `get` — answering a config question without a run.

Configuring an MCP server was write-and-hope: the only way to learn whether
the entry worked was to spend a whole run and read the log. Every mistake
available cost the same to discover, and three were hit by hand within one
afternoon of using the feature — the optional extra not installed, a server
with no fence (which `load_specs` REJECTS rather than limits), and a server
with no `snapshot:`/`state_probe:` (which costs the run its undo and leaves
the stall detector silent).

The tests are shaped around the distinction that decides the exit code:
**a check that could not be performed is not a check that failed.** The
first cut conflated them behind a single `None` bridge, so a server that
could not start printed `[not checked: ...]` and exited 0 — a health check
reporting a dead server and then claiming success, which is the
`empty_suite_reason` mistake pointed at a diagnostic.
"""
import logging
import sys
import textwrap

import pytest

from agentchanti import mcp_cli


def _write(tmp_path, body):
    (tmp_path / ".agentchanti.yaml").write_text(textwrap.dedent(body),
                                                encoding="utf-8")
    return str(tmp_path / ".agentchanti.yaml")


FENCED = """
    mcp:
      servers:
        - name: thing
          transport: stdio
          command: python
          args: ["server.py"]
          allow: [do_it]
"""

FULL = """
    mcp:
      servers:
        - name: thing
          transport: stdio
          command: python
          args: ["server.py"]
          allow: [do_it]
          state_probe: 'mcp:thing__do_it {}'
          snapshot:
            capture: 'mcp:thing__do_it {"p": "{path}"}'
            restore: 'mcp:thing__do_it {"p": "{path}"}'
"""


class _Def:
    def __init__(self, name):
        self.name = name


class _Withheld:
    def __init__(self, qualified, reason):
        self.qualified = qualified
        self.reason = reason


class _Bridge:
    def __init__(self, offered=(), withheld=()):
        self._offered = [_Def(n) for n in offered]
        self.withheld = [_Withheld(q, r) for q, r in withheld]

    def definitions(self):
        return self._offered


@pytest.fixture(autouse=True)
def _quiet(caplog):
    caplog.set_level(logging.WARNING)
    yield


def _run(args, capsys):
    code = mcp_cli.mcp_main(args)
    return code, capsys.readouterr().out


# ─── reading the config, with nothing started ────────────────────────

class TestWithoutStartingAnything:
    def test_no_config_section_is_not_a_failure(self, tmp_path, capsys):
        path = _write(tmp_path, "provider: openai\n")
        code, out = _run(["--config", path, "list", "--no-health"], capsys)
        assert code == 0
        assert "No usable server configured" in out

    def test_a_server_with_no_fence_is_reported_and_fails(self, tmp_path,
                                                          capsys):
        """`load_specs` DROPS such an entry, so the run gets no tools at
        all — the loudest possible outcome deserves a non-zero exit."""
        path = _write(tmp_path, """
            mcp:
              servers:
                - name: thing
                  command: python
        """)
        code, out = _run(["--config", path, "list", "--no-health"], capsys)
        assert code == 1
        assert "read_only" in out and "allow" in out
        assert "No usable server configured" in out

    def test_the_config_file_is_named(self, tmp_path, capsys):
        """`_find_config_file` returns the FIRST match, CWD then home, and
        never merges — so a project file shadows a home one entirely, and
        'I added a server and nothing happened' is most often that."""
        path = _write(tmp_path, FENCED)
        _code, out = _run(["--config", path, "list", "--no-health"], capsys)
        assert path in out

    def test_a_missing_undo_is_named_with_what_it_costs(self, tmp_path,
                                                        capsys):
        path = _write(tmp_path, FENCED)
        _code, out = _run(["--config", path, "list", "--no-health"], capsys)
        assert "no snapshot" in out
        assert "--restore" in out          # what it costs, not just the field
        assert "no state_probe" in out
        assert "stall detector" in out

    def test_a_declared_pair_is_not_reported_as_missing(self, tmp_path,
                                                        capsys):
        path = _write(tmp_path, FULL)
        _code, out = _run(["--config", path, "list", "--no-health"], capsys)
        assert "no snapshot" not in out
        assert "no state_probe" not in out

    def test_get_shows_the_fence_and_both_fields(self, tmp_path, capsys):
        path = _write(tmp_path, FULL)
        code, out = _run(["--config", path, "get", "thing", "--no-health"],
                         capsys)
        assert code == 0
        assert "allow: do_it" in out
        assert "snapshot     declared" in out
        assert "state_probe  declared" in out

    def test_get_on_an_unknown_name_lists_what_there_is(self, tmp_path,
                                                        capsys):
        path = _write(tmp_path, FENCED)
        code, out = _run(["--config", path, "get", "nope", "--no-health"],
                         capsys)
        assert code == 1
        assert "Configured: thing" in out

    def test_header_values_are_hidden(self, tmp_path, capsys):
        """A header value is usually a bearer token, and output people
        paste into an issue must not carry credentials."""
        path = _write(tmp_path, """
            mcp:
              servers:
                - name: remote
                  transport: http
                  url: https://example.test/mcp
                  allow: [read_thing]
                  headers:
                    Authorization: "Bearer SECRET-DO-NOT-PRINT"
        """)
        _code, out = _run(["--config", path, "get", "remote", "--no-health"],
                          capsys)
        assert "Authorization" in out
        assert "values hidden" in out
        assert "SECRET-DO-NOT-PRINT" not in out


# ─── the health check, and the distinction that decides the exit ─────

class TestHealth:
    def test_a_healthy_server_exits_zero_and_counts_its_tools(
            self, tmp_path, capsys, monkeypatch):
        path = _write(tmp_path, FULL)
        monkeypatch.setattr(mcp_cli, "_health",
                            lambda *a: (_Bridge(["thing__do_it"]), "", True))
        monkeypatch.setattr("agentchanti.mcp_bridge.stop_active", lambda: None)
        code, out = _run(["--config", path, "list"], capsys)
        assert code == 0
        assert "1 tool(s) offered" in out

    def test_a_server_that_cannot_start_exits_non_zero(
            self, tmp_path, capsys, monkeypatch):
        """The measured defect: it printed `[not checked]` and exited 0."""
        path = _write(tmp_path, FULL)
        monkeypatch.setattr(mcp_cli, "_health",
                            lambda *a: (None, "no tool was offered", True))
        code, out = _run(["--config", path, "list"], capsys)
        assert code == 1
        assert "FAILED" in out
        assert "not checked" not in out

    def test_a_check_that_could_not_run_says_so_and_still_exits_non_zero(
            self, tmp_path, capsys, monkeypatch):
        """Different words, same exit: a server was configured and nothing
        verified it, which is worth a script knowing about."""
        path = _write(tmp_path, FULL)
        monkeypatch.setattr(
            mcp_cli, "_health",
            lambda *a: (None, "the `mcp` package is not importable", False))
        code, out = _run(["--config", path, "list"], capsys)
        assert code == 1
        assert "not checked" in out
        assert "not importable" in out
        assert "FAILED" not in out

    def test_get_reports_withheld_tools_with_their_reason(
            self, tmp_path, capsys, monkeypatch):
        """The allow list is the fence; which tools it excluded is exactly
        what someone debugging 'the agent never called it' needs."""
        path = _write(tmp_path, FULL)
        bridge = _Bridge(["thing__do_it"],
                         [("thing__danger", "not in the allow list")])
        monkeypatch.setattr(mcp_cli, "_health", lambda *a: (bridge, "", True))
        monkeypatch.setattr("agentchanti.mcp_bridge.stop_active", lambda: None)
        code, out = _run(["--config", path, "get", "thing"], capsys)
        assert code == 0
        assert "1 offered, 1 withheld" in out
        assert "+ do_it" in out
        assert "- danger  (not in the allow list)" in out

    def test_no_health_starts_nothing(self, tmp_path, capsys, monkeypatch):
        def _boom(*_a):
            raise AssertionError("a server was started under --no-health")
        monkeypatch.setattr(mcp_cli, "_health", _boom)
        path = _write(tmp_path, FULL)
        assert _run(["--config", path, "list", "--no-health"], capsys)[0] == 0
        assert _run(["--config", path, "get", "thing", "--no-health"],
                    capsys)[0] == 0


# ─── the wiring ──────────────────────────────────────────────────────

def test_the_cli_dispatches_mcp_before_resolving_a_provider():
    """Someone asking why their MCP server is not working must not be told
    to supply credentials to find out — the argument `--restore` makes.
    A missing CALL is invisible to any behavioural test of the function."""
    import inspect

    from agentchanti.orchestrator import cli

    src = inspect.getsource(cli)
    assert 'sys.argv[1] == "mcp"' in src
    assert "from ..mcp_cli import mcp_main" in src
    # and before config/provider resolution, which happens at "0. Load config"
    assert src.index('sys.argv[1] == "mcp"') < src.index("# ── 0. Load config")

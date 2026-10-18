"""`--scope user`: one config section that merges, and only one.

Everything in `.agentchanti.yaml` is first-file-wins — `_find_config_file`
returns the project file when there is one and ignores a home file
entirely. That is right for a provider and an API key, where two
half-applied sources would be worse than one source, and it is WRONG for
MCP servers: "a server I want in every project" is exactly what a
user-level entry means, and under first-file-wins it was silently dead in
every project that sets a provider — which is every project. `mcp add
--scope user` had to refuse for that reason alone.

So the `mcp:` section merges and nothing else does. Project entries come
first, which gives them precedence, because `load_specs` keeps the first of
a duplicated name and reports the rest.
"""
import os
import textwrap

import pytest
import yaml

from agentchanti import mcp_cli
from agentchanti.config import Config
from agentchanti.mcp_bridge import load_specs


@pytest.fixture
def scoped(isolated_home, tmp_path, monkeypatch):
    """A home and a project directory, with the project as cwd.

    `~` already points somewhere private — tests/conftest.py redirects it
    for every test, after `mcp add --scope user` wrote a server into the
    developer's own config. This only adds the project side.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    assert os.path.abspath(os.path.expanduser("~")) == os.path.abspath(
        isolated_home)
    return isolated_home, proj


def _server(name, **kw):
    entry = {"name": name, "command": "python", "read_only": True, **kw}
    return {"mcp": {"servers": [entry]}}


def _write(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _run(args, capsys):
    code = mcp_cli.mcp_main(args)
    return code, capsys.readouterr().out


class TestTheMerge:
    def test_a_user_server_reaches_a_project_that_has_its_own_config(
            self, scoped):
        """The whole point. Before this, the project config shadowed the
        home one entirely and the user-level server simply did not exist."""
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("shared"))
        (proj / ".agentchanti.yaml").write_text("provider: openai\n",
                                                encoding="utf-8")
        specs, problems = load_specs(Config.load().MCP)
        assert [s.name for s in specs] == ["shared"]
        assert problems == []

    def test_both_are_offered_together(self, scoped):
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("from_home"))
        _write(proj / ".agentchanti.yaml", _server("from_proj"))
        specs, _p = load_specs(Config.load().MCP)
        assert sorted(s.name for s in specs) == ["from_home", "from_proj"]

    def test_the_project_wins_a_name_collision_and_it_is_reported(
            self, scoped):
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("same", command="HOME"))
        _write(proj / ".agentchanti.yaml", _server("same", command="PROJECT"))
        specs, problems = load_specs(Config.load().MCP)
        assert [s.command for s in specs] == ["PROJECT"]
        assert any("duplicate" in p and "ignored" in p for p in problems)

    def test_each_server_knows_which_file_declared_it(self, scoped):
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("from_home"))
        _write(proj / ".agentchanti.yaml", _server("from_proj"))
        specs, _p = load_specs(Config.load().MCP)
        by_name = {s.name: s.source for s in specs}
        assert by_name["from_home"] == str(home / ".agentchanti.yaml")
        assert by_name["from_proj"] == str(proj / ".agentchanti.yaml")

    def test_nothing_else_merges(self, scoped):
        """Only the `mcp:` section. A provider or API key resolved from two
        half-applied files would be worse than one."""
        home, proj = scoped
        (home / ".agentchanti.yaml").write_text(
            "provider: anthropic\nreview_mode: quick\n", encoding="utf-8")
        (proj / ".agentchanti.yaml").write_text("provider: openai\n",
                                                encoding="utf-8")
        cfg = Config.load()
        assert cfg.PROVIDER == "openai"
        assert cfg.REVIEW_MODE != "quick"

    def test_an_explicit_config_path_does_not_merge(self, scoped):
        """`--config X` means use X. Folding a home file into it would make
        that false."""
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("from_home"))
        _write(proj / ".agentchanti.yaml", _server("from_proj"))
        specs, _p = load_specs(
            Config.load(str(proj / ".agentchanti.yaml")).MCP)
        assert [s.name for s in specs] == ["from_proj"]

    def test_no_home_config_changes_nothing(self, scoped):
        home, proj = scoped
        _write(proj / ".agentchanti.yaml", _server("only"))
        specs, problems = load_specs(Config.load().MCP)
        assert [s.name for s in specs] == ["only"]
        assert problems == []
        assert not (home / ".agentchanti.yaml").exists()


class TestScopedWrites:
    def test_add_scope_user_writes_the_home_file(self, scoped, capsys):
        home, proj = scoped
        (proj / ".agentchanti.yaml").write_text("provider: openai\n",
                                                encoding="utf-8")
        code, out = _run(["add", "shared", "--scope", "user", "--read-only",
                          "python", "s.py"], capsys)
        assert code == 0
        assert str(home) in out
        assert (home / ".agentchanti.yaml").exists()
        assert "mcp" not in (proj / ".agentchanti.yaml").read_text(
            encoding="utf-8")

    def test_add_scope_project_never_writes_the_home_file(self, scoped,
                                                          capsys):
        """`_find_config_file` returns the HOME file when the project has
        none, so taking the write target from it would put a "project"
        server in the user config — the opposite of what was asked."""
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("shared"))
        code, _out = _run(["add", "mine", "--read-only", "python", "s.py"],
                          capsys)
        assert code == 0
        assert (proj / ".agentchanti.yaml").exists()
        home_servers = yaml.safe_load(
            (home / ".agentchanti.yaml").read_text(encoding="utf-8"))
        assert [s["name"] for s in home_servers["mcp"]["servers"]] == ["shared"]

    def test_remove_finds_the_file_the_server_came_from(self, scoped, capsys):
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("shared"))
        _write(proj / ".agentchanti.yaml", _server("mine"))
        code, out = _run(["remove", "shared"], capsys)
        assert code == 0
        assert str(home) in out
        assert "Remaining: mine" in out

    def test_remove_reports_what_is_left_across_both_files(self, scoped,
                                                           capsys):
        """Measured: it re-read only the edited file and said
        `Remaining: none` while the project still declared a server."""
        home, proj = scoped
        _write(home / ".agentchanti.yaml", _server("shared"))
        _write(proj / ".agentchanti.yaml", _server("mine"))
        _code, out = _run(["remove", "shared"], capsys)
        assert "Remaining: none" not in out
        assert "mine" in out

    def test_user_scope_is_refused_when_home_is_the_working_directory(
            self, isolated_home, monkeypatch, capsys):
        monkeypatch.chdir(isolated_home)
        code, out = _run(["add", "x", "--scope", "user", "--read-only",
                          "python", "s.py"], capsys)
        assert code == 2
        assert "same file" in out

    def test_an_unknown_scope_explains_itself(self, scoped, capsys):
        code, out = _run(["add", "x", "--scope", "local", "--read-only",
                          "python", "s.py"], capsys)
        assert code == 2
        assert "local" in out and "project" in out and "user" in out


def test_add_works_again_after_removing_the_last_server(scoped, capsys):
    """Removing the last server rewrites `servers:` to `servers: []` so the
    config still loads — and the first cut of `add` then refused that file
    with "no plain `servers:` key", so a config could be emptied and never
    refilled. Measured before it shipped."""
    _home, proj = scoped
    _write(proj / ".agentchanti.yaml", _server("only"))
    assert _run(["remove", "only"], capsys)[0] == 0
    assert "servers: []" in (proj / ".agentchanti.yaml").read_text(
        encoding="utf-8")
    assert _run(["add", "again", "--read-only", "python", "s.py"],
                capsys)[0] == 0
    specs, problems = load_specs(Config.load().MCP)
    assert [s.name for s in specs] == ["again"]
    assert problems == []


def test_the_merge_is_skipped_when_home_declares_no_servers(scoped):
    """Cheap and silent for the overwhelmingly common case: a home config
    that exists for a provider and knows nothing about MCP."""
    home, proj = scoped
    (home / ".agentchanti.yaml").write_text("provider: anthropic\n",
                                            encoding="utf-8")
    _write(proj / ".agentchanti.yaml", _server("only"))
    specs, _p = load_specs(Config.load().MCP)
    assert [s.name for s in specs] == ["only"]
    assert specs[0].source == ""      # untagged: no merge happened

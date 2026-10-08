"""`agentchanti mcp remove` — deletion that takes only the item's lines.

Insertion's promise mirrored. The assertion that covers both at once is a
ROUND TRIP: add a server, remove it, and the file is byte-for-byte as it
was found — comments, inline comments, key order, quoting style and all.

One case needs its own reasoning. Removing the LAST server cannot simply
delete the item and leave `servers:` with no value: `load_specs` reads a
bare `servers:` as None and reports `should be a list or mapping of
servers, got NoneType`, so tidying the file would leave it complaining.
It is rewritten to `servers: []` instead, which loads as no servers and no
problems — and keeps the section's comments, which a reader put there.
"""
import textwrap

import pytest
import yaml

from agentchanti import mcp_cli
from agentchanti.mcp_bridge import load_specs

THREE = """\
provider: openai
# keep me
mcp:
  servers:
    - name: one
      command: a
      allow: [x]
    - name: two          # inline comment on the doomed one
      command: b
      allow: [y]
    - name: three
      command: c
      read_only: true
review_mode: full
"""


def _run(args, capsys):
    code = mcp_cli.mcp_main(args)
    return code, capsys.readouterr().out


def _names(text):
    specs, _problems = load_specs((yaml.safe_load(text) or {}).get("mcp"))
    return [s.name for s in specs]


class TestRemoval:
    def test_only_the_named_item_goes(self):
        out, _note = mcp_cli._remove_server(THREE, "two")
        assert _names(out) == ["one", "three"]
        assert "# keep me" in out
        assert "inline comment on the doomed one" not in out

    def test_everything_else_survives_verbatim(self):
        out, _note = mcp_cli._remove_server(THREE, "two")
        removed = {"    - name: two          # inline comment on the doomed one",
                   "      command: b", "      allow: [y]"}
        for line in THREE.splitlines():
            if line.strip() and line not in removed:
                assert line in out.splitlines(), line

    @pytest.mark.parametrize("target,expect", [
        ("one", ["two", "three"]),
        ("two", ["one", "three"]),
        ("three", ["one", "two"]),
    ])
    def test_any_position_in_the_list(self, target, expect):
        out, _note = mcp_cli._remove_server(THREE, target)
        assert _names(out) == expect

    def test_the_name_need_not_be_on_the_dash_line(self):
        text = textwrap.dedent("""\
            mcp:
              servers:
                - transport: stdio
                  name: two
                  command: b
                  allow: [y]
                - name: one
                  command: a
                  allow: [x]
            """)
        out, _note = mcp_cli._remove_server(text, "two")
        assert _names(out) == ["one"]

    def test_a_nested_mapping_inside_the_item_goes_with_it(self):
        """A `snapshot:` pair is two levels deeper than the item's own keys,
        so an item's extent cannot be read off indentation of one line."""
        text = textwrap.dedent("""\
            mcp:
              servers:
                - name: one
                  command: a
                  allow: [x]
                  snapshot:
                    capture: 'mcp:one__go {}'
                    restore: 'mcp:one__go {}'
                - name: two
                  command: b
                  allow: [y]
            """)
        out, _note = mcp_cli._remove_server(text, "one")
        assert _names(out) == ["two"]
        assert "capture" not in out

    def test_removing_the_last_leaves_a_config_that_still_loads(self):
        text = textwrap.dedent("""\
            provider: openai
            # External tools
            mcp:
              servers:
                - name: one
                  command: a
                  allow: [x]
            review_mode: full
            """)
        out, note = mcp_cli._remove_server(text, "one")
        assert "last server" in note
        assert "servers: []" in out
        assert "# External tools" in out
        specs, problems = load_specs(yaml.safe_load(out)["mcp"])
        assert specs == [] and problems == []

    @pytest.mark.parametrize("text", [
        "mcp:\n  servers: [{name: one, command: a}]\n",
        "mcp:\n  - name: one\n    command: a\n",
    ])
    def test_an_ambiguous_shape_is_refused(self, text):
        out, note = mcp_cli._remove_server(text, "one")
        assert out is None
        assert "could corrupt" in note

    def test_a_name_that_is_not_there_is_reported(self):
        out, note = mcp_cli._remove_server(THREE, "nope")
        assert out is None
        assert "no server named 'nope'" in note


class TestRemoveCommand:
    def test_it_reports_what_is_left(self, tmp_path, capsys, monkeypatch):
        (tmp_path / ".agentchanti.yaml").write_text(THREE, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        code, out = _run(["remove", "two"], capsys)
        assert code == 0
        assert "Remaining: one, three" in out

    def test_a_missing_name_changes_nothing(self, tmp_path, capsys,
                                            monkeypatch):
        (tmp_path / ".agentchanti.yaml").write_text(THREE, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        code, out = _run(["remove", "nope"], capsys)
        assert code == 1
        assert (tmp_path / ".agentchanti.yaml").read_text(
            encoding="utf-8") == THREE

    def test_user_scope_is_rejected(self, tmp_path, capsys, monkeypatch):
        (tmp_path / ".agentchanti.yaml").write_text(THREE, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        code, _out = _run(["remove", "two", "--scope", "user"], capsys)
        assert code == 2
        assert (tmp_path / ".agentchanti.yaml").read_text(
            encoding="utf-8") == THREE

    def test_no_config_file_at_all(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, out = _run(["remove", "two"], capsys)
        assert code == 1
        assert "No .agentchanti.yaml" in out


def test_add_then_remove_is_a_byte_for_byte_round_trip(tmp_path, capsys,
                                                       monkeypatch):
    """The one assertion that covers insertion and deletion together."""
    original = textwrap.dedent("""\
        provider: openai
        # This comment and this shape must come back unchanged.
        mcp:
          servers:
            - name: blender      # inline
              transport: stdio
              command: python
              args: ["s.py"]
              allow: [get_scene_info]
        review_mode: full
        """)
    path = tmp_path / ".agentchanti.yaml"
    path.write_text(original, encoding="utf-8", newline="")
    monkeypatch.chdir(tmp_path)

    assert _run(["add", "second", "--read-only", "python", "x.py"],
                capsys)[0] == 0
    assert _names(path.read_text(encoding="utf-8")) == ["blender", "second"]
    assert _run(["remove", "second"], capsys)[0] == 0
    assert path.read_text(encoding="utf-8", newline="") == original

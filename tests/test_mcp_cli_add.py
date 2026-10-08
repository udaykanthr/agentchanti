"""`agentchanti mcp add` — writing the config without destroying it.

`--generate-config` rewrites `.agentchanti.yaml` wholesale from
`cfg.to_yaml()`. Doing that here would be destructive in a way nobody asked
for: a real MCP config is heavily commented — the one this feature was
developed against carries the reasoning for `allow:`, `env:` and the
snapshot pair inline — and a PyYAML round-trip drops every comment, blank
line and the key order with them.

So the only operation performed on an existing file is INSERTION of new
lines, and when the shape cannot be determined with certainty the edit is
REFUSED and the block printed to paste. These tests assert the promise
directly: not "does the result parse" but "is every line that was there
still there, and does the config still mean what it meant".
"""
import json

import pytest
import yaml

from agentchanti import mcp_cli

COMMENTED = """\
provider: openai
# This comment is load-bearing.
mcp:
  servers:
    - name: blender      # inline
      transport: stdio
      command: python
      args: ["s.py"]
      allow: [get_scene_info]
review_mode: full
"""

ENTRY = {"name": "added", "transport": "stdio", "command": "python",
         "args": ["x.py"], "allow": ["do_it"]}


def _run(args, capsys):
    code = mcp_cli.mcp_main(args)
    return code, capsys.readouterr().out


# ─── insertion ───────────────────────────────────────────────────────

class TestInsertion:
    def test_every_original_line_survives_verbatim(self):
        out, _note = mcp_cli._insert_server(COMMENTED, ENTRY)
        for line in COMMENTED.splitlines():
            if line.strip():
                assert line in out.splitlines(), line

    def test_the_config_still_means_what_it_meant(self):
        out, _note = mcp_cli._insert_server(COMMENTED, ENTRY)
        before, after = yaml.safe_load(COMMENTED), yaml.safe_load(out)
        assert after["provider"] == before["provider"]
        assert after["review_mode"] == before["review_mode"]
        assert [s["name"] for s in after["mcp"]["servers"]] == ["blender",
                                                                "added"]

    def test_no_mcp_section_appends_one(self):
        out, note = mcp_cli._insert_server("provider: openai\n", ENTRY)
        assert "appended" in note
        assert "provider: openai" in out
        assert yaml.safe_load(out)["mcp"]["servers"][0]["name"] == "added"

    def test_an_empty_file_is_fine(self):
        out, _note = mcp_cli._insert_server("", ENTRY)
        assert yaml.safe_load(out)["mcp"]["servers"][0]["name"] == "added"

    def test_a_file_with_no_trailing_newline_is_fine(self):
        text = ("mcp:\n  servers:\n    - name: one\n      command: a\n"
                "      allow: [x]")
        out, _note = mcp_cli._insert_server(text, ENTRY)
        assert [s["name"] for s in yaml.safe_load(out)["mcp"]["servers"]] \
            == ["one", "added"]

    def test_deeper_indentation_is_followed_not_assumed(self):
        text = ("mcp:\n    servers:\n        - name: one\n"
                "          command: a\n          allow: [x]\n")
        out, _note = mcp_cli._insert_server(text, ENTRY)
        assert [s["name"] for s in yaml.safe_load(out)["mcp"]["servers"]] \
            == ["one", "added"]

    @pytest.mark.parametrize("text", [
        "mcp:\n  servers: [{name: one, command: a}]\n",   # flow style
        "mcp:\n  - name: one\n    command: a\n",          # mcp as a list
    ])
    def test_an_ambiguous_shape_is_refused_not_guessed(self, text):
        out, note = mcp_cli._insert_server(text, ENTRY)
        assert out is None
        assert "could corrupt" in note


# ─── add ─────────────────────────────────────────────────────────────

class TestAdd:
    def test_a_fence_is_required_before_anything_is_written(
            self, tmp_path, capsys, monkeypatch):
        """A server with neither `allow:` nor `read_only: true` is REJECTED
        by `load_specs`, so writing one produces a file that parses and a
        run with no external tools — the failure this command prevents."""
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "thing", "python", "s.py", "--no-discover"],
                         capsys)
        assert code == 1
        assert "--allow" in out and "--read-only" in out
        assert not (tmp_path / ".agentchanti.yaml").exists()

    def test_discovery_offers_the_tools_the_server_really_has(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mcp_cli, "_discover",
                            lambda e: (["read_a", "read_b"], ""))
        monkeypatch.setattr("builtins.input", lambda _p: "read_a")
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "thing", "python", "s.py"], capsys)
        assert code == 0
        assert "offers 2 tool(s)" in out
        written = yaml.safe_load(
            (tmp_path / ".agentchanti.yaml").read_text(encoding="utf-8"))
        assert written["mcp"]["servers"][0]["allow"] == ["read_a"]

    def test_a_blank_reply_allows_everything_offered(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mcp_cli, "_discover",
                            lambda e: (["read_a", "read_b"], ""))
        monkeypatch.setattr("builtins.input", lambda _p: "")
        monkeypatch.chdir(tmp_path)
        assert _run(["add", "t", "python", "s.py"], capsys)[0] == 0
        written = yaml.safe_load(
            (tmp_path / ".agentchanti.yaml").read_text(encoding="utf-8"))
        assert written["mcp"]["servers"][0]["allow"] == ["read_a", "read_b"]

    def test_a_tool_the_server_does_not_offer_is_refused(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(mcp_cli, "_discover", lambda e: (["read_a"], ""))
        monkeypatch.setattr("builtins.input", lambda _p: "read_a nonesuch")
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "thing", "python", "s.py"], capsys)
        assert code == 1
        assert "does not offer: nonesuch" in out

    def test_a_closed_stdin_declines_rather_than_crashing(
            self, tmp_path, capsys, monkeypatch):
        """`isatty()` cannot answer this on Windows: the CRT reports every
        CHARACTER DEVICE as a tty, so a process whose stdin is NUL — what a
        CI runner and `subprocess.DEVNULL` both give — reports True and then
        raises EOFError on the first read. Measured before this shipped; the
        first cut guarded on `isatty()` and crashed with a traceback."""
        def _eof(_prompt):
            raise EOFError
        monkeypatch.setattr(mcp_cli, "_discover", lambda e: (["read_a"], ""))
        monkeypatch.setattr("builtins.input", _eof)
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "thing", "python", "s.py"], capsys)
        assert code == 1
        assert "--allow read_a" in out
        assert not (tmp_path / ".agentchanti.yaml").exists()

    def test_a_duplicate_name_is_refused_and_the_file_untouched(
            self, tmp_path, capsys, monkeypatch):
        (tmp_path / ".agentchanti.yaml").write_text(COMMENTED,
                                                    encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "blender", "--read-only", "python", "s.py"],
                         capsys)
        assert code == 1
        assert "already configured" in out
        assert (tmp_path / ".agentchanti.yaml").read_text(
            encoding="utf-8") == COMMENTED

    def test_user_scope_is_rejected_with_the_reason(self, tmp_path, capsys,
                                                    monkeypatch):
        """Silently omitting the flag would hide the limitation; rejecting
        it states it. `_find_config_file` returns the FIRST match and never
        merges, so a home-scoped server is dead in any project with a file."""
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "x", "--scope", "user", "--read-only",
                          "python", "s.py"], capsys)
        assert code == 2
        assert "NEVER merged" in out
        assert not (tmp_path / ".agentchanti.yaml").exists()

    def test_an_http_server_keeps_its_url_and_headers(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, _out = _run(["add", "api", "--transport", "http",
                           "https://x.test/mcp", "--read-only",
                           "-H", "Authorization: Bearer z"], capsys)
        assert code == 0
        written = yaml.safe_load(
            (tmp_path / ".agentchanti.yaml").read_text(encoding="utf-8"))
        entry = written["mcp"]["servers"][0]
        assert entry["url"] == "https://x.test/mcp"
        assert entry["headers"] == {"Authorization": "Bearer z"}

    def test_env_and_header_pairs_are_parsed(self):
        assert mcp_cli._kv(["A=1", "B=x=y"], "=", "--env") == {"A": "1",
                                                               "B": "x=y"}
        assert mcp_cli._kv(["Authorization: Bearer z"], ":", "-H") == {
            "Authorization": "Bearer z"}
        with pytest.raises(ValueError):
            mcp_cli._kv(["novalue"], "=", "--env")

    def test_the_written_entry_is_verified_by_re_reading_it(
            self, tmp_path, capsys, monkeypatch):
        """The question is whether the PIPELINE will accept the entry, and
        only the loader the pipeline uses answers that."""
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "thing", "--read-only", "python", "s.py"],
                         capsys)
        assert code == 0
        assert "Check it any time" in out
        specs, problems, _cfg, _p = mcp_cli._load(
            str(tmp_path / ".agentchanti.yaml"))
        assert [s.name for s in specs] == ["thing"]
        assert problems == []

    def test_a_refused_shape_prints_the_block_to_paste(
            self, tmp_path, capsys, monkeypatch):
        (tmp_path / ".agentchanti.yaml").write_text(
            "mcp:\n  servers: [{name: one, command: a}]\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add", "two", "--read-only", "python", "s.py"],
                         capsys)
        assert code == 1
        assert "Add this under" in out
        assert "- name: two" in out


class TestAddJson:
    def test_it_carries_snapshot_and_state_probe(self, tmp_path, capsys,
                                                 monkeypatch):
        """Why this subcommand exists: those values are `mcp:` tool calls
        carrying JSON payloads full of quotes and backslashes, and a shell
        is where this project has repeatedly lost such strings."""
        monkeypatch.chdir(tmp_path)
        blob = json.dumps({
            "transport": "stdio", "command": "python", "args": ["x.py"],
            "allow": ["execute_code"],
            "state_probe": "mcp:thing__get_info {}",
            "snapshot": {
                "capture": 'mcp:thing__execute_code {"code": "save \\"{path}\\""}',
                "restore": 'mcp:thing__execute_code {"code": "open \\"{path}\\""}',
            }})
        code, _out = _run(["add-json", "thing", blob], capsys)
        assert code == 0
        written = yaml.safe_load(
            (tmp_path / ".agentchanti.yaml").read_text(
                encoding="utf-8"))["mcp"]["servers"][0]
        assert written["state_probe"] == "mcp:thing__get_info {}"
        assert "{path}" in written["snapshot"]["capture"]
        assert '\\"' in written["snapshot"]["capture"]

    def test_bad_json_is_refused(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, out = _run(["add-json", "thing", "{not json"], capsys)
        assert code == 2
        assert "not valid JSON" in out
        assert not (tmp_path / ".agentchanti.yaml").exists()

    def test_the_name_argument_wins_over_one_inside_the_json(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, _out = _run(["add-json", "real",
                           json.dumps({"name": "other", "command": "a",
                                       "read_only": True})], capsys)
        assert code == 0
        written = yaml.safe_load(
            (tmp_path / ".agentchanti.yaml").read_text(encoding="utf-8"))
        assert [s["name"] for s in written["mcp"]["servers"]] == ["real"]

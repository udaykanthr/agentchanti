"""An undo for state that is not a file.

`snapshot.py` exists because *"neither guard is a guarantee"*. Every net in
this project is made of files and git — `wave_snapshots`,
`_enforce_monotonic_gates`' rollback, `_best_snapshot`,
`agentchanti --restore`. A run reaching a live external system through MCP has
none of them.

Measured 2026-10-06: a tool-only Blender run left an action with zero curves
and a constant 360-degree rotation partway through, and recovered only
because it had turns left to iterate with. Had it run out there, the user's
scene would have been left worse than found, with nothing able to put it
back.

The two design decisions these tests exist to hold:

**No guessing.** There is no general way to snapshot an arbitrary external
system, so the operator declares the pair and a server that declares nothing
is WARNED about rather than quietly assumed safe.

**No automatic restore.** The measured run FAILED while leaving a CORRECT
scene. Rolling back on failure would have destroyed the work the user
wanted — the `_check_advisory_stage` reasoning, and the smoke-test case where
restoring a crashing app is the wrong answer.
"""
import json
import os

import pytest

from agentchanti.orchestrator import external_state as ext


class _Spec:
    def __init__(self, name, snapshot=None):
        self.name = name
        self.snapshot = snapshot if snapshot is not None else {}


class _Bridge:
    def __init__(self, *names):
        self.names = set(names)

    def owns(self, name):
        return name in self.names


class _Tools:
    """Records the calls a capture or restore makes."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def execute_all(self, calls):
        from agentchanti.llm.chat_types import Message
        out = []
        for call in calls:
            self.calls.append((call.name, call.arguments))
            body = ("ERROR from the MCP tool: nope" if self.fail
                    else '{"saved": true}')
            out.append(Message(role="tool", content=body))
        return out


def _pair():
    return {
        "capture": 'mcp:b__save {"filepath": "{path}"}',
        "restore": 'mcp:b__open {"filepath": "{path}"}',
    }


class TestWhatCountsAsADeclaredUndo:
    def test_both_halves_present(self):
        assert ext.declared_snapshot(_Spec("b", _pair())) is not None

    def test_no_snapshot_section_is_none(self):
        assert ext.declared_snapshot(_Spec("b")) is None

    @pytest.mark.parametrize("half", ["capture", "restore"])
    def test_half_a_pair_is_refused_and_warned(self, half, caplog):
        """A capture with no restore is a file nobody can use; a restore
        with no capture has nothing to read. Either alone LOOKS like an
        undo, which is worse than plainly having none."""
        spec = _Spec("b", {half: 'mcp:b__save {"filepath": "{path}"}'})
        with caplog.at_level("WARNING"):
            assert ext.declared_snapshot(spec) is None
        assert "half a snapshot pair" in caplog.text

    def test_a_non_mapping_snapshot_is_ignored(self):
        assert ext.declared_snapshot(_Spec("b", "yes please")) is None


class TestCapture:
    def test_it_calls_the_declared_tool_with_the_path(self, tmp_path):
        tools = _Tools()
        captured = ext.capture_all(_Bridge("b__save", "b__open"), tools,
                                   {"b": _Spec("b", _pair())},
                                   root=str(tmp_path))
        assert list(captured) == ["b"]
        name, args = tools.calls[0]
        assert name == "b__save"
        assert captured["b"] in args["filepath"].replace("\\\\", "\\")

    def test_a_windows_path_survives_the_json_payload(self, tmp_path):
        r"""The command is embedded in JSON and a Windows path is full of
        backslashes; an unescaped substitution makes the payload unparseable
        and the capture is silently skipped — which would be a backstop that
        reports success and preserves nothing."""
        tools = _Tools()
        ext.capture_all(_Bridge("b__save"), tools,
                        {"b": _Spec("b", _pair())}, root=str(tmp_path))
        assert tools.calls, "the capture never reached the tool"
        # the payload parsed, which is the whole assertion
        assert "filepath" in tools.calls[0][1]

    def test_a_server_with_no_pair_is_skipped_not_failed(self, tmp_path):
        tools = _Tools()
        assert ext.capture_all(_Bridge("b__save"), tools,
                               {"b": _Spec("b")}, root=str(tmp_path)) == {}
        assert tools.calls == []

    def test_a_failing_capture_is_a_warning_not_an_exception(self, tmp_path,
                                                             caplog):
        """A backstop that can stop the thing it protects is worse than no
        backstop. But the operator asked for protection and has to learn
        they did not get it."""
        with caplog.at_level("WARNING"):
            got = ext.capture_all(_Bridge("b__save"), _Tools(fail=True),
                                  {"b": _Spec("b", _pair())},
                                  root=str(tmp_path))
        assert got == {}
        assert "could NOT capture" in caplog.text
        assert "cannot be undone" in caplog.text

    def test_an_uncallable_command_is_named_not_run(self, tmp_path, caplog):
        """A capture naming a tool nobody offers must not silently do
        nothing under the appearance of working."""
        spec = _Spec("b", {"capture": "cp -r scene /tmp/x",
                           "restore": "cp -r /tmp/x scene"})
        with caplog.at_level("WARNING"):
            assert ext.capture_all(_Bridge("b__save"), _Tools(),
                                   {"b": spec}, root=str(tmp_path)) == {}
        assert "not a runnable tool call" in caplog.text

    def test_no_bridge_captures_nothing(self, tmp_path):
        assert ext.capture_all(None, _Tools(), {"b": _Spec("b", _pair())},
                               root=str(tmp_path)) == {}


class TestTheWarningWhenThereIsNoUndo:
    def test_an_unprotected_server_is_named_once(self, caplog):
        with caplog.at_level("WARNING"):
            names = ext.warn_about_unprotected_servers(
                _Bridge(), {"blender": _Spec("blender")})
        assert names == ["blender"]
        assert "no undo for 'blender'" in caplog.text
        assert "files and git" in caplog.text

    def test_a_protected_server_is_silent(self, caplog):
        with caplog.at_level("WARNING"):
            assert ext.warn_about_unprotected_servers(
                _Bridge(), {"b": _Spec("b", _pair())}) == []
        assert "no undo" not in caplog.text

    def test_nothing_configured_is_silent(self, caplog):
        with caplog.at_level("WARNING"):
            ext.warn_about_unprotected_servers(None, {})
        assert caplog.text == ""


class TestRestore:
    def test_it_calls_the_restore_tool_when_a_capture_exists(self, tmp_path):
        bridge, tools = _Bridge("b__save", "b__open"), _Tools()
        ext.capture_all(bridge, tools, {"b": _Spec("b", _pair())},
                        root=str(tmp_path))
        # the fake tool wrote no file, so create it as a real capture would
        path = os.path.join(str(tmp_path), ext.EXTERNAL_ROOT, "b",
                            "pre_run_state")
        open(path, "w").close()
        tools2 = _Tools()
        rows = ext.restore_all(bridge, tools2, {"b": _Spec("b", _pair())},
                               root=str(tmp_path))
        assert rows and rows[0][0] == "b" and rows[0][1] is True
        assert tools2.calls[0][0] == "b__open"

    def test_restoring_without_a_capture_says_so(self, tmp_path):
        rows = ext.restore_all(_Bridge("b__open"), _Tools(),
                               {"b": _Spec("b", _pair())},
                               root=str(tmp_path))
        assert rows and rows[0][1] is False
        assert "nothing was captured" in rows[0][2]

    def test_a_server_with_no_pair_produces_no_row(self, tmp_path):
        assert ext.restore_all(_Bridge(), _Tools(), {"b": _Spec("b")},
                               root=str(tmp_path)) == []


class TestItIsWiredAndNeverAutomatic:
    def _cli_src(self):
        import pathlib

        import agentchanti
        return (pathlib.Path(agentchanti.__file__).parent / "orchestrator"
                / "cli.py").read_text(encoding="utf-8")

    def test_the_capture_runs_before_the_first_step(self):
        src = self._cli_src()
        assert "capture_all(" in src, "nothing ever captures external state"
        assert src.index("capture_all(") < src.index(
            "for wave_idx, wave in enumerate(waves)"), (
            "the capture happens after the first step could have changed "
            "the very state it is supposed to preserve")

    def test_the_warning_is_wired_too(self):
        assert "warn_about_unprotected_servers(" in self._cli_src()

    def test_restore_is_reachable_from_the_cli_flag(self):
        src = self._cli_src()
        assert "restore_all(" in src
        assert src.index("restore_all(") < src.index("Config.load(args.config)\n\n"
                                                    "    # CLI overrides"), (
            "external restore must sit in the --restore branch, which runs "
            "before any provider or API key is needed")

    def test_nothing_restores_automatically(self):
        """The measured run FAILED and left a CORRECT scene. A rollback on
        failure would have destroyed the work the user wanted kept."""
        src = self._cli_src()
        # the only call site is the --restore branch, above config loading
        assert src.count("restore_all(") == 1, (
            "a second restore_all call site would be an automatic rollback")

    def test_the_config_carries_the_pair(self):
        from agentchanti.mcp_bridge import load_specs
        specs, problems = load_specs({"servers": [{
            "name": "b", "command": "x", "allow": ["t"],
            "snapshot": _pair()}]})
        assert problems == []
        assert ext.declared_snapshot(specs[0]) is not None

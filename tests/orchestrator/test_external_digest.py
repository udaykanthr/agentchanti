"""The stall detector could not see a tool-only run.

`observe_gate_verdict` is the one mechanism designed to end a run stuck on a
gate that cannot pass, and it needs two things: repeated byte-identical
failing verdicts, AND at least two distinct artifact digests. The second is
what makes it evidence rather than impatience — without it a model that
edited nothing for three turns looks exactly like a broken gate.

`agent_loop._artifact_digest` builds that digest from the FILES the attempt
wrote. A tool-only step writes none, so the digest is constant, the
two-digest condition can never be met, and the check is silent **by
construction**. Measured 2026-10-06: two Blender runs each spent 80-90k
tokens and an escalation proving a correct artifact wrong, and this is the
check that existed to stop them at turn three.

Two decisions these tests hold:

**No guessing.** "What is the state of this system" has no general answer, so
the operator declares a read, exactly as they declare the snapshot pair. A
tool-call count would have been available for free and is the wrong signal —
it measures effort, not the artifact, and would fire on a model flailing
with real calls, which is the false-positive direction `observe_gate_verdict`
documents as the harmful one.

**Undeclared is said, not silent.** A detector structurally unable to fire
must not read as one that looked and found nothing — the
`empty_suite_reason` mistake pointed at a safety net.
"""
import pytest

from agentchanti.orchestrator import external_state as ext


class _Spec:
    def __init__(self, name, state_probe="", snapshot=None):
        self.name = name
        self.state_probe = state_probe
        self.snapshot = snapshot or {}


class _Bridge:
    """Returns a scripted body per call, so a digest can be made to move.

    The fake is the bridge because that is what `state_digest` executes
    through — an `AgentTools` only forwards to it, and a fake standing in
    for the forwarder is how `--restore` passed its tests while being
    unable to reach a tool at all.
    """

    def __init__(self, *names, bodies=()):
        self.names = set(names)
        self.bodies = list(bodies)
        self.calls = []

    def owns(self, name):
        return name in self.names

    def execute(self, name, arguments):
        self.calls.append(name)
        return self.bodies.pop(0) if self.bodies else "same"


PROBE = 'mcp:blender__get_scene_info {}'


def _bridge(*bodies):
    return _Bridge("blender__get_scene_info", bodies=bodies)


class TestWhatCountsAsADeclaredProbe:
    def test_a_declared_probe_is_found(self):
        assert ext.declared_state_probe(_Spec("b", PROBE)) == PROBE

    def test_no_probe_is_none(self):
        assert ext.declared_state_probe(_Spec("b")) is None

    def test_whitespace_is_not_a_probe(self):
        assert ext.declared_state_probe(_Spec("b", "   ")) is None


class TestTheDigest:
    def test_it_moves_when_the_state_moves(self):
        """The whole point: two distinct digests are what let
        `observe_gate_verdict` fire at all."""
        specs = {"blender": _Spec("blender", PROBE)}
        a = ext.state_digest(_bridge('{"objects": ["Cube"]}'), specs)
        b = ext.state_digest(_bridge('{"objects": ["Cube","Tree"]}'), specs)
        assert a and b and a != b

    def test_it_is_stable_when_the_state_is(self):
        """And equally important: it must NOT move on its own, or every gate
        would look like it was measuring something."""
        specs = {"blender": _Spec("blender", PROBE)}
        body = '{"objects": ["Cube"]}'
        assert (ext.state_digest(_bridge(body), specs)
                == ext.state_digest(_bridge(body), specs))

    def test_no_probe_declared_returns_none_not_a_constant(self):
        """None, deliberately — an empty string would be a digest that never
        changes, quietly restoring the exact blindness this removes."""
        assert ext.state_digest(_bridge(), {"blender": _Spec("blender")}) is None

    def test_no_bridge_returns_none(self):
        assert ext.state_digest(None,
                                {"blender": _Spec("blender", PROBE)}) is None

    def test_a_probe_that_is_not_a_tool_call_is_named(self, caplog):
        specs = {"blender": _Spec("blender", "python -c 'print(1)'")}
        with caplog.at_level("WARNING"):
            assert ext.state_digest(_bridge(), specs) is None
        assert "not a runnable tool call" in caplog.text

    def test_a_failing_probe_is_still_a_state(self):
        """A server that has gone away is not the same state as one
        answering normally, so an error contributes rather than aborting."""
        specs = {"blender": _Spec("blender", PROBE)}
        ok = ext.state_digest(_bridge('{"objects": []}'), specs)
        err = ext.state_digest(_bridge("ERROR from the MCP tool: gone"), specs)
        assert ok and err and ok != err

    def test_servers_are_ordered_so_the_digest_is_deterministic(self):
        specs = {"z": _Spec("z", "mcp:z__read {}"),
                 "a": _Spec("a", "mcp:a__read {}")}
        first = ext.state_digest(
            _Bridge("a__read", "z__read", bodies=("1", "2")), specs)
        # same bodies, dict built the other way round
        specs2 = {"a": _Spec("a", "mcp:a__read {}"),
                  "z": _Spec("z", "mcp:z__read {}")}
        second = ext.state_digest(
            _Bridge("a__read", "z__read", bodies=("1", "2")), specs2)
        assert first == second


class TestTheWarningWhenTheDetectorCannotSee:
    def test_an_unprobed_server_is_named(self, caplog):
        with caplog.at_level("WARNING"):
            names = ext.warn_about_unprobed_servers(
                _bridge(), {"blender": _Spec("blender")})
        assert names == ["blender"]
        assert "state_probe" in caplog.text
        assert "cannot be PROVEN not to measure" in caplog.text

    def test_a_probed_server_is_silent(self, caplog):
        with caplog.at_level("WARNING"):
            assert ext.warn_about_unprobed_servers(
                _bridge(), {"b": _Spec("b", PROBE)}) == []
        assert caplog.text == ""


class TestItIsWiredIntoTheLoop:
    def _loop_src(self):
        import inspect
        from agentchanti.orchestrator import agent_loop
        return inspect.getsource(agent_loop.run_agent_loop)

    def test_the_observation_uses_the_combined_digest(self):
        src = self._loop_src()
        assert "observe_gate_verdict(\n                verify_cmd, result, " \
               "_combined_digest())" in src.replace("\r\n", "\n"), (
            "the stall detector still sees only files, so a tool-only run "
            "can never meet its two-digest condition")

    def test_both_halves_are_kept(self):
        """A step may edit files and drive an external system in the same
        turn, so the file digest is not replaced."""
        src = self._loop_src()
        assert "_artifact_digest()" in src
        assert 'f"{files}|{external}"' in src

    def test_the_warning_fires_only_for_a_tool_gate(self):
        """An ordinary shell-gated run has nothing to warn about, and the
        warning must not become noise on every run with a server attached."""
        src = self._loop_src()
        i = src.index("warn_about_unprobed_servers")
        assert "is_tool_gate(" in src[max(0, i - 400):i]

    def test_the_warning_is_said_once(self):
        src = self._loop_src()
        assert "_warned_unprobed = True" in src

    def test_the_config_carries_the_probe(self):
        from agentchanti.mcp_bridge import load_specs
        specs, problems = load_specs({"servers": [{
            "name": "blender", "command": "x", "allow": ["get_scene_info"],
            "state_probe": PROBE}]})
        assert problems == []
        assert ext.declared_state_probe(specs[0]) == PROBE

    def test_a_server_without_a_probe_still_loads(self):
        from agentchanti.mcp_bridge import load_specs
        specs, problems = load_specs({"servers": [
            {"name": "b", "command": "x", "allow": ["t"]}]})
        assert problems == []
        assert ext.declared_state_probe(specs[0]) is None


class TestTheDetectorCanNowFire:
    """End to end against the real `observe_gate_verdict`: the same failing
    verdict three times, which is silent while the digest is constant and
    trips once the external digest moves."""

    def _reset(self):
        from agentchanti.orchestrator import gate_integrity as gi
        with gi._verdicts_lock:
            gi._verdicts.clear()

    def test_silent_while_the_digest_never_moves(self):
        from agentchanti.orchestrator.gate_integrity import observe_gate_verdict
        self._reset()
        shell_err = "exit: failure\nThe system cannot find the path specified."
        for _ in range(4):
            assert observe_gate_verdict("gate", shell_err, "same") is None

    def test_trips_once_the_external_state_is_visible(self):
        from agentchanti.orchestrator.gate_integrity import observe_gate_verdict
        self._reset()
        shell_err = "exit: failure\nThe system cannot find the path specified."
        got = None
        for digest in ("files|aaa", "files|bbb", "files|ccc"):
            got = observe_gate_verdict("gate", shell_err, digest) or got
        assert got, (
            "three identical failing verdicts over three distinct external "
            "states is the proof the gate is not measuring the artifact")

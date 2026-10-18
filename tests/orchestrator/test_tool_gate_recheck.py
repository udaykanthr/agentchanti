"""A tool gate has to be runnable from every route, not just the loop.

`tool_gates` made a `verify:` that names an external tool runnable, and was
wired into the two callers that knew about it: the agent loop's own
verification, and `external_state`. Every OTHER gate-running route in the
pipeline reaches `Executor.run_command`, so every other route sent a tool
gate to cmd.exe.

Measured 2026-10-07 on a live Blender run that renamed an object, recoloured
it and moved another — all three correct. `GateLedger` re-ran the step's gate
four times, after the wave and after each of the bulk-test, wiring and
smoke-test stages, and got the same answer every time::

    'mcp:blender__execute_code' is not recognized as an internal or
    external command

`_is_harness_error` reads that as "the gate can no longer launch", which is
right about what it sees and wrong about what happened. The consequence is
that the monotonic-gate protection — the one thing that catches a later
stage breaking an earlier verified step — is **structurally absent for every
tool-verified step, and says nothing about it**: the `empty_suite_reason`
mistake aimed at a safety net.

It mattered in that very run. Step 2 deleted an object the task said to
leave alone and gated itself on `assert len(scene.objects)==5`; the user's
acceptance check caught the deletion; the repair round put an object back;
and the step's own gate was therefore red at the end of a run that reported
success. Two layers had to be blind at once, and both are covered here.
"""
import pytest

from agentchanti import mcp_bridge
from agentchanti.executor import Executor
from agentchanti.orchestrator import tool_gates
from agentchanti.orchestrator.wave_snapshots import get_gate_ledger

# The run's own gate, shortened. The assertion is the one that went red
# after the acceptance repair put the deleted object back.
GATE = ('mcp:blender__execute_code {"code": "import bpy\\n'
        'assert len(bpy.context.scene.objects)==5"}')

# What cmd.exe said, verbatim and reproduced by hand afterwards.
SHELL_ERROR = ("'mcp:blender__execute_code' is not recognized as an internal "
               "or external command,\r\noperable program or batch file.")


class _Bridge:
    """Stands in for `MCPBridge`: `owns` for parsing, `execute` for running."""

    def __init__(self, *names, answers=None):
        self.names = set(names)
        self.answers = answers or {}
        self.calls = []

    def owns(self, name):
        return name in self.names

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return self.answers.get(name, "ok")


@pytest.fixture(autouse=True)
def _clean_ledger():
    get_gate_ledger().reset()
    yield
    get_gate_ledger().reset()


@pytest.fixture
def bridge(monkeypatch):
    """Install a bridge as the run's active one, as `ensure_started` would."""
    b = _Bridge("blender__execute_code", "blender__get_scene_info")
    monkeypatch.setattr(mcp_bridge, "_ACTIVE", b)
    return b


# ─── the executor seam ───────────────────────────────────────────────

def test_a_tool_gate_runs_through_the_bridge(bridge):
    ok, out = Executor().run_command(GATE, timeout=5)
    assert ok is True
    assert out.startswith("exit: success")
    assert bridge.calls == [("blender__execute_code",
                             {"code": "import bpy\nassert "
                                      "len(bpy.context.scene.objects)==5"})]


def test_a_failing_tool_gate_reads_as_a_failure_not_a_launch_error(bridge):
    """The whole defect: the shell's answer was a harness error, this is not."""
    bridge.answers["blender__execute_code"] = (
        "ERROR from the MCP tool: AssertionError")
    ex = Executor()
    ok, out = ex.run_command(GATE, timeout=5)
    assert ok is False
    assert out.startswith("exit: failure")
    assert "AssertionError" in out
    assert ex.last_exit_code == 1
    from agentchanti.orchestrator.wave_snapshots import _is_harness_error
    assert not _is_harness_error(out)


def test_the_shell_answer_really_did_read_as_a_launch_error():
    """Pins what the prior behaviour was, so the fix is not mistaken for noise."""
    from agentchanti.orchestrator.wave_snapshots import _is_harness_error
    assert _is_harness_error(SHELL_ERROR)


def test_without_a_bridge_nothing_changes(monkeypatch):
    monkeypatch.setattr(mcp_bridge, "_ACTIVE", None)
    assert Executor()._as_tool_gate(GATE) is None


def test_an_ordinary_command_carrying_the_separator_still_goes_to_the_shell(
        bridge):
    """`__` appears in real commands; only an OFFERED tool name may divert."""
    assert Executor()._as_tool_gate('python -c "print(__name__)"') is None
    assert Executor()._as_tool_gate("mcp:other__thing {}") is None
    assert bridge.calls == []


def test_the_fast_path_costs_nothing_on_an_ordinary_command(monkeypatch):
    """No separator, no bridge lookup — `run_command` is on every route."""
    def _boom():
        raise AssertionError("the bridge was consulted")
    monkeypatch.setattr("agentchanti.executor.active_bridge", _boom)
    assert Executor()._as_tool_gate("python -m pytest -q") is None


def test_a_mixed_gate_is_not_diverted(bridge):
    """Half a gate honoured is the other half silently dropped."""
    mixed = "python -m pytest && mcp:blender__execute_code {}"
    assert Executor()._as_tool_gate(mixed) is None


def test_a_destructive_payload_is_still_refused_before_dispatch(bridge):
    """The dispatch sits AFTER the destructive screen, deliberately."""
    nasty = ('mcp:blender__execute_code {"code": "taskkill /im python.exe /f"}')
    ok, out = Executor().run_command(nasty, timeout=5)
    assert ok is False
    assert "refused" in out.lower()
    assert bridge.calls == []


def test_run_and_run_via_bridge_agree(bridge):
    """One verdict path, reached two ways — the loop's and everyone else's."""
    from agentchanti.llm.chat_types import Message

    class _Tools:
        _mcp = bridge

        def execute_all(self, calls):
            return [Message(role="tool",
                            content=bridge.execute(c.name, c.arguments))
                    for c in calls]

    gates = tool_gates.parse(GATE, bridge)
    assert gates is not None
    assert (tool_gates.run(gates, _Tools())
            == tool_gates.run_via_bridge(gates, bridge))


def test_run_via_bridge_stops_at_the_first_failure(bridge):
    bridge.answers["blender__get_scene_info"] = "ERROR: server is gone"
    gates = tool_gates.parse(
        "mcp:blender__get_scene_info {} && mcp:blender__execute_code {}",
        bridge)
    out = tool_gates.run_via_bridge(gates, bridge)
    assert out.startswith("exit: failure")
    assert [n for n, _a in bridge.calls] == ["blender__get_scene_info"]


# ─── the ledger, which is what was blind ─────────────────────────────

def test_the_ledger_now_sees_a_tool_gate_regress(bridge):
    """The measured sequence: green on the step, red after a later stage."""
    led = get_gate_ledger()
    led.record(GATE, "2.1")
    assert led.recheck(Executor(), timeout=5) == []

    # The acceptance repair puts the deleted object back, so the step's own
    # count assertion is now false.
    bridge.answers["blender__execute_code"] = (
        "ERROR from the MCP tool: AssertionError")
    regressions = led.recheck(Executor(), timeout=5)
    assert [cmd for cmd, _l, _o in regressions] == [GATE]
    assert "AssertionError" in regressions[0][2]


def test_the_ledger_keeps_a_tool_gate_s_output_for_inspection(bridge):
    led = get_gate_ledger()
    led.record(GATE, "2.1")
    led.recheck(Executor(), timeout=5)
    assert (led.last_output(GATE) or "").startswith("exit: success")


def test_a_destructive_tool_gate_never_enters_the_ledger(bridge):
    """A ledger gate is re-run after every later wave, so it is worst here."""
    led = get_gate_ledger()
    led.record('mcp:blender__execute_code {"code": "rm -rf /"}', "2.1")
    assert led.gates() == {}


# ─── the stage that had no gate check at all ─────────────────────────
#
# The second half of the same incident. Even with a tool gate now runnable,
# nothing re-ran it after the acceptance repair round: the monotonic stages
# were each wave, bulk-test fixes, wiring fixes and smoke-test fixes, and the
# repair round came after all of them with nothing following it.
#
# It is also the stage most likely to break a plan gate, because it exists to
# change the artifact until an instrument outside the run goes green — which
# can directly contradict what a step asserted about its own work.


class _Snapshots:
    """Enough of `ProjectSnapshots` to see which branch was taken."""

    def __init__(self, managed=True):
        self.managed = managed
        self.committed = []
        self.marked_green = 0
        self.rollbacks = 0

    def commit_wave(self, stage):
        self.committed.append(stage)

    def mark_green(self):
        self.marked_green += 1

    def rollback_to_last(self):
        self.rollbacks += 1
        return True, "rolled back"


def test_an_authority_reports_the_conflict_and_does_not_roll_back(bridge, caplog):
    """`acceptance_cmds` passing outranks a gate the plan wrote.

    Rolling the repair back would restore a tree that instrument has already
    FAILED, which is strictly worse than keeping one it passes and naming the
    red gate. Still False: an unresolved red gate is never success.
    """
    from agentchanti.orchestrator.cli import _enforce_monotonic_gates

    led = get_gate_ledger()
    led.record(GATE, "2.1")
    led.recheck(Executor(), timeout=5)            # green on the step
    bridge.answers["blender__execute_code"] = (
        "ERROR from the MCP tool: AssertionError")

    snaps = _Snapshots()
    with caplog.at_level("ERROR"):
        ok = _enforce_monotonic_gates(
            snaps, Executor(), "acceptance repair",
            authority="the user's acceptance command(s)")
    assert ok is False
    assert snaps.rollbacks == 0
    text = caplog.text
    assert "GATE CONFLICT" in text
    assert "acceptance command(s)" in text
    assert "AssertionError" in text               # the output, not just the cmd


def test_without_an_authority_the_same_regression_rolls_back(bridge):
    """The ordinary stages keep their behaviour exactly."""
    from agentchanti.orchestrator.cli import _enforce_monotonic_gates

    led = get_gate_ledger()
    led.record(GATE, "2.1")
    led.recheck(Executor(), timeout=5)
    bridge.answers["blender__execute_code"] = (
        "ERROR from the MCP tool: AssertionError")

    snaps = _Snapshots()
    assert _enforce_monotonic_gates(
        snaps, Executor(), "wave 3") is False
    assert snaps.rollbacks == 1


def test_a_green_recheck_commits_the_stage_either_way(bridge):
    from agentchanti.orchestrator.cli import _enforce_monotonic_gates

    get_gate_ledger().record(GATE, "2.1")
    snaps = _Snapshots()
    assert _enforce_monotonic_gates(
        snaps, Executor(), "acceptance repair",
        authority="the user's acceptance command(s)") is True
    assert snaps.committed == ["acceptance repair"]
    assert snaps.marked_green == 1
    assert snaps.rollbacks == 0


def test_the_acceptance_repair_is_a_gate_checked_stage():
    """The defect is a missing CALL, so the test reads the source.

    No behavioural test of `_enforce_monotonic_gates` can see that one of
    its callers does not exist — the same reasoning the `attach_to` wiring
    tests use.
    """
    import inspect

    from agentchanti.orchestrator import cli

    src = inspect.getsource(cli)
    assert '"acceptance repair"' in src, (
        "the acceptance repair round writes source and, through a tool, live "
        "external state — it must be followed by a monotonic gate check")
    # And it must be the authority form: rolling this stage back restores a
    # tree the user's own acceptance command has already failed.
    where = src.index('"acceptance repair"')
    assert "authority=" in src[where:where + 400]


# ─── the undo that could never reach a tool ──────────────────────────


def test_capture_and_restore_execute_through_the_bridge(tmp_path):
    """Measured 2026-10-07, the first time an undo was actually needed.

    `--restore` built `build_step_tools(Executor(), FileMemory())` — a
    FRESH FileMemory, which no bridge had been attached to — so `tools._mcp`
    was None and the restore answered `ERROR: unknown tool
    'blender__execute_code'. Available: list_files, read_file, ...`. The one
    mechanism whose whole purpose is to put a live external system back
    could never have done it, on its only code path, and the tests passed
    because they hand in a fake `tools` that does answer.
    """
    from agentchanti.orchestrator import external_state as ext

    class _Spec:
        snapshot = {"capture": 'mcp:blender__execute_code {"p": "{path}"}',
                    "restore": 'mcp:blender__execute_code {"p": "{path}"}'}

    b = _Bridge("blender__execute_code")
    specs = {"blender": _Spec()}
    assert ext.capture_all(b, specs, root=str(tmp_path)) == {
        "blender": str(tmp_path / ".agentchanti" / "external" / "blender"
                       / "pre_run_state")}
    (tmp_path / ".agentchanti" / "external" / "blender"
     / "pre_run_state").write_bytes(b"x")
    rows = ext.restore_all(b, specs, root=str(tmp_path))
    assert rows and rows[0][1] is True
    assert [n for n, _a in b.calls] == ["blender__execute_code"] * 2


def test_the_restore_path_does_not_build_an_agent_tools():
    """A missing CALL and a surplus one are both invisible behaviourally.

    `restore_all` now takes the bridge itself, so there is no second object
    that can be wrong — and nothing should reintroduce one.
    """
    import inspect

    from agentchanti.orchestrator import cli

    src = inspect.getsource(cli)
    branch = src[src.index('if getattr(args, "restore", False):'):]
    branch = branch[:branch.index("# ── 0. Load config ──")]
    assert "restore_all" in branch
    assert "build_step_tools" not in branch, (
        "the restore path must not build an AgentTools: a FileMemory made "
        "here has no bridge attached, which is exactly how the undo was "
        "inert")


def test_a_failed_restore_prints_more_than_the_verdict_header():
    """`exit: failure` is the header and carries nothing.

    Printing only `row_detail.splitlines()[0]` is how the real error stayed
    invisible through the one restore that mattered.
    """
    import inspect

    from agentchanti.orchestrator import cli

    src = inspect.getsource(cli)
    branch = src[src.index('if getattr(args, "restore", False):'):]
    branch = branch[:branch.index("# ── 0. Load config ──")]
    assert "row_detail.strip()" in branch

"""An undo for state that is not a file.

`snapshot.py` copies the project before the run because *"neither guard is a
guarantee"* — the scan can form a wrong premise and the executor can be asked
to run something destructive, so there has to be a backstop that depends on
neither. Every one of agentchanti's nets is made of files and git:
`wave_snapshots`, `_enforce_monotonic_gates`' rollback, `_best_snapshot`
restore, `agentchanti --restore`.

A run that reaches a live external system through MCP has **none** of them.
Measured 2026-10-06: a tool-only Blender run wrote an action with zero
curves and a constant 360-degree rotation partway through, and recovered only
because it had turns left to iterate with. Had it run out there, the user's
scene would have been left worse than it was found, with nothing in the
system able to put it back — the artifact is not on disk, so there was
nothing to copy and nothing to restore.

**What this module does not do is guess.** There is no general way to
snapshot an arbitrary external system: saving a .blend file, dumping a
database, exporting a browser profile and committing a repository share no
vocabulary, and inventing one is how a backstop silently captures the wrong
thing. The operator — who knows the system — declares the pair, in the same
`mcp:` tool-call syntax a `verify:` uses::

    mcp:
      servers:
        - name: blender
          ...
          snapshot:
            capture: 'mcp:blender__execute_code {"code": "import bpy;
                      bpy.ops.wm.save_as_mainfile(filepath=r\\"{path}\\",
                      copy=True)"}'
            restore: 'mcp:blender__execute_code {"code": "import bpy;
                      bpy.ops.wm.open_mainfile(filepath=r\\"{path}\\")"}'

`{path}` is substituted with a run-specific file under
`.agentchanti/external/<server>/`. A server that declares no pair is
**warned about once, before the first step**, because the alternative is
silence that reads exactly like a net being present — the
`_prompt_for_acceptance_cmds` argument, decided while it is still cheap.

**Nothing is ever restored automatically, and that is deliberate.** The
measured run FAILED and left a CORRECT scene; an automatic rollback on
failure would have destroyed work the user wanted. It is the same reasoning
`_check_advisory_stage` records — a rollback must be measured against what
it rolls back to — and the smoke-test case where restoring a crashing app is
the wrong answer. Recovery is offered to the operator (`--restore`), never
taken on their behalf.
"""
from __future__ import annotations

import logging
import os

from . import tool_gates

log = logging.getLogger(__name__)

__all__ = (
    "EXTERNAL_ROOT",
    "capture_all",
    "declared_snapshot",
    "declared_state_probe",
    "restore_all",
    "state_digest",
    "warn_about_unprotected_servers",
    "warn_about_unprobed_servers",
)

EXTERNAL_ROOT = os.path.join(".agentchanti", "external")


def declared_snapshot(spec: object) -> tuple[str, str] | None:
    """The (capture, restore) pair this server declares, or None.

    Both halves are required. A capture with no restore is a file nobody can
    use, and a restore with no capture has nothing to read — either on its
    own would read as protection while providing none.
    """
    raw = getattr(spec, "snapshot", None)
    if not isinstance(raw, dict):
        return None
    capture = str(raw.get("capture") or "").strip()
    restore = str(raw.get("restore") or "").strip()
    if not capture or not restore:
        name = getattr(spec, "name", "?")
        if capture or restore:
            log.warning(
                "[External] server %r declares only half a snapshot pair "
                "(%s). Both `capture:` and `restore:` are required — one "
                "alone looks like an undo and is not one.",
                name, "capture" if capture else "restore")
        return None
    return capture, restore


def _path_for(server: str, root: str = ".") -> str:
    directory = os.path.join(os.path.abspath(root), EXTERNAL_ROOT, server)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "pre_run_state")


def _run(command: str, path: str, bridge) -> tuple[bool, str]:
    """Execute one declared command, with `{path}` substituted.

    Executed through the bridge rather than through an `AgentTools`, and
    that is the whole difference between this module working and being
    inert. Measured 2026-10-07, the first time an undo was actually needed:
    `agentchanti --restore` built its tools as
    `build_step_tools(Executor(), FileMemory())` — a FRESH FileMemory, which
    no bridge had been attached to — so `tools._mcp` was None and the
    restore answered::

        ERROR: unknown tool 'blender__execute_code'. Available: list_files,
        read_file, write_file, edit_file, run_command, search_code

    The one mechanism whose entire purpose is to put a live external system
    back could never have done it, on its only code path. The tests passed
    because they hand in a fake `tools` that does answer — the defect was in
    the WIRING, which is the `protect_acceptance_files` lesson verbatim.

    So this asks for the one object it actually needs. An `AgentTools`
    dispatches an MCP call straight to the bridge anyway, so nothing is
    lost, and there is no longer a second object that can be wrong.
    """
    # Backslashes are doubled because the command is embedded in a JSON
    # payload and a Windows path is full of them; a raw substitution would
    # make the payload unparseable and the capture would be silently skipped.
    filled = command.replace("{path}", path.replace("\\", "\\\\"))
    gates = tool_gates.parse(filled, bridge)
    if gates is None:
        return False, (
            "the command is not a runnable tool call — it must be "
            f"`mcp:<server>__<tool> {{json}}` naming an offered tool: {filled[:200]}")
    result = tool_gates.run_via_bridge(gates, bridge)
    return result.startswith("exit: success"), result


def capture_all(bridge, specs, root: str = ".") -> dict[str, str]:
    """Snapshot every server that declares how. Returns {server: path}.

    Never raises and never fails a run: a backstop that can stop the thing
    it protects is worse than no backstop. A capture that does not work is a
    WARNING naming the server, because the operator asked for protection and
    has to know they did not get it.
    """
    captured: dict[str, str] = {}
    if bridge is None:
        return captured
    for name, spec in (specs or {}).items():
        pair = declared_snapshot(spec)
        if pair is None:
            continue
        capture, _restore = pair
        path = _path_for(name, root)
        try:
            ok, detail = _run(capture, path, bridge)
        except Exception as exc:                  # pragma: no cover - env
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if ok:
            captured[name] = path
            log.info("[External] captured %r state before the run — "
                     "restore it with `agentchanti --restore`", name)
        else:
            log.warning("[External] could NOT capture %r state, so changes "
                        "this run makes to it cannot be undone: %s",
                        name, detail[:400])
    return captured


def warn_about_unprotected_servers(bridge, specs) -> list[str]:
    """Name every configured server that has no undo. Returns those names.

    Said once, before the first step, for the reason the unsatisfiable
    evidence policy is checked at startup: the answer is already fixed, and
    learning it afterwards costs a run. Silence here is indistinguishable
    from a net being present.
    """
    unprotected = [
        name for name, spec in (specs or {}).items()
        if declared_snapshot(spec) is None
    ]
    if bridge is None or not unprotected:
        return unprotected
    log.warning(
        "[External] no undo for %s. This run can change %s through its "
        "tools, and nothing in agentchanti can put it back — every other "
        "rollback here is made of files and git. Declare `snapshot: "
        "{capture: ..., restore: ...}` on the server to get one.",
        ", ".join(repr(n) for n in unprotected),
        "them" if len(unprotected) > 1 else "it")
    return unprotected


def restore_all(bridge, specs,
                root: str = ".") -> list[tuple[str, bool, str]]:
    """Put back what `capture_all` saved. One row per server attempted.

    Only ever called from `--restore`, never automatically: the measured run
    failed while leaving a CORRECT scene, so an automatic rollback would
    have destroyed the work the user wanted kept.
    """
    rows: list[tuple[str, bool, str]] = []
    if bridge is None:
        return rows
    for name, spec in (specs or {}).items():
        pair = declared_snapshot(spec)
        if pair is None:
            continue
        _capture, restore = pair
        path = _path_for(name, root)
        if not os.path.exists(path):
            rows.append((name, False,
                         f"nothing was captured for {name!r} "
                         f"({path} does not exist)"))
            continue
        try:
            ok, detail = _run(restore, path, bridge)
        except Exception as exc:                  # pragma: no cover - env
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        rows.append((name, ok, detail[:400]))
    return rows


# --------------------------------------------------------------------------
# A digest of state that is not a file.
#
# `observe_gate_verdict` is the one mechanism designed to stop a run burning
# its budget on a gate that cannot pass, and it needs two things to fire:
# repeated byte-identical failing verdicts, AND at least two distinct
# artifact digests. The second condition is what makes it evidence rather
# than impatience — without it, a model that edited nothing for three turns
# looks exactly like a broken gate.
#
# `agent_loop._artifact_digest` builds that digest from the FILES the attempt
# wrote. A tool-only run writes none, so the digest is constant, the
# two-digest condition can never be met, and the detector is silent by
# construction. Measured 2026-10-06: two Blender runs each spent 80-90k
# tokens and an escalation proving a correct artifact wrong, and this is the
# check that existed to end them at turn three.
#
# The fix is a read, and the operator declares it for the same reason they
# declare the snapshot pair: "what is the state of this system" has no
# general answer. Undeclared is warned about once rather than left silent,
# because a detector that cannot fire reads exactly like one that found
# nothing.
_PROBE_KEY = "state_probe"


def declared_state_probe(spec: object) -> str | None:
    """The read-only command that fingerprints this server's state."""
    probe = str(getattr(spec, _PROBE_KEY, None) or "").strip()
    return probe or None


def state_digest(bridge, specs) -> str | None:
    """A fingerprint of every declared external system, or None.

    None means no server declared a probe — deliberately distinct from an
    empty string, which would be a digest that never changes and would
    quietly restore the exact blindness this exists to remove.

    Never raises. A probe that fails contributes its error text, which is
    itself a state worth distinguishing: a server that has gone away is not
    the same state as one answering normally.
    """
    if bridge is None:
        return None
    parts: list[str] = []
    for name, spec in sorted((specs or {}).items()):
        probe = declared_state_probe(spec)
        if probe is None:
            continue
        gates = tool_gates.parse(probe, bridge)
        if gates is None:
            log.warning("[External] server %r: `state_probe` is not a "
                        "runnable tool call, so the stall detector stays "
                        "blind to it: %s", name, probe[:200])
            continue
        try:
            result = tool_gates.run_via_bridge(gates, bridge)
        except Exception as exc:                  # pragma: no cover - env
            result = f"probe raised {type(exc).__name__}: {exc}"
        parts.append(f"{name}:{result}")
    if not parts:
        return None
    import hashlib
    return hashlib.sha1(
        "\n".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def warn_about_unprobed_servers(bridge, specs) -> list[str]:
    """Name every server whose state the stall detector cannot see.

    Said once, when a tool gate is first observed. The alternative is a
    detector that is structurally unable to fire and says nothing about it,
    which is the `empty_suite_reason` mistake pointed at a safety net.
    """
    unprobed = [
        name for name, spec in (specs or {}).items()
        if declared_state_probe(spec) is None
    ]
    if bridge is None or not unprobed:
        return unprobed
    log.warning(
        "[External] no `state_probe` for %s, so a gate that cannot pass "
        "cannot be PROVEN not to measure: the stall detector needs two "
        "distinct artifact digests and a tool-only step writes no files. "
        "Declare `state_probe: 'mcp:<server>__<read tool> {}'` to get it.",
        ", ".join(repr(n) for n in unprobed))
    return unprobed

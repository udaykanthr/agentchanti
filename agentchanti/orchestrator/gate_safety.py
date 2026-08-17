"""Verify gates that damage the machine instead of measuring the code.

A gate is an *instrument*. It is run on the step, re-run by the agent
loop's early-exit check, re-run again by the platform-variant retry, and
re-run once more after every later wave by :class:`GateLedger` — so a
gate with a side effect does not have that side effect once, it has it a
dozen times. That makes a destructive command in a ``verify:`` line a
different class of defect from a weak one: a weak gate wastes a run, this
one can end it, and can take unrelated work on the same machine with it.

Measured, 2026-08-17 23:42. A planner emitted, as the acceptance gate for
a "beautify the game" step::

    python -c "from main import CubeCollectorGame; ..." && python main.py &
    timeout /t 2 /nobreak & taskkill /im python.exe /f 2>nul || exit /b 0

``taskkill /im python.exe /f`` names an *image*, not a process: it force-
kills every ``python.exe`` on the machine. The pipeline was one. The log
ends mid-line at the moment the executor ran it, with no ``Finished``, no
ghost reconciliation and no wave snapshot; the next run opened with
``[CrashDiag] Previous run (pid=33548) ended abnormally — no clean exit``.
The step's rewritten ``main.py`` — 7.4k to 14.7k — was left on disk with
nothing having checked it.

Note what did *not* catch it. ``check_gate_quality`` asks whether a gate
can fail on wrong behaviour; this one could. ``check_gate_consistency``
asks whether it runs in the right directory; it did. ``unrunnable_gate_
reason`` asks whether it parses; it parsed. Every existing check reads a
gate as a *measurement* and asks how good a measurement it is. None of
them asks what else it does.

The bias here is deliberate and the opposite of the other gate checks. A
false positive costs one gate — the step keeps whatever prefix survives,
and the existing repair/re-plan machinery gets a chance to write a better
one. A false negative costs the run and whatever else was running. So the
patterns are matched against the whole command text rather than only
outside quotes: a gate that merely *mentions* ``rm -rf`` inside a string
literal is rare enough, and cheap enough to lose, that guessing wrong in
that direction is the right trade.
"""

from __future__ import annotations

import re
from typing import Optional

# Top-level shell separators, longest first so `&&` is not read as `&`.
_SEPARATORS = ("&&", "||", ";", "&", "|", "\n")

# (pattern, reason). Every entry is a command that changes the machine
# rather than reporting on the code. Each says what it destroys, because
# the message is what the planner is asked to fix.
_DESTRUCTIVE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btaskkill\b(?![^\n]*\/pid\b)", re.I),
     "taskkill without /pid kills processes by image name across the "
     "whole machine, including the pipeline running the gate"),
    (re.compile(r"\b(pkill|killall)\b", re.I),
     "kills processes by name, which cannot distinguish the gate's own "
     "process from anything else on the machine"),
    (re.compile(r"\bkill\s+-(9|KILL)\b", re.I),
     "an unconditional SIGKILL in a check"),
    (re.compile(r"\bStop-Process\b(?![^\n]*-Id\b)", re.I),
     "Stop-Process without -Id kills by name across the session"),
    (re.compile(r"\bwmic\b[^\n]*\bprocess\b[^\n]*\bdelete\b", re.I),
     "deletes processes by WMI query"),
    # Recursive force delete, either flag order (-rf, -fr, -Rf, --force).
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b",
                re.I),
     "recursively force-deletes a directory tree"),
    (re.compile(r"\b(rmdir|rd)\s+/s\b", re.I),
     "recursively deletes a directory tree"),
    (re.compile(r"\bdel\b[^\n]*(/s\b|\*)", re.I),
     "deletes files by wildcard or recursively"),
    (re.compile(r"\bRemove-Item\b[^\n]*-Recurse\b", re.I),
     "recursively deletes a directory tree"),
    (re.compile(r"\bshutil\.rmtree\b"),
     "recursively deletes a directory tree"),
    (re.compile(r"\bos\.removedirs\b"),
     "removes a directory tree"),
    # The working tree is the run's deliverable — a gate must never
    # discard it to make itself pass.
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
     "discards the working tree the run is building"),
    (re.compile(r"\bgit\s+clean\s+-[a-z]*[dx]", re.I),
     "deletes untracked files, which on a greenfield run is every file "
     "the run has produced"),
    (re.compile(r"\b(shutdown|reboot|logoff|Restart-Computer|Stop-Computer)\b",
                re.I),
     "ends the session or powers the machine down"),
    (re.compile(r"\b(mkfs\S*|diskpart)\b", re.I),
     "operates on a filesystem or partition table"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.I),
     "formats a drive"),
    (re.compile(r"\bdd\s+if=", re.I),
     "writes a raw image, which can overwrite a device"),
    (re.compile(r"\bdocker\s+(system|volume|image)\s+prune\b", re.I),
     "deletes Docker state shared with everything else on the machine"),
]


def split_shell_segments(cmd: str) -> list[tuple[str, str]]:
    """Split *cmd* into ``(segment, trailing_separator)`` pairs.

    Quote-aware, so a separator inside a ``python -c "..."`` payload does
    not split the command — ``python -c "a and b"`` is one segment, and
    so is a payload containing a literal ``;``.

    The trailing separator is kept with its segment so a prefix can be
    rejoined exactly as written; the final pair's separator is ``""``.
    """
    segments: list[tuple[str, str]] = []
    buf: list[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(cmd or "")
    while i < n:
        ch = cmd[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(cmd[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        for sep in _SEPARATORS:
            if cmd.startswith(sep, i):
                segments.append(("".join(buf).strip(), sep))
                buf = []
                i += len(sep)
                break
        else:
            buf.append(ch)
            i += 1
    segments.append(("".join(buf).strip(), ""))
    return [(seg, sep) for seg, sep in segments if seg or sep]


def segment_destructive_reason(segment: str) -> Optional[str]:
    """Why this single command damages the machine, or None."""
    if not segment:
        return None
    for pattern, reason in _DESTRUCTIVE:
        if pattern.search(segment):
            return reason
    return None


def destructive_reason(cmd: str) -> Optional[str]:
    """Why *cmd* must never be run as a gate, or None.

    The reason names the offending command, because the caller quotes it
    back to the planner and "your gate is destructive" is not actionable.
    """
    if not cmd:
        return None
    for segment, _sep in split_shell_segments(cmd):
        reason = segment_destructive_reason(segment)
        if reason:
            return f"`{segment.strip()}` {reason}"
    return None


def sanitize_gate(cmd: str) -> tuple[str, Optional[str]]:
    """Return ``(safe_cmd, reason)`` — the gate truncated to its safe head.

    Everything from the first destructive command onwards is dropped,
    not just the offending segment: the tail of such a gate exists to
    clean up after the destructive part (``timeout /t 2 & taskkill ... ||
    exit /b 0``) and means nothing without it.

    The head is kept rather than discarded because it is usually the
    check the planner actually intended — in the measured incident it was
    a real assertion over the class's constants and public API. What
    survives is then judged by the ordinary gate-quality machinery, which
    can find it shallow and have it rewritten. ``reason`` is None when
    the command was already safe, in which case the text is unchanged.
    """
    reason = destructive_reason(cmd)
    if reason is None:
        return cmd, None
    kept: list[str] = []
    for segment, sep in split_shell_segments(cmd):
        if segment_destructive_reason(segment):
            break
        if segment:
            kept.append(segment + (f" {sep} " if sep else ""))
    safe = "".join(kept).strip()
    # A dangling separator is a syntax error in every shell.
    safe = re.sub(r"\s*(&&|\|\||;|&|\|)\s*$", "", safe).strip()
    return safe, reason


def check_gate_safety(steps) -> list[tuple[str, str]]:
    """Find steps whose ``verify:`` would damage the machine.

    Returns ``(step_id, reason)`` pairs in the same shape as
    :func:`check_gate_quality`, so the caller can feed them straight to
    ``repair_verify_commands`` and to the planner correction.
    """
    gaps: list[tuple[str, str]] = []
    for step in steps or ():
        reason = destructive_reason(getattr(step, "verify_cmd", "") or "")
        if reason:
            gaps.append((getattr(step, "id", "?"), reason))
    return gaps


def neutralize_destructive_gates(steps) -> list[tuple[str, str, str]]:
    """Strip destructive tails from every gate, in place. The backstop.

    Called at the point a plan is accepted, *after* repair and re-plan
    have had their chance, because those can fail — and the existing
    "proceeding after N attempts" path deliberately ships a plan whose
    gates are still imperfect. That is the right call for a weak gate and
    the wrong one for this: there is no number of attempts after which
    running ``taskkill /im python.exe /f`` becomes acceptable.

    Returns ``(step_id, original_cmd, reason)`` for each gate changed.
    """
    changed: list[tuple[str, str, str]] = []
    for step in steps or ():
        original = getattr(step, "verify_cmd", "") or ""
        if not original:
            continue
        safe, reason = sanitize_gate(original)
        if reason is None:
            continue
        step.verify_cmd = safe
        changed.append((getattr(step, "id", "?"), original, reason))
    return changed

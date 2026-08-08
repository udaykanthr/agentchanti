"""Is the GATE the defect, rather than the code it judges?

``verify_passed`` already encodes "exit 0 is not proof". This is the
mirror: **exit 1 is not proof either**, because a gate can be an invalid
instrument rather than a failing test.

WHY THIS EXISTS
---------------
Observed on a React/Vite run. The planner declared this acceptance gate::

    node -e "...if(!/@media \\\\(max-width: 48rem\\\\)[\\\\s\\\\S]*.../.test(s))process.exit(1)"

Note the DOUBLE backslashes. Under a POSIX shell the ``"..."`` quoting
collapses ``\\\\`` to ``\\`` before node ever sees it, so node compiles
``[\\s\\S]`` — "any character", the intended meaning. Under Windows
``cmd.exe`` there is no such collapsing: node receives ``[\\\\s\\\\S]``,
which is a character class matching a literal backslash, ``s`` or ``S``.
The regex can then never match ordinary CSS, so the gate was unsatisfiable
on Windows and satisfiable on Linux — from identical plan text.

The cost was not academic. The CSS edit was correct on the very first
turn. The primary loop, the escalation to a stronger model, and the
recovery loop all failed against that one gate — 24 turns across three
attempts, ~182k tokens, and the run reported failure on working code. The
escalated model even PROVED the gate wrong at turn 2 by printing each
sub-condition (all true) — and had nowhere to put that finding, because
the gate was the only thing allowed to decide.

WHY A DIFFERENTIAL RE-RUN, AND NOT A PARSER
-------------------------------------------
The broken payload is *syntactically valid* JavaScript — ``\\\\(`` parses
fine as "literal backslash, then a capture group". So a syntax check
(the ``ast.parse`` approach used by ``unrunnable_gate_reason``) cannot
catch it, in any language. Nor can "does this look like a mis-escape?",
which is a guess.

What IS decidable is behaviour: run the same text under the other shell
dialect's reading and see whether it passes. That needs no parser and no
knowledge of the payload's language — it works identically for
``python -c``, ``node -e``, ``ruby -e`` or anything else — and it yields
proof rather than suspicion.

THE SAFETY BOUNDARY
-------------------
A variant is only ever a **platform-equivalent re-reading of the identical
text**, produced by one whitelisted pure transform. It is never authored
by a model and never semantically different. Widen that and this stops
being a gate-integrity check and becomes a machine for manufacturing
false greens — "mutate the gate until something passes" — which is the
exact failure this project's verification layers exist to prevent.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Tuple

_logger = logging.getLogger(__name__)


def collapse_posix_escapes(cmd: str) -> Tuple[str, bool]:
    """Apply the backslash collapsing a POSIX shell would have applied.

    Returns ``(rewritten, changed)``.

    Only ``\\\\`` -> ``\\`` inside DOUBLE-QUOTED regions is collapsed:

    * That is the transform proven to differ between the platforms, and
      the one that broke a real run. POSIX also unescapes ``\\$`` and
      ``\\```; those are left alone because they are rarer, and every
      extra transform widens the blast radius of a wrong guess.
    * Outside quotes a backslash on Windows is usually a path separator
      (``venv\\Scripts\\activate``). Rewriting there would corrupt working
      commands to fix a bug that only occurs in quoted payloads.
    * ``\\"`` is deliberately preserved: POSIX turns it into a literal
      quote, and so do the Windows argv rules, so the two platforms
      already agree and there is nothing to reconcile.
    """
    out: List[str] = []
    in_double_quotes = False
    changed = False
    i, n = 0, len(cmd)

    while i < n:
        ch = cmd[i]

        if not in_double_quotes:
            if ch == '"':
                in_double_quotes = True
            out.append(ch)
            i += 1
            continue

        if ch == '\\' and i + 1 < n:
            nxt = cmd[i + 1]
            if nxt == '\\':
                # The pair a POSIX shell would have eaten one level of.
                out.append('\\')
                i += 2
                changed = True
                continue
            if nxt == '"':
                # Escaped quote: identical on both platforms, and it must
                # NOT toggle the quote state — treating it as a closing
                # quote would mis-scan the rest of the command.
                out.append('\\')
                out.append('"')
                i += 2
                continue

        if ch == '"':
            in_double_quotes = False
        out.append(ch)
        i += 1

    return ''.join(out), changed


def platform_equivalent_variants(cmd: str) -> List[Tuple[str, str]]:
    """Other readings of *cmd* under a different shell dialect.

    Empty on POSIX: the shell already performed the collapsing there, so
    the command the planner wrote and the command that ran already agree
    and there is no second reading to try.
    """
    if not cmd or os.name != 'nt':
        return []
    collapsed, changed = collapse_posix_escapes(cmd)
    if not changed or collapsed == cmd:
        return []
    return [("posix-backslash-collapse", collapsed)]


# ---------------------------------------------------------------------------
# Repairs, so a gate proven defective is not re-run in its broken form
# ---------------------------------------------------------------------------

# Keyed by the ORIGINAL command text rather than a step index: the same
# gate is enforced by the main loop, the escalation and the recovery loop,
# and is later re-run by the monotonic GateLedger. Keying by text means a
# repair proven once is known everywhere that command appears, without
# threading a new argument through four call sites.
_repairs: dict[str, str] = {}
_repairs_lock = threading.Lock()


def record_gate_repair(original: str, repaired: str, reason: str) -> None:
    """Remember that *original* is defective and *repaired* is equivalent."""
    if not original or not repaired or original == repaired:
        return
    with _repairs_lock:
        _repairs[original] = repaired
    _logger.warning(
        "[GateIntegrity] gate repaired (%s) — the ORIGINAL form can never "
        "pass on this platform:\n  original: %s\n  repaired: %s",
        reason, original, repaired)


def repaired_gate(cmd: str | None) -> str | None:
    """The proven-equivalent replacement for *cmd*, or None."""
    if not cmd:
        return None
    with _repairs_lock:
        return _repairs.get(cmd)


def effective_gate(cmd: str | None) -> str | None:
    """*cmd*, or its repaired form when one was proven."""
    return repaired_gate(cmd) or cmd


def reset_repairs() -> None:
    """Drop every recorded repair (tests, and between runs in-process)."""
    with _repairs_lock:
        _repairs.clear()

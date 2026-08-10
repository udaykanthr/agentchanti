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
import re
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


# ---------------------------------------------------------------------------
# A gate superseded by the command diagnosis proved equivalent
# ---------------------------------------------------------------------------
#
# The module above catches a gate broken by SHELL ESCAPING. This catches the
# other way a gate is the defect: it names something that does not exist.
#
# Observed twice on hello-world runs, both on working code:
#
#   gate: python -m pytest test_hello.py -q   -> exit 4, no such file
#   ran : python -m pytest tests/test_hello_world.py -> exit 0, 2 passed
#
# The tester had written a conventionally-named file, so the plan's gate
# pointed at a path nobody created. Diagnosis identified this correctly every
# round and proposed the working command; the pipeline RAN that command, saw
# it pass, then re-ran the gate and failed the step. Three rounds, then halt.
#
# `_handle_cmd_step` already has this idea for CMD steps: when a fix command
# is "the same core operation" as the failed one, the step is resolved rather
# than re-running something known to fail. It is restricted to CMD steps and
# compares against the failed command, so a broken CODE/TEST *gate* never
# benefits.
#
# The decision is deliberately behavioural, matching this module's existing
# stance that "what IS decidable is behaviour": re-run both, and only accept
# the substitution when the gate still fails AND the candidate still passes.

_RUNNERS = frozenset({
    "pytest", "unittest", "nose2", "tox",
    "jest", "vitest", "mocha", "jasmine", "ava",
    "rspec", "phpunit",
})
# Tools whose SUBCOMMAND decides what they do: `go test` is a gate,
# `go build` is not.
_SUBCOMMAND_RUNNERS = frozenset({"go", "cargo", "gradle", "mvn", "dotnet"})
_INTERPRETERS = frozenset({"python", "python3", "py", "node", "ruby", "perl"})
# Script runners — `npm test` really is a gate, so these carry an operation.
# pip is deliberately ABSENT: there is no `pip test`, and an installer
# proves nothing about the code. Python's actual gates (pytest, unittest,
# tox) are runners in their own right and are covered above, including the
# `poetry run pytest` / `uv run pytest` forms.
_PKG_MANAGERS = frozenset({"npm", "yarn", "pnpm", "bun"})
# Modules that provision an environment rather than judge code, even when
# invoked as `python -m <module>`.
_INSTALLER_MODULES = frozenset({"pip", "ensurepip", "venv", "virtualenv"})
# Verbs that mutate the environment rather than judge the code.
_INSTALL_VERBS = frozenset({
    "install", "add", "ci", "sync", "uninstall", "remove", "update",
    "upgrade", "restore", "fetch",
})
_SEGMENT_RE = re.compile(r"&&|\|\||;|\|")


def _basename(token: str) -> str:
    return re.split(r"[\\/]", token.strip().strip('"\''))[-1].lower()


def _segment_operation(tokens: List[str]) -> str | None:
    """The instrument one shell segment drives, or None.

    Position matters, and an early version of this ignored it: scanning
    every token for a known runner name made `pip install pytest` report
    "pytest" — the package being INSTALLED read as the instrument. Since
    that command exits 0 whenever pytest is already present, it could have
    been adopted as a stand-in for a test suite, which is exactly the
    "gate quietly replaced by something weaker" this whole mechanism has to
    refuse. An installer verifies nothing; it must have no operation at all.
    """
    if not tokens:
        return None
    head = _basename(tokens[0])
    positional = [t for t in tokens[1:] if not t.startswith("-")]

    # Installing/removing is a side effect, never a verification.
    if positional and positional[0].lower() in _INSTALL_VERBS:
        return None

    # `npm test`, `npm run test` — the runner lives in package.json, so the
    # script name is the identity. Without this a JS gate has no operation
    # and could never be superseded.
    if head in _PKG_MANAGERS:
        if not positional:
            return None
        script = positional[0].lower()
        if script == "run" and len(positional) > 1:
            script = positional[1].lower()
        return f"{head}:{script}"

    # `go test ./...` vs `go build` are different instruments.
    if head in _SUBCOMMAND_RUNNERS:
        return f"{head}:{positional[0].lower()}" if positional else head

    if head in _RUNNERS:
        return head

    if head in _INTERPRETERS:
        # `python -m pytest ...` — the module IS the runner.
        if "-m" in tokens:
            idx = tokens.index("-m")
            if idx + 1 < len(tokens):
                module = _basename(tokens[idx + 1])
                after = [t for t in tokens[idx + 2:] if not t.startswith("-")]
                # `python -m pip install ...` / `python -m venv env`: an
                # installer is still an installer when routed through -m.
                if (module in _INSTALLER_MODULES
                        or (after and after[0].lower() in _INSTALL_VERBS)):
                    return None
                return module if module in _RUNNERS else f"module:{module}"
        # `python hello.py` — the script identifies the operation.
        if positional:
            return "script:" + _basename(positional[0])
        return None

    # `poetry run pytest`, `uv run pytest -q`, `pipenv run pytest`.
    lowered = [t.lower() for t in tokens]
    if "run" in lowered:
        after = [t for t in tokens[lowered.index("run") + 1:]
                 if not t.startswith("-")]
        if after and _basename(after[0]) in _RUNNERS:
            return _basename(after[0])
    return None


def gate_operation(cmd: str) -> set[str]:
    """The tool identities *cmd* invokes — its "core operation".

    `python -m pytest a.py` and `pytest b.py` are both {"pytest"}: same
    instrument, different argument. That is the level at which one command
    can stand in for another. A command that verifies nothing — `echo ok`,
    `pip install pytest`, `cd build` — yields the empty set and can
    therefore never be accepted as a substitute, which is the point.
    """
    ops: set[str] = set()
    for segment in _SEGMENT_RE.split(cmd or ""):
        found = _segment_operation([t for t in segment.split() if t])
        if found:
            ops.add(found)
    return ops


def same_gate_operation(a: str, b: str) -> bool:
    """Do *a* and *b* drive the same instrument?"""
    return bool(gate_operation(a) & gate_operation(b))


def prove_gate_superseded(gate: str, candidate: str, run) -> bool:
    """Is *candidate* a working stand-in for a *gate* that cannot pass?

    *run* is called as ``run(cmd) -> (ok, output)``.

    Both are re-run at decision time rather than trusting the earlier
    observations: the files have changed since, and a stale "it passed once"
    is exactly the kind of evidence that lets a weak gate through. Accepting
    a substitute is only safe while the original genuinely fails, so a gate
    that has started passing is left completely alone.
    """
    if not gate or not candidate or gate.strip() == candidate.strip():
        return False
    if not same_gate_operation(gate, candidate):
        return False
    gate_ok, _ = run(gate)
    if gate_ok:
        return False        # the gate works — there is nothing to supersede
    candidate_ok, _ = run(candidate)
    return bool(candidate_ok)

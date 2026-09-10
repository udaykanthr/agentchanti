"""Acceptance tests written from the TASK, before any code exists.

`evidence.py` defines independence as a test the agent did not author:
user-supplied `acceptance_cmds`, or a pre-existing test file the run left
byte-identical. A greenfield build honestly has neither, so every such run
is judged by a suite it wrote itself — and three measured runs shipped
`exit 0` over artifacts that failed every external probe while their own
tests were green.

The point is not who writes the check, it is *when*. A test written after
`game.py` exists is written by an agent that has just read `game.py`, and
it agrees with the code by construction. A test written from the task text
alone cannot be shaped to fit an implementation that does not exist yet.

So this runs once, after the plan is final and before the first step:
generate a suite from the task, write it, and let `snapshot_test_files`
record it a moment later as pre-existing. Everything downstream is already
built — if a later step rewrites it, the hash changes, independence is
forfeited and the run says so.

What it deliberately does NOT do:

* Invent behaviour. The prompt asks only for assertions the task states in
  so many words; a task that says nothing checkable yields no file, and no
  file is an honest "nothing independent verified this".
* Overwrite anyone else's suite. A test file this module did not write is
  better evidence than anything generated here, and is never touched.
* Decide the run. A failing seed test is reported through the normal
  acceptance path; it never silently rewrites a verdict.

WHEN THE TASK CHANGES
---------------------
"A suite already exists, so skip" was too coarse by exactly one case: a
suite this module seeded **for a different task**. Measured 2026-08-17,
across four runs in one directory. Run 1 seeded a contract from a
"Panda3D cube collector" prompt. The prompt was then rewritten into a
Snake game, and every subsequent run logged ``skipped: 1 test file(s)
already predate the run and are stronger evidence`` and then
``Evidence: independent (pre-existing-tests)``. Both statements were true
under the old rule and the conclusion was worthless: the surviving file
asserts only that ``main.py`` does not exit within five seconds, which
holds for any Panda3D script that starts at all, Snake or otherwise. The
banner read exactly the same as it had when the check genuinely matched
the task.

So the seed stamps its own header — the task it was written for, and a
hash of the body it wrote — and re-seeds when the task no longer matches.
Both halves of the header are load-bearing. Without the task hash there
is no way to tell a current contract from a stale one; without the body
hash, regenerating would silently discard edits someone made by hand.
A file whose body no longer matches is treated as adopted by its editor
and left alone, the same refusal `ghost_heal` makes when a written file
declares something the plan's body does not.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

log = logging.getLogger("agentchanti")

# Written where every runner this project supports will collect it, and
# named so `_is_test_file` recognises it as a test.
SEED_BASENAME = "test_acceptance_contract.py"

# How many times to ask again when the first contract cannot fail. Two,
# because one attempt is lost outright to an unusable response, and the
# generation is cheap next to the run it is supposed to judge.
_REPAIR_ATTEMPTS = 2

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# First line of a seeded file. A comment, so it costs the suite nothing,
# and self-describing, so it survives a wiped .agentchanti directory —
# the state that decides whether to re-seed must live with the artifact
# it describes, not beside it.
_HEADER_RE = re.compile(
    r"^#\s*agentchanti:acceptance-seed\s+task=([0-9a-f]+)\s+body=([0-9a-f]+)\s*$"
)


def _fingerprint(text: str) -> str:
    """Stable short hash, insensitive to whitespace-only edits.

    Reflowing a prompt must not count as changing the task — the check
    is meant to fire when what is being ASKED FOR changes, not when a
    line was rewrapped.
    """
    normalized = " ".join((text or "").split())
    return hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()[:16]


def _header(task: str, body: str) -> str:
    return (f"# agentchanti:acceptance-seed task={_fingerprint(task)} "
            f"body={_fingerprint(body)}\n")


def seed_state(path: str) -> tuple[str, str, str] | None:
    """``(task_hash, body_hash, body)`` for a file this module wrote.

    None means "not ours" — no header, or an unreadable file — and is the
    answer that makes the module leave a user's own suite alone.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
            body = fh.read()
    except OSError:
        return None
    match = _HEADER_RE.match(first.strip())
    if not match:
        return None
    return match.group(1), match.group(2), body

_PROMPT = """\
You are writing the ACCEPTANCE TEST for a task, BEFORE any code exists.

TASK:
{task}

Write a single self-contained Python `unittest` file that checks ONLY the
behaviour the task states explicitly. Rules, all of them load-bearing:

1. Assert ONLY what the task says. Do not invent requirements, do not
   guess at internals, do not assume a file layout beyond what the task
   names. If the task states an exact API, use exactly those names.
2. Where the task gives a RANGE or a QUANTITY, exercise it properly. A
   range like "any dt from 0.001 to 0.5" means loop enough iterations at
   the SMALL end for the behaviour to actually be observable — hundreds of
   steps, not a handful. A claim about an invariant is only as good as the
   number of chances it had to fail.
3. Where the task states something must CHANGE, assert a strict relation
   between two observed values (`assertNotEqual`, `assertLess`). Never
   assert only invariants and monotonicity: `assertLessEqual(a, b)` passes
   against a frozen system.
4. Import the modules the task names inside each test method and let an
   ImportError fail that test. Do not guard with try/except or skip.
5. No mocks, no stubs, no reaching into private attributes, no assigning
   to state. Drive the system only through the public API the task names.
6. NEVER call `sys.exit()`, `quit()`, `os._exit()` or a framework's exit
   helper (`userExit()`, `destroy()` that exits) — not in the test, not
   in a `finally:`, not in `tearDown`. They raise SystemExit, which
   unittest records as an ERROR *after* your assertions have already
   passed, turning a green contract red. Let objects be garbage
   collected instead.
7. Never assert on the program's SOURCE TEXT. No `inspect.getsource`, no
   `ast.parse` of the module under test, no reading `__file__` and
   searching it for identifiers. "The source mentions `reset`" passes a
   stub that mentions it and fails correct code that calls the method
   `restart`; you would be testing vocabulary, not behaviour. Build the
   objects and call the methods.

8. NEVER require a human. No `input()`, no `tkinter` or any other GUI
   prompt, no "press a key", no asking someone to look at the screen and
   report what they saw. Nobody is watching this run: a contract that
   waits for a person never finishes, fails the run over code that may be
   perfect, and leaves an application window open that looks hung. Assert
   on the program's own state and returned values instead.

Output ONLY the Python file in one ``` fenced block. No commentary.
"""


# Rule 5 of the prompt says no mocks, and a suite that ignores it is not
# an acceptance contract — it is a description of the code's shape that
# agrees with itself. Measured 2026-08-17: a seeded contract patched
# `panda3d.core.ShowBase.__init__`, an attribute that does not exist, and
# 22 of its 23 tests ERRORED. It was still counted as this run's
# independent evidence, because nothing ran it (see evidence.classify).
# Refusing is the honest outcome: no file means "nothing independent
# verified this", which is true, where an unrunnable file means the same
# thing while looking like proof.
_MOCK_MARKERS = (
    "unittest.mock",
    "from mock import",
    "import mock",
    "MagicMock",
    "mock.patch",
    "@patch",
    "patch(",
)


def mocking_reason(src: str) -> str | None:
    """Which forbidden stubbing construct this suite uses, or None."""
    for marker in _MOCK_MARKERS:
        if marker in src:
            return marker
    return None


# A contract that needs a PERSON cannot pass an unattended run, whatever
# the code does — the same category as a gate this platform's shell cannot
# execute, which `unrunnable_gate_reason` already refuses at plan time. The
# weakness, mocking and source-grep screens all ask whether the contract
# CHECKS the right thing; none asked whether it can ever finish.
#
# Measured 2026-09-10, a 3D pinball run. The seeded contract opened a
# Tkinter prompt window and asked a human to observe the game and type
# scores in:
#
#     score_before = self.integer("Score before the final scoring hit")
#     self.action("Press Esc in the game. Do not close it with the window
#                  manager.")
#     print("\nThe game is still open; press Esc in its window ...")
#
# The run had every gate green, a suite that passed and a game that
# demonstrably worked — verified afterwards by screenshot, 1090 fps and a
# full-power launch that cleared the lane — and still exited non-zero,
# because nobody was there to answer. Worse, it is self-concealing: it
# leaves a real game window parked on the desktop, which reads as a hung
# application rather than as a contract waiting for input.
#
# Matched anywhere in the source, not line-anchored: the measured contract
# imported tkinter lazily inside a method (`    import tkinter as tk`), so
# an `^import` scan would have missed the one construct that mattered.
_INTERACTIVE_MARKERS = (
    "tkinter",
    "input(",
    "raw_input(",
    "getpass",
    "msvcrt.getch",
    "sys.stdin.read",        # also covers .readline()
    "PySimpleGUI",
    "pyautogui",
)


def interactive_reason(src: str) -> str | None:
    """Which human-in-the-loop construct makes this contract unattendable."""
    for marker in _INTERACTIVE_MARKERS:
        if marker in src:
            return marker
    return None


_INTERACTIVE_NOTE = """

Your contract CANNOT RUN UNATTENDED: it uses `{marker}`, which asks a
person to look at something, type something, or press a key. This run has
no operator. The contract will launch the application, wait forever for
input nobody will give, and fail — over code that may be perfect — while
leaving a window open that looks like a hung program.

Rewrite it to assert the same behaviours PROGRAMMATICALLY. Hard rules:

  - No `tkinter`, no `input()`, no `getpass`, no reading stdin, no
    screenshot-and-ask. Nothing that waits for a human.
  - Do NOT weaken what you check to achieve this. Drive the program's own
    public API, read its state back, and assert on the values.
  - If a behaviour is genuinely only observable on screen, assert on the
    state that produces it (positions, scores, phase, returned values)
    rather than asking someone to look.
  - The test must terminate on its own, every time, with no keypress.
"""


def _substantive_count(src: str) -> int:
    """How many discriminating assertions a candidate has, for ranking."""
    import ast

    from .seed_strength import _substantive_assertions
    try:
        return _substantive_assertions(ast.parse(src))
    except SyntaxError:
        return 0


def _generate(llm_client, task: str, extra: str = "") -> str | None:
    """One generation round: prompt, extract the fence, sanity-check it."""
    prompt = _PROMPT.format(task=task.strip())
    if extra:
        prompt += "\n\n" + extra
    try:
        raw = llm_client.generate_response(prompt)
    except Exception as exc:
        log.warning("[AcceptanceSeed] generation failed: %s", exc)
        return None
    match = _FENCE_RE.search(raw or "")
    src = (match.group(1) if match else (raw or "")).strip()
    if not _looks_like_a_suite(src):
        log.warning("[AcceptanceSeed] response was not a usable test module "
                    "— no independent check seeded")
        return None
    return src


def _looks_like_a_suite(src: str) -> bool:
    """Cheap sanity gate — it must at least be a runnable test module."""
    if not src or "import unittest" not in src:
        return False
    if "class " not in src or "def test" not in src:
        return False
    try:
        compile(src, "<seed>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def existing_test_files(root: str) -> list[str]:
    """Test files already on disk, using the pipeline's own definition."""
    from .evidence import _SKIP_DIRS
    from .pipeline import _is_test_file

    found: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                if _is_test_file(rel.replace(os.sep, "/")):
                    found.append(rel)
    except OSError:
        pass
    return found


def _should_seed(task: str, root: str, path: str) -> bool:
    """Whether to write a contract now, logging the reason either way.

    Four answers, and the order matters. Someone else's suite wins over
    everything — including a stale seed of ours sitting beside it —
    because the whole point of the skip is that independent evidence we
    did not author already exists.
    """
    others = [f for f in existing_test_files(root)
              if os.path.normpath(f) != os.path.normpath(SEED_BASENAME)]
    if others:
        log.info("[AcceptanceSeed] skipped: %d test file(s) already predate "
                 "the run and are stronger evidence (%s)",
                 len(others), ", ".join(sorted(others)[:3]))
        return False

    if not os.path.exists(path):
        return True

    state = seed_state(path)
    if state is None:
        # No header: written by a user, or by a build of this module
        # from before the header existed. Either way it is not ours to
        # replace, and it is still real pre-existing evidence.
        log.info("[AcceptanceSeed] skipped: %s predates the run and this "
                 "module did not write it", SEED_BASENAME)
        return False

    task_hash, body_hash, body = state
    if _fingerprint(body) != body_hash:
        log.info("[AcceptanceSeed] skipped: %s was edited since it was "
                 "seeded — whoever changed it owns it now", SEED_BASENAME)
        return False
    if task_hash == _fingerprint(task):
        log.info("[AcceptanceSeed] skipped: %s was seeded from this same "
                 "task and still applies", SEED_BASENAME)
        return False

    log.info("[AcceptanceSeed] re-seeding: %s was written for a DIFFERENT "
             "task, so it is not evidence about this one", SEED_BASENAME)
    return True


def seed_acceptance_tests(task: str, root: str, llm_client,
                          language: str | None = None,
                          identity_task: str | None = None) -> str | None:
    """Write a task-derived suite and return its path, or None.

    Returns None — silently, and without writing — whenever the honest
    answer is "no independent check was established": a non-Python
    project, a task with nothing checkable in it, an unusable response, or
    a suite already present that is better evidence than this one.

    *task* is what the suite is WRITTEN from — the caller passes the
    enriched requirement, which is the fuller statement. *identity_task*
    is what decides whether a contract on disk belongs to this same task,
    and must be the user's raw text: the enriched form is LLM output and
    differs between runs of an identical prompt, so fingerprinting it
    would re-seed every run and discard a contract that was fine.
    Defaults to *task* for callers that have only one of them.
    """
    if language and language.lower() not in ("python", "py"):
        log.debug("[AcceptanceSeed] skipped: language is %s", language)
        return None
    if not task or not task.strip():
        return None

    identity = identity_task if (identity_task or "").strip() else task
    path = os.path.join(root, SEED_BASENAME)
    if not _should_seed(identity, root, path):
        return None

    src = _generate(llm_client, task)
    if src is None:
        return None
    _mock = mocking_reason(src)
    if _mock:
        log.warning("[AcceptanceSeed] response mocks the system under test "
                    "(%s) — refusing it. A contract that stubs the code "
                    "cannot be evidence about the code, and this run has "
                    "no seeded independent check", _mock)
        return None

    # Repaired rather than refused outright: a contract that asks a human
    # usually checks the RIGHT behaviours and only reads them the wrong
    # way, so the strong draft is worth recovering. Refused if the retry is
    # still interactive — keeping it would guarantee a failed run.
    _human = interactive_reason(src)
    if _human:
        log.info("[AcceptanceSeed] the contract needs a human (%s) — it can "
                 "never pass an unattended run, asking again for one that "
                 "asserts programmatically", _human)
        for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
            retry = _generate(llm_client, task,
                              extra=_INTERACTIVE_NOTE.format(marker=_human))
            if retry is None or mocking_reason(retry):
                continue
            _still = interactive_reason(retry)
            if _still is None:
                log.info("[AcceptanceSeed] the repaired contract runs "
                         "unattended — using it")
                src, _human = retry, None
                break
            _human = _still
        if _human:
            log.warning(
                "[AcceptanceSeed] the contract still needs a human (%s) after "
                "%d attempt(s) — refusing it. It would launch the app, wait "
                "for input nobody will give, and fail the run over code that "
                "may be perfect; no file is the honest outcome",
                _human, _REPAIR_ATTEMPTS)
            return None

    # Strength, judged once and repaired once. Measured across three runs
    # of one prompt: 2 substantive tests, then 1 that asserted only that
    # the process had not exited — which passes over any program that
    # starts. Accurately measuring a weak instrument still reports a weak
    # measurement as a strong claim, so ask again with the specific
    # complaint, the same shape as `repair_verify_commands`.
    from .seed_strength import REPAIR_NOTE, weak_contract_reason
    _weak = weak_contract_reason(src)
    if _weak:
        log.info("[AcceptanceSeed] first contract is too weak (%s) — asking "
                 "again for one that can fail", _weak)
        # Two attempts, because an unusable RESPONSE is not the same as a
        # weak contract and must not be read as one. Measured 2026-08-18
        # 09:04: the repair came back unparseable, the single attempt was
        # spent, and the run silently kept a contract with zero
        # discriminating assertions — the strongest draft the model could
        # have written was never asked for a second time.
        for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
            retry = _generate(llm_client, task,
                              extra=REPAIR_NOTE.format(reason=_weak))
            if retry is None:
                log.info("[AcceptanceSeed] repair attempt %d/%d came back "
                         "unusable — that is a bad response, not a weak "
                         "contract, so asking again", _attempt,
                         _REPAIR_ATTEMPTS)
                continue
            if mocking_reason(retry):
                log.info("[AcceptanceSeed] repair attempt %d/%d mocks the "
                         "system under test — asking again", _attempt,
                         _REPAIR_ATTEMPTS)
                continue
            _weak_after = weak_contract_reason(retry)
            if _weak_after is None:
                log.info("[AcceptanceSeed] the repaired contract asserts real "
                         "behaviour — using it")
                src, _weak = retry, None
                break
            if _substantive_count(retry) > _substantive_count(src):
                # Not strong, but stronger; keeping the better of the two
                # is never worse than keeping the first.
                src, _weak = retry, _weak_after
    if _weak:
        # Kept rather than refused: a weak check that runs still catches a
        # crashing artifact, and refusing would trade a shallow instrument
        # for none at all. Said out loud, because "independent" will be
        # reported about it.
        log.warning("[AcceptanceSeed] the seeded contract remains SHALLOW "
                    "(%s) — this run's independent evidence can catch a "
                    "broken build but not wrong behaviour", _weak)

    if not src.endswith("\n"):
        src += "\n"
    # The body is hashed as it will sit on disk — everything after the
    # header line — so a later run can tell "unchanged since we wrote it"
    # from "someone has taken this over".
    body = "\n" + src
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_header(identity, body) + body)
    except OSError as exc:
        log.warning("[AcceptanceSeed] could not write %s: %s", path, exc)
        return None

    log.info("[AcceptanceSeed] wrote %s from the task text, before any step "
             "ran — it counts as independent evidence for exactly as long "
             "as the run leaves it byte-identical", SEED_BASENAME)
    return path


# ── Does the contract actually RUN? ──────────────────────────────────
#
# Everything above validates the contract STATICALLY: `_looks_like_a_suite`
# compiles it, `mocking_reason` reads its imports, `weak_contract_reason`
# walks its AST. Not one of those executes it, and every measured failure
# of a seeded contract has been a RUNTIME failure:
#
#   `ambient_lights[0].getColor()[:3]`   TypeError — LVecBase4f has no slice
#   `self.win.requestProperties(...)`    AttributeError — it is a
#                                        GraphicsBuffer, not a window
#
# Both compile perfectly. The reason they cluster in framework
# introspection is structural: the contract is written before the code
# exists, so it cannot name the artifact's own API, and the only
# vocabulary it can predict is the framework's — which is precisely the
# most environment- and version-fragile code there is to write blind.
#
# The contract cannot be executed when it is written; there is nothing to
# import yet. The first moment it CAN be is as soon as the code exists,
# and that is where this runs — instead of at the very end, which is
# where a broken contract used to surface, after the whole budget was
# spent. Measured 2026-09-08: one such contract failed a step's own gate
# and cost 30 turns, an escalation and 402k tokens over a game that
# scored 18/18 against external probes.

_UNRUNNABLE_NOTE = """

IGNORE the instruction above to write a NEW contract. This is a REPAIR of
one you have already written, and the code it tests now EXISTS.

Here is your contract exactly as it stands:

--- BEGIN CONTRACT ---
{contract}
--- END CONTRACT ---

Executed against the project as it now stands, it CRASHED instead of
judging it:

{output}

Return the COMPLETE corrected file. Hard constraints:

  - Keep the same behaviours under test, and keep them just as strict.
    Do NOT delete, weaken or skip an assertion to make the error go
    away, and do NOT wrap anything in try/except or unittest.skip.
  - Do NOT replace the contract with a test that calls `self.fail(...)`,
    `self.skipTest(...)` or otherwise refuses to check anything. The
    project exists and your contract already imports it successfully;
    a contract that gives up is worse than the crashing one.
  - The defect is in HOW the check inspects the program, not in WHAT it
    checks.

This kind of crash almost always comes from reaching into the
FRAMEWORK's internals — objects whose API you had to guess, or that only
exist when there is a real window or display. Prefer asserting on the
program's OWN public API and observable state (its classes, methods and
returned values) over the framework's internal objects. If a behaviour
can only be observed through the framework, assert on something the
program itself exposes instead.
"""

# unittest reports a crash as an ERROR and a judged disagreement as a
# FAILURE. Only the first says the instrument is broken; a failure may
# simply be code that is not finished yet, which at this point in the run
# is the normal state of affairs.
_FAILED_TAIL_RE = re.compile(r"FAILED \(([^)]*)\)")
_ERRORS_RE = re.compile(r"errors=(\d+)")

# The module under test does not exist yet — the contract is fine, the
# code has just not been written. Not a defect, and not this function's
# business; try again after the next wave.
_NOT_READY_RE = re.compile(
    r"ModuleNotFoundError|No module named|ImportError|cannot import name")

# One path -> repair attempts already spent. A contract that still cannot
# run afterwards is left alone: the end-of-run path reports it as
# inconclusive, which is the honest outcome and already implemented.
_REPAIR_STATE: dict = {}


def reset_contract_repairs() -> None:
    """Forget repair bookkeeping — for a second run in one process."""
    _REPAIR_STATE.clear()


def _errors_reported(output: str) -> int:
    """How many tests CRASHED, as opposed to failing an assertion."""
    tail = _FAILED_TAIL_RE.search(output or "")
    if not tail:
        return 0
    found = _ERRORS_RE.search(tail.group(1))
    return int(found.group(1)) if found else 0


def _run_contract(executor, root: str):
    cmd = "python -m unittest " + SEED_BASENAME
    try:
        return executor.run_command(cmd, timeout=180, cwd=root)
    except TypeError:
        # Executors that take no cwd — the caller is already rooted there.
        return executor.run_command(cmd, timeout=180)


def verify_contract_runs(executor, root: str, llm_client, task: str,
                         identity_task: str = None,
                         max_repairs: int = 1):
    """Execute the seeded contract; repair it once if it cannot run.

    Returns ``(relative path, new sha256)`` when the file was rewritten —
    the caller MUST update its pre-existing-test snapshot with it, or the
    repair would read as "the agent edited the contract" and forfeit the
    very independence this is protecting.

    Returns ``None`` whenever nothing was changed, including every case
    where the honest answer is "not yet": no contract, not ours, the code
    it imports does not exist, or the contract ran and merely disagreed
    with the code. An assertion FAILURE is deliberately never repaired —
    at this point in the run the code is usually incomplete, and rewriting
    the instrument because it says so is exactly the cheat this module
    exists to prevent.
    """
    if executor is None or llm_client is None:
        return None
    path = os.path.join(root, SEED_BASENAME)
    state = seed_state(path)
    if state is None:
        return None                       # absent, or not ours to touch
    if _REPAIR_STATE.get(path, 0) >= max_repairs:
        return None

    ok, out = _run_contract(executor, root)
    if ok:
        # It runs. Nothing more to check, and re-running it every wave
        # would cost a subprocess (and, for a graphical project, a window)
        # for no new information.
        _REPAIR_STATE[path] = max_repairs
        return None
    if _NOT_READY_RE.search(out or ""):
        log.debug("[AcceptanceSeed] contract cannot import the project yet "
                  "— deferring the runnability check")
        return None
    if _errors_reported(out) == 0:
        # It ran and judged. Whether it is RIGHT is the end-of-run
        # question, and a seeded contract may not convict the code anyway.
        _REPAIR_STATE[path] = max_repairs
        return None

    _REPAIR_STATE[path] = _REPAIR_STATE.get(path, 0) + 1
    tail = "\n".join((out or "").strip().splitlines()[-25:])
    log.warning("[AcceptanceSeed] the seeded contract CRASHES rather than "
                "judging the code — it is the instrument that is broken, "
                "not necessarily the artifact. Repairing it now rather "
                "than discovering this at the end of the run:\n%s", tail)

    try:
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
    except OSError as exc:
        log.warning("[AcceptanceSeed] could not read the contract: %s", exc)
        return None
    original_strength = _substantive_count(state[2])

    # Asked more than once, for the reason the weakness repair above
    # already documents: an unusable RESPONSE is not the same as a repair
    # that was judged and rejected, and must not spend the budget as if
    # it were. Measured live 2026-09-09: the single attempt came back
    # having burned all 16,384 output tokens on reasoning ("Response hit
    # the output-token limit"), the contract stayed broken, and the run
    # ended exactly as it would have without any of this.
    identity = identity_task if (identity_task or "").strip() else task
    latest = tail
    # Why the PREVIOUS attempt was rejected. Sending an identical prompt
    # twice asks a blind question: measured live 2026-09-09, both attempts
    # came back weaker than the original (0 then 1 substantive assertion
    # against 2) because nothing told the model that the first had been
    # refused, or why. The still-crashing branch below already feeds its
    # error back; every other rejection now does the same.
    rejected = ""
    for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
        # The contract itself goes in the prompt. Without it the model was
        # being asked to fix a file it could not see, under a base prompt
        # whose framing is "before any code exists" — and it did the only
        # thing that framing allows: it regenerated from scratch, decided
        # the task named no importable module, and returned a contract
        # whose single test was `self.fail(...)`. Its own prior output is
        # safe to show; the ARTIFACT is not, because a contract shaped by
        # the code it judges is no longer independent of it.
        candidate = _generate(
            llm_client, task,
            extra=_UNRUNNABLE_NOTE.format(output=latest,
                                          contract=state[2].strip())
                  + rejected)
        if candidate is None:
            log.info("[AcceptanceSeed] runnability repair %d/%d came back "
                     "unusable — that is a bad response, not a verdict on "
                     "the repair, so asking again", _attempt,
                     _REPAIR_ATTEMPTS)
            continue
        _mock = mocking_reason(candidate)
        if _mock:
            log.info("[AcceptanceSeed] runnability repair %d/%d mocks the "
                     "system under test (%s) — asking again", _attempt,
                     _REPAIR_ATTEMPTS, _mock)
            rejected = (
                "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it mocked or "
                "stubbed the system under test (%s). A contract that stubs "
                "the code cannot be evidence about the code — drive the "
                "real objects." % _mock)
            continue
        # A repair is free to introduce this defect even when the original
        # did not, so the runnability path screens for it too.
        _human = interactive_reason(candidate)
        if _human:
            log.info("[AcceptanceSeed] runnability repair %d/%d needs a "
                     "human (%s) — asking again", _attempt,
                     _REPAIR_ATTEMPTS, _human)
            rejected = _INTERACTIVE_NOTE.format(marker=_human)
            continue
        # A crash is trivially "fixed" by asserting less. Rank the repair
        # the same way the weakness repair does, and refuse a trade of
        # correctness for coverage.
        if _substantive_count(candidate) < original_strength:
            log.info("[AcceptanceSeed] runnability repair %d/%d is weaker "
                     "than the original (%d substantive assertion(s) vs "
                     "%d) — that trades a broken check for a shallow one, "
                     "so asking again", _attempt, _REPAIR_ATTEMPTS,
                     _substantive_count(candidate), original_strength)
            rejected = (
                "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it kept only %d "
                "meaningful assertion(s) where the contract above has %d. "
                "You removed checks instead of fixing the crash. Reproduce "
                "EVERY assertion from the contract above, changing only the "
                "expression that raised."
                % (_substantive_count(candidate), original_strength))
            continue

        # Static checks passed, so prove it the only way that counts: run
        # it. The write has to happen first — the contract is executed by
        # its own name, from the project root, exactly as the end-of-run
        # check will execute it.
        if not candidate.endswith("\n"):
            candidate += "\n"
        body = "\n" + candidate
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_header(identity, body) + body)
        except OSError as exc:
            log.warning("[AcceptanceSeed] could not write the repair: %s",
                        exc)
            return None

        ok_after, out_after = _run_contract(executor, root)
        if ok_after or _errors_reported(out_after) == 0:
            from .evidence import _digest
            digest = _digest(path)
            if digest is None:
                return None
            log.info("[AcceptanceSeed] the contract now runs (%s) — "
                     "repaired before the run spent its budget proving a "
                     "broken instrument right",
                     "and passes" if ok_after else "and judges the code")
            return SEED_BASENAME, digest

        # Still crashing. Restore the original — two broken contracts are
        # not better than one, and the original at least had whatever
        # strength the seeding checks approved — then feed the NEW error
        # back. Measured live 2026-09-09: a candidate that passed every
        # static check and then crashed ended the repair outright, which
        # threw away the most useful signal available (a fresh, specific
        # error about the attempt just made) while attempts remained.
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(original)
        except OSError:
            pass
        latest = "\n".join((out_after or "").strip().splitlines()[-25:])
        rejected = ("\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it still "
                    "crashed, with the error shown above. Fix that one, and "
                    "keep every assertion.")
        log.info("[AcceptanceSeed] runnability repair %d/%d still crashes — "
                 "restored the original and asking again with the new "
                 "error:\n%s", _attempt, _REPAIR_ATTEMPTS, latest)

    log.warning("[AcceptanceSeed] no usable repair after %d attempt(s) — "
                "keeping the original contract; this run's seeded evidence "
                "will be reported as inconclusive", _REPAIR_ATTEMPTS)
    return None

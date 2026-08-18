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
                 "once more for one that can fail", _weak)
        retry = _generate(llm_client, task,
                          extra=REPAIR_NOTE.format(reason=_weak))
        if retry is not None and not mocking_reason(retry):
            _weak_after = weak_contract_reason(retry)
            if _weak_after is None:
                log.info("[AcceptanceSeed] the second contract asserts real "
                         "behaviour — using it")
                src, _weak = retry, None
            elif _substantive_count(retry) > _substantive_count(src):
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

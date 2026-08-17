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

Output ONLY the Python file in one ``` fenced block. No commentary.
"""


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
                          language: str | None = None) -> str | None:
    """Write a task-derived suite and return its path, or None.

    Returns None — silently, and without writing — whenever the honest
    answer is "no independent check was established": a non-Python
    project, a task with nothing checkable in it, an unusable response, or
    a suite already present that is better evidence than this one.
    """
    if language and language.lower() not in ("python", "py"):
        log.debug("[AcceptanceSeed] skipped: language is %s", language)
        return None
    if not task or not task.strip():
        return None

    path = os.path.join(root, SEED_BASENAME)
    if not _should_seed(task, root, path):
        return None

    try:
        raw = llm_client.generate_response(_PROMPT.format(task=task.strip()))
    except Exception as exc:
        log.warning("[AcceptanceSeed] generation failed: %s", exc)
        return None

    match = _FENCE_RE.search(raw or "")
    src = (match.group(1) if match else (raw or "")).strip()
    if not _looks_like_a_suite(src):
        log.warning("[AcceptanceSeed] response was not a usable test module "
                    "— no independent check seeded")
        return None

    if not src.endswith("\n"):
        src += "\n"
    # The body is hashed as it will sit on disk — everything after the
    # header line — so a later run can tell "unchanged since we wrote it"
    # from "someone has taken this over".
    body = "\n" + src
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_header(task, body) + body)
    except OSError as exc:
        log.warning("[AcceptanceSeed] could not write %s: %s", path, exc)
        return None

    log.info("[AcceptanceSeed] wrote %s from the task text, before any step "
             "ran — it counts as independent evidence for exactly as long "
             "as the run leaves it byte-identical", SEED_BASENAME)
    return path

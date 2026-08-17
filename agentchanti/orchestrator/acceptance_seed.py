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
* Overwrite. If a suite is already there — the user's own, or a previous
  run's — that is better evidence than anything generated here.
* Decide the run. A failing seed test is reported through the normal
  acceptance path; it never silently rewrites a verdict.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("agentchanti")

# Written where every runner this project supports will collect it, and
# named so `_is_test_file` recognises it as a test.
SEED_BASENAME = "test_acceptance_contract.py"

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

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

    already = existing_test_files(root)
    if already:
        log.info("[AcceptanceSeed] skipped: %d test file(s) already predate "
                 "the run and are stronger evidence (%s)",
                 len(already), ", ".join(sorted(already)[:3]))
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
    path = os.path.join(root, SEED_BASENAME)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
    except OSError as exc:
        log.warning("[AcceptanceSeed] could not write %s: %s", path, exc)
        return None

    log.info("[AcceptanceSeed] wrote %s from the task text, before any step "
             "ran — it counts as independent evidence for exactly as long "
             "as the run leaves it byte-identical", SEED_BASENAME)
    return path

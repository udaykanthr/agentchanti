"""Path spelling helpers shared across the pipeline.

``str.lstrip("./")`` looks like "remove a leading ``./``" and is not: it
strips every leading ``.`` and ``/`` CHARACTER. That turns ``.gitignore``
into ``gitignore``, ``.env`` into ``env``, ``.github/workflows/ci.yml`` into
``github/workflows/ci.yml`` — and ``../x`` into ``x``.

Measured 2026-09-13: a plan declared ``target: .gitignore`` and the parser
recorded ``gitignore``, so the ghost reported the planned file missing and
the real ``.gitignore`` as an unplanned write — one correct step, two false
findings. The same idiom sat in about fifteen places. Where both sides of a
comparison were stripped it merely agreed with itself; where one side was
not, a dotfile silently failed to match — including the acceptance-
instrument write guard, which stored ``.ci/accept.py`` as ``ci/accept.py``
and so never refused a write to the real file.

Traversal is deliberately NOT handled here. Silently turning ``../x`` into
``x`` is a rewrite, not a refusal; the write paths refuse escapes themselves
(``Executor._sanitize_filename``, ``AgentTools._resolve``).
"""

from __future__ import annotations

import re


def strip_dot_slash(path: str) -> str:
    """Remove leading ``./`` and ``/`` PREFIXES; keep names that start with a dot.

    Expects forward slashes. ``"."`` alone becomes ``""``, which is what the
    character-stripping idiom produced and what callers treat as the root.
    """
    p = path or ""
    while True:
        if p.startswith("./"):
            p = p[2:]
        elif p.startswith("/"):
            p = p[1:]
        else:
            break
    return "" if p == "." else p


def norm_rel_path(path: str) -> str:
    """Collapse separators to single ``/`` and strip leading ``./`` and ``/``."""
    return strip_dot_slash(re.sub(r"[\\/]+", "/", (path or "").strip()))


def package_shadow_reason(root: str, path: str) -> "str | None":
    """Why writing ``path`` would collide with a package, or None.

    ``core/tests.py`` and ``core/tests/`` are two claimants to the single
    module name ``core.tests``. Promoting a module to a package — adding
    ``tests/__init__.py`` and splitting the cases across
    ``tests/test_views.py``, ``tests/test_forms.py`` — is ordinary, and
    idiomatic in Django, where ``startapp`` hands you the module and any
    real app outgrows it. Once the package exists the module name is taken,
    and a file re-claiming it breaks discovery for the whole tree:

        ImportError: 'tests' module incorrectly imported from
        '...core/tests'. Expected '...core'.

    Measured twice on 2026-09-16, benchmark task ``django-webapp``, by two
    different routes to the same end state — which is why this lives here
    rather than at either seam. First GhostHeal restored the deleted module
    from the plan's body; then, with that guarded, the plan's own step 8.1
    wrote it as its declared ``target:`` while another step had already
    built the package. Both runs were graded FAIL over an application that
    is otherwise complete: remove the one file and the suite runs 10 tests
    OK, ``/`` returns 200 and ``/dashboard/`` returns 302.

    ``__init__.py`` is never in scope: it lives *inside* a package and
    names no module of its own. A directory holding no Python is not a
    package and claims no name — ``data/`` beside ``data.py`` is fine.
    """
    import os

    rel = path.replace("\\", "/").strip("/")
    if not rel.endswith(".py") or os.path.basename(rel) == "__init__.py":
        return None
    pkg = os.path.join(root, *rel[:-3].split("/"))
    if not os.path.isdir(pkg):
        return None
    try:
        entries = os.listdir(pkg)
    except OSError:
        return None
    if "__init__.py" not in entries and not any(
            e.endswith(".py") for e in entries):
        return None
    return (f"a package directory {rel[:-3]}/ already owns the module name "
            f"this file would claim")


def _declares_nothing(text: str) -> bool:
    """True when this module defines no name — imports and a docstring only.

    The test is what the module *declares*, not how big it is: a scaffold
    carries no information whatever its wording, and a file that defines
    even one thing carries some. Unparseable is deliberately False — no
    opinion means keep the file.
    """
    import ast

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue                        # docstring
        return False
    return True


def superseded_scaffold_module(root: str, written_rel: str) -> "str | None":
    """The empty scaffold module a newly written package file supersedes.

    The third form of the collision `package_shadow_reason` refuses, and
    the one a refusal cannot fix. Measured 2026-09-16, benchmark task
    ``django-webapp``, third consecutive run:

        13:27:55  manage.py startapp core     -> core/tests.py (63 bytes)
        13:28:56  Written: core/tests/__init__.py, core/tests/test_forms.py
        13:30:12  Written: core/tests/test_views.py

    Here the package is created *beside* a module that a scaffold command
    wrote first, and every write is going exactly where it should — so
    refusing would block the correct behaviour. What is left over is
    `startapp`'s stub, which is `from django.test import TestCase` and a
    comment. The run's 10 tests are all correct and all in the right place;
    `manage.py test` dies on the stub alone, and deleting those 63 bytes
    turns the suite green (`Found 10 test(s) ... OK`).

    Removing it is therefore bounded by what the file *declares*, not by
    matching any framework's exact text: a module that defines nothing
    cannot be the thing anyone meant to keep, and the package beside it now
    owns the name. A stub with real content in it is never touched — that
    is a genuine conflict for `package_shadow_reason` to report and a human
    or the model to resolve. The agent reached the same conclusion on its
    own in the first measured incident, deleting the module when it built
    the package; this only stops the outcome depending on whether it
    remembers to.
    """
    import os

    rel = strip_dot_slash(written_rel.replace("\\", "/")).strip("/")
    if not rel.endswith(".py"):
        return None
    parts = rel.split("/")
    if len(parts) < 2:
        return None
    pkg = "/".join(parts[:-1])
    pkg_dir = os.path.join(root, *pkg.split("/"))
    if not os.path.isdir(pkg_dir):
        return None
    candidate = pkg + ".py"
    full = os.path.join(root, *candidate.split("/"))
    if not os.path.isfile(full):
        return None
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    return candidate if _declares_nothing(text) else None

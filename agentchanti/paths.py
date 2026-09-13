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

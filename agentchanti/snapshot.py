"""A copy of the project as it was before the run touched it.

WHY THIS EXISTS

Measured 2026-09-17. A user created a Next.js + TypeScript application in
``my-app/`` by hand, never committed it, and asked for a home page. The
project scan reported ``0 files detected`` — it counted only what git
already tracked — so the planner was handed an empty directory and wrote::

    > rmdir /s /q my-app && npm create next-app@latest my-app -- --js

Nineteen files of the user's work were deleted and replaced with a
JavaScript scaffold. Both halves of that are fixed now: the scan sees
untracked files, and `gate_safety.command_destructive_reason` refuses the
command. This module exists because **neither fix is a guarantee**.

The recovery that afternoon worked by luck. `git_utils.create_checkpoint_
branch` did auto-commit the files — but it is gated on the directory
already being a git repository, and in `cli.py` it runs *324 lines after*
the scan that formed the wrong premise. Run agentchanti in a plain folder,
which is exactly what a new user does, and there was no snapshot at all.

So the guarantee this module makes is narrow and absolute: **whatever
existed before the run can be restored afterwards**, with no dependency on
git, on the model behaving, or on any check correctly identifying a
dangerous command in advance. Guards decide what is allowed to happen;
this decides what can be undone.

WHAT IT DOES NOT DO

It is not a backup system. It copies the project's own files and skips
build output and dependency trees (`node_modules`, `.venv`, `.next`,
`dist`), because those are large, reproducible, and not what anyone
mourns. It refuses rather than half-copies when the tree is bigger than
the bounds below: a partial snapshot that looks like a complete one is
worse than an honest refusal, and the log says which bound was hit.
"""

from __future__ import annotations

import json
import os
import shutil
import time

from .cli_display import log
from .project_scanner import SKIP_DIRS

SNAPSHOT_ROOT = os.path.join(".agentchanti", "snapshots")
MANIFEST_NAME = "manifest.json"

# Never copied: reproducible, large, or ours.
_SKIP_DIRS = frozenset(SKIP_DIRS) | {".agentchanti", ".git"}

# Bounds. A source tree this big is not what this module is for, and
# copying it would cost more than the run it protects.
MAX_FILES = 4000
MAX_BYTES = 200 * 1024 * 1024


def _candidates(root: str) -> tuple[list[str], int]:
    """Relative paths worth preserving, and their total size."""
    out: list[str] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            try:
                total += os.path.getsize(full)
            except OSError:
                continue
            out.append(rel)
            if len(out) > MAX_FILES or total > MAX_BYTES:
                return out, total
    return out, total


def take_snapshot(root: str = ".") -> str | None:
    """Copy the project as it stands. Returns the snapshot directory.

    Called before anything reads or plans against the project, so that the
    premise a plan is built on cannot also be the thing that destroys the
    evidence of what was there.
    """
    root = os.path.abspath(root)
    files, total = _candidates(root)
    if not files:
        log.debug("[Preserve] nothing to preserve — empty project")
        return None
    if len(files) > MAX_FILES:
        log.warning("[Preserve] skipped: more than %d files. Nothing was "
                    "copied, so an undo will not be available for this run",
                    MAX_FILES)
        return None
    if total > MAX_BYTES:
        log.warning("[Preserve] skipped: project exceeds %d MB. Nothing was "
                    "copied, so an undo will not be available for this run",
                    MAX_BYTES // (1024 * 1024))
        return None

    dest = os.path.join(root, SNAPSHOT_ROOT, time.strftime("%Y%m%d_%H%M%S"))
    try:
        os.makedirs(dest, exist_ok=True)
        for rel in files:
            src = os.path.join(root, *rel.split("/"))
            dst = os.path.join(dest, *rel.split("/"))
            os.makedirs(os.path.dirname(dst) or dest, exist_ok=True)
            shutil.copy2(src, dst)
        with open(os.path.join(dest, MANIFEST_NAME), "w",
                  encoding="utf-8") as fh:
            json.dump({"files": files, "bytes": total,
                       "taken": time.time()}, fh)
    except OSError as exc:
        log.warning("[Preserve] could not preserve the project: %s", exc)
        return None

    log.info("[Preserve] preserved %d file(s) before the run — restore with "
             "`agentchanti --restore`", len(files))
    return dest


def latest_snapshot(root: str = ".") -> str | None:
    """The most recent snapshot directory, or None."""
    base = os.path.join(os.path.abspath(root), SNAPSHOT_ROOT)
    if not os.path.isdir(base):
        return None
    dirs = [os.path.join(base, d) for d in sorted(os.listdir(base))]
    dirs = [d for d in dirs
            if os.path.isfile(os.path.join(d, MANIFEST_NAME))]
    return dirs[-1] if dirs else None


def restore_snapshot(root: str = ".",
                     snapshot: str | None = None) -> tuple[bool, str]:
    """Put back every file the snapshot holds. ``(ok, detail)``.

    Deliberately ADDITIVE: it restores what was preserved and does not
    delete whatever the run has since created. Removing files would make
    the undo itself destructive, which is the behaviour this module exists
    to prevent — and a file the run added is the user's to keep or bin.
    """
    root = os.path.abspath(root)
    snapshot = snapshot or latest_snapshot(root)
    if not snapshot:
        return False, "no snapshot to restore from"
    try:
        with open(os.path.join(snapshot, MANIFEST_NAME),
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, f"unreadable snapshot: {exc}"

    restored = 0
    for rel in manifest.get("files", []):
        src = os.path.join(snapshot, *rel.split("/"))
        dst = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(src):
            continue
        try:
            os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
        except OSError as exc:
            return False, f"could not restore {rel}: {exc}"
    return True, (f"restored {restored} file(s) from "
                  f"{os.path.basename(snapshot)}")

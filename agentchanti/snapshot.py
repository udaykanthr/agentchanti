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

# Deliberately NOT under SNAPSHOT_ROOT: `latest_snapshot` sorts that
# directory's names and would pick a `pre-restore` entry as the newest,
# so the next `--restore` would undo the undo instead of repeating it.
PRE_RESTORE_ROOT = os.path.join(".agentchanti", "pre-restore")

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


def _differing(root: str, snapshot: str, files: list) -> list:
    """Snapshot files whose content on disk is no longer the same.

    Not a hash: the bytes are compared directly, because a snapshot is a
    copy sitting beside the original and reading both is what a copy costs
    anyway.
    """
    changed = []
    for rel in files:
        src = os.path.join(snapshot, *rel.split("/"))
        dst = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(src) or not os.path.isfile(dst):
            continue
        try:
            if os.path.getsize(src) != os.path.getsize(dst):
                changed.append(rel)
                continue
            with open(src, "rb") as a, open(dst, "rb") as b:
                if a.read() != b.read():
                    changed.append(rel)
        except OSError:
            # Unreadable either side: assume it differs. Preserving a copy
            # we did not need costs a few kilobytes; not preserving one we
            # did costs the work.
            changed.append(rel)
    return changed


def _preserve_before_restore(root: str, snapshot: str,
                             changed: list) -> str | None:
    """Keep a copy of what the restore is about to overwrite.

    Measured 2026-10-07, on the second real use of this module. The undo
    had just put a Blender scene back correctly — and `shutil.copy2`
    overwrote `acceptance_check_caseC.py` with the pre-run version,
    silently discarding edits made to it after the snapshot was taken. The
    log said `restored 8 file(s)`, which reads identically whether a
    restore changed nothing or threw away an afternoon.

    The docstring's promise was "additive", and it was true about
    EXISTENCE and never about CONTENT. Overwriting is the point of an undo;
    doing it irreversibly is not — this module's whole argument is that an
    undo which destroys work is the behaviour it exists to prevent.

    So: only the files that actually differ, so an ordinary restore writes
    nothing extra, and a targeted copy rather than `take_snapshot`, which
    would re-copy the whole tree and could refuse on its own bounds at the
    one moment refusing is least acceptable.
    """
    if not changed:
        return None
    dest = os.path.join(root, PRE_RESTORE_ROOT, time.strftime("%Y%m%d_%H%M%S"))
    try:
        os.makedirs(dest, exist_ok=True)
        for rel in changed:
            src = os.path.join(root, *rel.split("/"))
            dst = os.path.join(dest, *rel.split("/"))
            os.makedirs(os.path.dirname(dst) or dest, exist_ok=True)
            shutil.copy2(src, dst)
        with open(os.path.join(dest, MANIFEST_NAME), "w",
                  encoding="utf-8") as fh:
            json.dump({"files": changed, "restored_from": snapshot,
                       "taken": time.time()}, fh)
    except OSError as exc:
        # Never fails the restore: someone reaching for an undo needs it
        # more than they need this. But they are told, because a safety
        # net nobody mentions is indistinguishable from one that is there.
        log.warning("[Preserve] could not keep a copy of the %d file(s) "
                    "about to be overwritten (%s) — the restore will "
                    "proceed and those edits will be lost", len(changed), exc)
        return None
    return dest


def restore_snapshot(root: str = ".",
                     snapshot: str | None = None) -> tuple[bool, str]:
    """Put back every file the snapshot holds. ``(ok, detail)``.

    Deliberately ADDITIVE about existence: it restores what was preserved
    and does not delete whatever the run has since created. Removing files
    would make the undo itself destructive, which is the behaviour this
    module exists to prevent — and a file the run added is the user's to
    keep or bin.

    It is **not** additive about content, and cannot be: putting the old
    bytes back is what an undo means. What it does instead is keep a copy
    of anything whose content has moved since the snapshot, and say how
    many — see `_preserve_before_restore` for the afternoon that cost.
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

    files = list(manifest.get("files", []))
    changed = _differing(root, snapshot, files)
    kept = _preserve_before_restore(root, snapshot, changed)

    restored = 0
    for rel in files:
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
    detail = f"restored {restored} file(s) from {os.path.basename(snapshot)}"
    if changed:
        # Named, not just counted, up to a few: "3 files had changed" sends
        # a reader looking, while naming them answers the question.
        shown = ", ".join(changed[:5]) + ("..." if len(changed) > 5 else "")
        detail += (f" — {len(changed)} had changed since it was taken and "
                   f"were overwritten ({shown})")
        detail += (f"; the previous content is in "
                   f"{os.path.join(PRE_RESTORE_ROOT, os.path.basename(kept))}"
                   if kept else "; that content could NOT be preserved")
    return True, detail

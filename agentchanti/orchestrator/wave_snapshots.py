"""Per-wave git snapshots of the target project + monotonic gate ledger.

Safety by checkpoint/rollback instead of edit vetoes: after every green
wave the project workdir is committed to a machine-managed git repo, and
a ledger records every per-step acceptance gate that has passed. When a
later fix round leaves a previously-green gate red (a regression), the
project rolls back to the last snapshot instead of shipping the
regression.

The snapshot repo is only ever one agentchanti creates itself (marked
with ``.git/agentchanti-managed``). When the workdir already lives in a
git repository — the user's own — snapshots are disabled: nesting a
machine repo inside a user repo would hide their files from their own
tracking, and their git history is already a better safety net.
"""

from __future__ import annotations

import logging
import os
import subprocess
from threading import Lock

_logger = logging.getLogger(__name__)

# Identity flags so snapshot commits work on machines with no git
# user configured; --no-verify at the call sites keeps user-level
# hook templates from interfering with machine commits.
_GIT_ID = [
    "-c", "user.name=agentchanti",
    "-c", "user.email=agentchanti@local",
    "-c", "commit.gpgsign=false",
]

_MARKER_NAME = "agentchanti-managed"

_DEFAULT_GITIGNORE = """\
.agentchanti/
venv/
.venv/
node_modules/
__pycache__/
*.pyc
.pytest_cache/
db.sqlite3
"""


# ── Gate ledger ───────────────────────────────────────────────────────


# Signatures of a gate command that could not even launch — the
# interpreter/executable was not found or the cwd was wrong. These mean
# the recorded command is un-runnable in the current environment, NOT
# that the project's code regressed, so they must never trigger a
# rollback of otherwise-green code. Genuine code failures (assertion
# errors, ImportError/ModuleNotFoundError from the project's own modules,
# non-zero test exits) are deliberately excluded.
_HARNESS_ERROR_SIGNATURES = (
    "the system cannot find the path specified",
    "the system cannot find the file specified",
    "is not recognized as an internal or external command",
    "no such file or directory",
    "command not found",
    "can't open file",
    "cannot open file",
)


def _is_harness_error(out: str | None) -> bool:
    """True when *out* shows the gate command failed to launch (env/cwd),
    rather than the project's code failing the check."""
    low = (out or "").lower()
    return any(sig in low for sig in _HARNESS_ERROR_SIGNATURES)


class GateLedger:
    """Acceptance commands that have passed at least once this run.

    Thread-safe (steps in the same wave run in parallel). Keyed by the
    exact command string — the same gate recorded by several steps is
    rechecked once.
    """

    def __init__(self) -> None:
        self._gates: dict[str, str] = {}  # cmd -> step label
        self._lock = Lock()

    def record(self, cmd: str, step_label: str = "") -> None:
        if not cmd:
            return
        with self._lock:
            if cmd not in self._gates:
                self._gates[cmd] = step_label
                _logger.info("[GateLedger] recorded gate (%s): %s",
                             step_label or "?", cmd)

    def gates(self) -> dict[str, str]:
        with self._lock:
            return dict(self._gates)

    def reset(self) -> None:
        with self._lock:
            self._gates.clear()

    def recheck(self, executor, timeout: int = 300) -> list[tuple[str, str, str]]:
        """Re-run every recorded gate; return the ones that now fail.

        Each regression is ``(cmd, step_label, output_tail)``. An empty
        list means monotonic progress holds.
        """
        regressions: list[tuple[str, str, str]] = []
        for cmd, label in self.gates().items():
            ok, out = executor.run_command(cmd, timeout=timeout)
            if ok:
                continue
            if _is_harness_error(out):
                # The command can no longer launch (missing interpreter /
                # wrong cwd) — inconclusive, not a code regression. Rolling
                # back over this would discard good code (observed: a
                # `cd sub && venv\Scripts\python.exe ...` gate whose venv
                # path stopped resolving from the sub-dir).
                _logger.warning(
                    "[GateLedger] gate no longer launches — harness/env "
                    "error, treating as inconclusive rather than a "
                    "regression (%s): %s", label or "?", cmd)
                continue
            _logger.warning(
                "[GateLedger] REGRESSION — previously-passing gate "
                "now fails (%s): %s", label or "?", cmd)
            regressions.append((cmd, label, (out or "")[-1500:]))
        return regressions


_ledger = GateLedger()


def get_gate_ledger() -> GateLedger:
    return _ledger


# ── Project snapshots ─────────────────────────────────────────────────


class ProjectSnapshots:
    """Wave-granular git snapshots of the pipeline's working directory."""

    def __init__(self, root: str = ".", enabled: bool = True) -> None:
        self.root = os.path.abspath(root)
        self.enabled = enabled
        self.managed = False           # True only for a repo we created
        self._last_sha: str | None = None
        # Last snapshot whose gate recheck actually passed. Distinct from
        # _last_sha: a wave is committed before its gates are rechecked, so
        # _last_sha may be the very commit that introduced a regression.
        # Rolling back must target the last *verified* state, not HEAD.
        self._last_green_sha: str | None = None

    # -- plumbing ------------------------------------------------------

    def _git(self, *args: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git", *_GIT_ID, *args],
                cwd=self.root, capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        out = ((result.stdout or "") + (result.stderr or "")).strip()
        return result.returncode == 0, out

    def _marker_path(self) -> str:
        return os.path.join(self.root, ".git", _MARKER_NAME)

    def _ensure_gitignore(self) -> None:
        """Make sure every default ignore rule is present in .gitignore.

        Appends missing rules instead of overwriting, so a user-authored
        .gitignore is preserved. Idempotent on both fresh and resumed repos.
        """
        path = os.path.join(self.root, ".gitignore")
        existing = ""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except OSError:
                return
        present = {ln.strip() for ln in existing.splitlines()}
        missing = [ln for ln in _DEFAULT_GITIGNORE.splitlines()
                   if ln.strip() and ln.strip() not in present]
        if not missing:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write("\n".join(missing) + "\n")
        except OSError:
            pass

    def _is_tracked(self, path: str) -> bool:
        ok, out = self._git("ls-files", "--", path)
        return ok and bool(out.strip())

    def _purge_tracked_volatile(self) -> None:
        """Untrack volatile runtime dirs that an earlier run may have committed.

        `.gitignore` never untracks already-committed files, so a stale
        `.agentchanti/` (live logs, cache) stays tracked and every
        `reset --hard` tries to restore/unlink the log file the running
        process holds open — on Windows that fails with "Invalid argument"
        and disables rollback. Dropping them from the index (working tree
        untouched) removes that hazard.
        """
        removed: list[str] = []
        for rel in (".agentchanti", "venv", ".venv", "node_modules",
                    "__pycache__"):
            if self._is_tracked(rel):
                self._git("rm", "-r", "--cached", "--ignore-unmatch", "--", rel)
                removed.append(rel)
        if removed:
            self._commit("agentchanti: stop tracking volatile runtime state")
            _logger.info("[Snapshots] Untracked volatile dir(s) so rollback "
                         "can't be blocked by open handles: %s",
                         ", ".join(removed))

    def _head_sha(self) -> str | None:
        ok, out = self._git("rev-parse", "HEAD")
        return out.strip() if ok else None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> bool:
        """Initialise snapshotting. Returns True when snapshots are live."""
        if not self.enabled:
            return False

        if os.path.isfile(self._marker_path()):
            # Resumed run inside a repo this tool created earlier.
            self.managed = True
            # Older runs may have committed .agentchanti/ (logs, cache) before
            # the ignore rule existed; while those stay tracked, reset --hard
            # tries to unlink the live log Windows holds open and rollback
            # dies. Re-assert the ignore rules and untrack volatile state.
            self._ensure_gitignore()
            self._purge_tracked_volatile()
            self._last_sha = self._head_sha()
            # Nothing has run yet this session, so the resumed HEAD is the
            # baseline we can safely fall back to.
            self._last_green_sha = self._last_sha
            _logger.info("[Snapshots] Resuming managed snapshot repo at %s",
                         self.root)
            return True

        ok, _ = self._git("rev-parse", "--is-inside-work-tree")
        if ok:
            # The workdir belongs to an existing (user) repository.
            self.enabled = False
            _logger.info(
                "[Snapshots] Workdir is inside an existing git repo — "
                "wave snapshots disabled (the user's git is the safety net)")
            return False

        ok, out = self._git("init", "-q")
        if not ok:
            self.enabled = False
            _logger.warning("[Snapshots] git init failed — snapshots "
                            "disabled: %s", out[:200])
            return False

        self._ensure_gitignore()

        try:
            with open(self._marker_path(), "w", encoding="utf-8") as f:
                f.write("snapshot repo created by agentchanti\n")
        except OSError:
            pass

        self.managed = True
        self._commit("agentchanti: baseline (pre-run)")
        self._last_green_sha = self._last_sha
        _logger.info("[Snapshots] Initialised snapshot repo at %s", self.root)
        return True

    def _commit(self, message: str) -> str | None:
        ok, out = self._git("status", "--porcelain")
        if ok and not out.strip():
            return self._last_sha  # nothing new to snapshot
        self._git("add", "-A")
        ok, out = self._git("commit", "-q", "--no-verify", "-m", message)
        if not ok:
            _logger.warning("[Snapshots] commit failed: %s", out[:200])
            return self._last_sha
        self._last_sha = self._head_sha()
        return self._last_sha

    def commit_wave(self, label: str) -> str | None:
        """Snapshot the workdir after a wave. Returns the commit sha.

        The commit is *not* yet a rollback target — call :meth:`mark_green`
        once the wave's gates have been rechecked and still pass.
        """
        if not self.managed:
            return None
        sha = self._commit(f"agentchanti: {label}")
        if sha:
            _logger.info("[Snapshots] %s -> %s", label, sha[:12])
        return sha

    def mark_green(self) -> str | None:
        """Promote the latest snapshot to "last verified green".

        Only call this after the gate ledger has been rechecked and found
        clean. Everything committed since the previous green snapshot then
        becomes permanent as far as rollback is concerned.
        """
        if not self.managed:
            return None
        self._last_green_sha = self._last_sha
        return self._last_green_sha

    def last_green_sha(self) -> str | None:
        return self._last_green_sha

    def rollback_to_last(self) -> tuple[bool, str]:
        """Hard-restore the workdir to the last *verified green* snapshot.

        Deliberately not HEAD: waves are committed before their gates are
        rechecked, so HEAD can be the very commit that introduced the
        regression — resetting to it would "roll back" to the broken state
        and report success while discarding nothing.

        Untracked files are removed too (``clean -fd``) — but ignored
        ones (venv/, node_modules/, .agentchanti/) survive, so the
        environment does not need rebuilding after a rollback.
        """
        if not self.managed or not self._last_green_sha:
            return False, "no snapshot available"
        ok1, out1 = self._git("reset", "--hard", self._last_green_sha)
        ok2, out2 = self._git("clean", "-fd")
        ok = ok1 and ok2
        if ok:
            _logger.warning("[Snapshots] Rolled back workdir to %s",
                            self._last_green_sha[:12])
            self._last_sha = self._last_green_sha
        return ok, f"{out1}\n{out2}".strip()

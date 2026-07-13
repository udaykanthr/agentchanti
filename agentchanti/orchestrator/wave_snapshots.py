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
            if not ok:
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
            self._last_sha = self._head_sha()
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

        gitignore = os.path.join(self.root, ".gitignore")
        if not os.path.exists(gitignore):
            try:
                with open(gitignore, "w", encoding="utf-8") as f:
                    f.write(_DEFAULT_GITIGNORE)
            except OSError:
                pass

        try:
            with open(self._marker_path(), "w", encoding="utf-8") as f:
                f.write("snapshot repo created by agentchanti\n")
        except OSError:
            pass

        self.managed = True
        self._commit("agentchanti: baseline (pre-run)")
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
        """Snapshot the workdir after a green wave. Returns the commit sha."""
        if not self.managed:
            return None
        sha = self._commit(f"agentchanti: {label}")
        if sha:
            _logger.info("[Snapshots] %s -> %s", label, sha[:12])
        return sha

    def rollback_to_last(self) -> tuple[bool, str]:
        """Hard-restore the workdir to the last snapshot.

        Untracked files are removed too (``clean -fd``) — but ignored
        ones (venv/, node_modules/, .agentchanti/) survive, so the
        environment does not need rebuilding after a rollback.
        """
        if not self.managed or not self._last_sha:
            return False, "no snapshot available"
        ok1, out1 = self._git("reset", "--hard", self._last_sha)
        ok2, out2 = self._git("clean", "-fd")
        ok = ok1 and ok2
        if ok:
            _logger.warning("[Snapshots] Rolled back workdir to %s",
                            self._last_sha[:12])
        return ok, f"{out1}\n{out2}".strip()

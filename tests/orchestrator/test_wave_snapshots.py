"""Tests for per-wave snapshots + the monotonic gate ledger."""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.orchestrator.wave_snapshots import (
    GateLedger,
    ProjectSnapshots,
    get_gate_ledger,
)


class TestGateLedger(unittest.TestCase):

    def test_record_and_dedupe(self):
        ledger = GateLedger()
        ledger.record("pytest -q", "1.1")
        ledger.record("pytest -q", "2.1")  # same command — first label wins
        ledger.record("npm test", "3.1")
        self.assertEqual(ledger.gates(),
                         {"pytest -q": "1.1", "npm test": "3.1"})

    def test_empty_command_ignored(self):
        ledger = GateLedger()
        ledger.record("", "1.1")
        ledger.record(None, "1.2")
        self.assertEqual(ledger.gates(), {})

    def test_recheck_reports_only_regressions(self):
        ledger = GateLedger()
        ledger.record("good-cmd", "1.1")
        ledger.record("bad-cmd", "2.1")
        executor = MagicMock()
        executor.run_command.side_effect = lambda cmd, timeout=300: (
            (True, "ok") if cmd == "good-cmd" else (False, "FAILED (errors=1)"))
        regressions = ledger.recheck(executor)
        self.assertEqual(len(regressions), 1)
        cmd, label, out = regressions[0]
        self.assertEqual((cmd, label), ("bad-cmd", "2.1"))
        self.assertIn("FAILED", out)

    def test_reset(self):
        ledger = GateLedger()
        ledger.record("x", "1.1")
        ledger.reset()
        self.assertEqual(ledger.gates(), {})

    def test_module_singleton(self):
        get_gate_ledger().reset()
        get_gate_ledger().record("cmd", "1.1")
        self.assertIn("cmd", get_gate_ledger().gates())
        get_gate_ledger().reset()


def _git_available() -> bool:
    try:
        return subprocess.run(["git", "--version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


@unittest.skipUnless(_git_available(), "git not available")
class TestProjectSnapshots(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="wavesnap_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, rel, content):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_start_creates_managed_repo(self):
        self._write("a.txt", "one")
        snaps = ProjectSnapshots(self.root)
        self.assertTrue(snaps.start())
        self.assertTrue(snaps.managed)
        self.assertTrue(os.path.isdir(os.path.join(self.root, ".git")))
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, ".git", "agentchanti-managed")))
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, ".gitignore")))

    def test_commit_wave_and_rollback(self):
        self._write("a.txt", "original")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        sha = snaps.commit_wave("wave 1")
        self.assertIsNotNone(sha)
        # A regressing fix round: mutate a file, add a stray one
        self._write("a.txt", "broken by fix round")
        self._write("stray.txt", "should be cleaned")
        ok, _ = snaps.rollback_to_last()
        self.assertTrue(ok)
        with open(os.path.join(self.root, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "original")
        self.assertFalse(os.path.exists(os.path.join(self.root, "stray.txt")))

    def test_ignored_dirs_survive_rollback(self):
        self._write("a.txt", "v1")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        snaps.commit_wave("wave 1")
        # venv/ is gitignored — clean -fd must not delete it
        self._write("venv/pyvenv.cfg", "home = somewhere")
        self._write("a.txt", "v2")
        ok, _ = snaps.rollback_to_last()
        self.assertTrue(ok)
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, "venv", "pyvenv.cfg")))

    def test_no_new_commit_when_clean(self):
        self._write("a.txt", "x")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        sha1 = snaps.commit_wave("wave 1")
        sha2 = snaps.commit_wave("wave 2")  # nothing changed
        self.assertEqual(sha1, sha2)

    def test_disabled_inside_existing_repo(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        snaps = ProjectSnapshots(self.root)
        self.assertFalse(snaps.start())
        self.assertFalse(snaps.managed)
        self.assertIsNone(snaps.commit_wave("wave 1"))
        ok, msg = snaps.rollback_to_last()
        self.assertFalse(ok)
        self.assertIn("no snapshot", msg)

    def test_resume_reuses_managed_repo(self):
        self._write("a.txt", "x")
        first = ProjectSnapshots(self.root)
        first.start()
        first.commit_wave("wave 1")
        # New run in the same workdir — the marker makes it resume
        second = ProjectSnapshots(self.root)
        self.assertTrue(second.start())
        self.assertTrue(second.managed)
        self._write("a.txt", "regressed")
        ok, _ = second.rollback_to_last()
        self.assertTrue(ok)
        with open(os.path.join(self.root, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "x")

    def test_disabled_flag(self):
        snaps = ProjectSnapshots(self.root, enabled=False)
        self.assertFalse(snaps.start())
        self.assertFalse(os.path.isdir(os.path.join(self.root, ".git")))

    def test_existing_gitignore_not_clobbered(self):
        self._write(".gitignore", "custom-entry/\n")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        with open(os.path.join(self.root, ".gitignore"),
                  encoding="utf-8") as f:
            self.assertEqual(f.read(), "custom-entry/\n")


if __name__ == "__main__":
    unittest.main()

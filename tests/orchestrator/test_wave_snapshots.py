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

    def test_recheck_skips_harness_errors(self):
        """A gate that can no longer launch (missing interpreter / wrong
        cwd) is inconclusive — it must NOT be reported as a code
        regression (which would roll back good code)."""
        ledger = GateLedger()
        ledger.record("cd sub && venv\\Scripts\\python.exe -c \"import x\"",
                      "2.1")
        executor = MagicMock()
        executor.run_command.return_value = (
            False, "The system cannot find the path specified.")
        regressions = ledger.recheck(executor)
        self.assertEqual(regressions, [])

    def test_abnormal_exit_is_retried_then_treated_as_inconclusive(self):
        """A crashed gate produced no verdict, so it is not a regression.

        Observed: a green `python -m unittest -v` gate was re-run by the
        monotonic check, the interpreter fast-failed with 0xC0000409
        (3221226505) after printing ordinary test output, and the ledger
        read that as "the tests regressed" — rolling back a correct test
        file that had passed twice moments earlier.
        """
        from agentchanti.orchestrator.wave_snapshots import is_abnormal_exit
        self.assertTrue(is_abnormal_exit(3221226505))   # 0xC0000409
        self.assertTrue(is_abnormal_exit(3221225477))   # 0xC0000005
        self.assertTrue(is_abnormal_exit(-11))          # SIGSEGV
        self.assertFalse(is_abnormal_exit(1))           # ordinary failure
        self.assertFalse(is_abnormal_exit(0))
        self.assertFalse(is_abnormal_exit(None))

        ledger = GateLedger()
        ledger.record("python -m unittest -v", "4.1")
        executor = MagicMock()
        executor.run_command.return_value = (False, "ran 7 tests")
        executor.last_exit_code = 3221226505
        self.assertEqual(ledger.recheck(executor), [])
        # Retried once before believing the crash.
        self.assertEqual(executor.run_command.call_count, 2)

    def test_a_crash_that_passes_on_retry_is_clean(self):
        ledger = GateLedger()
        ledger.record("python -m unittest -v", "4.1")
        executor = MagicMock()
        outcomes = [(False, "died"), (True, "OK")]
        codes = [3221226505, 0]

        def _run(cmd, timeout=300):
            executor.last_exit_code = codes.pop(0)
            return outcomes.pop(0)

        executor.run_command.side_effect = _run
        self.assertEqual(ledger.recheck(executor), [])

    def test_a_real_failure_after_a_crash_still_regresses(self):
        """The retry must not swallow a genuine failure."""
        ledger = GateLedger()
        ledger.record("python -m unittest -v", "4.1")
        executor = MagicMock()
        outcomes = [(False, "died"), (False, "FAILED (errors=1)")]
        codes = [3221226505, 1]

        def _run(cmd, timeout=300):
            executor.last_exit_code = codes.pop(0)
            return outcomes.pop(0)

        executor.run_command.side_effect = _run
        regressions = ledger.recheck(executor)
        self.assertEqual(len(regressions), 1)
        self.assertIn("FAILED", regressions[0][2])

    def test_recheck_real_failure_still_regresses(self):
        """A genuine code failure (assertion / non-zero test exit) is still
        a regression even alongside the harness-error guard."""
        ledger = GateLedger()
        ledger.record("pytest -q", "3.1")
        executor = MagicMock()
        executor.run_command.return_value = (False, "1 failed, 0 passed")
        regressions = ledger.recheck(executor)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0][1], "3.1")

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


class TestEnforceMonotonicGates(unittest.TestCase):
    """The shared guard applied after every source-mutating stage."""

    def setUp(self):
        get_gate_ledger().reset()

    def tearDown(self):
        get_gate_ledger().reset()

    def _snapshots(self, managed=True):
        snaps = MagicMock()
        snaps.managed = managed
        snaps.rollback_to_last.return_value = (True, "")
        return snaps

    def test_clean_stage_is_committed_and_marked_green(self):
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("pytest -q", "4.1")
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        snaps = self._snapshots()

        self.assertTrue(_enforce_monotonic_gates(snaps, executor, "wave 1"))
        snaps.commit_wave.assert_called_once_with("wave 1")
        snaps.mark_green.assert_called_once()
        snaps.rollback_to_last.assert_not_called()

    def test_smoke_fix_that_breaks_a_gate_fails_and_rolls_back(self):
        """The exact shipped-broken case.

        The smoke-test repair changed `Player.update()`'s signature, which
        turned the already-green unittest gate red. Nothing rechecked it and
        the run reported success.
        """
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("python -m unittest discover -v", "4.1")
        executor = MagicMock()
        executor.run_command.return_value = (
            False, "TypeError: <lambda>() missing 1 required positional "
                   "argument: 'game_map'\nFAILED (errors=1)")
        snaps = self._snapshots()

        self.assertFalse(
            _enforce_monotonic_gates(snaps, executor, "smoke-test fixes"))
        snaps.rollback_to_last.assert_called_once()
        snaps.mark_green.assert_not_called()

    def test_unmanaged_repo_is_left_alone(self):
        """Inside the user's own git repo, never roll back or re-run gates."""
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("pytest -q", "4.1")
        executor = MagicMock()
        snaps = self._snapshots(managed=False)

        self.assertTrue(
            _enforce_monotonic_gates(snaps, executor, "smoke-test fixes"))
        executor.run_command.assert_not_called()
        snaps.rollback_to_last.assert_not_called()

    def test_successful_repair_keeps_the_fix_instead_of_rolling_back(self):
        """A stale test, not a bad fix.

        The smoke-test repair legitimately changed `Player.update()`; the
        test still stubbed the old signature. Rolling back would re-break a
        working app to satisfy the stub, so the repair round runs first and
        its success keeps the fix.
        """
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("python -m unittest discover -v", "4.1")
        executor = MagicMock()
        # Red before the repair, green after it.
        executor.run_command.side_effect = [
            (False, "FAILED (errors=1)"),
            (True, "OK"),
        ]
        snaps = self._snapshots()
        repair = MagicMock()

        self.assertTrue(_enforce_monotonic_gates(
            snaps, executor, "smoke-test fixes", repair=repair))
        repair.assert_called_once()
        snaps.rollback_to_last.assert_not_called()
        snaps.mark_green.assert_called_once()

    def test_failed_repair_still_rolls_back(self):
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("python -m unittest discover -v", "4.1")
        executor = MagicMock()
        executor.run_command.return_value = (False, "FAILED (errors=1)")
        snaps = self._snapshots()
        repair = MagicMock()

        self.assertFalse(_enforce_monotonic_gates(
            snaps, executor, "smoke-test fixes", repair=repair))
        repair.assert_called_once()
        snaps.rollback_to_last.assert_called_once()

    def test_repair_is_skipped_when_nothing_regressed(self):
        """The repair round costs LLM calls — never run it speculatively."""
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("pytest -q", "4.1")
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        snaps = self._snapshots()
        repair = MagicMock()

        self.assertTrue(_enforce_monotonic_gates(
            snaps, executor, "smoke-test fixes", repair=repair))
        repair.assert_not_called()

    def test_raising_repair_is_contained(self):
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("pytest -q", "4.1")
        executor = MagicMock()
        executor.run_command.return_value = (False, "FAILED")
        snaps = self._snapshots()

        def _boom():
            raise RuntimeError("fix loop exploded")

        self.assertFalse(_enforce_monotonic_gates(
            snaps, executor, "smoke-test fixes", repair=_boom))
        snaps.rollback_to_last.assert_called_once()

    def test_harness_error_is_not_treated_as_a_regression(self):
        """A gate that can no longer launch must not discard good code."""
        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        get_gate_ledger().record("venv\\Scripts\\python.exe -c \"import x\"",
                                 "2.1")
        executor = MagicMock()
        executor.run_command.return_value = (
            False, "The system cannot find the path specified.")
        snaps = self._snapshots()

        self.assertTrue(
            _enforce_monotonic_gates(snaps, executor, "smoke-test fixes"))
        snaps.rollback_to_last.assert_not_called()


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
        snaps.mark_green()   # wave 1's gates rechecked clean
        # A regressing fix round: mutate a file, add a stray one
        self._write("a.txt", "broken by fix round")
        self._write("stray.txt", "should be cleaned")
        ok, _ = snaps.rollback_to_last()
        self.assertTrue(ok)
        with open(os.path.join(self.root, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "original")
        self.assertFalse(os.path.exists(os.path.join(self.root, "stray.txt")))

    def test_unverified_wave_is_not_a_rollback_target(self):
        """A committed-but-not-green wave must never become the target.

        This is the bug that made rollback a silent no-op: waves are
        committed *before* their gates are rechecked, so HEAD was the very
        commit carrying the regression. `reset --hard HEAD` then restored
        the broken state and reported success while discarding nothing.
        """
        self._write("a.txt", "green")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        snaps.commit_wave("wave 1")
        snaps.mark_green()
        green_sha = snaps.last_green_sha()

        # Wave 2 introduces a regression and is committed before its gates
        # are rechecked — so it is never marked green.
        self._write("a.txt", "regressed by wave 2")
        wave2_sha = snaps.commit_wave("wave 2")
        self.assertNotEqual(wave2_sha, green_sha)
        self.assertEqual(snaps.last_green_sha(), green_sha)

        ok, _ = snaps.rollback_to_last()
        self.assertTrue(ok)
        with open(os.path.join(self.root, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "green")

    def test_rollback_without_any_green_wave_uses_baseline(self):
        """A regression in the very first wave falls back to the pre-run state."""
        self._write("a.txt", "pre-run")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        self._write("a.txt", "regressed by wave 1")
        self._write("new.py", "written by wave 1")
        snaps.commit_wave("wave 1")   # never marked green
        ok, _ = snaps.rollback_to_last()
        self.assertTrue(ok)
        with open(os.path.join(self.root, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "pre-run")
        self.assertFalse(os.path.exists(os.path.join(self.root, "new.py")))

    def test_rollback_is_repeatable(self):
        """After a rollback, HEAD tracking must not strand the green pointer."""
        self._write("a.txt", "green")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        snaps.commit_wave("wave 1")
        snaps.mark_green()
        self._write("a.txt", "bad")
        snaps.commit_wave("wave 2")
        self.assertTrue(snaps.rollback_to_last()[0])
        self._write("a.txt", "bad again")
        self.assertTrue(snaps.rollback_to_last()[0])
        with open(os.path.join(self.root, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "green")

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

    def test_existing_gitignore_preserved_and_augmented(self):
        # A user-authored .gitignore must be preserved (not clobbered), but the
        # default rules — critically `.agentchanti/` — must be appended so the
        # tool's volatile state never gets tracked and blocks rollback.
        self._write(".gitignore", "custom-entry/\n")
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        with open(os.path.join(self.root, ".gitignore"),
                  encoding="utf-8") as f:
            content = f.read()
        self.assertIn("custom-entry/", content)   # preserved
        self.assertIn(".agentchanti/", content)   # augmented
        self.assertIn("venv/", content)

    def test_gitignore_augment_is_idempotent(self):
        # Re-running start() on the same repo must not duplicate default rules.
        snaps = ProjectSnapshots(self.root)
        snaps.start()
        again = ProjectSnapshots(self.root)
        again.start()
        with open(os.path.join(self.root, ".gitignore"),
                  encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content.count(".agentchanti/"), 1)


if __name__ == "__main__":
    unittest.main()


class TestCrashDiagnostics(unittest.TestCase):
    """A bare exit code tells nobody anything.

    Runs died with "exit code 3221226505" repeatedly across sessions and
    it read as a mystery, when it is a native crash inside the child
    interpreter that the pipeline already retries around.
    """

    def setUp(self):
        from agentchanti.orchestrator.wave_snapshots import reset_crash_hint
        reset_crash_hint()

    def test_names_the_codes_seen_in_real_runs(self):
        from agentchanti.orchestrator.wave_snapshots import (
            describe_abnormal_exit,
        )
        self.assertIn("0xC0000409", describe_abnormal_exit(3221226505))
        self.assertIn("fast-fail", describe_abnormal_exit(3221226505))
        self.assertIn("0xC0000005", describe_abnormal_exit(3221225477))
        self.assertIn("ACCESS_VIOLATION", describe_abnormal_exit(3221225477))

    def test_an_unknown_native_code_still_renders_as_hex(self):
        from agentchanti.orchestrator.wave_snapshots import (
            describe_abnormal_exit,
        )
        out = describe_abnormal_exit(0xC0000123)
        self.assertIn("0xC0000123", out)
        self.assertIn("native crash", out)

    def test_posix_signals(self):
        from agentchanti.orchestrator.wave_snapshots import (
            describe_abnormal_exit,
        )
        self.assertIn("signal 11", describe_abnormal_exit(-11))

    def test_ordinary_exits_describe_nothing(self):
        from agentchanti.orchestrator.wave_snapshots import (
            describe_abnormal_exit,
        )
        for code in (0, 1, 5, None, True, "1"):
            with self.subTest(code=code):
                self.assertEqual(describe_abnormal_exit(code), "")

    def test_the_dump_instructions_appear_once_per_run(self):
        import os

        from agentchanti.orchestrator.wave_snapshots import (
            log_crash_diagnostics,
        )
        with self.assertLogs("agentchanti", level="WARNING") as first:
            log_crash_diagnostics(3221226505, "python -m unittest -v")
        joined = "\n".join(first.output)
        self.assertIn("0xC0000409", joined)
        self.assertIn("python -m unittest -v", joined)
        if os.name == "nt":
            self.assertIn("LocalDumps", joined)

        with self.assertLogs("agentchanti", level="WARNING") as second:
            log_crash_diagnostics(3221226505, "python -m unittest -v")
        self.assertNotIn("LocalDumps", "\n".join(second.output),
                         "the dump instructions must not repeat every crash")

    def test_a_clean_exit_logs_nothing(self):
        from agentchanti.orchestrator.wave_snapshots import (
            log_crash_diagnostics,
        )
        import logging
        with self.assertNoLogs("agentchanti", level="WARNING"):
            log_crash_diagnostics(1, "python -m unittest -v")

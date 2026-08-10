"""A command that CRASHED reported no verdict, so it must not read as one.

On Windows a pygame/SDL suite fast-fails (0xC0000409) or access-violates
(0xC0000005) in a substantial fraction of invocations, printing ordinary
test output first. Believing that exit status turns a green suite into "the
tests regressed" and rolls back correct code — the failure this project has
already been bitten by more than once.

The pipeline hand-rolled the same detect-log-retry block at four call sites
(GateLedger.recheck, the BulkTest plan gate, the BulkTest runner,
AgentLoop._run_verify — whose own comment records it as having been
*missing* there until someone hit it), which left every site nobody thought
of unprotected. `Executor.run_command` now owns the behaviour so new call
sites inherit it. These tests pin both halves of the contract: a crash is
retried, and an ordinary failure is NOT — masking a real red suite would be
a far worse bug than the one being fixed.
"""

import unittest
from unittest.mock import patch

from agentchanti.executor import CRASHED_MARKER, Executor

FAST_FAIL = 3221226505      # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN
ACCESS_VIOLATION = 3221225477  # 0xC0000005
SIGSEGV = -11               # POSIX: killed by a signal


def _scripted(codes):
    """A stand-in for `_run_command_once` that replays scripted exit codes.

    Deliberately a plain function, not a callable object: `patch.object`
    stores it as a class attribute, and only a function is a descriptor, so
    only a function gets bound and receives the Executor as `self` — which
    the retry logic needs in order to publish `last_exit_code`.
    """
    remaining = list(codes)
    calls = []

    def _run_once(self, cmd, env=None, timeout=120, background=False,
                  cwd=None):
        code = remaining.pop(0)
        self.last_exit_code = code
        calls.append(cmd)
        return code == 0, f"output(code={code})"

    return _run_once, calls


def _run(codes, **kwargs):
    """Run with scripted exit codes; returns (ok, output, attempts)."""
    fake, calls = _scripted(codes)
    with patch.object(Executor, "_run_command_once", fake):
        ok, out = Executor().run_command("python -m unittest -v", **kwargs)
    return ok, out, len(calls)


class CrashIsRetriedTest(unittest.TestCase):
    def test_fast_fail_then_pass_is_a_pass(self):
        ok, _, attempts = _run([FAST_FAIL, 0])
        self.assertTrue(ok)
        self.assertEqual(attempts, 2)

    def test_access_violation_is_retried(self):
        self.assertEqual(_run([ACCESS_VIOLATION, 0])[2], 2)

    def test_posix_signal_death_is_retried(self):
        self.assertEqual(_run([SIGSEGV, 0])[2], 2)

    def test_crash_then_genuine_failure_reports_the_failure(self):
        """The retry decides nothing on its own — a real red result stands."""
        ok, _, attempts = _run([FAST_FAIL, 1])
        self.assertFalse(ok)
        self.assertEqual(attempts, 2)

    def test_crashing_twice_is_flagged_inconclusive_but_still_fails(self):
        ok, out, attempts = _run([FAST_FAIL, FAST_FAIL])
        self.assertFalse(ok, "suppressing it would invent a pass")
        self.assertEqual(attempts, 2, "retry is bounded at one")
        self.assertIn(CRASHED_MARKER, out)


class RealVerdictsAreNeverRetriedTest(unittest.TestCase):
    """The safety property: genuine results must reach the caller untouched."""

    def test_success_runs_once(self):
        ok, _, attempts = _run([0])
        self.assertTrue(ok)
        self.assertEqual(attempts, 1)

    def test_ordinary_failure_runs_once_and_is_not_masked(self):
        ok, out, attempts = _run([1])
        self.assertFalse(ok)
        self.assertEqual(attempts, 1)
        self.assertNotIn(CRASHED_MARKER, out)

    def test_assertion_failure_exit_code_is_a_verdict(self):
        self.assertEqual(_run([2])[2], 1)


class OptOutTest(unittest.TestCase):
    def test_retry_can_be_disabled_for_side_effecting_commands(self):
        ok, _, attempts = _run([FAST_FAIL, 0], retry_on_crash=False)
        self.assertFalse(ok)
        self.assertEqual(attempts, 1)

    def test_background_commands_are_never_retried(self):
        """They have not finished, so there is no exit status to judge."""
        self.assertEqual(_run([FAST_FAIL, 0], background=True)[2], 1)


class LastExitCodeTest(unittest.TestCase):
    """Consumers read `last_exit_code`; it must describe the FINAL attempt."""

    def test_reflects_the_successful_retry(self):
        fake, _ = _scripted([FAST_FAIL, 0])
        with patch.object(Executor, "_run_command_once", fake):
            ex = Executor()
            ex.run_command("python -m unittest -v")
            self.assertEqual(ex.last_exit_code, 0)

    def test_reflects_the_second_crash(self):
        fake, _ = _scripted([FAST_FAIL, ACCESS_VIOLATION])
        with patch.object(Executor, "_run_command_once", fake):
            ex = Executor()
            ex.run_command("python -m unittest -v")
            self.assertEqual(ex.last_exit_code, ACCESS_VIOLATION)


if __name__ == "__main__":
    unittest.main()

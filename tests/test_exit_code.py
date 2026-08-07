"""Tests that a halted pipeline exits non-zero.

`_main_impl`'s failure branch logged "Pipeline failed", wrote the HTML
report, and fell through returning None, so the process exited 0. Observed
2026-08-07: a classic-mode benchmark run halted at step 11 of 12 after three
failed diagnosis attempts, having never written its tests, and still returned
`EXIT=0`. Anything reading `$?` — CI, a `&&` chain, a benchmark harness —
records that as a pass.
"""

import unittest
from unittest import mock

from agentchanti.orchestrator import cli


class MainExitCodeTest(unittest.TestCase):
    def _run_main_with(self, impl_result):
        """Call cli.main() with _main_impl stubbed; return the exit code."""
        with mock.patch.object(cli, "_main_impl", return_value=impl_result), \
                mock.patch.object(cli, "install_sigint_handler"), \
                mock.patch.object(cli, "_arm_faulthandler"), \
                mock.patch.object(cli, "install_crash_diagnostics"):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        return caught.exception.code

    # ── the regression ────────────────────────────────────────────────
    def test_failed_pipeline_exits_non_zero(self):
        self.assertEqual(self._run_main_with(1), 1)

    # ── success and the early-return paths stay 0 ─────────────────────
    def test_successful_pipeline_exits_zero(self):
        self.assertEqual(self._run_main_with(0), 0)

    def test_early_return_none_exits_zero(self):
        """--version, the kb subcommand and an aborted prompt return None."""
        self.assertEqual(self._run_main_with(None), 0)

    # ── the pre-existing exit codes are unchanged ─────────────────────
    def test_keyboard_interrupt_still_exits_130(self):
        with mock.patch.object(cli, "_main_impl",
                               side_effect=KeyboardInterrupt), \
                mock.patch.object(cli, "install_sigint_handler"), \
                mock.patch.object(cli, "_arm_faulthandler"), \
                mock.patch.object(cli, "install_crash_diagnostics"):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        self.assertEqual(caught.exception.code, 130)

    def test_unhandled_exception_still_propagates(self):
        """Re-raised, so the process exits non-zero with a traceback."""
        with mock.patch.object(cli, "_main_impl",
                               side_effect=RuntimeError("boom")), \
                mock.patch.object(cli, "install_sigint_handler"), \
                mock.patch.object(cli, "_arm_faulthandler"), \
                mock.patch.object(cli, "install_crash_diagnostics"):
            with self.assertRaises(RuntimeError):
                cli.main()


if __name__ == "__main__":
    unittest.main()

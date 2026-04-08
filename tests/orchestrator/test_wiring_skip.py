"""Regression tests for the post-pipeline wiring-verification skip gate.

Wiring verification is an expensive LLM call (60-90s) that checks for
cross-file integration issues. After a successful bulk test run, every
failure mode it looks for would already have crashed the test runner —
running it again is pure waste. ``should_run_wiring_verification`` is the
single source of truth for that gate; these tests pin its behaviour so
the optimisation can never silently regress.

See: bugfix branch — pipeline.py:should_run_wiring_verification.
"""
import unittest
from unittest.mock import MagicMock

from multi_agent_coder.orchestrator.pipeline import (
    should_run_wiring_verification,
)


def _make_memory(files: dict[str, str]):
    """Build a minimal mock memory whose ``all_files`` returns *files*."""
    mem = MagicMock()
    mem.all_files.return_value = files
    return mem


class TestShouldRunWiringVerification(unittest.TestCase):
    """Pin the boolean truth table for the wiring-skip gate."""

    # ── Skip cases ─────────────────────────────────────────────────────────

    def test_skip_when_bulk_tests_existed_and_passed(self):
        """The headline optimisation: green bulk tests prove wiring."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/__tests__/App.test.jsx": "...",
        })
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=True,
                wiring_enabled=True,
            )
        )

    def test_skip_when_wiring_disabled_in_config(self):
        """Config opt-out wins regardless of bulk-test state."""
        memory = _make_memory({"src/App.jsx": "..."})
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=False,
            )
        )

    def test_skip_when_pipeline_failed(self):
        """No point verifying wiring on a failed pipeline."""
        memory = _make_memory({"src/App.jsx": "..."})
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=False,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )

    # ── Run cases ──────────────────────────────────────────────────────────

    def test_run_when_no_test_files_exist(self):
        """No tests = wiring is the only integration check we have."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/main.jsx": "...",
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,  # bulk test didn't run
                wiring_enabled=True,
            )
        )

    def test_run_when_bulk_tests_existed_but_failed(self):
        """Failed bulk test does not prove wiring is correct."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/__tests__/App.test.jsx": "...",
        })
        # Note: in practice pipeline_success would also be False here, but
        # the helper handles that case independently — verify both axes.
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )

    def test_run_when_only_metadata_files_present(self):
        """Underscore-prefixed memory keys (e.g. _cmd_output/) are not tests."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "_cmd_output/step_1.txt": "...",
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )

    def test_run_when_only_non_source_files_in_test_dir(self):
        """A snapshot.json inside __tests__/ is not a real test file."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/__tests__/snapshot.json": "...",
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )


class TestWiringSkipRealLogScenario(unittest.TestCase):
    """Reproduce the exact scenario from the bug report."""

    def test_user_test_fix_run_skips_wiring(self):
        """Task: 'fix all test cases' — bulk tests passed, wiring should skip.

        From the user's logs:
          02:13:01 [INFO] [BulkTest] All tests passed on first run.
          02:13:01 [INFO] [WiringVerification] Starting cross-file wiring check
        """
        memory = _make_memory({
            "myapp/src/App.jsx": "...",
            "myapp/src/main.jsx": "...",
            "myapp/src/components/Header.jsx": "...",
            "myapp/src/__tests__/App.test.jsx": "...",
            "myapp/src/__tests__/main.test.jsx": "...",
            "myapp/src/components/__tests__/Header.test.jsx": "...",
            "myapp/src/components/__tests__/HeroBanner.test.jsx": "...",
        })
        # After the test fix landed, bulk test re-ran green:
        result = should_run_wiring_verification(
            memory,
            pipeline_success=True,
            bulk_test_verif_ok=True,
            wiring_enabled=True,  # default config
        )
        self.assertFalse(
            result,
            "Wiring verification must be skipped after a green bulk test run",
        )


if __name__ == '__main__':
    unittest.main()

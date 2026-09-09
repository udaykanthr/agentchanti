"""An empty suite is a verdict in NEITHER direction.

Measured 2026-09-10, a pinball bug-fix run. Every gate passed, the ghost
reported 18/18 expectations holding with 0 violated, pytest was green, the
smoke test launched the app, and the seeded contract passed — and the run
exited non-zero with::

    [Evidence] pre-existing suite(s) FAILED — tests/__init__.py:
               Ran 0 tests in 0.000s |  | NO TESTS RAN
    Pipeline failed: require_independent_evidence is set and nothing
                     outside this run's own output verified it

`tests/__init__.py` was a 71-byte docstring-only package marker. Three
separate defects had to line up:

1. `_is_test_file` matched it, because `_TEST_DIR_RE` matches every ``.py``
   under ``tests/`` and nothing excluded the package marker.
2. `run_pre_existing_tests` read the exit code before asking whether
   anything had run. unittest on Python 3.12+ exits **5** on an empty
   collection, so "nothing was checked" was recorded as "the code failed".
   The same blindness in the other direction is worse and was also
   present: most runners exit **0** on an empty collection, which was
   silently counted as a pass and established independent evidence.
3. The genuine pass was then discarded anyway, because the inconclusive
   branch was ordered above it.

The codebase already drew this distinction twice — `verify_passed()`
refuses to call an empty gate run a pass, and `_green_suites_contradicting`
refuses to let an empty suite overrule a failing gate — but never here.
"""
import unittest

from agentchanti.orchestrator.evidence import (
    EMPTY_SUITE_RE, empty_suite_reason, run_pre_existing_tests,
)
from agentchanti.orchestrator.pipeline import _is_test_file

# Real output from the incident: unittest on Python 3.12+, exit code 5.
EMPTY_UNITTEST = "Ran 0 tests in 0.000s\n\nNO TESTS RAN"
PASSING_UNITTEST = "Ran 4 tests in 0.012s\n\nOK"
FAILING_UNITTEST = (
    "FAIL: test_launch (tests.test_pinball_game.PinballTest)\n"
    "AssertionError: 640.0 not less than 640.0\n\n"
    "Ran 4 tests in 0.012s\n\nFAILED (failures=1)"
)


class ScriptedExecutor:
    """Returns a canned ``(ok, output)`` per test file, recording calls."""

    def __init__(self, results):
        self._results = results          # {relpath: (ok, output)}
        self.commands = []

    def run_command(self, cmd, timeout=300, **kw):
        self.commands.append(cmd)
        for rel, result in self._results.items():
            if rel.replace("/", "\\") in cmd or rel in cmd:
                return result
        raise AssertionError("unexpected command: %s" % cmd)


class PackageMarkerIsNotASuite(unittest.TestCase):

    def test_tests_init_is_not_a_test_file(self):
        """The trigger. No runner collects a package marker."""
        self.assertFalse(_is_test_file("tests/__init__.py"))
        self.assertFalse(_is_test_file("tests/helpers/__init__.py"))

    def test_real_test_files_are_unaffected(self):
        self.assertTrue(_is_test_file("tests/test_pinball_game.py"))
        self.assertTrue(_is_test_file("test_acceptance_contract.py"))
        self.assertTrue(_is_test_file("src/__tests__/App.test.jsx"))


class EmptyCollectionIsNotAVerdict(unittest.TestCase):

    def test_every_runner_spelling_is_recognised(self):
        for output in (EMPTY_UNITTEST,
                       "collected 0 items\n\nno tests ran in 0.01s",
                       "No test files found, exiting with code 1",
                       "No tests found, exiting with code 1",
                       "ok  \texample/pkg\t[no test files]"):
            with self.subTest(output=output.splitlines()[0]):
                self.assertTrue(EMPTY_SUITE_RE.search(output))
                self.assertIsNotNone(empty_suite_reason(output))

    def test_a_real_result_is_never_called_empty(self):
        """Or this becomes a machine for explaining away disagreement."""
        self.assertIsNone(empty_suite_reason(PASSING_UNITTEST))
        self.assertIsNone(empty_suite_reason(FAILING_UNITTEST))


class RunPreExistingTests(unittest.TestCase):

    def test_the_incident_no_longer_fails_the_run(self):
        """The exact survivor set from 2026-09-10."""
        executor = ScriptedExecutor({
            "test_acceptance_contract.py": (True, PASSING_UNITTEST),
            "tests/__init__.py": (False, EMPTY_UNITTEST),
        })

        passed, detail = run_pre_existing_tests(
            executor, ".", ["test_acceptance_contract.py", "tests/__init__.py"])

        self.assertIs(passed, True,
                      "an empty package marker convicted working code")
        self.assertIn("passed", detail)
        self.assertIn("no verdict", detail,
                      "the empty file should be reported, not hidden")

    def test_an_empty_suite_exiting_zero_is_not_a_pass(self):
        """The other direction, and the more dangerous one.

        Most runners exit 0 on an empty collection, so reading the status
        first manufactures independent evidence out of a file that ran
        nothing at all.
        """
        executor = ScriptedExecutor({
            "tests/__init__.py": (True, EMPTY_UNITTEST),
        })

        passed, detail = run_pre_existing_tests(
            executor, ".", ["tests/__init__.py"])

        self.assertIsNone(passed, "an empty suite established evidence")
        self.assertIn("collected no tests", detail)

    def test_a_real_failure_still_convicts(self):
        """The protection this layer exists for must not be softened."""
        executor = ScriptedExecutor({
            "tests/test_pinball_game.py": (False, FAILING_UNITTEST),
        })

        passed, detail = run_pre_existing_tests(
            executor, ".", ["tests/test_pinball_game.py"])

        self.assertIs(passed, False)
        # The detail is the last three lines of the run, which is where a
        # runner puts its tally — see `truncate_middle`'s reasoning about
        # conclusions living at the end of test output.
        self.assertIn("FAILED (failures=1)", detail)

    def test_a_real_failure_outranks_a_sibling_pass(self):
        """A pass does not excuse a disagreement from another instrument."""
        executor = ScriptedExecutor({
            "test_acceptance_contract.py": (True, PASSING_UNITTEST),
            "tests/test_pinball_game.py": (False, FAILING_UNITTEST),
        })

        passed, _detail = run_pre_existing_tests(
            executor, ".",
            ["test_acceptance_contract.py", "tests/test_pinball_game.py"])

        self.assertIs(passed, False)

    def test_all_empty_is_could_not_run_not_a_pass(self):
        executor = ScriptedExecutor({
            "tests/__init__.py": (False, EMPTY_UNITTEST),
            "tests/helpers/__init__.py": (True, EMPTY_UNITTEST),
        })

        passed, detail = run_pre_existing_tests(
            executor, ".", ["tests/__init__.py", "tests/helpers/__init__.py"])

        self.assertIsNone(passed)
        self.assertIn("collected no tests", detail)


class OneDefinitionOfEmpty(unittest.TestCase):

    def test_cli_and_evidence_cannot_drift_apart(self):
        """Both callers ask the same question about the same file."""
        from agentchanti.orchestrator import cli
        self.assertIs(cli._EMPTY_SUITE_RE, EMPTY_SUITE_RE)


if __name__ == "__main__":
    unittest.main()

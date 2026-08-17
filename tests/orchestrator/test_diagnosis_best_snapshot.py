"""The diagnosis loop must ship the best state it reached, not the newest.

Both measured incidents came from the same blindness: `_error_signature`
says whether two failures are *different*, never which one is worse.

  2026-08-16  Inequality KEPT two fixes that took a suite from 4 failures
              to 19 errors to 39 errors, and the final restore shipped the
              39-error state — a Game that could not be constructed.
  2026-08-17  Equality DISCARDED a fix that took a suite from 9 errors + 1
              failure down to 1 failure, and the run shipped the 9-error
              file. Every wave commit in the project's own git was clean;
              only the working tree was broken.

`_diagnosis_score` gives the comparison a direction by counting failing
tests, and falls back to `None` — never to a guess — for output no test
runner parser can read.
"""

import pytest

from agentchanti.orchestrator.pipeline import _diagnosis_score
from agentchanti.orchestrator.step_handlers import _parse_test_counts


def unittest_output(total, failures=0, errors=0):
    """Realistic `python -m unittest` output, wrapped the way the TEST step
    handler wraps it (`step_handlers.py`, "Tests partially failing:")."""
    body = "\n".join(f"test_{i} (t.T.test_{i}) ... ok" for i in range(total))
    tail = f"\n\n{'-' * 70}\nRan {total} tests in 0.074s\n\n"
    if failures or errors:
        parts = []
        if failures:
            parts.append(f"failures={failures}")
        if errors:
            parts.append(f"errors={errors}")
        tail += f"FAILED ({', '.join(parts)})"
    else:
        tail += "OK"
    return ("Tests partially failing: 0/1 test files passed. Failed: t.py\n"
            "Last output:\n" + body + tail)


# ── the parser gap that made scoring impossible ──────────────────────────

def test_unittest_counts_are_parsed():
    """Before this branch existed every failing unittest run collapsed to
    the (0, 1) fallback, so 10 failures and 1 failure scored identically."""
    assert _parse_test_counts(unittest_output(8, failures=1, errors=9))[:2] == (0, 8)
    assert _parse_test_counts(unittest_output(8, failures=1))[:2] == (7, 8)
    assert _parse_test_counts(unittest_output(8))[:2] == (8, 8)


def test_subtest_failures_never_produce_a_negative_pass_count():
    """subTest can report more failures than there are test methods."""
    passed, total, _ = _parse_test_counts(unittest_output(2, errors=9))
    assert passed == 0
    assert total == 2


@pytest.mark.parametrize("output,expected", [
    ("collected 5 items\n3 passed, 2 failed", (3, 5)),
    ("Tests: 5 failed, 3 passed, 8 total", (3, 8)),
    ("--- PASS: TestA\n--- FAIL: TestB", (1, 2)),
])
def test_other_runners_still_parse(output, expected):
    assert _parse_test_counts(output)[:2] == expected


# ── the score itself ─────────────────────────────────────────────────────

def test_score_counts_failing_tests_lower_is_better():
    assert _diagnosis_score(unittest_output(8, failures=1, errors=9)) == 8
    assert _diagnosis_score(unittest_output(8, failures=1)) == 1
    assert _diagnosis_score(unittest_output(8)) == 0


@pytest.mark.parametrize("err", [
    "",
    None,
    "Traceback (most recent call last):\nTypeError: 'bool' object is not callable",
    "ModuleNotFoundError: No module named 'pygame'",
])
def test_unscorable_errors_return_none(err):
    """A CODE step's gate failure is a bare traceback with no counts in it.
    An unknown score must never read as an improvement."""
    assert _diagnosis_score(err) is None


# ── the two incidents ────────────────────────────────────────────────────

def test_2026_08_17_correct_fix_is_recognised_as_progress():
    """9 errors + 1 failure -> 1 failure is progress, and was discarded."""
    before = _diagnosis_score(unittest_output(8, failures=1, errors=9))
    after = _diagnosis_score(unittest_output(8, failures=1))
    assert after < before


def test_2026_08_16_compounding_regressions_are_rejected():
    """4 failures -> 19 errors -> 39 errors is not progress, and was kept."""
    baseline = _diagnosis_score(unittest_output(40, failures=4))
    worse = _diagnosis_score(unittest_output(40, failures=1, errors=19))
    worst = _diagnosis_score(unittest_output(40, failures=1, errors=39))

    assert not worse < baseline
    assert not worst < baseline
    # The state the loop should be left holding is the original one.
    assert min(baseline, worse, worst) == baseline

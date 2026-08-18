"""Whose evidence is the run claiming success on?

Measured over six benchmark runs of one task: both agent-loop failures
printed "All tasks completed successfully!" over a game whose player
could not move at 1/60, because the suite judging them was written in
the same run. Both classic-path failures failed their own tests and
reported failure. The pipeline was not lying — nothing in it was in a
position to notice that all of its evidence was self-authored.
"""

import os

from agentchanti.orchestrator.evidence import (
    INDEPENDENT_ACCEPTANCE, INDEPENDENT_PRE_EXISTING, NO_TESTS,
    SELF_AUTHORED, Evidence, classify, run_acceptance_commands,
    snapshot_test_files, surviving_pre_existing_tests,
)


def _write(root, rel, text):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ── the snapshot ─────────────────────────────────────────────────────


def test_snapshot_finds_test_files_by_convention(tmp_path):
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): pass\n")
    _write(root, "tests/test_more.py", "def test_x(): pass\n")
    _write(root, "src/app.spec.ts", "it('works', () => {})\n")
    _write(root, "calc.py", "def add(a, b): return a + b\n")
    snap = snapshot_test_files(root)
    assert set(snap) == {"test_calc.py", "tests/test_more.py",
                         "src/app.spec.ts"}


def test_snapshot_skips_vendored_and_internal_dirs(tmp_path):
    """A dependency's own tests are not this project's evidence."""
    root = str(tmp_path)
    _write(root, "node_modules/lib/test_vendor.py", "def test_v(): pass\n")
    _write(root, "venv/Lib/test_stdlib.py", "def test_s(): pass\n")
    _write(root, ".agentchanti/test_internal.py", "def test_i(): pass\n")
    assert snapshot_test_files(root) == {}


def test_a_modified_test_file_does_not_survive(tmp_path):
    """The oldest cheat: 'fix' the failing test by editing the assertion."""
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): assert add(2, 2) == 4\n")
    snap = snapshot_test_files(root)
    _write(root, "test_calc.py", "def test_add(): assert True\n")
    assert surviving_pre_existing_tests(root, snap) == []


def test_an_untouched_test_file_survives(tmp_path):
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): assert add(2, 2) == 4\n")
    snap = snapshot_test_files(root)
    _write(root, "calc.py", "def add(a, b): return a + b\n")   # source only
    assert surviving_pre_existing_tests(root, snap) == ["test_calc.py"]


def test_a_deleted_test_file_does_not_survive(tmp_path):
    root = str(tmp_path)
    path = _write(root, "test_calc.py", "def test_add(): pass\n")
    snap = snapshot_test_files(root)
    os.remove(path)
    assert surviving_pre_existing_tests(root, snap) == []


# ── the verdict ──────────────────────────────────────────────────────


def test_greenfield_run_is_completed_but_unverified(tmp_path):
    """The measured case: no pre-existing suite, so the only tests that
    passed are the ones the run wrote."""
    root = str(tmp_path)
    _write(root, "test_game.py", "def test_x(): pass\n")     # written by the run
    verdict = classify(root, {}, tests_ran=True)
    assert not verdict.independent
    assert verdict.kind == SELF_AUTHORED
    assert "marked its own homework" in verdict.detail
    assert "successfully" not in verdict.headline


def test_untouched_seeded_suite_is_independent(tmp_path):
    """Untouched AND actually run and passed.

    `survivors_passed` is what makes this a measurement. Before it
    existed the verdict rested on `tests_ran` — a flag about the
    pipeline's OWN tests — and two measured runs (2026-08-17 and
    2026-08-18) reported a seeded contract as having passed while every
    test in it errored.
    """
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): assert add(2, 2) == 4\n")
    snap = snapshot_test_files(root)
    verdict = classify(root, snap, tests_ran=True, survivors_passed=True)
    assert verdict.independent
    assert verdict.kind == INDEPENDENT_PRE_EXISTING
    assert "test_calc.py" in verdict.detail


def test_a_surviving_suite_nobody_ran_is_not_evidence(tmp_path):
    """The measured defect, pinned: surviving is not the same as passing."""
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): assert add(2, 2) == 4\n")
    snap = snapshot_test_files(root)
    verdict = classify(root, snap, tests_ran=True, survivors_passed=None)
    assert not verdict.independent
    assert "none could be run" in verdict.detail


def test_a_failing_pre_existing_suite_is_reported_as_such(tmp_path):
    """The one instrument the run did not author disagreeing with it is
    a stronger statement than a generic "unverified"."""
    from agentchanti.orchestrator.evidence import PRE_EXISTING_FAILED

    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): assert add(2, 2) == 4\n")
    snap = snapshot_test_files(root)
    verdict = classify(root, snap, tests_ran=True, survivors_passed=False,
                       survivors_detail="test_calc.py: AssertionError")
    assert not verdict.independent
    assert verdict.kind == PRE_EXISTING_FAILED
    assert "AssertionError" in verdict.detail


def test_running_survivors_reports_pass_fail_and_unknown(tmp_path):
    from agentchanti.orchestrator.evidence import run_pre_existing_tests

    class _Exec:
        def __init__(self, ok):
            self.ok = ok

        def run_command(self, cmd, timeout=None):
            return self.ok, "Ran 1 test" if self.ok else "FAILED (errors=1)"

    root = str(tmp_path)
    ok, detail = run_pre_existing_tests(_Exec(True), root, ["test_a.py"])
    assert ok is True and "passed" in detail

    ok, detail = run_pre_existing_tests(_Exec(False), root, ["test_a.py"])
    assert ok is False and "test_a.py" in detail

    # No runnable file, and no executor: both mean "cannot answer",
    # which must never read as a pass.
    assert run_pre_existing_tests(_Exec(True), root, [])[0] is None
    assert run_pre_existing_tests(None, root, ["test_a.py"])[0] is None


def test_rewriting_the_seeded_suite_forfeits_independence(tmp_path):
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): assert add(2, 2) == 4\n")
    snap = snapshot_test_files(root)
    _write(root, "test_calc.py", "def test_add(): assert True\n")
    verdict = classify(root, snap, tests_ran=True)
    assert not verdict.independent
    assert "every one of the 1 pre-existing test file(s) was modified" \
        in verdict.detail


def test_pre_existing_suite_that_never_ran_is_not_evidence(tmp_path):
    """A test file nobody executed proves nothing."""
    root = str(tmp_path)
    _write(root, "test_calc.py", "def test_add(): pass\n")
    snap = snapshot_test_files(root)
    verdict = classify(root, snap, tests_ran=False)
    assert not verdict.independent


def test_no_tests_at_all_says_so(tmp_path):
    verdict = classify(str(tmp_path), {}, tests_ran=False)
    assert not verdict.independent
    assert verdict.kind == NO_TESTS


def test_user_acceptance_commands_are_independent(tmp_path):
    """The one instrument the model neither wrote nor can edit."""
    verdict = classify(str(tmp_path), {}, tests_ran=True,
                       acceptance_passed=True,
                       acceptance_cmds=["python probe.py"])
    assert verdict.independent
    assert verdict.kind == INDEPENDENT_ACCEPTANCE


def test_acceptance_absence_is_not_acceptance_failure(tmp_path):
    """`None` means "none supplied" and must never read as a failure."""
    verdict = classify(str(tmp_path), {}, tests_ran=True,
                       acceptance_passed=None, acceptance_cmds=[])
    assert verdict.kind == SELF_AUTHORED


# ── acceptance commands ──────────────────────────────────────────────


class _FakeExecutor:
    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.ran = []

    def run_command(self, cmd, **kw):
        self.ran.append(cmd)
        return self.outcomes.get(cmd, (True, ""))


def test_no_acceptance_commands_returns_none():
    passed, failures = run_acceptance_commands(_FakeExecutor({}), [])
    assert passed is None and failures == []


def test_all_acceptance_commands_must_pass():
    ex = _FakeExecutor({"b": (False, "exit 1: boom")})
    passed, failures = run_acceptance_commands(ex, ["a", "b", "c"])
    assert passed is False
    assert len(failures) == 1 and "boom" in failures[0]
    assert ex.ran == ["a", "b", "c"]      # every one runs, for the report


def test_a_raising_command_is_a_failure_not_a_crash():
    class Boom:
        def run_command(self, cmd, **kw):
            raise OSError("no shell")

    passed, failures = run_acceptance_commands(Boom(), ["x"])
    assert passed is False and "no shell" in failures[0]


# ── the headline ─────────────────────────────────────────────────────


def test_unverified_headline_does_not_claim_success():
    verdict = Evidence(False, SELF_AUTHORED, "…")
    assert "nothing independent verified" in verdict.headline
    assert "✓" not in verdict.headline


def test_verified_headline_is_the_original_claim():
    verdict = Evidence(True, INDEPENDENT_ACCEPTANCE, "…")
    assert verdict.headline == "All tasks completed successfully!"

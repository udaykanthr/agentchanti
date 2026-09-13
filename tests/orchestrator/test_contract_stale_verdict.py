"""A verdict measured before the artifact existed.

Measured 2026-09-12, a single-file pygame pinball run that shipped a working
game and exited 1. The seeded contract was executed **once** in the whole
run — at 22:20:05, after wave 1, which was a CMD step that created a venv.
`main.py` was written by wave 2 at 22:22:23. The contract's harness reaches
the project through a subprocess::

    completed = subprocess.run([sys.executable, "-c", harness], ...)
    # harness: runpy.run_path("main.py", run_name="__main__")
    self.assertEqual(completed.returncode, 0, ...)

so a project that does not exist yet is reported as `FAILED (failures=1)`
over a `FileNotFoundError` — an assertion about a return code, with no
import error anywhere. Three defects then lined up:

1. `_NOT_READY_RE` reads import errors, so the deferral never fired.
2. "it ran and judged" spent the repair budget, which retired the check —
   so the `final=True` call returned before executing anything.
3. `_LAST_RUN` still held the 22:20:05 result, and `cli.py` handed it to
   `run_pre_existing_tests(prerun=...)`.

The verdict therefore convicted a finished artifact on a measurement taken
two minutes and eighteen seconds before its only source file was written::

    22:22:30 the seeded contract failed (FAILED (failures=1) | pygame 2.6.1 ...)
    22:22:30 Pipeline failed: require_independent_evidence is set ...

Run by hand afterwards, the same bytes pass 5/5. The cached result was only
ever valid for the bytes of BOTH the instrument and the artifact; it tracked
the instrument alone.
"""
import os
import textwrap

import pytest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, last_contract_run, reset_contract_repairs,
    verify_contract_runs,
)
from tests.orchestrator.test_contract_runnability import (  # noqa: F401
    PROJECT, REPAIRED, RealExecutor, ScriptedClient, _write_contract,
)

# The measured shape: the project is reached through a subprocess, so a
# missing project surfaces as an assertion about a return code.
SUBPROCESS_HARNESS = '''
    import unittest


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            import runpy
            import subprocess
            import sys
            import textwrap

            harness = textwrap.dedent("""
                import runpy
                runpy.run_path("project.py", run_name="__main__")
                print("RESULT=ok")
                """)
            completed = subprocess.run([sys.executable, "-c", harness],
                                       capture_output=True, text=True,
                                       timeout=30)
            self.assertEqual(completed.returncode, 0,
                             msg=completed.stdout + completed.stderr)
            self.assertIn("RESULT=ok", completed.stdout)
'''


@pytest.fixture
def bare(tmp_path):
    """A project root with the contract but no code yet — wave 1."""
    reset_contract_repairs()
    _write_contract(tmp_path, SUBPROCESS_HARNESS)
    yield tmp_path
    reset_contract_repairs()


def _build(root):
    """What wave 2 does: write the code the contract drives."""
    with open(os.path.join(str(root), "project.py"), "w",
              encoding="utf-8") as fh:
        fh.write(textwrap.dedent(PROJECT))


def test_a_project_reached_by_subprocess_is_not_built_yet_either(bare):
    """The deferral must not depend on HOW the contract reaches the code.

    And a deferral is not a verdict: the contract must be tried again once
    a later wave has written the code. Reading this as "it ran and judged"
    is what settled the question while the answer was still meaningless.
    """
    client = ScriptedClient()             # any generation would raise
    executor = RealExecutor(bare)

    result = verify_contract_runs(executor, str(bare), client, "a game")
    assert result is None
    assert not client.prompts, \
        "a project that does not exist yet was read as a contract defect"

    after_defer = len(executor.commands)
    _build(bare)
    verify_contract_runs(executor, str(bare), client, "a game")
    assert len(executor.commands) > after_defer, \
        "a deferral was recorded as a verdict, so the next wave never looked"


def test_a_wave_one_failure_does_not_retire_the_final_check(bare):
    """The defect itself: the check ran once, too early, and never again."""
    client = ScriptedClient()
    executor = RealExecutor(bare)

    verify_contract_runs(executor, str(bare), client, "a game")
    runs_before_build = len(executor.commands)

    _build(bare)                          # wave 2 writes the code
    verify_contract_runs(executor, str(bare), client, "a game", final=True)

    assert len(executor.commands) > runs_before_build, \
        "the final check never executed the contract against the built code"


def test_the_verdict_is_never_handed_a_pre_build_result(bare):
    """A per-wave run measures an unfinished tree. It must not escape."""
    client = ScriptedClient()
    verify_contract_runs(RealExecutor(bare), str(bare), client, "a game")

    assert last_contract_run(str(bare)) is None, \
        "a result measured before the code existed was offered to the verdict"


def test_the_finished_artifact_is_judged_on_its_own_bytes(bare):
    """End to end: the run that failed must now come out green."""
    client = ScriptedClient()
    executor = RealExecutor(bare)

    verify_contract_runs(executor, str(bare), client, "a game")
    _build(bare)
    verify_contract_runs(executor, str(bare), client, "a game", final=True)

    cached = last_contract_run(str(bare))
    assert cached is not None, "the final check offered nothing to the verdict"
    assert cached[0] is True, \
        "a contract that passes against the built code was reported failed"


def test_a_mid_run_disagreement_is_still_only_measured_once_per_run(bare):
    """Not retiring the check must not mean re-running it every wave."""
    _build(bare)
    # Passes, so it is settled for the rest of the waves.
    _write_contract(bare, REPAIRED)
    client = ScriptedClient()
    executor = RealExecutor(bare)

    verify_contract_runs(executor, str(bare), client, "a game")
    after_first = len(executor.commands)
    verify_contract_runs(executor, str(bare), client, "a game")

    assert len(executor.commands) == after_first, \
        "a settled contract was re-executed on a later wave"


def test_a_genuine_missing_data_file_is_still_a_verdict(bare):
    """The not-built reading is scoped to a missing PYTHON SOURCE file.

    A contract asserting that the code produced a save-file has judged;
    reading that as "not built yet" would silently lose the finding.
    """
    _build(bare)
    # Caught and reported as a FAILURE, so the output carries the words
    # "FileNotFoundError" and a filename while still being a verdict.
    _write_contract(bare, '''
    import unittest

    import project


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            try:
                with open("highscores.json", encoding="utf-8") as fh:
                    self.assertIn("score", fh.read())
            except FileNotFoundError as exc:
                self.fail("the game never wrote its scores: %r" % exc)
''')
    client = ScriptedClient()
    result = verify_contract_runs(RealExecutor(bare), str(bare), client,
                                  "a game", final=True)

    assert result is None
    cached = last_contract_run(str(bare))
    assert cached is not None and cached[0] is False, \
        "a missing data file was excused as 'the code is not built yet'"
    assert not client.prompts, "an assertion failure triggered a rewrite"

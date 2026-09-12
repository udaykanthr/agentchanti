"""The last chance to catch a broken instrument, and paying for it once.

The per-wave runnability check defers while the contract cannot import the
project, which is the right answer mid-run. Measured 2026-09-12: the project
only became importable during the post-wave phases (bulk test, wiring, smoke),
so all seven per-wave checks deferred, the check never ran once, and the
contract's first execution was the end-of-run verdict itself — too late for
any repair::

    13:10:33 contract cannot import the project yet — deferring
    ... x7 ...
    13:15:28 contract cannot import the project yet — deferring   ← last wave
    13:17:36 Evidence: self-authored — ... none could be run

A final check closes that gap. It costs no LLM tokens unless the contract
actually crashes — a pass, an assertion failure and an unimportable project
all return before any generation — and the one subprocess it spends is handed
to the verdict rather than paid for twice.
"""
import textwrap
import unittest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, last_contract_run, reset_contract_repairs,
    verify_contract_runs,
)
from agentchanti.orchestrator.evidence import run_pre_existing_tests
from tests.orchestrator.test_contract_runnability import (  # noqa: F401
    CRASHING, PROJECT, REPAIRED, RealExecutor, ScriptedClient, _write_contract,
    project,
)

# Imports a module the project has not produced — the measured shape.
NOT_READY = """
    import unittest

    import not_built_yet


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            self.assertTrue(not_built_yet.works())
            self.assertEqual(not_built_yet.score(), 10)
"""

# Runs, judges, and disagrees. Never repaired — by design.
DISAGREES = """
    import unittest

    import project


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            self.assertEqual(game.score, 99)
"""


class CountingExecutor(RealExecutor):
    """Counts how many times the contract is actually executed."""

    def __init__(self, root):
        super().__init__(root)
        self.runs = 0

    def run_command(self, cmd, timeout=120, cwd=None, **kw):
        self.runs += 1
        return super().run_command(cmd, timeout=timeout, cwd=cwd, **kw)


class _Clean(unittest.TestCase):
    def setUp(self):
        reset_contract_repairs()

    def tearDown(self):
        reset_contract_repairs()


class FinalCheckStopsDeferring(_Clean):

    def test_mid_run_an_unimportable_project_still_defers(self):
        """Deferring is the right answer while waves remain."""
        _write_contract(project_dir := self._proj(), NOT_READY)
        client = ScriptedClient()          # any generation would raise
        result = verify_contract_runs(RealExecutor(project_dir),
                                      str(project_dir), client, "a game")
        self.assertIsNone(result)
        self.assertFalse(client.prompts, "deferral must not call the model")

    def test_at_the_end_it_reports_instead_of_deferring_silently(self):
        """There is no later wave; 'not built yet' is no longer meaningful."""
        project_dir = self._proj()
        _write_contract(project_dir, NOT_READY)
        client = ScriptedClient()
        with self.assertLogs("agentchanti", level="WARNING") as caught:
            result = verify_contract_runs(
                RealExecutor(project_dir), str(project_dir), client,
                "a game", final=True)
        self.assertIsNone(result)
        self.assertFalse(client.prompts, "still no LLM call")
        self.assertTrue(
            any("cannot import the project at the end" in m
                for m in caught.output),
            "the end-of-run failure was reported as a silent deferral")

    def _proj(self):
        import os
        import tempfile
        self._tmp = getattr(self, "_tmp", None) or tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        with open(os.path.join(root, "project.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(PROJECT))
        import pathlib
        return pathlib.Path(root)


class NoLLMTokensUnlessItCrashes(_Clean):

    def test_a_passing_contract_costs_no_generation(self, ):
        _write_contract(self.p, REPAIRED)
        client = ScriptedClient()
        verify_contract_runs(RealExecutor(self.p), str(self.p), client,
                             "a game", final=True)
        self.assertFalse(client.prompts)

    def test_an_assertion_failure_costs_no_generation(self):
        """It judged the code. Rewriting it would be the cheat."""
        _write_contract(self.p, DISAGREES)
        client = ScriptedClient()
        verify_contract_runs(RealExecutor(self.p), str(self.p), client,
                             "a game", final=True)
        self.assertFalse(client.prompts,
                         "an assertion failure triggered a rewrite")

    def setUp(self):
        super().setUp()
        import os
        import pathlib
        import tempfile
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        with open(os.path.join(self._t.name, "project.py"), "w",
                  encoding="utf-8") as f:
            f.write(textwrap.dedent(PROJECT))
        self.p = pathlib.Path(self._t.name)


class TheSubprocessIsPaidForOnce(_Clean):

    def setUp(self):
        super().setUp()
        import os
        import pathlib
        import tempfile
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        with open(os.path.join(self._t.name, "project.py"), "w",
                  encoding="utf-8") as f:
            f.write(textwrap.dedent(PROJECT))
        self.p = pathlib.Path(self._t.name)

    def test_the_verdict_reuses_the_final_checks_run(self):
        _write_contract(self.p, DISAGREES)
        ex = CountingExecutor(self.p)
        verify_contract_runs(ex, str(self.p), ScriptedClient(), "a game",
                             final=True)
        after_check = ex.runs
        self.assertEqual(after_check, 1, "the check should run it once")

        cached = last_contract_run(str(self.p))
        self.assertIsNotNone(cached, "nothing was offered to the verdict")

        run_pre_existing_tests(ex, str(self.p), [SEED_BASENAME],
                               prerun={SEED_BASENAME: cached})
        self.assertEqual(ex.runs, after_check,
                         "the verdict re-ran a file it was handed")

    def test_without_the_handoff_it_runs_twice(self):
        """States the cost the reuse avoids."""
        _write_contract(self.p, DISAGREES)
        ex = CountingExecutor(self.p)
        verify_contract_runs(ex, str(self.p), ScriptedClient(), "a game",
                             final=True)
        run_pre_existing_tests(ex, str(self.p), [SEED_BASENAME])
        self.assertEqual(ex.runs, 2)

    def test_the_reused_result_gives_the_same_verdict(self):
        """Reuse must not change the answer, only the cost."""
        _write_contract(self.p, DISAGREES)
        ex = CountingExecutor(self.p)
        fresh = run_pre_existing_tests(ex, str(self.p), [SEED_BASENAME])

        reset_contract_repairs()
        _write_contract(self.p, DISAGREES)
        ex2 = CountingExecutor(self.p)
        verify_contract_runs(ex2, str(self.p), ScriptedClient(), "a game",
                             final=True)
        reused = run_pre_existing_tests(
            ex2, str(self.p), [SEED_BASENAME],
            prerun={SEED_BASENAME: last_contract_run(str(self.p))})
        self.assertEqual(fresh[0], reused[0])

    def test_a_restored_original_invalidates_the_cached_run(self):
        """Or the verdict reports a result for bytes nobody can read."""
        _write_contract(self.p, CRASHING)
        still_broken = textwrap.dedent(CRASHING).strip()
        ex = CountingExecutor(self.p)
        verify_contract_runs(ex, str(self.p),
                             ScriptedClient(still_broken, still_broken),
                             "a game", final=True)
        self.assertIsNone(
            last_contract_run(str(self.p)),
            "a run of the discarded repair was offered to the verdict")


if __name__ == "__main__":
    unittest.main()

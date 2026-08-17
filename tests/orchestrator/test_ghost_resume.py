"""A resume must not be reported as a run that changed nothing.

The ghost's pre-state snapshot is taken when the run starts. On a resume
that is *after* the earlier run's steps wrote their files, so every one
of their postconditions reads as "bytes identical to the pre-run state —
the step changed nothing".

Measured 2026-08-18 00:12: a resume that executed a single step reported
eleven disagreements, ten of them `violated-touched` against four steps
that had finished in the previous run. The one real finding was in the
same list.
"""

from agentchanti.orchestrator.ghost import (
    HOLDS, INAPPLICABLE, VIOLATED, GhostPlan,
)
from agentchanti.orchestrator.plan_step import PlanStep


def _step(sid, **kw):
    kw.setdefault("step_type", "CODE")
    return PlanStep(id=sid, **kw)


def _write(root, rel, text):
    import os
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _plan():
    return [_step("1.1", target_files=["audio.py"]),
            _step("2.1", target_files=["main.py"])]


class TestCarriedStepsAreNotJudged:
    def test_without_the_fix_the_carried_step_reads_as_untouched(self, tmp_path):
        """The premise: this is exactly what the measured run reported."""
        root = str(tmp_path)
        _write(root, "audio.py", "DONE_LAST_RUN = 1\n")
        _write(root, "main.py", "OLD = 1\n")
        ghost = GhostPlan.build(_plan(), root)          # no carried ids

        ghost.resolve(["1.1", "2.1"], language="python")
        gaps = ghost.disagreements(["1.1", "2.1"])
        assert any(g.kind == "violated-touched" and g.step_id == "1.1"
                   for g in gaps)

    def test_a_carried_step_produces_no_finding(self, tmp_path):
        root = str(tmp_path)
        _write(root, "audio.py", "DONE_LAST_RUN = 1\n")
        _write(root, "main.py", "OLD = 1\n")
        ghost = GhostPlan.build(_plan(), root, carried_step_ids=["1.1"])

        ghost.resolve(["1.1", "2.1"], language="python")
        gaps = ghost.disagreements(["1.1", "2.1"])
        assert not any(g.step_id == "1.1" for g in gaps)

    def test_the_step_this_run_did_execute_is_still_judged(self, tmp_path):
        # The fix must not buy silence by disabling the check.
        root = str(tmp_path)
        _write(root, "audio.py", "DONE_LAST_RUN = 1\n")
        _write(root, "main.py", "OLD = 1\n")
        ghost = GhostPlan.build(_plan(), root, carried_step_ids=["1.1"])

        ghost.resolve(["1.1", "2.1"], language="python")
        gaps = ghost.disagreements(["1.1", "2.1"])
        assert any(g.kind == "violated-touched" and g.step_id == "2.1"
                   for g in gaps)

    def test_a_carried_expectation_is_retired_from_the_tally(self, tmp_path):
        root = str(tmp_path)
        _write(root, "audio.py", "DONE_LAST_RUN = 1\n")
        ghost = GhostPlan.build(_plan(), root, carried_step_ids=["1.1"])

        assert ghost.expectations[
            "file:audio.py#touched"].verdict == INAPPLICABLE
        # And it stays retired: resolve must not revive it.
        ghost.resolve(["1.1", "2.1"], language="python")
        assert ghost.expectations[
            "file:audio.py#touched"].verdict == INAPPLICABLE

    def test_an_expectation_a_pending_step_shares_is_still_checked(self,
                                                                   tmp_path):
        """Interning means one node can belong to both kinds of step."""
        root = str(tmp_path)
        _write(root, "shared.py", "OLD = 1\n")
        steps = [_step("1.1", target_files=["shared.py"]),
                 _step("2.1", target_files=["shared.py"])]
        ghost = GhostPlan.build(steps, root, carried_step_ids=["1.1"])

        exp = ghost.expectations["file:shared.py#touched"]
        assert exp.verdict != INAPPLICABLE      # 2.1 still claims it

        _write(root, "shared.py", "NEW = 2\n")
        ghost.resolve(["2.1"], language="python")
        assert exp.verdict == HOLDS

    def test_an_unknown_carried_id_is_ignored(self, tmp_path):
        ghost = GhostPlan.build(_plan(), str(tmp_path),
                                carried_step_ids=["9.9"])
        assert ghost.carried == set()

    def test_no_carried_ids_leaves_behaviour_identical(self, tmp_path):
        root = str(tmp_path)
        _write(root, "audio.py", "OLD = 1\n")
        ghost = GhostPlan.build(_plan(), root, carried_step_ids=[])
        ghost.resolve(["1.1"], language="python")
        assert ghost.expectations["file:audio.py#touched"].verdict == VIOLATED

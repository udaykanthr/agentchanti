"""`_run_diagnosis_loop` must leave the step holding its BEST state.

The unit-level scoring lives in test_diagnosis_best_snapshot.py; this
drives the real loop, with `_apply_fix` and `_execute_step` standing in for
the model, and asserts on what is left on disk after every attempt fails.

Both measured incidents are replayed:

  2026-08-17  attempt 1 produced the correct fix (9 errors + 1 failure -> 1
              failure) and the loop shipped the 9-error file anyway.
  2026-08-16  attempts 1-2 compounded regressions (4 failures -> 19 -> 39
              errors) and attempt 3 — the escalated one — was correct; the
              final restore shipped the 39-error state.
"""

import unittest
from unittest.mock import MagicMock, patch

from agentchanti.orchestrator.pipeline import _run_diagnosis_loop

# (fix_applied, cmds_succeeded, has_fix_commands, cmds_run, cmds_passed)
FIX_APPLIED = (True, False, False, [], [])


def unittest_output(total, failures=0, errors=0):
    tail = f"\n{'-' * 70}\nRan {total} tests in 0.07s\n\n"
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
            "Last output:\n" + tail)


class FakeMemory:
    """Just enough FileMemory to observe snapshot/restore against a disk."""

    def __init__(self, state):
        self.state = state
        self.restored = []

    def snapshot(self):
        return self.state

    def restore(self, snap, executor=None):
        self.restored.append(snap)
        self.state = snap

    def all_files(self):
        return {}

    def update(self, files):
        pass


def _kwargs(memory):
    cfg = MagicMock()
    cfg.AGENT_LOOP = False          # classic path
    display = MagicMock()
    display.steps = [{"type": "TEST"}]
    llm_client = MagicMock()
    llm_client.escalation_client = None
    coder = MagicMock()
    coder.escalation_client = None
    return dict(
        steps=["step"], llm_client=llm_client, executor=MagicMock(),
        coder=coder, reviewer=MagicMock(), tester=MagicMock(),
        task="task", memory=memory, display=display,
        language="python", cfg=cfg)


def _run(memory, outcomes, entry_error):
    """Drive the loop; `outcomes` is one (disk_state, error_info) per attempt."""
    seq = iter(outcomes)

    def fake_execute(*_a, **_kw):
        state, err = next(seq)
        memory.state = state           # the "fix" lands on disk
        return None, False, err

    with patch("agentchanti.orchestrator.pipeline._diagnose_failure",
               return_value="1. ROOT CAUSE: x"), \
         patch("agentchanti.orchestrator.pipeline._apply_fix",
               return_value=FIX_APPLIED), \
         patch("agentchanti.orchestrator.pipeline._execute_step",
               side_effect=fake_execute):
        return _run_diagnosis_loop(0, "step text", entry_error,
                                   **_kwargs(memory))


class TestDiagnosisRestoresBestState(unittest.TestCase):

    def test_2026_08_17_correct_first_fix_survives(self):
        """Attempt 1 fixes it; 2 and 3 make it worse. Ship attempt 1's."""
        memory = FakeMemory("broken")
        result = _run(memory, [
            ("fixed", unittest_output(8, failures=1)),              # score 1
            ("worse", unittest_output(8, failures=2, errors=3)),    # score 5
            ("worst", unittest_output(8, failures=1, errors=7)),    # score 8
        ], entry_error=unittest_output(8, failures=1, errors=9))    # score 8

        self.assertFalse(result)                 # the step still failed
        self.assertEqual(memory.state, "fixed")  # but it shipped its best

    def test_2026_08_16_escalated_final_fix_survives(self):
        """Attempts 1-2 compound regressions; attempt 3 (escalated) is right."""
        memory = FakeMemory("baseline")
        result = _run(memory, [
            ("regressed", unittest_output(40, failures=1, errors=19)),  # 20
            ("worse", unittest_output(40, failures=1, errors=39)),      # 40
            ("escalated_fix", unittest_output(40, failures=2)),         # 2
        ], entry_error=unittest_output(40, failures=4))                 # 4

        self.assertFalse(result)
        self.assertEqual(memory.state, "escalated_fix")

    def test_compounding_regressions_never_ship(self):
        """Every attempt is worse than the entry state — keep the entry."""
        memory = FakeMemory("pre_diagnosis")
        result = _run(memory, [
            ("bad1", unittest_output(40, failures=1, errors=19)),
            ("bad2", unittest_output(40, failures=1, errors=29)),
            ("bad3", unittest_output(40, failures=1, errors=39)),
        ], entry_error=unittest_output(40, failures=4))                 # 4

        self.assertFalse(result)
        self.assertEqual(memory.state, "pre_diagnosis")

    def test_unscorable_errors_keep_the_pre_diagnosis_state(self):
        """A CODE gate's bare traceback has no counts; nothing may be read
        as an improvement, so the loop falls back to the old behaviour."""
        memory = FakeMemory("pre_diagnosis")
        tb = ("Traceback (most recent call last):\n"
              "TypeError: 'bool' object is not callable")
        result = _run(memory, [("a", tb + " 1"), ("b", tb + " 2"),
                               ("c", tb + " 3")], entry_error=tb)

        self.assertFalse(result)
        self.assertEqual(memory.state, "pre_diagnosis")


if __name__ == "__main__":
    unittest.main()

"""The diff guard must not discard the escalated model's answer.

Only the last diagnosis attempt escalates to `models.escalation`, and on
that attempt the change-ratio threshold costs more than it protects.
Rejecting means the step fails regardless — there is no attempt 4 — so
the only thing the threshold buys is never finding out what the stronger
model said.

Measured 2026-08-17: gpt-5.6-sol spent 70 seconds on the most expensive
call of a run and returned a fix changing 63% of lines. It was rejected
against a 40% threshold, the "test-only constraint" fallback then produced
nothing actionable twice, and the run halted having discarded the one
answer it had paid most for.

The downside is bounded in a way it was not when the threshold was
written: `_run_diagnosis_loop` now keeps the best-scoring snapshot and
restores it, so a large fix that makes things worse is measured and
reverted rather than shipped.

The full-file-replacement block is NOT relaxed. "This fix is large" and
"this fix replaces the file wholesale" are different claims, and only the
first stops being worth acting on at the last attempt.
"""

import unittest
from unittest.mock import MagicMock

from agentchanti.orchestrator.diagnosis import _apply_fix

ORIGINAL = "\n".join(f"line_{i} = {i}" for i in range(60)) + "\n"

# Changes ~63% of the lines but keeps well over 30% of them, so it clears
# the full-file-replacement block and only trips the change ratio.
LARGE_FIX = "\n".join(
    (f"line_{i} = {i}" if i % 3 == 0 else f"line_{i} = {i} + 1")
    for i in range(60)) + "\n"

# Shares almost nothing with the original.
WHOLESALE = "\n".join(f"replaced_{i} = {i}" for i in range(60)) + "\n"


def _run(new_content, final_attempt):
    executor = MagicMock()
    executor.parse_code_blocks.return_value = {"game.py": new_content}
    executor.parse_code_blocks_fuzzy.return_value = {}
    executor.parse_commands.return_value = []
    memory = MagicMock()
    memory.get.return_value = ORIGINAL
    _apply_fix("1. ROOT CAUSE: x\n2. FIX: y", executor, memory,
               MagicMock(), 0, step_type="CODE",
               step_target_files=["game.py"],
               final_attempt=final_attempt)
    # Read the calls rather than stubbing them: giving write_files a
    # side_effect replaces its return value, which the caller consumes.
    written = {}
    for call in executor.write_files.call_args_list:
        written.update(call.args[0] if call.args else {})
    return written


class TestDiffGuardOnTheFinalAttempt(unittest.TestCase):

    def test_a_large_fix_is_rejected_on_an_early_attempt(self):
        """Unchanged behaviour: attempts 1 and 2 are cheap and repeatable,
        so a sprawling rewrite is still not worth adopting."""
        self.assertNotIn("game.py", _run(LARGE_FIX, final_attempt=False))

    def test_the_same_fix_is_allowed_on_the_final_attempt(self):
        """The escalated answer gets tried rather than thrown away."""
        self.assertIn("game.py", _run(LARGE_FIX, final_attempt=True))

    def test_a_wholesale_replacement_is_still_blocked_on_the_final_attempt(self):
        """The stronger guard is untouched — this is a different claim."""
        self.assertNotIn("game.py", _run(WHOLESALE, final_attempt=True))


if __name__ == "__main__":
    unittest.main()

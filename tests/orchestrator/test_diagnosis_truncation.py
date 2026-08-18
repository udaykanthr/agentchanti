"""A diagnosis cut off at the output cap must be retried, not parsed.

Measured 2026-08-15 on the `pacman-strict` task, classic path: step 4's
first diagnosis attempt spent the full 16,384-token output budget and came
back ending at a bare ``if`` — the second ``#### [EDIT]:`` block never
closed. The client logged ``result is likely truncated`` and the pipeline
went on to parse it anyway. The chunk editor did the right thing and
refused ("no unambiguous match; refusing to overwrite it"), so ~85s and a
full output budget bought nothing, and the refusal then pushed the step
into the test-only fallback for a reason that had nothing to do with the
diff guard.

The fix is to treat the client's own truncation flag as authoritative:
discard the fragment, retry once asking for a much smaller scope, and if
that is cut off too, salvage only the blocks that actually closed.
"""

import unittest
from unittest.mock import MagicMock

from agentchanti.orchestrator.diagnosis import (
    _COMPACTION_DIRECTIVE, _diagnose_failure, _drop_dangling_code_block,
    _has_code_block,
)
from agentchanti.orchestrator.memory import FileMemory


COMPLETE = (
    "1. ROOT CAUSE: typo'd attribute.\n\n"
    "#### [EDIT]: game.py:Game._step (lines 334-335)\n"
    "```python\n"
    "        if self.pellet_count == 0:\n"
    "            self.state = 'win'\n"
    "```\n"
)

# Shape of the real truncated response: one complete block, then a header
# and a fence that never closed.
TRUNCATED = COMPLETE + (
    "\n#### [EDIT]: game.py:Game.__init__ (lines around self.power_timer)\n"
    "```python\n"
    "        if"
)


class FakeClient:
    """Minimal LLM client that can report a truncated call.

    ``truncated`` is a list of flags, one per expected call, mirroring
    ``LLMClient._last_truncated`` — which the real client rewrites on
    every ``generate_response``.
    """

    def __init__(self, responses, truncated):
        self.responses = list(responses)
        self.truncated = list(truncated)
        self.prompts = []
        self._last_truncated = False

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        self._last_truncated = self.truncated.pop(0)
        return self.responses.pop(0)


def _diagnose(client):
    display = MagicMock()
    return _diagnose_failure(
        "Write the game tests", "TEST", "AssertionError: 'game_over' != 'win'",
        FileMemory(), client, display, 0)


class TestDropDanglingCodeBlock(unittest.TestCase):

    def test_unterminated_block_and_its_header_are_dropped(self):
        cleaned, dropped = _drop_dangling_code_block(TRUNCATED)
        self.assertTrue(dropped)
        self.assertNotIn("Game.__init__", cleaned)
        self.assertIn("Game._step", cleaned)
        # The surviving text is balanced, so nothing half-open reaches
        # the chunk editor.
        self.assertEqual(cleaned.count("```") % 2, 0)

    def test_balanced_response_is_untouched(self):
        cleaned, dropped = _drop_dangling_code_block(COMPLETE)
        self.assertFalse(dropped)
        self.assertEqual(cleaned, COMPLETE)

    def test_prose_only_response_is_untouched(self):
        text = "1. ROOT CAUSE: the service is unreachable.\n"
        self.assertEqual(_drop_dangling_code_block(text), (text, False))

    def test_a_single_truncated_block_leaves_root_cause_only(self):
        text = ("1. ROOT CAUSE: off-by-one.\n\n"
                "#### [EDIT]: game.py:Game.advance (lines 10-12)\n"
                "```python\n"
                "        if")
        cleaned, dropped = _drop_dangling_code_block(text)
        self.assertTrue(dropped)
        self.assertFalse(_has_code_block(cleaned))
        self.assertIn("ROOT CAUSE", cleaned)


class TestTruncatedDiagnosisRetry(unittest.TestCase):

    def test_clean_response_is_not_retried(self):
        client = FakeClient([COMPLETE], [False])
        result = _diagnose(client)
        self.assertEqual(len(client.prompts), 1)
        self.assertIn("Game._step", result)

    def test_truncated_response_is_retried_with_a_compaction_directive(self):
        client = FakeClient([TRUNCATED, COMPLETE], [True, False])
        result = _diagnose(client)

        self.assertEqual(len(client.prompts), 2)
        self.assertNotIn(_COMPACTION_DIRECTIVE, client.prompts[0])
        self.assertIn(_COMPACTION_DIRECTIVE, client.prompts[1])
        # The retry's answer is what gets applied — the fragment is gone.
        self.assertNotIn("Game.__init__", result)
        self.assertTrue(_has_code_block(result))

    def test_both_truncated_salvages_the_complete_blocks(self):
        client = FakeClient([TRUNCATED, TRUNCATED], [True, True])
        result = _diagnose(client)

        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(result.count("```") % 2, 0)
        self.assertTrue(_has_code_block(result))
        self.assertNotIn("Game.__init__", result)

    def test_both_truncated_with_no_complete_block_yields_no_code(self):
        """Root cause survives; nothing applyable is invented.

        A prose-only diagnosis is the honest outcome here — it makes
        ``_apply_fix`` report no actionable fix instead of the chunk
        editor refusing a fragment, which is the same result reached for
        a reason the log can explain.
        """
        partial = ("1. ROOT CAUSE: ghosts outrun the player.\n\n"
                   "#### [EDIT]: game.py:Ghost.update (lines 1-2)\n"
                   "```python\n"
                   "        if")
        client = FakeClient([partial, partial], [True, True])
        result = _diagnose(client)
        self.assertFalse(_has_code_block(result))
        self.assertIn("ROOT CAUSE", result)

    def test_a_raising_retry_falls_back_to_the_salvaged_first_answer(self):
        class Boom(FakeClient):
            def generate_response(self, prompt):
                if self.prompts:
                    raise RuntimeError("provider down")
                return super().generate_response(prompt)

        client = Boom([TRUNCATED], [True])
        result = _diagnose(client)
        self.assertTrue(_has_code_block(result))
        self.assertNotIn("Game.__init__", result)


if __name__ == "__main__":
    unittest.main()

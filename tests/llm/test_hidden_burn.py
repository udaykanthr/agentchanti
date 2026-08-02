"""A cap hit with almost no visible output is a hidden-reasoning burn.

The retry loop already handled EMPTY responses that hit the output cap:
the budget went to hidden reasoning, so it dials reasoning down and tries
again. A response with a *sliver* of visible text fell through that check
and was returned to the caller as a truncated result.

Observed on Gemini writing a whole module: 654 visible tokens of a 16,384
budget, cut mid-file, twice with identical counts — a verbatim retry
reproduces a deterministic burn exactly. Neither attempt contained a
parseable file, the step failed, and the pipeline halted having written
nothing after 12 minutes.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentchanti.llm.base import LLMClient


class _Fake(LLMClient):
    """Minimal client: scripted (text, stop_reason, visible_tokens) turns."""

    def __init__(self, turns, **kw):
        super().__init__(**kw)
        self._turns = list(turns)
        self.calls = 0
        self.downgraded = 0

    def _generate(self, prompt):
        self.calls += 1
        text, stop, visible = self._turns.pop(0)
        self._last_stop_reason = stop
        self._last_completion_tokens = visible
        return text

    def _generate_stream(self, prompt):
        return self._generate(prompt)

    def generate_embedding(self, text, model=None, **kw):
        return []

    def _prepare_token_limit_retry(self):
        self.downgraded += 1


def _client(turns):
    with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
        return _Fake(turns, stream=False, max_output_tokens=16384,
                     max_retries=3)


class TestHiddenBurnDetection(unittest.TestCase):

    def test_a_sliver_of_output_at_the_cap_is_a_burn(self):
        c = _client([])
        c._last_completion_tokens = 654
        self.assertTrue(c._looks_like_hidden_burn())

    def test_a_genuinely_long_answer_is_not(self):
        """Cut at 97% of budget is an answer too long, not a burn —
        dialling reasoning down would not help it."""
        c = _client([])
        c._last_completion_tokens = 15993
        self.assertFalse(c._looks_like_hidden_burn())

    def test_an_unreported_count_never_triggers_a_retry(self):
        c = _client([])
        c._last_completion_tokens = 0
        self.assertFalse(c._looks_like_hidden_burn())


class TestRetryBehaviour(unittest.TestCase):

    def _run(self, turns):
        c = _client(turns)
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            return c, c.generate_response("write the module")

    def test_a_burn_is_retried_with_reasoning_dialled_down(self):
        c, out = self._run([
            ("prose, cut mid-file", "length", 654),      # burn
            ("#### [FILE]: a.py\ncode", "stop", 1200),    # rescued
        ])
        self.assertEqual(c.calls, 2)
        self.assertEqual(c.downgraded, 1)
        self.assertIn("[FILE]", out)

    def test_a_long_truncated_answer_is_returned_not_retried(self):
        """The caller is told it is truncated; retrying wastes a call."""
        c, out = self._run([("a very long answer", "length", 15993)])
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.downgraded, 0)
        self.assertTrue(c._last_truncated)
        self.assertEqual(out, "a very long answer")

    def test_a_clean_response_is_untouched(self):
        c, out = self._run([("all good", "stop", 500)])
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.downgraded, 0)
        self.assertFalse(c._last_truncated)

    def test_a_burn_on_the_last_attempt_still_returns_the_text(self):
        """Never lose the only result we have to a retry that cannot run."""
        c = _client([("sliver", "length", 100)] * 3)
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            out = c.generate_response("go")
        self.assertEqual(out, "sliver")
        self.assertTrue(c._last_truncated)


if __name__ == "__main__":
    unittest.main()

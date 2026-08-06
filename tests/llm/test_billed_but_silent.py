"""An empty response that still billed output tokens is a reasoning burn.

The retry loop already recognised a burn when the response hit the output
cap. But a reasoning model can also spend a few hundred tokens thinking
and then stop cleanly with nothing visible — same burn, ``stop``
finish_reason, so the cap check missed it.

Observed on gpt-5.6-terra at ``reasoning_effort: high`` driving a
diagnosis step: ``completion=521``, empty text, three identical retries,
then ``LLM returned empty response after all retries`` killed the step and
halted the pipeline. Each retry paid the full prompt again, and the
generic anti-``<think>`` preamble the empty path falls back to cannot help
— the reasoning is server-side and never reaches the stream. Worse, the
preamble mutates the prompt, so the retries lose prompt-cache reuse too.

The distinction that matters: text that ARRIVED and was stripped is a
``<think>``-tag model (the preamble is the right lever); text that never
arrived while tokens were billed is server-side reasoning (only an effort
downgrade changes the outcome).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentchanti.llm.base import LLMClient


class _Fake(LLMClient):
    """Scripted client: each turn is (text, stop_reason, billed_tokens)."""

    def __init__(self, turns, **kw):
        super().__init__(**kw)
        self._turns = list(turns)
        self.calls = 0
        self.downgraded = 0
        self.prompts: list[str] = []

    def _generate(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        text, stop, billed = self._turns.pop(0)
        self._last_stop_reason = stop
        self._last_completion_tokens = billed
        return text

    def _generate_stream(self, prompt):
        return self._generate(prompt)

    def generate_embedding(self, text, model=None, **kw):
        return []

    def _prepare_token_limit_retry(self):
        self.downgraded += 1


def _client(turns):
    return _Fake(turns, stream=False, max_output_tokens=16384, max_retries=3)


class TestBilledButSilent(unittest.TestCase):

    def test_billed_tokens_with_no_text_is_a_burn(self):
        c = _client([("", "stop", 521), ("real answer", "stop", 40)])
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            out = c.generate_response("write the fix")
        self.assertEqual(out, "real answer")
        self.assertEqual(c.downgraded, 1,
                         "a silent-but-billed turn must dial reasoning down")

    def test_burn_retry_keeps_the_prompt_byte_identical(self):
        """The anti-<think> preamble would break prompt-cache reuse on a
        path where it cannot possibly help."""
        c = _client([("", "stop", 521), ("real answer", "stop", 40)])
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            c.generate_response("write the fix")
        self.assertEqual(c.prompts[0], c.prompts[1])

    def test_a_stripped_think_block_still_gets_the_preamble(self):
        """Text arrived and was stripped to nothing — a <think>-tag model.
        Dialling effort down is the wrong lever here."""
        c = _client([("<think>reasoning only</think>", "stop", 300),
                     ("real answer", "stop", 40)])
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            out = c.generate_response("write the fix")
        self.assertEqual(out, "real answer")
        self.assertEqual(c.downgraded, 0)
        self.assertIn("Do NOT use <think> tags", c.prompts[1])

    def test_a_truly_empty_turn_with_no_billing_is_not_a_burn(self):
        """No tokens billed means nothing was spent thinking — the empty
        response came from somewhere else and the preamble is the lever."""
        c = _client([("", "stop", 0), ("real answer", "stop", 40)])
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            out = c.generate_response("write the fix")
        self.assertEqual(out, "real answer")
        self.assertEqual(c.downgraded, 0)
        self.assertIn("Do NOT use <think> tags", c.prompts[1])

    def test_a_stale_count_does_not_fake_a_burn(self):
        """A provider that reports no usage must not inherit the previous
        call's count and look like a burn."""
        c = _client([("", "stop", 0), ("real answer", "stop", 40)])
        c._last_completion_tokens = 999          # left over from an earlier call
        with patch.object(LLMClient, "_backoff", lambda *a, **k: None):
            c.generate_response("write the fix")
        self.assertEqual(c.downgraded, 0)


if __name__ == "__main__":
    unittest.main()

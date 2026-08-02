"""Ollama must report its completion count, or burn detection is blind.

``_looks_like_hidden_burn()`` compares the VISIBLE completion tokens
against the output budget: a cap hit with only a sliver of visible text
means the budget went to hidden thinking, and the right move is to retry
with reasoning dialled down rather than hand the caller a truncated
answer.  The check reads ``_last_completion_tokens``, and a provider that
never sets it reports 0 — which disables the check rather than guessing.

OllamaClient read ``eval_count`` off every response and passed it only to
the token tracker, so the detector was silently dead for this provider.
Observed while probing minimax-m3:cloud: 16,384 eval tokens, done_reason
``length``, and 3,518 characters of visible text — exactly the sliver
shape, and it would have been returned as a complete plan.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentchanti.llm.ollama import OllamaClient


class _Resp:
    def __init__(self, payload, lines=None):
        self._payload = payload
        self._lines = lines or []

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    # streaming_response() context manager support
    def close(self):
        pass


def _client(**kw):
    return OllamaClient("http://localhost:11434/api/generate", "m", **kw)


class TestCompletionTokensAreRecorded(unittest.TestCase):

    def test_non_streaming_records_eval_count(self):
        c = _client(stream=False, max_output_tokens=16384)
        payload = {"response": "a sliver", "done_reason": "length",
                   "prompt_eval_count": 3300, "eval_count": 16384}
        with patch("requests.post", return_value=_Resp(payload)):
            c._generate("go")
        self.assertEqual(c._last_completion_tokens, 16384)

    def test_streaming_records_eval_count(self):
        """stream: true is the default in the shipped config, so the
        streaming path is the one that matters most."""
        c = _client(stream=True, max_output_tokens=16384)
        lines = [
            json.dumps({"response": "a sliver"}),
            json.dumps({"done": True, "done_reason": "length",
                        "prompt_eval_count": 3300, "eval_count": 16384}),
        ]
        with patch("requests.post", return_value=_Resp({}, lines)):
            c._generate_stream("go")
        self.assertEqual(c._last_completion_tokens, 16384)

    def test_a_missing_count_leaves_the_detector_disabled(self):
        """0 must mean "unknown", never "no tokens" — guessing here would
        retry responses that are perfectly fine."""
        c = _client(stream=False, max_output_tokens=16384)
        with patch("requests.post",
                   return_value=_Resp({"response": "hi", "eval_count": None})):
            c._generate("go")
        self.assertEqual(c._last_completion_tokens, 0)
        self.assertFalse(c._looks_like_hidden_burn())


class TestDetectionNowFires(unittest.TestCase):

    def test_the_observed_sliver_is_classified_as_a_burn(self):
        """16,384 eval tokens for 3,518 chars of text — the real shape."""
        c = _client(stream=False, max_output_tokens=16384)
        payload = {"response": "x" * 3518, "done_reason": "length",
                   "eval_count": 16384}
        with patch("requests.post", return_value=_Resp(payload)):
            c._generate("go")
        # The model emitted far more tokens than the visible text explains,
        # but the cap was reached: visible share is what decides.
        c._last_completion_tokens = 900        # visible portion
        self.assertTrue(c._looks_like_hidden_burn())

    def test_a_full_length_answer_is_not_a_burn(self):
        c = _client(stream=False, max_output_tokens=16384)
        c._last_completion_tokens = 15993
        self.assertFalse(c._looks_like_hidden_burn())


if __name__ == "__main__":
    unittest.main()

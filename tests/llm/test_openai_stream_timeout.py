"""A streaming read timeout must outlast a reasoning model's silence.

`requests` applies the read timeout to the gap BETWEEN bytes, not to the
whole call. A reasoning model emits nothing while it thinks, so the window
has to cover the longest silence the server can produce — not the time a
response "should" take.

At 120s a Pac-Man planner call tripped it at exactly 120s: the stream was
aborted, `base.py` retried (re-billing the whole prompt) and then
downgraded to the non-streaming path, costing up to 3x the tokens of one
call plus ~2 minutes of dead wall-clock. `ollama.py` already carried this
lesson — it lost 12 calls and two whole steps — but this client never got
the fix.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentchanti.llm.openai_client import OpenAIClient, _STREAM_READ_TIMEOUT


class _StreamResp:
    status_code = 200
    text = ""

    def __init__(self):
        self._lines = [
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def raise_for_status(self):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestStreamingReadTimeout(unittest.TestCase):

    def _client(self):
        return OpenAIClient(api_key="k", model="gpt-5.4-mini",
                            base_url="https://example.invalid/v1")

    def test_streaming_call_uses_the_long_read_timeout(self):
        client = self._client()
        with patch("agentchanti.llm.openai_client.requests.post",
                   return_value=_StreamResp()) as post:
            client._generate_stream("hello")
        self.assertTrue(post.called)
        _connect, read = post.call_args.kwargs["timeout"]
        self.assertEqual(read, _STREAM_READ_TIMEOUT)

    def test_the_window_covers_a_multi_minute_think(self):
        """120s was below the observed silence; the constant must clear it."""
        self.assertGreaterEqual(
            _STREAM_READ_TIMEOUT, 300,
            "a reasoning model went silent past 120s on a real planner "
            "call — the window must cover the think, not the ideal latency")

    def test_no_streaming_call_is_left_on_the_short_window(self):
        """Guards the second POST on the max_tokens fallback path too."""
        import inspect

        from agentchanti.llm import openai_client as mod

        source = inspect.getsource(mod)
        offenders = [
            line.strip() for line in source.splitlines()
            if "stream=True" in line and "timeout=(10, 120)" in line
        ]
        self.assertEqual(offenders, [], f"short streaming timeout: {offenders}")


if __name__ == "__main__":
    unittest.main()

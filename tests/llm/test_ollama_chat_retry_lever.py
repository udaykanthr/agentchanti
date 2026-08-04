"""The burn retry must actually change the request on the chat path.

_prepare_token_limit_retry() arms `think: false`; _apply_generate_options()
puts it on the payload. Both text paths called it. _chat did not — so a
detected reasoning burn armed the lever and then sent a byte-identical
request, three times, deterministically reproducing the same burn.

Every retry exhaustion across a long benchmark session was on this path:
the classic path burned 7 times in one run and recovered every time,
while the loop path lost two whole steps to ~5.5-minute repeats. The
model in question does honour `think: false` — probed directly, thinking
dropped from 45 characters to 0 — so the lever works; it was simply
never pulled.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentchanti.llm.ollama import OllamaClient
from agentchanti.llm.chat_types import Message


class _StreamResp:
    status_code = 200
    text = ""

    def __init__(self, chunks):
        self._lines = [json.dumps(c) for c in chunks]

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        pass


MSGS = [Message(role="user", content="go")]
_BURN = [{"done": True, "done_reason": "length", "eval_count": 32768}]


def _client(**kw):
    return OllamaClient("http://localhost:11434/api/generate", "m", **kw)


class TestChatHonoursTheRetryLever(unittest.TestCase):

    def _payloads(self, client, calls):
        seen = []

        def fake_post(url, **kw):
            seen.append(kw["json"])
            return _StreamResp(_BURN)

        with patch("requests.post", side_effect=fake_post):
            for _ in range(calls):
                client._chat(MSGS)
        return seen

    def test_a_plain_chat_does_not_disable_thinking(self):
        seen = self._payloads(_client(), 1)
        self.assertNotIn("think", seen[0])

    def test_an_armed_retry_disables_thinking(self):
        c = _client()
        c._prepare_token_limit_retry()
        seen = self._payloads(c, 1)
        self.assertIs(seen[0].get("think"), False)

    def test_the_retry_request_actually_differs(self):
        """The whole point: attempt 2 must not be byte-identical to 1."""
        c = _client()
        seen = self._payloads(c, 1)
        c._prepare_token_limit_retry()
        seen += self._payloads(c, 1)
        self.assertNotEqual(seen[0], seen[1])

    def test_the_flag_is_one_shot(self):
        c = _client()
        c._prepare_token_limit_retry()
        seen = self._payloads(c, 2)
        self.assertIs(seen[0].get("think"), False)
        self.assertNotIn("think", seen[1])

    def test_it_does_not_leak_into_a_later_generate(self):
        """_chat leaving the flag armed dialled reasoning down on the next
        unrelated text call."""
        c = _client(stream=False)
        c._prepare_token_limit_retry()
        self._payloads(c, 1)          # consumed here
        self.assertFalse(c._retry_disable_think)


if __name__ == "__main__":
    unittest.main()

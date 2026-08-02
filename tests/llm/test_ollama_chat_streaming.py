"""/api/chat must stream, or long generations die on the read timeout.

A read timeout applies to the gap BETWEEN bytes, not to the whole call.
Streaming resets it on every chunk; non-streaming has to deliver
everything inside one window. Cloud reasoning models routinely think for
minutes before emitting a token, and a measured run lost 12 calls and two
whole steps to `Read timed out (read timeout=300)` — each costing the full
300s, three times over, for nothing. The text path had always streamed,
which is exactly why it saw one timeout to the chat path's twelve.

The reassembled result must be indistinguishable from what the
non-streaming endpoint used to return: same text, same tool calls, same
usage accounting.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentchanti.llm.ollama import OllamaClient
from agentchanti.llm.chat_types import Message, ToolDef


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


def _client(**kw):
    return OllamaClient("http://localhost:11434/api/generate", "m", **kw)


MSGS = [Message(role="user", content="go")]


class TestStreamedChat(unittest.TestCase):

    def test_text_chunks_are_reassembled_in_order(self):
        chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": ", "}},
            {"message": {"content": "world"}},
            {"done": True, "done_reason": "stop",
             "prompt_eval_count": 10, "eval_count": 3},
        ]
        c = _client()
        with patch("requests.post", return_value=_StreamResp(chunks)):
            r = c._chat(MSGS)
        self.assertEqual(r.text, "Hello, world")
        self.assertEqual(r.stop_reason, "stop")

    def test_tool_calls_survive_the_stream(self):
        chunks = [
            {"message": {"tool_calls": [
                {"function": {"name": "read_file",
                              "arguments": {"path": "a.py"}}}]}},
            {"done": True, "done_reason": "stop", "eval_count": 7},
        ]
        c = _client()
        with patch("requests.post", return_value=_StreamResp(chunks)):
            r = c._chat(MSGS, tools=[ToolDef("read_file", "d", {})])
        self.assertEqual(len(r.tool_calls), 1)
        self.assertEqual(r.tool_calls[0].name, "read_file")
        self.assertEqual(r.tool_calls[0].arguments, {"path": "a.py"})

    def test_string_encoded_arguments_are_still_parsed(self):
        """Some models return arguments as a JSON string, not an object."""
        chunks = [
            {"message": {"tool_calls": [
                {"function": {"name": "run_command",
                              "arguments": '{"command": "pytest"}'}}]}},
            {"done": True, "eval_count": 4},
        ]
        c = _client()
        with patch("requests.post", return_value=_StreamResp(chunks)):
            r = c._chat(MSGS)
        self.assertEqual(r.tool_calls[0].arguments, {"command": "pytest"})

    def test_usage_is_read_from_the_final_chunk(self):
        chunks = [
            {"message": {"content": "hi"}},
            {"done": True, "done_reason": "length",
             "prompt_eval_count": 2255, "eval_count": 16384},
        ]
        c = _client(max_output_tokens=16384)
        with patch("requests.post", return_value=_StreamResp(chunks)):
            c._chat(MSGS)
        # Burn detection depends on this being recorded.
        self.assertEqual(c._last_completion_tokens, 16384)

    def test_a_malformed_line_does_not_abort_the_stream(self):
        resp = _StreamResp([{"message": {"content": "a"}},
                            {"done": True, "eval_count": 1}])
        resp._lines.insert(1, "{not json")
        c = _client()
        with patch("requests.post", return_value=resp):
            r = c._chat(MSGS)
        self.assertEqual(r.text, "a")

    def test_the_request_actually_asks_for_a_stream(self):
        captured = {}

        def fake_post(url, **kw):
            captured.update(kw)
            return _StreamResp([{"done": True, "eval_count": 1}])

        c = _client()
        with patch("requests.post", side_effect=fake_post):
            c._chat(MSGS)
        self.assertTrue(captured["json"]["stream"])
        self.assertTrue(captured["stream"])


class TestReadTimeout(unittest.TestCase):

    def test_default_is_generous_enough_for_a_thinking_model(self):
        self.assertGreaterEqual(_client().read_timeout, 900)

    def test_it_is_configurable(self):
        self.assertEqual(_client(read_timeout=120).read_timeout, 120)

    def test_every_generation_path_uses_it(self):
        """A per-call timeout that only covers one of three paths leaves
        the other two able to die exactly as before."""
        seen = []

        def fake_post(url, **kw):
            seen.append(kw.get("timeout"))
            if kw.get("stream"):
                return _StreamResp([{"done": True, "eval_count": 1}])
            return _FakeJson({"response": "x", "eval_count": 1})

        c = _client(read_timeout=777, stream=True)
        with patch("requests.post", side_effect=fake_post):
            c._chat(MSGS)
            c._generate_stream("go")
            c._generate("go")
        self.assertEqual(len(seen), 3, "expected chat + stream + generate")
        self.assertTrue(all(t == (10, 777) for t in seen), seen)


class _FakeJson:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


if __name__ == "__main__":
    unittest.main()

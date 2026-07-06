"""Tests for the chat-native LLM layer: LLMClient.chat(), Message/ToolDef
serialization for Ollama and Anthropic, and the text-flattening fallback."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from agentchanti.llm.base import LLMClient, LLMError, ToolsNotSupportedError
from agentchanti.llm.chat_types import (
    ChatResponse,
    Message,
    ToolCall,
    ToolDef,
    flatten_messages,
)
from agentchanti.llm.anthropic_client import AnthropicClient
from agentchanti.llm.ollama import OllamaClient
from agentchanti.llm.openai_client import OpenAIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data, status_code=200, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


class _TextOnlyClient(LLMClient):
    """Provider without native chat — exercises the fallback path."""

    def __init__(self, reply="hello", **kwargs):
        kwargs.setdefault("max_retries", 1)
        kwargs.setdefault("retry_delay", 0)
        kwargs.setdefault("stream", False)
        super().__init__(**kwargs)
        self.reply = reply
        self.last_prompt = None

    def _generate(self, prompt):
        self.last_prompt = prompt
        return self.reply

    def _generate_stream(self, prompt):
        return self._generate(prompt)

    def generate_embedding(self, text, model=None, **kwargs):
        return []


_SAMPLE_TOOLS = [
    ToolDef(name="read_file",
            description="Read a file",
            parameters={"type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]}),
]


# ---------------------------------------------------------------------------
# flatten_messages
# ---------------------------------------------------------------------------

class TestFlattenMessages(unittest.TestCase):

    def test_roles_and_order(self):
        prompt = flatten_messages([
            Message(role="system", content="You are a coder."),
            Message(role="user", content="Fix the bug."),
        ])
        self.assertIn("### System\nYou are a coder.", prompt)
        self.assertIn("### User\nFix the bug.", prompt)
        self.assertLess(prompt.index("System"), prompt.index("User"))
        self.assertTrue(prompt.rstrip().endswith("### Assistant"))

    def test_tool_calls_and_results_rendered(self):
        prompt = flatten_messages([
            Message(role="assistant", content="Reading.",
                    tool_calls=[ToolCall(name="read_file",
                                         arguments={"path": "a.py"})]),
            Message(role="tool", content="print('hi')",
                    tool_name="read_file"),
        ])
        self.assertIn('[tool call] read_file({"path": "a.py"})', prompt)
        self.assertIn("### Tool result (read_file)\nprint('hi')", prompt)


# ---------------------------------------------------------------------------
# Fallback path (no native chat)
# ---------------------------------------------------------------------------

class TestChatFallback(unittest.TestCase):

    def test_chat_flattens_and_returns_text(self):
        client = _TextOnlyClient(reply="done")
        result = client.chat([
            Message(role="system", content="Rules."),
            Message(role="user", content="Do the thing."),
        ])
        self.assertIsInstance(result, ChatResponse)
        self.assertEqual(result.text, "done")
        self.assertEqual(result.tool_calls, [])
        self.assertIn("### System\nRules.", client.last_prompt)
        self.assertIn("### User\nDo the thing.", client.last_prompt)

    def test_supports_tools_false(self):
        self.assertFalse(_TextOnlyClient().supports_tools())

    def test_chat_with_tools_still_returns_text(self):
        client = _TextOnlyClient(reply="text answer")
        result = client.chat(
            [Message(role="user", content="hi")], tools=_SAMPLE_TOOLS)
        self.assertEqual(result.text, "text answer")
        self.assertFalse(result.has_tool_calls)


# ---------------------------------------------------------------------------
# Ollama native chat
# ---------------------------------------------------------------------------

class TestOllamaChat(unittest.TestCase):

    def _client(self, **kwargs):
        kwargs.setdefault("max_retries", 1)
        kwargs.setdefault("retry_delay", 0)
        kwargs.setdefault("stream", False)
        return OllamaClient(
            base_url="http://localhost:11434/api/generate",
            model="qwen2.5-coder", **kwargs)

    @patch("agentchanti.llm.ollama.requests.post")
    def test_payload_serialization(self, mock_post):
        mock_post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 5,
        })
        client = self._client()
        result = client.chat([
            Message(role="system", content="Rules."),
            Message(role="user", content="Read a.py"),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(name="read_file",
                                         arguments={"path": "a.py"})]),
            Message(role="tool", content="print('hi')", tool_name="read_file"),
        ], tools=_SAMPLE_TOOLS)

        url = mock_post.call_args[0][0]
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0],
                         {"role": "system", "content": "Rules."})
        self.assertEqual(
            payload["messages"][2]["tool_calls"],
            [{"function": {"name": "read_file",
                           "arguments": {"path": "a.py"}}}])
        self.assertEqual(payload["messages"][3]["role"], "tool")
        self.assertEqual(payload["messages"][3]["tool_name"], "read_file")
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.stop_reason, "stop")

    @patch("agentchanti.llm.ollama.requests.post")
    def test_tool_calls_parsed(self, mock_post):
        mock_post.return_value = _mock_response({
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file",
                                  "arguments": {"path": "a.py"}}},
                ],
            },
            "done_reason": "stop",
        })
        result = self._client().chat(
            [Message(role="user", content="go")], tools=_SAMPLE_TOOLS)
        self.assertTrue(result.has_tool_calls)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.py"})

    @patch("agentchanti.llm.ollama.requests.post")
    def test_string_arguments_decoded(self, mock_post):
        mock_post.return_value = _mock_response({
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file",
                                  "arguments": '{"path": "b.py"}'}},
                ],
            },
        })
        result = self._client().chat(
            [Message(role="user", content="go")], tools=_SAMPLE_TOOLS)
        self.assertEqual(result.tool_calls[0].arguments, {"path": "b.py"})

    @patch("agentchanti.llm.ollama.requests.post")
    def test_tools_rejected_falls_back_to_text(self, mock_post):
        mock_post.side_effect = [
            _mock_response({}, status_code=400,
                           text="registry.ollama.ai/library/x does not support tools"),
            _mock_response({"response": "text fallback",
                            "prompt_eval_count": 5, "eval_count": 3}),
        ]
        client = self._client()
        result = client.chat(
            [Message(role="user", content="go")], tools=_SAMPLE_TOOLS)
        self.assertEqual(result.text, "text fallback")
        self.assertFalse(client.supports_tools())
        # Second call went to the generate endpoint, not /api/chat
        fallback_url = mock_post.call_args[0][0]
        self.assertEqual(fallback_url, "http://localhost:11434/api/generate")

    @patch("agentchanti.llm.ollama.requests.post")
    def test_empty_response_raises_after_retries(self, mock_post):
        mock_post.return_value = _mock_response(
            {"message": {"role": "assistant", "content": ""}})
        client = self._client(max_retries=2)
        with self.assertRaises(LLMError):
            client.chat([Message(role="user", content="go")])
        self.assertEqual(mock_post.call_count, 2)

    @patch("agentchanti.llm.ollama.requests.post")
    def test_reasoning_stripped_from_chat_text(self, mock_post):
        mock_post.return_value = _mock_response({
            "message": {"role": "assistant",
                        "content": "<think>hmm</think>answer"},
        })
        result = self._client().chat([Message(role="user", content="go")])
        self.assertEqual(result.text, "answer")


# ---------------------------------------------------------------------------
# Anthropic native chat
# ---------------------------------------------------------------------------

class TestAnthropicChat(unittest.TestCase):

    def _client(self, **kwargs):
        kwargs.setdefault("max_retries", 1)
        kwargs.setdefault("retry_delay", 0)
        return AnthropicClient(
            base_url="https://api.anthropic.com/v1",
            model="claude-sonnet-5", api_key="test-key", **kwargs)

    @patch("agentchanti.llm.anthropic_client.requests.post")
    def test_system_extracted_and_tools_sent(self, mock_post):
        mock_post.return_value = _mock_response({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        })
        result = self._client().chat([
            Message(role="system", content="Rules."),
            Message(role="user", content="hi"),
        ], tools=_SAMPLE_TOOLS)

        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["system"], "Rules.")
        # No system message left in the messages array
        self.assertTrue(all(m["role"] != "system" for m in payload["messages"]))
        self.assertEqual(payload["tools"][0]["name"], "read_file")
        self.assertEqual(payload["tools"][0]["input_schema"],
                         _SAMPLE_TOOLS[0].parameters)
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.stop_reason, "end_turn")

    @patch("agentchanti.llm.anthropic_client.requests.post")
    def test_tool_use_parsed(self, mock_post):
        mock_post.return_value = _mock_response({
            "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "a.py"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        result = self._client().chat(
            [Message(role="user", content="go")], tools=_SAMPLE_TOOLS)
        self.assertEqual(result.text, "Reading.")
        self.assertEqual(result.tool_calls[0].id, "toolu_1")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(result.stop_reason, "tool_use")

    @patch("agentchanti.llm.anthropic_client.requests.post")
    def test_tool_results_become_user_blocks_and_merge(self, mock_post):
        mock_post.return_value = _mock_response({
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        self._client().chat([
            Message(role="user", content="go"),
            Message(role="assistant",
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "a.py"},
                                 id="toolu_1"),
                        ToolCall(name="read_file", arguments={"path": "b.py"},
                                 id="toolu_2"),
                    ]),
            Message(role="tool", content="aaa", tool_call_id="toolu_1"),
            Message(role="tool", content="bbb", tool_call_id="toolu_2"),
        ])
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        # user, assistant(tool_use), single merged user(tool_result x2)
        self.assertEqual(len(messages), 3)
        assistant = messages[1]
        self.assertEqual(
            [b["type"] for b in assistant["content"]],
            ["tool_use", "tool_use"])
        results = messages[2]
        self.assertEqual(results["role"], "user")
        self.assertEqual(
            [b["tool_use_id"] for b in results["content"]],
            ["toolu_1", "toolu_2"])

    def test_supports_tools_true(self):
        self.assertTrue(self._client().supports_tools())


# ---------------------------------------------------------------------------
# OpenAI native chat
# ---------------------------------------------------------------------------

class TestOpenAIChat(unittest.TestCase):

    def _client(self, **kwargs):
        kwargs.setdefault("max_retries", 1)
        kwargs.setdefault("retry_delay", 0)
        kwargs.setdefault("stream", False)
        return OpenAIClient(
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini", api_key="test-key", **kwargs)

    @patch("agentchanti.llm.openai_client.requests.post")
    def test_payload_serialization(self, mock_post):
        mock_post.return_value = _mock_response({
            "choices": [{"message": {"content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })
        result = self._client().chat([
            Message(role="system", content="Rules."),
            Message(role="user", content="go"),
            Message(role="assistant",
                    tool_calls=[ToolCall(name="read_file",
                                         arguments={"path": "a.py"},
                                         id="call_1")]),
            Message(role="tool", content="data", tool_call_id="call_1"),
        ], tools=_SAMPLE_TOOLS)

        url = mock_post.call_args[0][0]
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(payload["messages"][0],
                         {"role": "system", "content": "Rules."})
        assistant = payload["messages"][2]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"],
                         '{"path": "a.py"}')
        tool_msg = payload["messages"][3]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.stop_reason, "stop")

    @patch("agentchanti.llm.openai_client.requests.post")
    def test_tool_calls_parsed_with_json_string_args(self, mock_post):
        mock_post.return_value = _mock_response({
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "call_9", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "b.py"}'}},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })
        result = self._client().chat(
            [Message(role="user", content="go")], tools=_SAMPLE_TOOLS)
        self.assertTrue(result.has_tool_calls)
        self.assertEqual(result.tool_calls[0].id, "call_9")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "b.py"})
        self.assertEqual(result.stop_reason, "tool_calls")

    def test_supports_tools_true(self):
        self.assertTrue(self._client().supports_tools())


if __name__ == "__main__":
    unittest.main()

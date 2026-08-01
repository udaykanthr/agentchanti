"""Error handling in the OpenAI-compatible client.

A model that 400'd on every tool-calling request took a whole pipeline
down, and the logs showed nothing but `400 Client Error: Bad Request for
url: ...` — the provider's explanation was thrown away, the request was
retried three times regardless, and the resulting LLMError escaped.
"""

from __future__ import annotations

import unittest
import unittest.mock
from unittest.mock import MagicMock

from agentchanti.llm.chat_types import Message

from agentchanti.llm.base import LLMError, NonRetryableLLMError, \
    ToolsNotSupportedError
from agentchanti.llm.openai_client import (
    _looks_like_tools_rejection,
    _param_rejected,
    _raise_for_status_with_body,
    _reasoning_blocks_tools,
)


def _response(status: int, payload=None, text: str = "",
              url: str = "https://api.openai.com/v1/chat/completions"):
    resp = MagicMock()
    resp.status_code = status
    resp.url = url
    resp.text = text
    if payload is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = payload
    resp.raise_for_status.side_effect = AssertionError(
        "raise_for_status must not be reached for this status")
    return resp


class TestRaiseForStatusWithBody(unittest.TestCase):

    def test_success_is_a_no_op(self):
        resp = _response(200, {"ok": True})
        _raise_for_status_with_body(resp)   # must not raise

    def test_400_keeps_the_provider_message(self):
        resp = _response(400, {"error": {
            "message": "Invalid schema for function 'write_file'.",
            "type": "invalid_request_error"}})
        with self.assertRaises(NonRetryableLLMError) as ctx:
            _raise_for_status_with_body(resp, model="gpt-5.6-terra",
                                        tool_count=6)
        msg = str(ctx.exception)
        self.assertIn("Invalid schema for function", msg)
        self.assertIn("gpt-5.6-terra", msg)
        self.assertIn("6 tool(s)", msg)
        self.assertIn("400", msg)

    def test_names_the_request_parameters_that_were_sent(self):
        """Turns "the model rejected it" into something actionable."""
        resp = _response(400, {"error": {"message": "Unsupported parameter"}})
        with self.assertRaises(NonRetryableLLMError) as ctx:
            _raise_for_status_with_body(
                resp, model="m",
                payload={"model": "m", "messages": [], "stream": False,
                         "max_completion_tokens": 16384, "tools": []})
        msg = str(ctx.exception)
        self.assertIn("params=", msg)
        self.assertIn("max_completion_tokens", msg)
        self.assertIn("tools", msg)
        # The request payload, not the response body.
        self.assertNotIn("params=error", msg)

    def test_non_json_body_still_surfaces(self):
        resp = _response(400, payload=None, text="upstream rejected request")
        with self.assertRaises(NonRetryableLLMError) as ctx:
            _raise_for_status_with_body(resp)
        self.assertIn("upstream rejected", str(ctx.exception))

    def test_400_is_non_retryable(self):
        """A malformed request does not become valid by being resent."""
        resp = _response(400, {"error": {"message": "bad"}})
        with self.assertRaises(NonRetryableLLMError):
            _raise_for_status_with_body(resp)
        # NonRetryableLLMError is an LLMError, which chat() re-raises
        # immediately rather than feeding into the retry loop.
        self.assertTrue(issubclass(NonRetryableLLMError, LLMError))

    def test_429_and_408_stay_retryable(self):
        for status in (408, 409, 429):
            with self.subTest(status=status):
                resp = _response(status, {"error": {"message": "slow down"}})
                resp.raise_for_status.side_effect = RuntimeError("retryable")
                with self.assertRaises(RuntimeError):
                    _raise_for_status_with_body(resp)

    def test_5xx_stays_retryable(self):
        resp = _response(503, {"error": {"message": "overloaded"}})
        resp.raise_for_status.side_effect = RuntimeError("retryable")
        with self.assertRaises(RuntimeError):
            _raise_for_status_with_body(resp)


class TestToolsRejectionDetection(unittest.TestCase):

    def test_recognises_tool_rejection_phrasings(self):
        for body in (
            '{"error":{"message":"model does not support tools"}}',
            '{"error":{"message":"Function calling is not supported"}}',
            '{"error":{"message":"Unsupported parameter: \'tools\'"}}',
            '{"error":{"message":"Tools are not supported for this model"}}',
        ):
            with self.subTest(body=body):
                self.assertTrue(
                    _looks_like_tools_rejection(_response(400, text=body)))

    def test_unrelated_400_is_not_a_tools_rejection(self):
        """Misreading this would silently disable the agent loop session-wide."""
        for body in (
            '{"error":{"message":"context_length_exceeded"}}',
            '{"error":{"message":"Invalid value for temperature"}}',
            '{"error":{"message":"Incorrect API key provided"}}',
            "",
        ):
            with self.subTest(body=body):
                self.assertFalse(
                    _looks_like_tools_rejection(_response(400, text=body)))

    def test_error_type_is_the_one_chat_downgrades_on(self):
        self.assertFalse(issubclass(ToolsNotSupportedError, LLMError))


class TestReasoningBlocksTools(unittest.TestCase):
    """Reasoning-on + function tools is recoverable, not a lack of support.

    The real message from a model that defaults reasoning on server-side:
    "Function tools with reasoning_effort are not supported for
    gpt-5.6-terra in /v1/chat/completions. To use function tools, use
    /v1/responses or set reasoning_effort to 'none'."
    """

    REAL_BODY = ('{"error":{"message":"Function tools with reasoning_effort '
                 'are not supported for gpt-5.6-terra in '
                 '/v1/chat/completions. To use function tools, use '
                 '/v1/responses or set reasoning_effort to \'none\'."}}')

    def test_recognises_the_real_message(self):
        self.assertTrue(
            _reasoning_blocks_tools(_response(400, text=self.REAL_BODY)))

    def test_is_not_confused_with_no_tool_support(self):
        """These take different paths: one retries with reasoning off and
        KEEPS tools, the other abandons tools for the session."""
        resp = _response(400, text=self.REAL_BODY)
        self.assertTrue(_reasoning_blocks_tools(resp))
        self.assertFalse(_looks_like_tools_rejection(resp))

    def test_unrelated_400s_do_not_match(self):
        for body in (
            '{"error":{"message":"Invalid schema for function write_file"}}',
            '{"error":{"message":"reasoning_effort must be one of low, high"}}',
            '{"error":{"message":"context_length_exceeded"}}',
            "",
        ):
            with self.subTest(body=body):
                self.assertFalse(
                    _reasoning_blocks_tools(_response(400, text=body)))


class TestResponsesApiFallover(unittest.TestCase):
    """End to end: the 400 switches endpoints, keeping tools AND reasoning."""

    def _client(self, **kw):
        from agentchanti.llm.openai_client import OpenAIClient
        return OpenAIClient(base_url="https://api.openai.com/v1",
                            model="gpt-5.6-terra", api_key="k", **kw)

    def _tools(self):
        from agentchanti.llm.chat_types import ToolDef
        return [ToolDef(name="ping", description="p",
                        parameters={"type": "object", "properties": {}})]

    def _ok_responses(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.url = "https://api.openai.com/v1/responses"
        resp.json.return_value = {
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "done"}]},
                {"type": "function_call", "call_id": "fc_1",
                 "name": "ping", "arguments": '{"x": "1"}'},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 1}},
        }
        return resp

    def _blocked(self):
        return _response(400, text=TestReasoningBlocksTools.REAL_BODY)

    def _recorder(self, responses):
        """Snapshot each payload — the client mutates one dict in place, so
        inspecting call_args afterwards would only ever show the last state."""
        import copy
        sent: list[tuple[str, dict]] = []
        it = iter(responses)

        def _post(url, **kwargs):
            sent.append((url, copy.deepcopy(kwargs.get("json"))))
            return next(it)

        return _post, sent

    def test_switches_endpoint_and_latches(self):
        client = self._client()
        post_fn, sent = self._recorder([self._blocked(),
                                        self._ok_responses()])
        with unittest.mock.patch(
                "agentchanti.llm.openai_client.requests.post", post_fn):
            result = client._chat([Message(role="user", content="hi")],
                                  tools=self._tools())

        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[0][0].endswith("/chat/completions"))
        self.assertTrue(sent[1][0].endswith("/responses"))
        # Tools survive, and nothing forced reasoning off.
        self.assertIn("tools", sent[1][1])
        self.assertNotIn("reasoning_effort", sent[1][1])
        self.assertTrue(client._tools_need_responses_api)
        # The response was parsed out of the Responses shape.
        self.assertEqual(result.text, "done")
        self.assertEqual([tc.name for tc in result.tool_calls], ["ping"])
        self.assertEqual(result.tool_calls[0].arguments, {"x": "1"})
        self.assertEqual(result.tool_calls[0].id, "fc_1")

        # Latched: the next tool call goes straight to /responses.
        post_fn, sent = self._recorder([self._ok_responses()])
        with unittest.mock.patch(
                "agentchanti.llm.openai_client.requests.post", post_fn):
            client._chat([Message(role="user", content="again")],
                         tools=self._tools())
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0][0].endswith("/responses"))

    def test_a_sibling_latching_mid_flight_does_not_lose_the_retry(self):
        """Wave steps share one client across threads.

        Two parallel tool calls both got the 400. The first latched and
        recovered; the second — already past the latch check when it was
        flipped — saw it set, skipped its own retry and raised, costing
        that step a failure and a whole recovery loop for a condition
        already known to be handleable.
        """
        client = self._client()
        responses = iter([self._blocked(), self._ok_responses()])
        sent: list[str] = []

        def _post(url, **kwargs):
            sent.append(url)
            resp = next(responses)
            if resp.status_code == 400:
                # A sibling thread latches while this request is in flight.
                client._tools_need_responses_api = True
            return resp

        with unittest.mock.patch(
                "agentchanti.llm.openai_client.requests.post", _post):
            result = client._chat([Message(role="user", content="hi")],
                                  tools=self._tools())

        self.assertEqual(len(sent), 2, "the retry was skipped")
        self.assertTrue(sent[0].endswith("/chat/completions"))
        self.assertTrue(sent[1].endswith("/responses"))
        self.assertEqual(result.text, "done")

    def test_reasoning_effort_is_sent_when_configured(self):
        client = self._client(reasoning_effort="high")
        client._tools_need_responses_api = True
        post_fn, sent = self._recorder([self._ok_responses()])
        with unittest.mock.patch(
                "agentchanti.llm.openai_client.requests.post", post_fn):
            client._chat([Message(role="user", content="hi")],
                         tools=self._tools())
        self.assertEqual(sent[0][1]["reasoning"], {"effort": "high"})

    def test_toolless_turns_stay_on_chat_completions(self):
        """Only tool calls need the other endpoint; don't move everything."""
        client = self._client()
        client._tools_need_responses_api = True
        ok = MagicMock()
        ok.status_code = 200
        ok.url = "https://api.openai.com/v1/chat/completions"
        ok.json.return_value = {
            "choices": [{"message": {"content": "hi", "tool_calls": []},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        post_fn, sent = self._recorder([ok])
        with unittest.mock.patch(
                "agentchanti.llm.openai_client.requests.post", post_fn):
            client._chat([Message(role="user", content="hi")], tools=None)
        self.assertTrue(sent[0][0].endswith("/chat/completions"))


class TestConfiguredReasoningEffort(unittest.TestCase):
    """`reasoning_effort:` in config previously only reached LM Studio."""

    def _client(self, model="gpt-5.6-terra", effort="high"):
        from agentchanti.llm.openai_client import OpenAIClient
        return OpenAIClient(base_url="https://api.openai.com/v1",
                            model=model, api_key="k", reasoning_effort=effort)

    def test_config_key_resolves_for_openai(self):
        from agentchanti.config import Config
        cfg = Config({"reasoning_effort": "high"})
        self.assertEqual(cfg.OPENAI_REASONING_EFFORT, "high")

    def test_openai_section_outranks_top_level(self):
        from agentchanti.config import Config
        cfg = Config({"reasoning_effort": "high",
                      "openai": {"reasoning_effort": "low"}})
        self.assertEqual(cfg.OPENAI_REASONING_EFFORT, "low")

    def test_absent_by_default(self):
        from agentchanti.config import Config
        self.assertIsNone(Config({}).OPENAI_REASONING_EFFORT)

    def test_sent_for_a_reasoning_model(self):
        self.assertEqual(self._client()._effort(), "high")

    def test_withheld_from_a_non_reasoning_model(self):
        """Sending the parameter to a model that lacks it is a 400."""
        self.assertIsNone(self._client(model="gpt-4o-mini")._effort())

    def test_burn_downgrade_outranks_configured_effort(self):
        """The one-shot downgrade exists because the model just spent its
        whole budget thinking — honouring 'high' again would repeat it."""
        client = self._client()
        client._prepare_token_limit_retry()
        self.assertEqual(client._effort(), "low")
        # One-shot: the configured value applies again afterwards.
        self.assertEqual(client._effort(), "high")

    def test_reaches_the_chat_completions_payload(self):
        client = self._client()
        ok = MagicMock()
        ok.status_code = 200
        ok.url = "https://api.openai.com/v1/chat/completions"
        ok.json.return_value = {
            "choices": [{"message": {"content": "hi", "tool_calls": []},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        sent = {}

        def _post(url, **kwargs):
            sent.update(kwargs.get("json") or {})
            return ok

        with unittest.mock.patch(
                "agentchanti.llm.openai_client.requests.post", _post):
            client._chat([Message(role="user", content="hi")], tools=None)
        self.assertEqual(sent.get("reasoning_effort"), "high")


class TestResponsesTranslation(unittest.TestCase):
    """The two wire formats differ in ways that silently break tool loops."""

    def _client(self):
        from agentchanti.llm.openai_client import OpenAIClient
        return OpenAIClient(base_url="https://api.openai.com/v1",
                            model="m", api_key="k")

    def test_tool_call_and_result_become_top_level_items(self):
        from agentchanti.llm.chat_types import ToolCall
        items = self._client()._responses_input([
            Message(role="system", content="sys"),
            Message(role="user", content="go"),
            Message(role="assistant", content="thinking",
                    tool_calls=[ToolCall(name="ping",
                                         arguments={"x": 1}, id="fc_1")]),
            Message(role="tool", content="pong", tool_call_id="fc_1",
                    tool_name="ping"),
        ])
        self.assertEqual(items[0], {"role": "system", "content": "sys"})
        self.assertEqual(items[1], {"role": "user", "content": "go"})
        self.assertEqual(items[2], {"role": "assistant",
                                    "content": "thinking"})
        self.assertEqual(items[3], {"type": "function_call",
                                    "call_id": "fc_1", "name": "ping",
                                    "arguments": '{"x": 1}'})
        # Not a "tool" role, and linked by call_id not tool_call_id.
        self.assertEqual(items[4], {"type": "function_call_output",
                                    "call_id": "fc_1", "output": "pong"})

    def test_tool_call_without_an_id_gets_a_stable_one(self):
        from agentchanti.llm.chat_types import ToolCall
        items = self._client()._responses_input([
            Message(role="assistant", tool_calls=[ToolCall(name="ping")]),
        ])
        self.assertTrue(items[0]["call_id"])

    def test_parses_text_tool_calls_and_skips_reasoning_items(self):
        from agentchanti.llm.openai_client import OpenAIClient
        text, calls = OpenAIClient._parse_responses_output({
            "output": [
                {"type": "reasoning", "summary": ["hidden"]},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "a"},
                    {"type": "output_text", "text": "b"}]},
                {"type": "function_call", "call_id": "c1", "name": "f",
                 "arguments": '{"k": "v"}'},
            ]})
        self.assertEqual(text, "ab")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"k": "v"})

    def test_malformed_arguments_are_preserved_not_dropped(self):
        from agentchanti.llm.openai_client import OpenAIClient
        _, calls = OpenAIClient._parse_responses_output({
            "output": [{"type": "function_call", "call_id": "c1",
                        "name": "f", "arguments": "{not json"}]})
        self.assertEqual(calls[0].arguments, {"_raw": "{not json"})

    def test_empty_output_is_an_empty_response(self):
        from agentchanti.llm.openai_client import OpenAIClient
        text, calls = OpenAIClient._parse_responses_output({"output": []})
        self.assertEqual((text, calls), ("", []))


class TestTokenParamFallbackTrigger(unittest.TestCase):
    """The max_completion_tokens -> max_tokens fallback must be targeted.

    Firing it on *any* 400 meant an unrelated rejection was silently
    retried with the legacy parameter, and the reported error became
    "Unsupported parameter: 'max_tokens' ... Use 'max_completion_tokens'
    instead" — the exact opposite of the truth, with the real first error
    discarded. Cost a full debugging round to see through.
    """

    def test_fires_when_the_parameter_is_named(self):
        for body in (
            "{\"error\":{\"message\":\"Unsupported parameter: "
            "'max_completion_tokens' is not supported with this model.\"}}",
            "{\"error\":{\"message\":\"Unrecognized request argument "
            "supplied: max_completion_tokens\"}}",
        ):
            with self.subTest(body=body):
                self.assertTrue(_param_rejected(
                    _response(400, text=body), "max_completion_tokens"))

    def test_does_not_fire_on_an_unrelated_400(self):
        for body in (
            '{"error":{"message":"Invalid schema for function write_file"}}',
            '{"error":{"message":"context_length_exceeded"}}',
            '{"error":{"message":"Incorrect API key provided"}}',
            "",
        ):
            with self.subTest(body=body):
                self.assertFalse(_param_rejected(
                    _response(400, text=body), "max_completion_tokens"))

    def test_does_not_fire_when_named_without_a_rejection_word(self):
        """A body that merely mentions the parameter is not a rejection."""
        self.assertFalse(_param_rejected(
            _response(400, text='{"error":{"message":"max_completion_tokens '
                                'was 4096 and the prompt was too long"}}'),
            "max_completion_tokens"))


if __name__ == "__main__":
    unittest.main()

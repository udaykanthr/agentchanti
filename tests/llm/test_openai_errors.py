"""Error handling in the OpenAI-compatible client.

A model that 400'd on every tool-calling request took a whole pipeline
down, and the logs showed nothing but `400 Client Error: Bad Request for
url: ...` — the provider's explanation was thrown away, the request was
retried three times regardless, and the resulting LLMError escaped.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agentchanti.llm.base import LLMError, NonRetryableLLMError, \
    ToolsNotSupportedError
from agentchanti.llm.openai_client import (
    _looks_like_tools_rejection,
    _param_rejected,
    _raise_for_status_with_body,
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

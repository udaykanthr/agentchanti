"""Native tool calling for Gemini.

Gemini previously had ``NATIVE_CHAT = False``, so ``supports_tools()``
returned False, ``agent_loop_enabled()`` was False, and every CODE/TEST
step fell back to the classic generate -> review -> retry pipeline. The
whole agent-loop path — and every optimisation living in it — was
unreachable on a Gemini config.

Two wire-format details are load-bearing and were both found against the
live API rather than assumed:

* Gemini has no ``system`` role and only knows ``user``/``model``; tool
  results go back as a ``functionResponse`` part on a ``user`` turn, and
  its ``response`` must be an object, not a bare string.
* Gemini 3.x rejects a replayed ``functionCall`` whose
  ``thoughtSignature`` is absent: "Function call is missing a
  thought_signature ... required for tools to work correctly". It must be
  echoed back verbatim on the same part.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agentchanti.llm.chat_types import Message, ToolCall, ToolDef
from agentchanti.llm.gemini_client import GeminiClient


def _client(**kw):
    return GeminiClient(base_url="https://gen.example/v1beta",
                        model="gemini-3.6-flash", api_key="k", **kw)


def _response(status=200, payload=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.return_value = payload if payload is not None else {}
    return r


class TestToolSupportIsOn(unittest.TestCase):

    def test_native_chat_enables_the_agent_loop(self):
        self.assertTrue(GeminiClient.NATIVE_CHAT)
        self.assertTrue(_client().supports_tools())


class TestMessageSerialisation(unittest.TestCase):

    def test_system_is_hoisted_out_of_contents(self):
        """Gemini has no system role — it goes to systemInstruction."""
        system, contents = GeminiClient._system_and_contents([
            Message(role="system", content="SYS"),
            Message(role="user", content="hi"),
        ])
        self.assertEqual(system, "SYS")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")

    def test_multiple_system_messages_are_joined(self):
        system, _ = GeminiClient._system_and_contents([
            Message(role="system", content="A"),
            Message(role="system", content="B"),
        ])
        self.assertEqual(system, "A\n\nB")

    def test_assistant_becomes_model_role(self):
        _, contents = GeminiClient._system_and_contents(
            [Message(role="assistant", content="ok")])
        self.assertEqual(contents[0]["role"], "model")

    def test_tool_result_is_a_function_response_on_a_user_turn(self):
        _, contents = GeminiClient._system_and_contents([
            Message(role="tool", content="exit: success",
                    tool_name="run_command"),
        ])
        part = contents[0]["parts"][0]["functionResponse"]
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(part["name"], "run_command")
        # The API requires an object here; a bare string is a 400.
        self.assertIsInstance(part["response"], dict)
        self.assertEqual(part["response"]["result"], "exit: success")

    def test_an_assistant_tool_call_serialises_as_function_call(self):
        _, contents = GeminiClient._system_and_contents([
            Message(role="assistant", tool_calls=[
                ToolCall(name="write_file", arguments={"path": "a.py"})]),
        ])
        fc = contents[0]["parts"][0]["functionCall"]
        self.assertEqual(fc["name"], "write_file")
        self.assertEqual(fc["args"], {"path": "a.py"})

    def test_an_empty_assistant_turn_still_has_a_part(self):
        """Gemini rejects a content entry with no parts."""
        _, contents = GeminiClient._system_and_contents(
            [Message(role="assistant")])
        self.assertTrue(contents[0]["parts"])


class TestThoughtSignatureRoundTrip(unittest.TestCase):
    """Without this the SECOND turn of every loop 400s."""

    def test_the_signature_is_captured_from_the_response(self):
        payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"functionCall": {"name": "run_command",
                              "args": {"command": "ls"}},
             "thoughtSignature": "SIG123"}]}}]}
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(payload=payload)):
            r = _client()._chat([Message(role="user", content="go")],
                                tools=[ToolDef(name="run_command",
                                               description="d")])
        self.assertEqual(r.tool_calls[0].provider_state["thoughtSignature"],
                         "SIG123")

    def test_the_signature_is_echoed_back_when_replayed(self):
        _, contents = GeminiClient._system_and_contents([
            Message(role="assistant", tool_calls=[ToolCall(
                name="run_command", arguments={},
                provider_state={"thoughtSignature": "SIG123"})]),
        ])
        self.assertEqual(contents[0]["parts"][0]["thoughtSignature"], "SIG123")

    def test_absent_signature_adds_no_key(self):
        """Providers that do not use it must not gain a stray null field."""
        _, contents = GeminiClient._system_and_contents([
            Message(role="assistant",
                    tool_calls=[ToolCall(name="x", arguments={})]),
        ])
        self.assertNotIn("thoughtSignature", contents[0]["parts"][0])


class TestSchemaCleaning(unittest.TestCase):
    """Gemini 400s on JSON Schema keywords it does not implement."""

    def test_unsupported_keywords_are_stripped_recursively(self):
        cleaned = GeminiClient._clean_schema({
            "type": "object",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "path": {"type": "string", "default": "x"},
                "opts": {"type": "object", "additionalProperties": True},
            },
            "required": ["path"],
        })
        self.assertNotIn("additionalProperties", cleaned)
        self.assertNotIn("$schema", cleaned)
        self.assertNotIn("default", cleaned["properties"]["path"])
        self.assertNotIn("additionalProperties", cleaned["properties"]["opts"])
        # The parts Gemini needs survive.
        self.assertEqual(cleaned["type"], "object")
        self.assertEqual(cleaned["required"], ["path"])
        self.assertEqual(cleaned["properties"]["path"]["type"], "string")


class TestErrorHandling(unittest.TestCase):

    def test_a_tools_rejection_is_recoverable(self):
        """Must downgrade to the text path, not fail the step."""
        from agentchanti.llm.base import ToolsNotSupportedError
        payload = {"error": {"message": "Function calling is not supported "
                                        "for this model"}}
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(status=400, payload=payload)):
            with self.assertRaises(ToolsNotSupportedError):
                _client()._chat([Message(role="user", content="go")],
                                tools=[ToolDef(name="t", description="d")])

    def test_an_unrelated_400_keeps_the_provider_message(self):
        from agentchanti.llm.base import LLMError, ToolsNotSupportedError
        payload = {"error": {"message": "API key not valid"}}
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(status=400, payload=payload)):
            with self.assertRaises(LLMError) as ctx:
                _client()._chat([Message(role="user", content="go")],
                                tools=[ToolDef(name="t", description="d")])
        self.assertNotIsInstance(ctx.exception, ToolsNotSupportedError)
        self.assertIn("API key not valid", str(ctx.exception))
        self.assertIn("gemini-3.6-flash", str(ctx.exception))


class TestResponseParsing(unittest.TestCase):

    def test_text_and_tool_calls_together(self):
        payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "I will run it."},
            {"functionCall": {"name": "run_command", "args": {"command": "ls"}}},
        ]}}], "usageMetadata": {"promptTokenCount": 11,
                                "candidatesTokenCount": 5}}
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(payload=payload)):
            r = _client()._chat([Message(role="user", content="go")],
                                tools=[ToolDef(name="run_command",
                                               description="d")])
        self.assertEqual(r.text, "I will run it.")
        self.assertTrue(r.has_tool_calls)
        self.assertEqual(r.tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(r.stop_reason, "STOP")

    def test_no_candidates_is_an_empty_response(self):
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(payload={"candidates": []})):
            r = _client()._chat([Message(role="user", content="go")])
        self.assertTrue(r.is_empty)


if __name__ == "__main__":
    unittest.main()


class TestCacheAccounting(unittest.TestCase):
    """Gemini caches a repeated prefix automatically and reports the hit.

    Measured live: a repeated 24,011-token prompt came back with
    cachedContentTokenCount = 16,362 (68%). Not reading that field made
    every Gemini token look full-price and overstated a run's cost
    roughly threefold against the OpenAI client, which does report its
    cache hits.
    """

    def test_reads_the_cache_field(self):
        self.assertEqual(GeminiClient._cached_tokens(
            {"promptTokenCount": 24011, "cachedContentTokenCount": 16362}),
            16362)

    def test_absent_or_malformed_field_is_zero(self):
        self.assertEqual(GeminiClient._cached_tokens({}), 0)
        self.assertEqual(GeminiClient._cached_tokens(None), 0)
        self.assertEqual(
            GeminiClient._cached_tokens({"cachedContentTokenCount": "x"}), 0)

    def test_chat_reports_cached_tokens_to_the_tracker(self):
        payload = {"candidates": [{"finishReason": "STOP",
                                   "content": {"parts": [{"text": "ok"}]}}],
                   "usageMetadata": {"promptTokenCount": 1000,
                                     "candidatesTokenCount": 10,
                                     "cachedContentTokenCount": 700}}
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(payload=payload)), \
             patch("agentchanti.llm.gemini_client.token_tracker") as tt:
            _client()._chat([Message(role="user", content="go")])
        self.assertEqual(tt.record.call_args.kwargs["cached_tokens"], 700)


class TestAuthByHeader(unittest.TestCase):
    """The key used to travel as ?key=... in the URL.

    That put it into every request exception, proxy log and debug trace —
    it leaked into a traceback during development.
    """

    def test_key_is_sent_as_a_header(self):
        self.assertEqual(_client()._headers()["x-goog-api-key"], "k")

    def test_no_url_carries_the_key(self):
        captured = {}

        def fake_post(url, **kw):
            captured["url"] = url
            captured["headers"] = kw.get("headers") or {}
            return _response(payload={"candidates": []})

        with patch("agentchanti.llm.gemini_client.requests.post",
                   side_effect=fake_post):
            _client()._chat([Message(role="user", content="go")])
        self.assertNotIn("key=", captured["url"])
        self.assertIn("x-goog-api-key", captured["headers"])


class TestThinkingBurn(unittest.TestCase):
    """Gemini spends output tokens on hidden thoughts before any text.

    Measured live: a 200-token cap returned thoughtsTokenCount 190 and
    six visible tokens with finishReason MAX_TOKENS. At the real 16k cap
    that is an empty response. The base class already retries an empty
    response that hit the cap — but only if the provider records the stop
    reason, which Gemini never did, so the burn was invisible.
    """

    def test_max_tokens_is_recognised_as_a_cap_hit(self):
        c = _client()
        c._last_stop_reason = "MAX_TOKENS"
        self.assertTrue(c._generate_hit_token_limit())

    def test_a_clean_stop_is_not(self):
        c = _client()
        c._last_stop_reason = "STOP"
        self.assertFalse(c._generate_hit_token_limit())

    def test_the_retry_pins_a_thinking_budget(self):
        c = _client()
        self.assertNotIn("thinkingConfig", c._generation_config())
        c._prepare_token_limit_retry()
        self.assertEqual(
            c._generation_config()["thinkingConfig"]["thinkingBudget"], 512)

    def test_the_cap_latches_for_the_session(self):
        """A model that burned once burns again on the next request."""
        c = _client()
        c._prepare_token_limit_retry()
        c._prepare_token_limit_retry()
        self.assertEqual(
            c._generation_config()["thinkingConfig"]["thinkingBudget"], 512)

    def test_the_non_streaming_path_records_the_stop_reason(self):
        payload = {"candidates": [{"finishReason": "MAX_TOKENS",
                                   "content": {"parts": [{"text": ""}]}}]}
        c = _client()
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(payload=payload)):
            c._generate("hi")
        self.assertEqual(c._last_stop_reason, "MAX_TOKENS")
        self.assertTrue(c._generate_hit_token_limit())


class TestNonRetryableErrors(unittest.TestCase):
    """A malformed request does not become valid by being resent."""

    def test_a_400_is_non_retryable(self):
        from agentchanti.llm.base import LLMError, NonRetryableLLMError
        with patch("agentchanti.llm.gemini_client.requests.post",
                   return_value=_response(
                       status=400,
                       payload={"error": {"message": "API key not valid"}})):
            with self.assertRaises(NonRetryableLLMError):
                _client()._chat([Message(role="user", content="go")])
        self.assertTrue(issubclass(NonRetryableLLMError, LLMError))

    def test_429_and_5xx_stay_retryable(self):
        from agentchanti.llm.base import LLMError, NonRetryableLLMError
        for status in (429, 500, 503):
            with self.subTest(status=status):
                with patch("agentchanti.llm.gemini_client.requests.post",
                           return_value=_response(
                               status=status,
                               payload={"error": {"message": "busy"}})):
                    with self.assertRaises(LLMError) as ctx:
                        _client()._chat([Message(role="user", content="go")])
                self.assertNotIsInstance(ctx.exception, NonRetryableLLMError)

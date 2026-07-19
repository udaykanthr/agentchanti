"""Tests for OpenAI prompt-cache accounting (P1): the client reads
``usage.prompt_tokens_details.cached_tokens`` and the TokenTracker reports
gross vs. cached-net input and discounts the cached slice in cost."""

import unittest
from unittest.mock import patch

from agentchanti.cli_display import TokenTracker
from agentchanti.llm.chat_types import Message
from agentchanti.llm.openai_client import OpenAIClient


def _mock_post(json_data):
    """Minimal requests.post stand-in returning a 200 with json_data."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TokenTrackerCacheTest(unittest.TestCase):
    def test_records_cached_subset(self):
        t = TokenTracker()
        t.record(prompt_tokens=1000, completion_tokens=50, cached_tokens=600)
        self.assertEqual(t.total_prompt_tokens, 1000)
        self.assertEqual(t.total_cached_tokens, 600)
        # Full-price input = gross − cache hits.
        self.assertEqual(t.full_price_prompt_tokens, 400)

    def test_cached_is_clamped_to_prompt(self):
        # A cached count larger than prompt (never valid) must not push
        # full-price negative or inflate the cached total.
        t = TokenTracker()
        t.record(prompt_tokens=100, completion_tokens=0, cached_tokens=999)
        self.assertEqual(t.total_cached_tokens, 100)
        self.assertEqual(t.full_price_prompt_tokens, 0)

    def test_default_no_cache_is_backward_compatible(self):
        t = TokenTracker()
        t.record(prompt_tokens=200, completion_tokens=10)  # no cached arg
        self.assertEqual(t.total_cached_tokens, 0)
        self.assertEqual(t.full_price_prompt_tokens, 200)

    def test_cached_tokens_discount_cost(self):
        pricing = {"gpt-5": {"input": 1.0, "output": 1.0}}  # $1 / 1M
        full = TokenTracker(pricing=pricing)
        full.record(1000, 0, model_name="gpt-5-mini", cached_tokens=0)
        half = TokenTracker(pricing=pricing)
        half.record(1000, 0, model_name="gpt-5-mini", cached_tokens=1000)
        # 1000 fully cached tokens at the default 0.5 multiplier cost half.
        self.assertAlmostEqual(half.total_cost, full.total_cost * 0.5, places=9)

    def test_pricing_can_override_cached_multiplier(self):
        pricing = {"gpt-5": {"input": 1.0, "output": 1.0, "cached_input": 0.25}}
        t = TokenTracker(pricing=pricing)
        t.record(1000, 0, model_name="gpt-5-mini", cached_tokens=1000)
        self.assertAlmostEqual(t.total_cost, 1000 * (1.0 / 1_000_000) * 0.25,
                               places=9)


class OpenAICachedExtractionTest(unittest.TestCase):
    def _client(self):
        return OpenAIClient(base_url="https://api.openai.com/v1",
                            model="gpt-5-mini", api_key="k",
                            max_retries=1, retry_delay=0, stream=False)

    def test_extract_helper(self):
        self.assertEqual(OpenAIClient._cached_tokens(
            {"prompt_tokens_details": {"cached_tokens": 640}}), 640)
        # Absent field (non-OpenAI backends) → 0, not a crash.
        self.assertEqual(OpenAIClient._cached_tokens({"prompt_tokens": 10}), 0)
        self.assertEqual(OpenAIClient._cached_tokens(
            {"prompt_tokens_details": {}}), 0)

    @patch("agentchanti.llm.openai_client.token_tracker")
    @patch("agentchanti.llm.openai_client.requests.post")
    def test_chat_records_cached_tokens(self, mock_post, mock_tracker):
        mock_post.return_value = _mock_post({
            "choices": [{"message": {"content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2685, "completion_tokens": 116,
                      "prompt_tokens_details": {"cached_tokens": 2048}},
        })
        self._client().chat([Message(role="user", content="go")])
        # cached_tokens must be threaded into token_tracker.record().
        _, kwargs = mock_tracker.record.call_args
        self.assertEqual(kwargs.get("cached_tokens"), 2048)


if __name__ == "__main__":
    unittest.main()

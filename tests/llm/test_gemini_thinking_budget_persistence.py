"""Gemini's thinking cap is remembered across runs, like OpenAI's effort floor.

Gemini 3.x spends output tokens on hidden thoughts before any visible
text. The client already latched a 512-token ``thinkingBudget`` after the
first burn — but only for the session, so every run rediscovered it the
expensive way. Measured on a Pac-Man run: the very first call (intent
analysis) spent a 16,384 budget to return 655 visible tokens, hit
MAX_TOKENS, and retried — ~90s of wall clock and one wasted call, paid
again on every run.

The store is the same home cache the OpenAI floor uses, under a
``gemini:`` namespace so a numeric budget and an effort string can never
be read as one another.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agentchanti.llm import openai_client
from agentchanti.llm.gemini_client import GeminiClient
from agentchanti.llm.openai_client import load_effort_floor


def _client():
    return GeminiClient(base_url="https://example.invalid",
                        model="gemini-3.6-flash", api_key="k",
                        max_output_tokens=16384)


class GeminiThinkingBudgetPersistenceTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        path = os.path.join(self._dir.name, "effort_floors.json")
        p = patch.object(openai_client, "_effort_floor_store", lambda: path)
        p.start()
        self.addCleanup(p.stop)

    def test_a_fresh_model_starts_uncapped(self):
        self.assertIsNone(_client()._thinking_budget)

    def test_a_burn_is_remembered_for_the_next_run(self):
        first = _client()
        first._prepare_token_limit_retry()
        self.assertEqual(first._thinking_budget, 512)

        # A brand-new client stands in for the next run.
        self.assertEqual(_client()._thinking_budget, 512)

    def test_the_budget_is_namespaced_away_from_openai_efforts(self):
        _client()._prepare_token_limit_retry()
        self.assertIsNone(load_effort_floor("gemini-3.6-flash"))
        self.assertEqual(load_effort_floor("gemini:gemini-3.6-flash"), "512")

    def test_a_non_numeric_entry_is_ignored(self):
        """An OpenAI-style 'low' must never become a thinking budget."""
        openai_client.save_effort_floor("gemini:gemini-3.6-flash", "low")
        self.assertIsNone(_client()._thinking_budget)

    def test_a_remembered_budget_is_applied_to_requests(self):
        _client()._prepare_token_limit_retry()
        cfg = _client()._generation_config()
        self.assertEqual(cfg["thinkingConfig"]["thinkingBudget"], 512)


if __name__ == "__main__":
    unittest.main()

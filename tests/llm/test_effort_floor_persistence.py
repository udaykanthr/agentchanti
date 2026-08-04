"""Tests for the durable reasoning-effort floor.

A reasoning burn — the model spending its whole completion budget on
hidden reasoning and returning nothing — is a fact about the model at a
given effort, not about one run. The in-session latch alone still cost a
full completion cap to rediscover on every run (measured: 16,384
completion tokens with tool_calls=0 and no text, ~8% of a Pac-Man run),
so the latch is persisted per model and seeded back on the next client.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from agentchanti.llm import openai_client
from agentchanti.llm.openai_client import (
    OpenAIClient, load_effort_floor, save_effort_floor)


class EffortFloorStoreTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "effort_floors.json")
        p = patch.object(openai_client, "_effort_floor_store",
                         lambda: self.path)
        p.start()
        self.addCleanup(p.stop)

    def test_unknown_model_has_no_floor(self):
        self.assertIsNone(load_effort_floor("gpt-5.4-mini"))

    def test_roundtrip(self):
        save_effort_floor("gpt-5.4-mini", "low")
        self.assertEqual(load_effort_floor("gpt-5.4-mini"), "low")

    def test_floor_is_per_model(self):
        save_effort_floor("gpt-5.4-mini", "low")
        self.assertIsNone(load_effort_floor("gpt-5.6-terra"))

    def test_expired_entry_is_ignored(self):
        """A pinned model must get retested, not be stuck on 'low' forever."""
        stale = time.time() - openai_client._EFFORT_FLOOR_TTL_SECONDS - 1
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"m": {"floor": "low", "ts": stale}}, fh)
        self.assertIsNone(load_effort_floor("m"))

    def test_corrupt_store_is_not_fatal(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertIsNone(load_effort_floor("m"))
        save_effort_floor("m", "low")          # must not raise
        self.assertEqual(load_effort_floor("m"), "low")

    def test_unwritable_store_is_not_fatal(self):
        with patch.object(openai_client, "_effort_floor_store",
                          lambda: os.path.join(self._dir.name, "no", "x.json")):
            with patch("os.makedirs", side_effect=OSError("denied")):
                save_effort_floor("m", "low")   # must not raise


class EffortFloorClientTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "effort_floors.json")
        p = patch.object(openai_client, "_effort_floor_store",
                         lambda: self.path)
        p.start()
        self.addCleanup(p.stop)

    def _client(self, model="gpt-5.4-mini", effort="high"):
        return OpenAIClient(base_url="https://example.invalid/v1",
                            model=model, api_key="k",
                            reasoning_effort=effort)

    def test_burn_latches_and_persists(self):
        c = self._client()
        self.assertEqual(c._effort(), "high")
        c._prepare_token_limit_retry()
        self.assertEqual(load_effort_floor("gpt-5.4-mini"), "low")

    def test_next_client_starts_at_the_floor(self):
        """The whole point: run 2 must not re-pay the burn run 1 found."""
        self._client()._prepare_token_limit_retry()
        fresh = self._client()
        self.assertEqual(fresh._effort(), "low")

    def test_floor_does_not_leak_to_other_models(self):
        self._client()._prepare_token_limit_retry()
        other = self._client(model="gpt-5.6-terra")
        self.assertEqual(other._effort(), "high")

    def test_non_reasoning_model_never_latches(self):
        c = self._client(model="gpt-4o-mini")
        c._prepare_token_limit_retry()
        self.assertIsNone(load_effort_floor("gpt-4o-mini"))
        self.assertIsNone(c._effort())


if __name__ == "__main__":
    unittest.main()

"""Tests for the A/B benchmark harness's parsing helpers."""

import unittest

from benchmarks.run_ab import (
    _build_config,
    parse_loop_stats,
    parse_pipeline_claim,
    parse_total_tokens,
)
from benchmarks.tasks import TASKS


LOG_SUCCESS = """
18:29:43 [INFO] [AgentLoop] step 4 verified in 8 turn(s)
18:29:50 [INFO] [AgentLoop] session: 2 loop run(s), 14 total turns (avg 7.0), 0 recovery run(s), outcomes: {'verified': 2}
18:29:50 [INFO] Finished. Total tokens: 12,345 (sent 8,000 / recv 4,345)
"""

LOG_FAILURE = """
18:13:35 [INFO] Pipeline failed. Total tokens: 40630
"""


class TestParsers(unittest.TestCase):

    def test_tokens_with_commas(self):
        self.assertEqual(parse_total_tokens(LOG_SUCCESS), 12345)

    def test_tokens_last_match_wins(self):
        text = "Total tokens: 100\n...\nTotal tokens: 200"
        self.assertEqual(parse_total_tokens(text), 200)

    def test_tokens_missing(self):
        self.assertIsNone(parse_total_tokens("no totals here"))

    def test_pipeline_claim(self):
        self.assertTrue(parse_pipeline_claim(LOG_SUCCESS))
        self.assertFalse(parse_pipeline_claim(LOG_FAILURE))
        self.assertIsNone(parse_pipeline_claim("nothing conclusive"))

    def test_loop_stats(self):
        self.assertIn("2 loop run(s)", parse_loop_stats(LOG_SUCCESS))
        self.assertIsNone(parse_loop_stats(LOG_FAILURE))


class TestBuildConfig(unittest.TestCase):

    def test_overrides_existing_flags(self):
        base = "provider: openai\nagent_loop: true\nagent_loop_max_turns: 3\n"
        out = _build_config(base, agent_loop=False)
        self.assertIn("provider: openai", out)
        self.assertIn("agent_loop: false", out)
        self.assertIn("agent_loop_max_turns: 8", out)
        self.assertEqual(out.count("agent_loop:"), 1)

    def test_appends_when_absent(self):
        out = _build_config("provider: ollama\n", agent_loop=True)
        self.assertIn("agent_loop: true", out)


class TestTaskDefinitions(unittest.TestCase):

    def test_tasks_well_formed(self):
        ids = set()
        for t in TASKS:
            self.assertNotIn(t["id"], ids)
            ids.add(t["id"])
            self.assertTrue(t["task"])
            self.assertIsInstance(t["files"], dict)
            self.assertTrue(t["success_cmds"])


if __name__ == "__main__":
    unittest.main()

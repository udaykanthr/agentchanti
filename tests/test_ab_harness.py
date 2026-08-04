"""Tests for the A/B benchmark harness's parsing helpers."""

import unittest

from benchmarks.run_ab import (
    _build_config,
    parse_loop_stats,
    parse_pipeline_claim,
    parse_token_breakdown,
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


class TestTokenBreakdown(unittest.TestCase):
    """Totals hide what moves cost: cached input bills at a discount, so
    full-price and completion are the comparable numbers across runs."""

    # Real summary lines from a Pac-Man A/B pair.
    OK = ("Finished. Total tokens: 229288 "
          "(sent=174835 [cached=116224 (66%), full-price=58611], "
          "recv=54453)")
    FAILED = ("Pipeline failed. Total tokens: 65773 "
              "(sent=36696 [cached=1792 (4%), full-price=34904], "
              "recv=29077)")

    def test_parses_success_line(self):
        self.assertEqual(parse_token_breakdown(self.OK), {
            "sent": 174835, "cached": 116224, "cached_pct": 66,
            "full_price": 58611, "recv": 54453})

    def test_parses_failure_line(self):
        """The failure path reports the same breakdown — a failed run is
        the expensive one, so it must not degrade to a bare total."""
        self.assertEqual(parse_token_breakdown(self.FAILED), {
            "sent": 36696, "cached": 1792, "cached_pct": 4,
            "full_price": 34904, "recv": 29077})

    def test_no_cache_detail_means_nothing_was_cached(self):
        got = parse_token_breakdown(
            "Finished. Total tokens: 900 (sent=600, recv=300)")
        self.assertEqual(got["cached"], 0)
        self.assertEqual(got["full_price"], 600)
        self.assertEqual(got["recv"], 300)

    def test_absent_line_yields_nones(self):
        got = parse_token_breakdown("nothing to see here")
        self.assertIsNone(got["sent"])
        self.assertIsNone(got["recv"])
        self.assertIsNone(got["cached_pct"])

    def test_last_run_wins(self):
        got = parse_token_breakdown(self.FAILED + "\n" + self.OK)
        self.assertEqual(got["sent"], 174835)


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

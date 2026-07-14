"""Tests for plan_mode: intent — planner emits goals + gates, no file bodies."""

import unittest
from unittest.mock import MagicMock

from agentchanti.config import Config
from agentchanti.agents.planner import (
    PlannerAgent,
    done_contradicted_by_briefing,
)


def _captured_prompt(plan_mode: str) -> str:
    planner = PlannerAgent("P", "Architect", "Plan tasks", MagicMock())
    planner.llm_client.generate_response = lambda prompt: prompt
    return planner.process("build an api", context="ctx",
                           plan_mode=plan_mode)


class TestPlanModeConfig(unittest.TestCase):

    def test_default_is_content(self):
        self.assertEqual(Config({}).PLAN_MODE, "content")

    def test_yaml_intent(self):
        self.assertEqual(Config({"plan_mode": "intent"}).PLAN_MODE, "intent")

    def test_invalid_value_falls_back(self):
        self.assertEqual(Config({"plan_mode": "yolo"}).PLAN_MODE, "content")

    def test_case_insensitive(self):
        self.assertEqual(Config({"plan_mode": "INTENT"}).PLAN_MODE, "intent")


class TestPlannerPromptModes(unittest.TestCase):

    def test_content_mode_keeps_inline_code_instructions(self):
        prompt = _captured_prompt("content")
        self.assertIn("<<<FIND>>>", prompt)
        self.assertIn("---file-content-end---", prompt)
        self.assertIn("MANDATORY", prompt)
        self.assertNotIn("NO inline code (intent mode)", prompt)

    def test_default_mode_is_content(self):
        planner = PlannerAgent("P", "Architect", "Plan tasks", MagicMock())
        planner.llm_client.generate_response = lambda prompt: prompt
        prompt = planner.process("build an api", context="ctx")
        self.assertIn("<<<FIND>>>", prompt)

    def test_intent_mode_bans_inline_code(self):
        prompt = _captured_prompt("intent")
        self.assertNotIn("<<<FIND>>>", prompt)
        self.assertNotIn("MANDATORY", prompt)
        self.assertIn("NO inline code (intent mode)", prompt)
        self.assertIn("DO NOT include file contents", prompt)

    def test_intent_mode_requires_verify(self):
        prompt = _captured_prompt("intent")
        self.assertIn("verify: is REQUIRED", prompt)
        self.assertIn(
            "Every CODE/TEST step has a verify: line", prompt)

    def test_intent_example_has_no_content_blocks(self):
        prompt = _captured_prompt("intent")
        self.assertIn("==PLAN==", prompt)
        self.assertIn("verify: npm test --silent", prompt)
        # Shared rules survive in both modes
        self.assertIn("Tests import ONLY frameworks", prompt)
        self.assertIn("QUALITY CHECKLIST", prompt)

    def test_shared_metadata_reference_in_both_modes(self):
        for mode in ("content", "intent"):
            prompt = _captured_prompt(mode)
            self.assertIn("LINE REFERENCE", prompt)
            self.assertIn("verify: <shell command>", prompt)
            self.assertIn("imported_by:", prompt)

    def test_done_requires_zero_changes_in_both_modes(self):
        # Regression: the planner emitted ==DONE== while DESCRIBING the
        # required one-line fix in the reason — nothing executed it.
        for mode in ("content", "intent"):
            prompt = _captured_prompt(mode)
            self.assertIn("ZERO changes are required", prompt)
            self.assertIn("nothing executes it", prompt)

    def test_pinned_url_test_rule_in_both_modes(self):
        # Regression: task pinned /dashboard/, app served the dashboard
        # elsewhere, generated tests never asserted the pinned path.
        for mode in ("content", "intent"):
            prompt = _captured_prompt(mode)
            self.assertIn("MUST assert those exact paths", prompt)

    def test_no_placeholder_rule_in_both_modes(self):
        # Regression: a plan emitted `cd <project_name>` in every CMD and
        # verify line; the resolver guessed 'react-app' for a Django task.
        for mode in ("content", "intent"):
            prompt = _captured_prompt(mode)
            self.assertIn("NEVER emit angle-bracket placeholders", prompt)
            self.assertIn("NEVER activate a virtualenv", prompt)


class TestAgentModelOverrides(unittest.TestCase):
    """Regression: a hardcoded allowlist dropped models.escalation —
    the user configured it correctly and it silently never fired."""

    def test_escalation_key_survives_load(self):
        cfg = Config({"models": {"escalation": "gpt-5.4"}})
        self.assertEqual(cfg.get_agent_model("escalation"), "gpt-5.4")

    def test_intent_and_analyser_survive(self):
        cfg = Config({"models": {"intent": "a", "analyser": "b",
                                 "coder": "c"}})
        self.assertEqual(cfg.get_agent_model("intent"), "a")
        self.assertEqual(cfg.get_agent_model("analyser"), "b")
        self.assertEqual(cfg.get_agent_model("coder"), "c")

    def test_case_insensitive_keys(self):
        cfg = Config({"models": {"Escalation": "gpt-5.4"}})
        self.assertEqual(cfg.get_agent_model("escalation"), "gpt-5.4")

    def test_unconfigured_returns_none(self):
        self.assertIsNone(Config({}).get_agent_model("escalation"))


class TestDoneContradictionGuard(unittest.TestCase):
    """Deterministic ==DONE== rejection — prompt rules lost this fight
    twice (the DONE reason literally described the required fix)."""

    def test_agent_directive_contradicts_done(self):
        briefing = ("Task summary: fix the failing tests\n"
                    "Agent directive: Change the body of `calc.py`'s "
                    "`add(a, b)` from `return a - b` to `return a + b` "
                    "and do nothing else\n"
                    "New packages: NONE\n")
        hit = done_contradicted_by_briefing(briefing)
        self.assertIsNotNone(hit)
        self.assertIn("Agent directive:", hit)
        self.assertIn("return a + b", hit)

    def test_required_changes_contradicts_done(self):
        briefing = "Required changes: add LOGIN_URL to settings.py\n"
        self.assertIn("LOGIN_URL", done_contradicted_by_briefing(briefing))

    def test_noop_directives_allow_done(self):
        for value in ("NONE", "none", "No changes needed", "N/A"):
            briefing = f"Agent directive: {value}\nRequired changes: NONE\n"
            self.assertIsNone(done_contradicted_by_briefing(briefing),
                              value)

    def test_already_satisfied_allows_done(self):
        briefing = ("Already satisfied: Yes\n"
                    "Agent directive: Change add to return a + b\n")
        self.assertIsNone(done_contradicted_by_briefing(briefing))

    def test_empty_and_non_string_allow_done(self):
        self.assertIsNone(done_contradicted_by_briefing(""))
        self.assertIsNone(done_contradicted_by_briefing(None))


if __name__ == "__main__":
    unittest.main()

"""Tests for plan_mode: intent — planner emits goals + gates, no file bodies."""

import unittest
from unittest.mock import MagicMock

from agentchanti.config import Config
from agentchanti.agents.planner import PlannerAgent


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

    def test_no_placeholder_rule_in_both_modes(self):
        # Regression: a plan emitted `cd <project_name>` in every CMD and
        # verify line; the resolver guessed 'react-app' for a Django task.
        for mode in ("content", "intent"):
            prompt = _captured_prompt(mode)
            self.assertIn("NEVER emit angle-bracket placeholders", prompt)
            self.assertIn("NEVER activate a virtualenv", prompt)


if __name__ == "__main__":
    unittest.main()

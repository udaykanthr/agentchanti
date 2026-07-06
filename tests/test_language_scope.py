"""Tests for language scope guards: the planner's LANGUAGE CONSTRAINT and
the intent agent's off-language KB-topic filter."""

import unittest
from unittest.mock import MagicMock

from agentchanti.agents.intent import (
    _filter_kb_topics_language,
    _split_top_level_commas,
)
from agentchanti.agents.planner import PlannerAgent


SPEC = (
    "Task type: FEATURE\n"
    "Goal: print a symmetric sequence without loops\n"
    "KB topics: Python (recursion, sys.setrecursionlimit), "
    "C++ (recursion and stack limitations), Error handling\n"
    "Create: peak_sequence.py\n"
)


class TestSplitTopLevelCommas(unittest.TestCase):

    def test_commas_inside_parens_preserved(self):
        self.assertEqual(
            _split_top_level_commas(
                "Python (recursion, sys.setrecursionlimit), C++ (stack)"),
            ["Python (recursion, sys.setrecursionlimit)", "C++ (stack)"])

    def test_plain_list(self):
        self.assertEqual(_split_top_level_commas("a, b, c"), ["a", "b", "c"])


class TestKbTopicsLanguageFilter(unittest.TestCase):

    def test_drops_foreign_language_topic(self):
        result = _filter_kb_topics_language(SPEC, "python")
        self.assertNotIn("C++", result)
        self.assertIn("Python (recursion, sys.setrecursionlimit)", result)
        self.assertIn("Error handling", result)  # generic topic kept
        self.assertIn("Create: peak_sequence.py", result)  # rest of spec intact

    def test_keeps_own_language(self):
        result = _filter_kb_topics_language(
            "KB topics: Python (recursion)", "python")
        self.assertIn("Python (recursion)", result)

    def test_all_foreign_becomes_none(self):
        result = _filter_kb_topics_language(
            "KB topics: Java (streams), C++ (templates)", "python")
        self.assertIn("KB topics: none", result)

    def test_js_ts_family_not_dropped(self):
        result = _filter_kb_topics_language(
            "KB topics: JavaScript (promises), TypeScript (generics)",
            "typescript")
        self.assertIn("JavaScript (promises)", result)
        self.assertIn("TypeScript (generics)", result)

    def test_no_language_is_noop(self):
        self.assertEqual(_filter_kb_topics_language(SPEC, None), SPEC)

    def test_no_topics_line_is_noop(self):
        spec = "Task type: FEATURE\nGoal: something\n"
        self.assertEqual(_filter_kb_topics_language(spec, "python"), spec)

    def test_language_name_inside_word_not_matched(self):
        # "Go" must not match inside "Google"; "java" not inside "javascript"
        result = _filter_kb_topics_language(
            "KB topics: Google Maps API, JavaScript modules", "javascript")
        self.assertIn("Google Maps API", result)
        self.assertIn("JavaScript modules", result)


class TestPlannerLanguageConstraint(unittest.TestCase):

    def _planner(self):
        llm = MagicMock()
        llm.generate_response.return_value = "==PLAN==\n==END=="
        return PlannerAgent(name="planner", role="r", goal="g",
                            llm_client=llm), llm

    def test_constraint_present_when_language_known(self):
        planner, llm = self._planner()
        planner.process("write a function", language="python")
        prompt = llm.generate_response.call_args[0][0]
        self.assertIn("LANGUAGE CONSTRAINT (HARD RULE)", prompt)
        self.assertIn("Implement ONLY in Python", prompt)
        self.assertIn("g++", prompt)  # toolchain examples named

    def test_constraint_absent_without_language(self):
        planner, llm = self._planner()
        planner.process("write a function")
        prompt = llm.generate_response.call_args[0][0]
        self.assertNotIn("LANGUAGE CONSTRAINT", prompt)


if __name__ == "__main__":
    unittest.main()

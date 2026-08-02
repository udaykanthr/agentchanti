"""Framework rules ship only for the language that can use them.

A Pygame task was being asked to reason about Django's LogoutView GET/POST
semantics and React Testing Library's within() scoping — rules it cannot
possibly apply.

NOTE ON SCOPE: this gating is a correctness change, not a fix for the
minimax-m3 reasoning burn.  Measured directly: the gated Python prompt
still burned 16,384 tokens with zero visible output, and a prompt with
ALL rules removed burned too.  Prompt size is not the trigger.  Do not
let this test's existence imply otherwise.
"""

from __future__ import annotations

import unittest

from agentchanti.agents.planner import PlannerAgent

TASK = "Build a Pac-Man clone using Python and Pygame with a tile maze."

DJANGO_MARKER = "LogoutView"
REACT_MARKER = "@testing-library/react"
COMPONENT_MARKER = "Leaf components BEFORE parents"


class _Stub:
    max_output_tokens = 16384

    def __init__(self):
        self.prompt = ""

    def generate_response(self, prompt):
        self.prompt = prompt
        return "1. step"


def _prompt(language, plan_mode="intent"):
    stub = _Stub()
    PlannerAgent("Planner", "Planner", "plan", stub).process(
        TASK, language=language, plan_mode=plan_mode)
    return stub.prompt


class TestLanguageGating(unittest.TestCase):

    def test_python_gets_django_rules_but_not_react(self):
        p = _prompt("python")
        self.assertIn(DJANGO_MARKER, p)
        self.assertNotIn(REACT_MARKER, p)
        self.assertNotIn(COMPONENT_MARKER, p)

    def test_javascript_gets_react_rules_but_not_django(self):
        p = _prompt("javascript")
        self.assertIn(REACT_MARKER, p)
        self.assertIn(COMPONENT_MARKER, p)
        self.assertNotIn(DJANGO_MARKER, p)

    def test_a_test_runner_suffix_resolves_to_its_base_language(self):
        """language.py emits 'typescript:vitest' — the suffix must not
        defeat the match and silently drop every frontend rule."""
        p = _prompt("typescript:vitest")
        self.assertIn(REACT_MARKER, p)
        self.assertIn(COMPONENT_MARKER, p)

    def test_go_and_rust_get_neither(self):
        for lang in ("go", "rust"):
            p = _prompt(lang)
            self.assertNotIn(DJANGO_MARKER, p, lang)
            self.assertNotIn(REACT_MARKER, p, lang)
            self.assertNotIn(COMPONENT_MARKER, p, lang)

    def test_an_unknown_language_keeps_everything(self):
        """Dropping a rule that might apply is worse than carrying one
        that does not — an unknown language must not silently lose rules."""
        for lang in (None, ""):
            p = _prompt(lang)
            self.assertIn(DJANGO_MARKER, p)
            self.assertIn(REACT_MARKER, p)
            self.assertIn(COMPONENT_MARKER, p)

    def test_gating_applies_in_content_mode_too(self):
        p = _prompt("go", plan_mode="content")
        self.assertNotIn(DJANGO_MARKER, p)
        self.assertNotIn(REACT_MARKER, p)

    def test_the_shared_rules_survive_gating(self):
        """Only framework-specific blocks are gated; the general step
        rules and output format must be present for every language."""
        for lang in (None, "python", "go", "javascript"):
            p = _prompt(lang)
            self.assertIn("STEP RULES (CRITICAL)", p, lang)
            self.assertIn("OUTPUT FORMAT", p, lang)
            self.assertIn("QUALITY CHECKLIST", p, lang)


if __name__ == "__main__":
    unittest.main()

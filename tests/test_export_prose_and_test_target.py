"""Two small defects from hello-world runs on a local model.

1. `exports:` is documented as `exports: <Symbol1>, <Symbol2>` (planner.py),
   but a weaker planner answers in English:

       exports: main function that prints "Hello World" when executed

   `_export_satisfied` compared that whole sentence against the file's
   symbol set, found nothing, and warned — while the file really did export
   `main`. Both export warnings in an otherwise clean, passing run were
   wrong. That function's own docstring already states the cost: "a warning
   that is always wrong is worse than none, because it trains the reader to
   skip the line that will one day be right."

2. `TesterAgent.process` was never told the plan's declared target, so its
   OUTPUT FORMAT showed an invented path built from framework conventions
   (`tests/test_example.py`). A plan declaring `target: test_hello.py` got
   `tests/test_hello_world.py` — conventional, and invisible to the step's
   own gate `pytest test_hello.py`. Three diagnosis rounds and a halted run,
   while `pytest tests/test_hello_world.py` passed every time.
"""

import unittest
from unittest.mock import MagicMock

# Aliased: a module-level name starting with "Test" makes pytest try to
# collect the agent class itself and emit a PytestCollectionWarning.
from agentchanti.agents.tester import TesterAgent as _TesterAgent
from agentchanti.orchestrator.plan_graph import _export_satisfied


class ProseExportsTest(unittest.TestCase):
    def test_the_two_false_positives_from_the_clean_run(self):
        self.assertTrue(_export_satisfied(
            'main function that prints "Hello World" when executed', {"main"}))
        self.assertTrue(_export_satisfied(
            'test that runs hello_world.py and checks for "Hello World"',
            {"test_hello_world"}))

    def test_no_exports_spelled_any_way_is_not_a_missing_symbol(self):
        """A later run warned twice on `exports: (none)` — the planner
        answering "nothing" instead of omitting the line, and the check
        hunting for a symbol literally called "(none)"."""
        for spec in ("(none)", "none", "None", "(None)", "n/a", "N/A",
                     "-", "--", "()", "nothing", "no exports", "*"):
            with self.subTest(spec=spec):
                self.assertTrue(_export_satisfied(spec, {"main", "run"}))

    def test_a_real_symbol_named_none_still_matches_exactly(self):
        """The no-declaration check sits after the exact match for this."""
        self.assertTrue(_export_satisfied("none", {"none", "other"}))

    def test_a_bare_name_that_is_genuinely_absent_still_warns(self):
        """The check must keep its teeth for the case it was built for."""
        self.assertFalse(_export_satisfied("startServer", {"app"}))
        self.assertFalse(_export_satisfied("Footer", {"Header", "Nav"}))

    def test_existing_behaviour_is_unchanged(self):
        for spec, actual in (
            ("app", {"app", "startServer"}),          # exact
            ("default Footer", {"Footer", "default"}),  # default-prefixed
            ("Footer as default", {"default"}),       # as-default
            ("Footer", {"default"}),                  # only a default export
            ("", set()),                              # nothing declared
        ):
            with self.subTest(spec=spec):
                self.assertTrue(_export_satisfied(spec, actual))


class TesterHonoursDeclaredTargetTest(unittest.TestCase):
    def _prompt(self, **kwargs):
        llm = MagicMock()
        llm.generate_response.return_value = ""
        _TesterAgent("Tester", "tester", "write tests", llm).process(
            "Create a test script", context="", language="python", **kwargs)
        return llm.generate_response.call_args.args[0]

    def test_the_declared_path_is_the_one_demanded(self):
        prompt = self._prompt(target_files=["test_hello.py"])
        self.assertIn("#### [FILE]: test_hello.py", prompt)

    def test_the_invented_example_path_is_gone(self):
        """It is what the tester copied, and why the gate could not see it."""
        self.assertNotIn("test_example.py",
                         self._prompt(target_files=["test_hello.py"]))

    def test_shared_strict_rules_survive_both_branches(self):
        """They carry the 'only test files / forward slashes' constraints."""
        self.assertIn("STRICT RULES",
                      self._prompt(target_files=["test_hello.py"]))
        self.assertIn("STRICT RULES", self._prompt())

    def test_windows_paths_are_normalised(self):
        prompt = self._prompt(target_files=[r"tests\sub\test_a.py"])
        self.assertIn("#### [FILE]: tests/sub/test_a.py", prompt)

    def test_no_declared_target_keeps_the_conventional_example(self):
        prompt = self._prompt()
        self.assertIn("test_example.py", prompt)

    def test_blank_targets_are_ignored(self):
        self.assertIn("test_example.py", self._prompt(target_files=["", "  "]))


if __name__ == "__main__":
    unittest.main()

"""A contract must not judge documentation it could only guess at.

Measured 2026-09-13, "create a 2 snake game in python include production
ready features". Every gate green, ghost 23/24 holding with 0 violated,
the plan's own suite passing — and the run exited 1 on the seeded contract.
It had two tests. The behaviour test launched the game, checked the
high-score file was created, and checked recovery from a corrupted one: it
PASSED. The other read README.md and made eleven wording assertions, one
of which was::

    self.assertRegex(readme, re.compile(r"python\\s*(?:>=|3\\.10|3\\.1[0-9])"))

over a README reading ``- Python **3.10 or newer**``. The Markdown bold sits
between the two words, so the regex cannot match a document that states the
requirement exactly. The source-grep screen already refused the same defect
aimed at the program's own text; documentation was one file over and
unguarded, and worse, because the contract predates the README and must
guess both its words and its markup.
"""
import os
import re
import textwrap
import unittest

from agentchanti.orchestrator.acceptance_seed import (
    _PROMPT, SEED_BASENAME, reset_contract_repairs, seed_acceptance_tests,
    seed_state, verify_contract_runs,
)
from agentchanti.orchestrator.seed_strength import (
    documentation_grep_reason, documentation_reading_tests,
    weak_contract_reason,
)
from tests.orchestrator.test_contract_runnability import (
    CRASHING, PROJECT, REPAIRED, RealExecutor, ScriptedClient,
    _write_contract,
)

import ast

# Reduced from the measured contract: its two tests, shape preserved.
MEASURED = textwrap.dedent('''
    import re
    import unittest
    from pathlib import Path


    class SnakeGameAcceptanceTests(unittest.TestCase):
        def test_required_project_files_dependency_and_documentation_exist(self):
            project_root = Path(__file__).resolve().parent
            readme = (project_root / "README.md").read_text(encoding="utf-8").lower()
            self.assertIn("python snake_game.py", readme)
            self.assertIn("wasd", readme)
            self.assertRegex(readme, re.compile(r"python\\s*(?:>=|3\\.10|3\\.1[0-9])"))

        def test_game_rules(self):
            import snake_game
            game = snake_game.SnakeGame()
            self.assertEqual(game.score, 0)
            game.step()
            self.assertEqual(len(game.snake), 3)
''')

BEHAVIOUR_ONLY = textwrap.dedent('''
    import unittest


    class SnakeGameAcceptanceTests(unittest.TestCase):
        def test_game_rules(self):
            import snake_game
            game = snake_game.SnakeGame()
            self.assertEqual(game.score, 0)
            game.step()
            self.assertEqual(len(game.snake), 3)
''')


def _tree(src):
    return ast.parse(textwrap.dedent(src))


class TheMeasuredDefect(unittest.TestCase):

    def test_the_regex_really_cannot_match_a_correct_readme(self):
        """Pins the incident, so the rationale cannot quietly rot."""
        line = "- python **3.10 or newer**"
        pattern = re.compile(r"python\s*(?:>=|3\.10|3\.1[0-9])")
        self.assertIsNone(pattern.search(line))
        self.assertIsNotNone(pattern.search(line.replace("**", "")))

    def test_the_measured_contract_is_caught(self):
        reason = documentation_grep_reason(MEASURED)
        self.assertIsNotNone(reason)
        self.assertIn("test_required_project_files", reason)

    def test_only_the_documentation_test_is_named(self):
        """The behaviour test beside it is real evidence and not accused."""
        self.assertEqual(
            documentation_reading_tests(_tree(MEASURED)),
            ["test_required_project_files_dependency_and_documentation_exist"])

    def test_a_behaviour_contract_is_left_alone(self):
        self.assertIsNone(documentation_grep_reason(BEHAVIOUR_ONLY))


class WhatCountsAsReadingDocumentation(unittest.TestCase):

    def _flagged(self, body):
        src = ("import unittest\nfrom pathlib import Path\n"
               "class T(unittest.TestCase):\n    def test_x(self):\n"
               + textwrap.indent(textwrap.dedent(body), "        "))
        return documentation_grep_reason(src) is not None

    def test_every_way_of_reading_a_document(self):
        for body in (
            'text = Path("README.md").read_text()\nself.assertIn("a", text)',
            'readme = Path("docs") / "USAGE.md"\n'
            'self.assertIn("a", readme.read_text())',
            'with open("CHANGELOG.rst") as fh:\n'
            '    self.assertIn("1.0", fh.read())',
            'self.assertIn("a", open("README").read())',
            'p = Path("README.txt")\nq = p\nself.assertIn("a", q.read_text())',
        ):
            with self.subTest(body=body[:40]):
                self.assertTrue(self._flagged(body))

    def test_existence_is_not_reading(self):
        """A task that asks for docs may fairly check they were written."""
        for body in (
            'self.assertTrue(Path("README.md").is_file())',
            'self.assertGreater(Path("README.md").stat().st_size, 0)',
            'self.assertTrue(os.path.exists("README.md"))',
        ):
            with self.subTest(body=body[:40]):
                self.assertFalse(self._flagged(body))

    def test_reading_what_the_program_wrote_is_behaviour(self):
        """A save-file or a manifest is not prose."""
        for body in (
            'self.assertIn("score", Path("data/high_score.json").read_text())',
            'self.assertIn("pygame", Path("requirements.txt").read_text())',
            'with open("scores.txt") as fh:\n'
            '    self.assertEqual(fh.read(), "10")',
        ):
            with self.subTest(body=body[:40]):
                self.assertFalse(self._flagged(body))


class StrengthNoLongerCountsWording(unittest.TestCase):

    def test_wording_checks_do_not_make_a_contract_strong(self):
        """Eleven README assertions looked like a strong contract."""
        docs_only = textwrap.dedent('''
            import unittest
            from pathlib import Path


            class T(unittest.TestCase):
                def test_docs(self):
                    readme = Path("README.md").read_text().lower()
                    self.assertEqual(readme.count("wasd"), 1)
                    self.assertRegex(readme, r"python 3")
                    self.assertIn("esc", readme)
        ''')
        self.assertIsNotNone(weak_contract_reason(docs_only))

    def test_a_users_suite_is_not_called_weak_for_having_a_readme_test(self):
        """weak_contract_reason also judges USER suites at verdict time.

        A user's README check was written against a README that existed,
        so the blind guess that makes this a defect does not apply; the
        documentation screen lives at seeding only.
        """
        self.assertIsNone(weak_contract_reason(MEASURED))

    def test_the_prompt_forbids_it_up_front(self):
        self.assertIn("NEVER assert on documentation wording", _PROMPT)


class Seeding(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.root = self._t.name

    def test_a_documentation_contract_is_repaired_before_it_is_written(self):
        client = ScriptedClient(MEASURED.strip(), BEHAVIOUR_ONLY.strip())
        path = seed_acceptance_tests("a snake game", self.root, client,
                                     language="python")
        self.assertIsNotNone(path)
        written = seed_state(os.path.join(self.root, SEED_BASENAME))[2]
        self.assertIsNone(documentation_grep_reason(written))
        self.assertEqual(len(client.prompts), 2)
        # The retry must see what it is revising, or "keep every behaviour
        # test you already had" asks about a file it cannot see.
        self.assertIn("documentation wording", client.prompts[1].lower())
        self.assertIn("def test_game_rules", client.prompts[1])

    def test_kept_with_a_warning_when_every_retry_still_reads_docs(self):
        """Refusing would leave nothing; its behaviour test is real."""
        client = ScriptedClient(MEASURED.strip(), MEASURED.strip(),
                                MEASURED.strip())
        with self.assertLogs("agentchanti", level="WARNING") as caught:
            path = seed_acceptance_tests("a snake game", self.root, client,
                                         language="python")
        self.assertIsNotNone(path, "a contract with real behaviour tests "
                                   "was discarded")
        self.assertTrue(any("documentation wording" in m
                            for m in caught.output))


class RunnabilityRepair(unittest.TestCase):

    def setUp(self):
        import pathlib
        import tempfile
        reset_contract_repairs()
        self.addCleanup(reset_contract_repairs)
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        with open(os.path.join(self._t.name, "project.py"), "w",
                  encoding="utf-8") as fh:
            fh.write(textwrap.dedent(PROJECT))
        self.root = pathlib.Path(self._t.name)

    def test_a_repair_that_starts_reading_the_readme_is_refused(self):
        with_readme = textwrap.dedent(REPAIRED).rstrip() + textwrap.indent(
            textwrap.dedent('''

                def test_docs(self):
                    from pathlib import Path
                    self.assertIn("snake", Path("README.md").read_text())
            '''), "    ")
        # A README that satisfies the wording check, so the candidate would
        # RUN GREEN — the refusal must come from the screen, not a crash.
        (self.root / "README.md").write_text("A snake game.\n",
                                             encoding="utf-8")
        _write_contract(self.root, CRASHING)
        client = ScriptedClient(with_readme.strip(),
                                textwrap.dedent(REPAIRED).strip())

        result = verify_contract_runs(RealExecutor(self.root), str(self.root),
                                      client, "a snake game")

        self.assertIsNotNone(result)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("documentation wording", client.prompts[1])
        on_disk = seed_state(os.path.join(str(self.root), SEED_BASENAME))[2]
        self.assertIsNone(documentation_grep_reason(on_disk))


if __name__ == "__main__":
    unittest.main()

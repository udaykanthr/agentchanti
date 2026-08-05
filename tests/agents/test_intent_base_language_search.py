"""A bare language must not trigger a setup-guide web search.

The blank-project pre-seed asks the model for "framework/library names
that need a setup guide" and routinely gets the language back too — a
Pac-Man task returned ``['Python', 'Pygame']``. Each entry costs a
Perplexity fetch plus a summarisation LLM call: measured at prompt 2012 +
completion 691 (~2.7k tokens) and ~10s for "Python", to be told to run
`pip install`. The plan's own CMD step already creates the venv.

Frameworks stay searched — their scaffolds and config-file formats are
exactly what the model's training data goes stale on.
"""

from __future__ import annotations

import unittest

from agentchanti.agents.intent import _BASE_LANGUAGES


class TestBaseLanguageSetupSearchSkipped(unittest.TestCase):

    @staticmethod
    def _needing_web(techs):
        """Mirrors the filter applied in intent.py."""
        return [t for t in techs
                if t.strip().lower() not in _BASE_LANGUAGES]

    def test_the_observed_pair_drops_only_the_language(self):
        self.assertEqual(self._needing_web(["Python", "Pygame"]), ["Pygame"])

    def test_common_languages_are_all_recognised(self):
        for lang in ("Python", "JavaScript", "TypeScript", "Java", "Go",
                     "Rust", "Ruby", "PHP", "C#", "Node.js"):
            self.assertEqual(
                self._needing_web([lang]), [],
                f"{lang} is a language, not a framework")

    def test_frameworks_are_still_searched(self):
        for fw in ("Pygame", "Django", "React", "Vite", "Tailwind",
                   "FastAPI", "Next.js", "Spring"):
            self.assertEqual(
                self._needing_web([fw]), [fw],
                f"{fw} is a framework — its setup guide still earns its cost")

    def test_matching_ignores_case_and_padding(self):
        self.assertEqual(self._needing_web(["  PYTHON  ", " Go "]), [])

    def test_a_framework_sharing_a_language_name_is_not_dropped(self):
        """'python-dateutil' is a library; only the bare language matches."""
        self.assertEqual(self._needing_web(["python-dateutil"]),
                         ["python-dateutil"])


if __name__ == "__main__":
    unittest.main()

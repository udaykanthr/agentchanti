"""A language keyword must be a whole token, not a substring.

`detect_language_from_task` tested each keyword with a plain `in`, and its
answer outranks `detect_language()` — the function that reads the manifests
and source files actually on disk. So three letters inside an ordinary
English word decide the language of a project that is sitting right there.

Measured 2026-09-10 on an existing Panda3D project with thirteen `.py`
files and a `requirements.txt`::

    Task: Fix the ball is getting struck in the corners by changing the
          shapes of board, Also add more bouncing trigger objects
    Language: go (Go)
    Performing pre-execution baseline test analysis via go test ./...
    [Executor] Test runner `go` is not installed or not on PATH.

`gin` is a Go web framework, and it is inside "chan-GIN-g". The run then
spent an LLM call correcting itself (`LLM corrected language: go → python`)
after having already run the wrong baseline command and fed a wrong
directive ("Initialize or repair the test environment") into planning.

Surveyed across ten ordinary task strings, nine picked up a false language.
"""
import unittest

from agentchanti.language import detect_language_from_task


class SubstringMatchesAreRefused(unittest.TestCase):
    """Each of these is a real word that contains a real keyword."""

    CASES = [
        ("Fix the ball is getting struck in the corners by changing the "
         "shapes of board, Also add more bouncing trigger objects", "gin/changing"),
        ("refactor the engine so the login page loads faster", "gin/engine"),
        ("add a plugin system with margin and padding options", "gin/plugin"),
        ("imagine a smoother origin animation when logging in", "gin/imagine"),
        ("the user is frustrated by the crusty layout", "rust/frustrated"),
        ("render the graph nodes and edges with better spacing", "node/nodes"),
        ("invite a second player and show the invitation banner", "vite/invite"),
        ("the nested interest calculation is honest but slow", "nest/nested"),
        ("suggest a fix for the broken image grid", "none"),
    ]

    def test_ordinary_english_names_no_language(self):
        for task, why in self.CASES:
            with self.subTest(why=why):
                self.assertIsNone(
                    detect_language_from_task(task),
                    "%r matched via %s" % (task[:48], why))


class RealMentionsStillDetect(unittest.TestCase):
    """Or the fix would trade a false positive for a false negative."""

    CASES = [
        ("build a REST API with gin", "go"),
        ("create a go module for auth", "go"),
        ("use the echo framework", "go"),
        ("write a Django view", "python"),
        ("add pytest coverage", "python"),
        ("use pandas and numpy", "python"),
        ("a JavaScript module", "javascript"),
        ("build a Next.js page", "javascript"),
        ("scaffold with create-next-app", "javascript"),
        ("use vite for bundling", "javascript"),
        ("a NestJS service", "typescript"),
        ("write it in TypeScript", "typescript"),
        ("a Rust crate with cargo", "rust"),
        ("Spring Boot with maven", "java"),
        ("write it in C#", "csharp"),
        ("target .NET 8", "csharp"),
        ("use C++ and cmake", "cpp"),
    ]

    def test_whole_word_mentions_are_detected(self):
        for task, expected in self.CASES:
            with self.subTest(task=task):
                self.assertEqual(detect_language_from_task(task), expected)


class BoundaryShape(unittest.TestCase):
    """Punctuation-bearing keywords need alphanumeric bounds, not \\b."""

    def test_punctuation_keywords_survive(self):
        # `\b` would place its boundary between 'c' and '#', so a word-
        # boundary implementation silently loses these.
        for task, expected in (("port it to c#", "csharp"),
                               ("build with c++", "cpp"),
                               ("an asp.net endpoint", "csharp")):
            with self.subTest(task=task):
                self.assertEqual(detect_language_from_task(task), expected)

    def test_hyphen_is_a_boundary(self):
        """Needed by `create-next-app`; also makes `pip-free` a real match."""
        self.assertEqual(
            detect_language_from_task("scaffold with create-next-app"),
            "javascript")

    def test_case_is_ignored(self):
        self.assertEqual(detect_language_from_task("Use GOLANG here"), "go")

    def test_empty_and_none_are_safe(self):
        self.assertIsNone(detect_language_from_task(""))
        self.assertIsNone(detect_language_from_task(None))


if __name__ == "__main__":
    unittest.main()

"""C was absent from task detection entirely.

`_TASK_KEYWORDS` had a `cpp` entry and no `c` one, so
`detect_language_from_task("Build a todo manager in C")` returned None.
That matters more than it looks, because of what the caller does with it:

    detect_language_from_task(task) or detect_language()

A greenfield C task in an empty directory has no files for
`detect_language()` to read, so the answer was None all the way down — and
the acceptance seeder treats None as "fall back to Python". A C run would
therefore have been handed a *Python* contract, which is the sixth
per-language evidence gap reappearing one layer up from where it was
fixed.

WHY C IS PHRASES AND NOT A BARE LETTER

`detect_language_from_task` OUTRANKS `detect_language()` — the function
that reads the manifests and sources actually on disk. So a false positive
here decides the language of a project that is sitting right there, which
is exactly what `gin` inside "chan-gin-g" did to a Pygame run: wrong
baseline command, wrong directive into planning.

Surveyed 2026-09-30 over fifteen realistic prompts, a whole-token `c`
matched the four genuine C tasks and also "vitamin c", "plan c" and
"option c". Three false positives is far worse than the single `pip-free`
case the whole-token fix was accepted with, so C is matched by phrases
that are unambiguous in a coding task.
"""
import pytest

from agentchanti.language import _TASK_KEYWORDS, detect_language_from_task


class TestCIsDetected:

    @pytest.mark.parametrize("task", [
        "Build a production-ready command-line todo manager in C",
        "write a small program in C that reverses a string",
        "Create a C library for parsing INI files",
        "compile it with gcc and make",
        "use c99 features only",
        "a C compiler exercise",
        "keep it pure C, no dependencies",
        "link against libc only",
    ])
    def test_a_c_task_is_recognised(self, task):
        assert detect_language_from_task(task) == "c"


class TestItDoesNotStealOtherLanguages:
    """"in c" matches "in c++" too, because the boundary rule treats `+` as
    a boundary. Ordering is what settles it — cpp and csharp are consulted
    first — and insertion order is the only thing keeping that true, so it
    needs a test rather than a comment."""

    @pytest.mark.parametrize("task", [
        "build a todo CLI in C++ with cmake",
        "write a C++ program",
        "implement it in c++",
        "use cpp for the hot path",
    ])
    def test_cpp_still_wins(self, task):
        assert detect_language_from_task(task) == "cpp"

    @pytest.mark.parametrize("task", [
        "make me a C# web api with asp.net",
        "a .net service in c#",
        "write it in csharp",
    ])
    def test_csharp_still_wins(self, task):
        assert detect_language_from_task(task) == "csharp"

    def test_c_is_declared_after_cpp_and_csharp(self):
        """The mechanism, not just the outcome: first match wins, and the
        dict is ordered. Moving `c` above either of these would silently
        reassign every C++ and C# task."""
        order = list(_TASK_KEYWORDS)
        assert order.index("c") > order.index("cpp")
        assert order.index("c") > order.index("csharp")

    def test_no_bare_c_keyword(self):
        """The whole point. A bare token would match "vitamin c"."""
        assert "c" not in _TASK_KEYWORDS["c"]


class TestItDoesNotFireOnProse:
    """Each of these matched a whole-token bare `c` in the survey."""

    @pytest.mark.parametrize("task", [
        "add vitamin c tracking to the nutrition app",
        "implement plan c as a fallback when the primary fails",
        "use option c for the default configuration",
        "refactor the pipeline to cache responses",
        "Fix the ball getting stuck in the corners by changing the board",
    ])
    def test_prose_does_not_select_c(self, task):
        assert detect_language_from_task(task) != "c"

    @pytest.mark.parametrize("task,expected", [
        ("create a 2 snake game in python with production features", "python"),
        ("Build a command-line todo manager in Java", "java"),
        ("a react responsive project as FE and express js API as BE", "javascript"),
        ("build a todo CLI in Go using go module layout", "go"),
        ("write a todo manager in rust with cargo", "rust"),
    ])
    def test_other_languages_are_unaffected(self, task, expected):
        assert detect_language_from_task(task) == expected


class TestTheSeedingConsequence:
    """The reason this file exists at all."""

    def test_a_greenfield_c_task_reaches_the_c_seeder(self, tmp_path):
        from agentchanti.orchestrator.acceptance_seed import (
            BUILTIN_C_CONTRACT, seed_acceptance_tests)

        class _Client:
            def generate_response(self, _prompt):
                raise AssertionError("seeding C must not call an LLM")

        language = detect_language_from_task(
            "Build a production-ready command-line todo manager in C")
        assert language == "c"
        path = seed_acceptance_tests("Build a todo manager in C.",
                                     str(tmp_path), _Client(),
                                     language=language)
        assert path is not None
        body = open(path, encoding="utf-8").read()
        shipped = open(BUILTIN_C_CONTRACT, encoding="utf-8").read()
        assert shipped in body, "a C task must get the C contract, not Python's"

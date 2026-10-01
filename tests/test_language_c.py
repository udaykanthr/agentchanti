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


class TestTheCTestRunner:
    """C and C++ fell through to the PYTHON backend, so the TesterAgent was
    told to write pytest for a C project and `TEST_FRAMEWORKS` handed it
    `python -m pytest`.

    Neither ecosystem has a standard test framework — Unity, CMocka,
    Criterion, Check, GoogleTest and Catch2 all exist and every one needs
    an install this project cannot assume — so the runner is the project's
    own `make test` target. That is the same answer
    `verify_cmd_for_language` gives for C, for the same reason.
    """

    @pytest.mark.parametrize("lang,ext,name", [("c", ".c", "C"),
                                               ("cpp", ".cpp", "C++")])
    def test_there_is_a_backend(self, lang, ext, name):
        from agentchanti.language_backend import get_backend
        b = get_backend(lang)
        assert b.language == lang, f"{lang} fell through to {b.language}"
        assert b.display_name == name
        fw = b.get_test_framework()
        assert fw["command"] == "make test"
        assert fw["ext"] == ext

    @pytest.mark.parametrize("lang", ["c", "cpp"])
    def test_the_rules_forbid_adding_a_framework(self, lang):
        """The one instruction that matters: a generated suite needing an
        install is a suite that cannot run."""
        from agentchanti.language_backend import get_backend
        rules = get_backend(lang).get_test_rules()
        assert "framework" in rules.lower()
        assert "make test" in rules

    @pytest.mark.parametrize("lang", ["c", "cpp"])
    def test_get_test_framework_agrees_with_the_backend(self, lang):
        from agentchanti.language import get_test_framework
        assert get_test_framework(lang)["command"] == "make test"

    def test_the_dict_and_the_backends_cannot_drift(self):
        """`get_test_framework` prefers the backend, but `test_analyzer`
        reads TEST_FRAMEWORKS directly — so the two must agree or a C
        project gets a different answer depending on which one asked.

        Checked for EVERY language, not just C: the other seven already
        satisfied this, which is what makes it an invariant rather than a
        new rule.
        """
        from agentchanti.language import TEST_FRAMEWORKS
        from agentchanti.language_backend import get_backend
        for key, entry in TEST_FRAMEWORKS.items():
            if ":" in key:                      # runner variants (vitest)
                continue
            backend = get_backend(key)
            if backend.language != key:         # no backend of its own
                continue
            bf = backend.get_test_framework()
            assert entry["command"] == bf["command"], key
            assert entry["ext"] == bf["ext"], key

    def test_a_c_project_no_longer_gets_pytest(self):
        """The defect, stated as the thing a reader would check."""
        from agentchanti.language import get_test_framework
        for lang in ("c", "cpp"):
            assert "pytest" not in get_test_framework(lang)["command"]


class TestTheBackendReadsCAndCpp:
    """`extract_exports` feeds the wiring and dependency graphs, so a wrong
    answer here becomes a false finding there. Three drafts were wrong
    before this was pinned: the first missed `typedef struct {...} Name;`
    because the body contains its own semicolons, the second listed
    `static` functions, and the third returned NOTHING for C++ because a
    `std::` return type has a colon.
    """

    def _c(self):
        from agentchanti.language_backend import get_backend
        return get_backend("c")

    def _cpp(self):
        from agentchanti.language_backend import get_backend
        return get_backend("cpp")

    def test_a_definition_is_an_export(self):
        assert "todo_add" in self._c().extract_exports(
            "int todo_add(const char *t) { return 0; }\n")

    def test_a_declaration_is_not(self):
        """A prototype in a header promises nothing about this file."""
        assert self._c().extract_exports("void declared_only(void);\n") == []

    def test_a_static_function_is_not_an_export(self):
        """`static` is INTERNAL linkage. Listing one would make find_gaps
        report an export nothing imports — forever, because nothing outside
        the file can import it."""
        out = self._c().extract_exports("static void helper(int n) { (void)n; }\n")
        assert "helper" not in out

    @pytest.mark.parametrize("src,name", [
        ("typedef struct { int x; char n[8]; } Task;\n", "Task"),
        ("typedef enum { A, B } Kind;\n", "Kind"),
        ("typedef unsigned long handle_t;\n", "handle_t"),
    ])
    def test_typedefs_in_every_shape(self, src, name):
        """The braced form carries semicolons inside it, which is why a
        single `[^;]` scan could never reach its name."""
        assert name in self._c().extract_exports(src)

    @pytest.mark.parametrize("src,name", [
        ('std::string join(const std::vector<int>& v) { return ""; }\n', "join"),
        ("std::map<int, int> build(int n) { return {}; }\n", "build"),
        ("int Store::count() const { return 0; }\n", "count"),
        ("void plain() {}\n", "plain"),
    ])
    def test_cpp_return_types_do_not_defeat_it(self, src, name):
        assert name in self._cpp().extract_exports(src)

    def test_cpp_static_is_still_excluded(self):
        assert self._cpp().extract_exports("static int hidden() { return 1; }\n") == []

    @pytest.mark.parametrize("src,want", [
        ('#include <stdio.h>\n', ["stdio.h"]),
        ('#include "todo.h"\n', ["todo.h"]),
        ('#  include   <stdint.h>\n', ["stdint.h"]),
    ])
    def test_both_include_forms(self, src, want):
        assert self._c().extract_imports(src) == want

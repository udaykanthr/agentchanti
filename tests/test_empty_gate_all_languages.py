"""`verify_passed` refused an empty collection for Python only.

The gate exists so a step cannot exit green on a suite that collected
nothing — "the exit code cannot distinguish 'nothing was wrong' from
'nothing was checked'", the rule `empty_suite_reason` and
`_green_suites_contradicting` both rest on. Measured 2026-10-02 it held for
Python's runners and no others:

    go test ./...     exit 0  "[no test files]"        -> ACCEPTED
    make test         exit 0  (silent)                 -> ACCEPTED
    cargo test        exit 0  "running 0 tests"        -> ACCEPTED
    mvn -B test       exit 0  "[INFO] No tests to run" -> ACCEPTED

Go is the instructive one. `go test` was already in `_TEST_RUNNER_TOKENS`,
so the list read as covering it — but Go reports an empty package with
`[no test files]` and exit **0**, never the exit 5 the check keyed on. A
token list is not a guard.

C is where it mattered most, because a silent zero exit is the NORMAL
output of a generated C suite: measured across six artifacts from a
fixed-plan A/B, four printed nothing at all from `make test`. With
`verify_cmd_for_language("c")` answering `make test`, a C TEST step could
go green having run no tests.

`mvn -B test` is why runner detection is patterns rather than substrings:
it does not contain the substring "mvn test". It was the single wrong
answer out of sixteen on the first attempt at this fix.
"""
import pytest

from agentchanti.agent_tools import _no_tests_collected
from agentchanti.orchestrator.agent_loop import verify_passed


def _gate_accepts(command, exit_code, output):
    """What the agent loop's exit gate concludes for this verify run."""
    flagged = _no_tests_collected(command, exit_code, output)
    result = ("exit: success" if exit_code == 0 else "exit: FAILED")
    if output:
        result += "\n" + output
    if flagged:
        result += "\n\nNOTE: the runner exited having COLLECTED NO TESTS."
    return verify_passed(result)


class TestAnEmptyCollectionIsRefused:
    """One case per language family, each with that runner's own spelling
    of "I found nothing" — which is the part a shared exit code cannot
    express."""

    @pytest.mark.parametrize("command,exit_code,output", [
        ("python -m pytest -q", 5, ""),
        ("python -m unittest discover", 0, "Ran 0 tests in 0.000s\nNO TESTS RAN"),
        ("python -m pytest -q", 0, "collected 0 items"),
        ("go test ./...", 0, "?   todo  [no test files]"),
        ("cargo test --quiet", 0, "running 0 tests"),
        ("mvn test", 0, "[INFO] No tests to run."),
        ("mvn -B test", 0, "[INFO] No tests to run."),
        ("./gradlew -q test", 0, "no tests found"),
        ("ctest", 0, "No tests were found!!!"),
    ])
    def test_the_runner_says_so(self, command, exit_code, output):
        assert not _gate_accepts(command, exit_code, output)


class TestASilentOpaqueRunnerProvesNothing:
    """`make test` runs whatever the Makefile says, so there is no
    vocabulary to match — the proof has to be positive."""

    @pytest.mark.parametrize("command", ["make test", "make -C build test",
                                         "ctest --output-on-failure"])
    def test_silence_is_not_a_pass(self, command):
        assert not _gate_accepts(command, 0, "")

    def test_build_chatter_alone_is_not_a_pass(self):
        """Compiler output proves the suite COMPILED, not that it ran."""
        assert not _gate_accepts(
            "make test", 0,
            "cc -c -o todo.o todo.c\ncc -o todo todo.o\n"
            "tests/test_todo.c:187:5: warning: unused variable")

    @pytest.mark.parametrize("output", [
        "4 tests passed",
        "All todo tests passed",
        "OK",
        "running 12 checks",
        "12 assertions, 0 failures",
        "ok - add appends",
        "PASS",
    ])
    def test_any_sign_a_test_ran_is_accepted(self, output):
        """Deliberately broad: a false "it ran" leaves the gate exactly as
        it was, while a false "nothing ran" fails a correct step."""
        assert _gate_accepts("make test", 0, output)

    def test_a_failing_run_is_an_ordinary_failure(self):
        """Not "collected nothing" — calling it that would send the model
        after a discovery problem that does not exist, which is the
        mistake `_no_tests_collected` was written to stop."""
        assert not _no_tests_collected("make test", 1, "")
        assert not _gate_accepts("make test", 1, "")


class TestRealRunsStillPass:

    @pytest.mark.parametrize("command,output", [
        ("python -m pytest -q", "4 passed in 0.12s"),
        ("go test ./...", "ok   todo  0.004s"),
        ("cargo test --quiet", "test result: ok. 7 passed; 0 failed"),
        ("mvn -B test", "Tests run: 9, Failures: 0, Errors: 0"),
        ("npm test --silent", "# pass 4\n# fail 0"),
    ])
    def test_a_green_suite_is_accepted(self, command, output):
        assert _gate_accepts(command, 0, output)


class TestNonTestCommandsAreUntouched:
    """The guard keys on the command looking like a test runner. A build
    that legitimately prints nothing must not be accused — it is not a
    suite and has no tests to collect."""

    @pytest.mark.parametrize("command", [
        "make", "make build", "make -C build all", "gcc -o x x.c",
        "cmake --build build", "npm run build", "cargo build --quiet",
        "mvn -B -DskipTests package",
    ])
    def test_a_silent_build_is_still_a_pass(self, command):
        assert _gate_accepts(command, 0, "")
        assert not _no_tests_collected(command, 0, "")


class TestTheDetectionIsPatternsNotSubstrings:
    """`mvn -B test` does not contain "mvn test". That single case was the
    one wrong answer out of sixteen before runner detection became
    pattern-based, and it is the shape every flagged-subcommand runner
    takes."""

    @pytest.mark.parametrize("command", [
        "mvn -B test", "mvn --batch-mode test", "mvn -q -B test",
        "./gradlew --console=plain test", "cargo test --quiet",
        "go test -count=1 ./...", "make -C build test",
        "dotnet test --no-build", "swift test --parallel",
    ])
    def test_flags_between_program_and_subcommand(self, command):
        from agentchanti.agent_tools import _looks_like_a_test_runner
        assert _looks_like_a_test_runner(command.lower()), command

    @pytest.mark.parametrize("command", [
        "mvn -B package", "cargo build", "go build ./...", "make install",
        "dotnet build", "npm run lint",
    ])
    def test_the_same_programs_without_test_are_not_runners(self, command):
        from agentchanti.agent_tools import _looks_like_a_test_runner
        assert not _looks_like_a_test_runner(command.lower()), command


class TestTheRegexesSurvivedBeingWritten:
    """Five times in this work a `\\b` written through a shell heredoc
    became an ASCII backspace (0x08) instead of a word boundary, silently
    disabling the pattern. The source is checked for control characters
    because a broken pattern still compiles and still matches nothing."""

    def test_no_control_characters_in_the_module(self):
        import pathlib

        import agentchanti.agent_tools as mod
        raw = pathlib.Path(mod.__file__).read_bytes()
        stray = sorted({b for b in raw if b < 32 and b not in (9, 10, 13)})
        assert stray == [], f"control bytes in source: {stray}"

    def test_the_patterns_use_real_word_boundaries(self):
        from agentchanti.agent_tools import (_OPAQUE_RUNNER_RES,
                                             _RAN_SOMETHING_RE,
                                             _TEST_RUNNER_RES)
        for r in _TEST_RUNNER_RES + _OPAQUE_RUNNER_RES + (_RAN_SOMETHING_RE,):
            assert "\x08" not in r.pattern, r.pattern
            assert "\\b" in r.pattern, r.pattern


class TestTheAdviceMatchesTheRunner:
    """One hint used to end with "named test_*.py ... __init__.py ...
    project root" — correct for a Python runner and actively misleading for
    `make test`, where it would send the model looking for Python test
    files in a C project.

    A tool misreporting its own situation is the shape `_read_file_range`
    (a 300-line read headed "full source") and `NO TESTS RAN` were both
    fixed for.
    """

    def _hint(self, command):
        from agentchanti.agent_tools import _no_tests_hint
        return _no_tests_hint(command)

    @pytest.mark.parametrize("command", ["make test", "make -C build test",
                                         "ctest"])
    def test_an_opaque_runner_is_told_to_print_what_passed(self, command):
        h = self._hint(command)
        assert "printed nothing" in h
        assert "tests passed" in h          # the concrete thing to emit
        assert "test_*.py" not in h, "Python advice leaked into a make hint"

    @pytest.mark.parametrize("command", ["python -m pytest -q",
                                         "python -m unittest discover",
                                         "python manage.py test"])
    def test_a_python_runner_keeps_the_python_advice(self, command):
        h = self._hint(command)
        assert "test_*.py" in h
        assert "printed nothing" not in h

    @pytest.mark.parametrize("command", ["go test ./...", "mvn -B test",
                                         "cargo test --quiet",
                                         "./gradlew test"])
    def test_other_runners_get_runner_neutral_advice(self, command):
        """Neither Python's filenames nor make's print-a-line, because
        neither is true for them."""
        h = self._hint(command)
        assert "test_*.py" not in h
        assert "printed nothing" not in h
        assert "where this runner looks for them" in h

    def test_every_hint_keeps_the_part_that_matters(self):
        """The shared warning is the load-bearing half: a zero exit is not
        evidence, and there is no bug in the code to chase."""
        for command in ("make test", "python -m pytest", "go test ./..."):
            h = self._hint(command)
            assert "COLLECTED NO TESTS" in h
            assert "is NOT evidence that anything passed" in h
            assert "not a failing assertion" in h

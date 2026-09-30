"""A greenfield C or C++ build can earn independent evidence.

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a contract this pipeline seeds
— and the seeder could write one for Python, JavaScript, Go, Rust and
Java. A C project had none of the three however well it went. That is the
sixth instance of one structural gap: evidence seeding is per language,
and every new language starts with none.

C is the first of the six with **no manifest at all**. Go has `go.mod`,
Rust `Cargo.toml`, Java `pom.xml`, and each of those contracts opens by
asking the manifest what the project is. C's nearest equivalent is a build
system, which says how to build and never what the project is — so the
floor is weaker, and the interesting tests here are the ones pinning what
the contract REFUSES to judge.

These tests deliberately exercise the contract's own functions rather than
grepping its source: a test that asserts a docstring contains a sentence
passes over code that does the opposite. No compiler is required, which is
also true of CI.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from agentchanti.orchestrator.acceptance_seed import (BUILTIN_C_CONTRACT,
                                                      SEED_BASENAME,
                                                      seed_acceptance_tests,
                                                      seedable_language)


def _contract():
    """Import the shipped contract as a module, to call its helpers."""
    spec = importlib.util.spec_from_file_location("_c_contract",
                                                  BUILTIN_C_CONTRACT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Client:
    """Seeding C must cost nothing — if this is called, it is a bug."""

    def __init__(self):
        self.calls = 0

    def generate_response(self, _prompt):
        self.calls += 1
        raise AssertionError("the built-in C contract must not call an LLM")


class TestSeedingC:

    @pytest.mark.parametrize("language", ["c", "cpp", "c++", "C", "CPP"])
    def test_a_c_project_gets_a_contract(self, tmp_path, language):
        client = _Client()
        path = seed_acceptance_tests("Build a todo CLI in C.", str(tmp_path),
                                     client, language=language)
        assert path is not None
        assert Path(path).name == SEED_BASENAME
        assert client.calls == 0, "seeding C must spend no tokens"

    def test_the_written_file_is_the_shipped_contract(self, tmp_path):
        path = seed_acceptance_tests("t", str(tmp_path), _Client(),
                                     language="c")
        written = Path(path).read_text(encoding="utf-8")
        shipped = Path(BUILTIN_C_CONTRACT).read_text(encoding="utf-8")
        assert shipped in written, "the body must be the shipped contract"

    def test_it_carries_a_seed_header(self, tmp_path):
        """Without the stamp it is indistinguishable from a user's suite,
        and the demotion rule could not apply to it."""
        path = seed_acceptance_tests("t", str(tmp_path), _Client(),
                                     language="c")
        first = Path(path).read_text(encoding="utf-8").splitlines()[0]
        assert "agentchanti:acceptance-seed" in first

    def test_the_language_gate_agrees_with_the_seeder(self):
        """`cli.py` warns "nothing can satisfy this flag" from
        `seedable_language`, and a stale answer there tells a user their
        run cannot succeed when it can. It drifted twice before."""
        for lang in ("c", "cpp", "c++", "cxx", "C"):
            assert seedable_language(lang) is True
        assert seedable_language("haskell") is False

    def test_it_is_python_not_c(self):
        """A `.c` file in the project is swept into the build by make, by a
        CMake glob, or by `cc *.c` — so a contract written in C would be
        compiled and linked into the artifact it judges, and a second
        `main` would break the very build it exists to measure."""
        assert BUILTIN_C_CONTRACT.endswith(".py")

    def test_it_parses(self):
        import ast
        ast.parse(Path(BUILTIN_C_CONTRACT).read_text(encoding="utf-8"))


class TestACrashIsASignal:
    """C's spelling of a crash, and the one thing no sibling contract's
    regex can express: a segfaulting program prints nothing at all."""

    def setup_method(self):
        self.mod = _contract()

    @pytest.mark.parametrize("rc,name", [(-11, "SIGSEGV"), (-6, "SIGABRT"),
                                         (-9, "SIGKILL"), (-4, "SIGILL")])
    def test_a_posix_signal_death_is_a_crash(self, rc, name):
        assert self.mod._died_on_a_signal(rc) is True, name

    @pytest.mark.parametrize("rc,name", [
        (0xC0000005, "STATUS_ACCESS_VIOLATION"),
        (0xC0000409, "STATUS_STACK_BUFFER_OVERRUN"),
        (0xC000001D, "STATUS_ILLEGAL_INSTRUCTION"),
    ])
    def test_an_ntstatus_is_a_crash(self, rc, name):
        """Windows has no signals; an access violation arrives as an
        NTSTATUS with the high bit set, i.e. a very large positive int."""
        assert self.mod._died_on_a_signal(rc) is True, name

    @pytest.mark.parametrize("rc", [0, 1, 2, 64, 127, 255])
    def test_a_non_zero_exit_is_not_a_crash(self, rc):
        """The refusal four generated contracts had to be corrected for. A
        CLI invoked with no arguments is SUPPOSED to refuse: printing usage
        and exiting 1 is correct behaviour, not a defect."""
        assert self.mod._died_on_a_signal(rc) is False

    def test_no_result_at_all_is_not_a_crash(self):
        """A timeout yields None. That is inconclusive, not a crash."""
        assert self.mod._died_on_a_signal(None) is False


class TestWhatItRefusesToBuild:
    """The heart of the C contract. Every sibling reads a manifest; this
    one has to decide how to build, and a wrong guess means reporting a
    missing `-lm` as a defect in the code."""

    def setup_method(self):
        self.mod = _contract()

    def _with_tools(self, monkeypatch, *available):
        monkeypatch.setattr(self.mod.shutil, "which",
                            lambda name: f"/usr/bin/{name}"
                            if name in available else None)

    def test_cmake_is_preferred_over_a_makefile(self, tmp_path, monkeypatch):
        """A CMake project's generated Makefile lives in a build directory,
        so bare `make` at the top would fail over a project that is fine."""
        (tmp_path / "CMakeLists.txt").write_text("project(x)")
        (tmp_path / "Makefile").write_text("all:\n\techo hi\n")
        self._with_tools(monkeypatch, "cmake", "make")
        plan, label = self.mod._build_plan(tmp_path)
        assert label == "cmake"
        assert plan[0][0] == "cmake"

    def test_a_makefile_is_used_when_there_is_no_cmake(self, tmp_path,
                                                      monkeypatch):
        (tmp_path / "Makefile").write_text("all:\n\techo hi\n")
        self._with_tools(monkeypatch, "make")
        plan, label = self.mod._build_plan(tmp_path)
        assert label == "make"

    def test_one_source_file_is_unambiguous(self, tmp_path, monkeypatch):
        """A single translation unit needs no build system: `cc main.c -o a`
        is the only reading there is."""
        (tmp_path / "main.c").write_text("int main(void){return 0;}")
        self._with_tools(monkeypatch, "cc")
        plan, label = self.mod._build_plan(tmp_path)
        assert label == "direct"
        assert "main.c" in plan[0]

    def test_a_cpp_source_gets_a_cpp_compiler(self, tmp_path, monkeypatch):
        (tmp_path / "main.cpp").write_text("int main(){return 0;}")
        self._with_tools(monkeypatch, "g++", "cc")
        plan, label = self.mod._build_plan(tmp_path)
        assert label == "direct"
        assert "g++" in plan[0][0]

    def test_several_sources_without_a_build_system_is_refused(
            self, tmp_path, monkeypatch):
        """THE refusal. Two translation units make the link order, the
        include paths and the libraries the build system's decisions —
        guessing them would judge the contract's guess, not the project."""
        (tmp_path / "main.c").write_text("int main(void){return 0;}")
        (tmp_path / "todo.c").write_text("void f(void){}")
        self._with_tools(monkeypatch, "cc", "make", "cmake")
        plan, reason = self.mod._build_plan(tmp_path)
        assert plan is None
        assert "build system" in reason
        assert "guessing it" in reason

    def test_no_source_at_all_does_not_apply(self, tmp_path, monkeypatch):
        (tmp_path / "README.md").write_text("hello")
        self._with_tools(monkeypatch, "cc")
        plan, reason = self.mod._build_plan(tmp_path)
        assert plan is None
        assert "does not apply" in reason

    @pytest.mark.parametrize("marker,tool", [("CMakeLists.txt", "cmake"),
                                             ("Makefile", "make"),
                                             ("meson.build", "meson")])
    def test_a_missing_tool_skips_rather_than_fails(self, tmp_path,
                                                    monkeypatch, marker, tool):
        """An absent toolchain is the instrument being unavailable and must
        never convict the code."""
        (tmp_path / marker).write_text("x")
        self._with_tools(monkeypatch)          # nothing installed
        plan, reason = self.mod._build_plan(tmp_path)
        assert plan is None
        assert tool in reason
        assert "proves nothing about the code" in reason


class TestWhichFilesCountAsCommands:
    """The build's output name is unpredictable — a Makefile emits whatever
    its author chose — so the contract asks the tree instead of guessing.
    That only works if it knows what is NOT a command."""

    def setup_method(self):
        self.mod = _contract()

    def _make_executable(self, path):
        """An ELF header, because that is what the check now reads."""
        path.write_bytes(b"\x7fELF fake")
        if os.name != "nt":
            path.chmod(0o755)

    def _exe_name(self, stem):
        return f"{stem}.exe" if os.name == "nt" else stem

    def test_the_execute_bit_is_not_evidence(self, tmp_path):
        """THE defect this check exists for.

        Measured 2026-09-30: on a Windows bind mount every file reports
        mode 777, so `Makefile`, `main.c` and `store.h` were all
        "executable". `Makefile` sorts before `todo`, so the crash check
        ran the MAKEFILE, got an exec-format error, and read that as "did
        not crash" — a segfaulting program passed the contract. The same
        holds on any FAT, exFAT or CIFS mount.
        """
        for name in ("Makefile", "CMakeLists.txt", "README", "tests.sh.txt"):
            f = tmp_path / name
            f.write_text("all:\n\tcc -o todo main.c\n")
            if os.name != "nt":
                f.chmod(0o777)
        self._make_executable(tmp_path / self._exe_name("todo"))
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert found == {self._exe_name("todo")}, (
            f"only the real program is a command, got {found}")

    def test_a_script_with_a_shebang_is_a_command(self, tmp_path):
        """A Makefile may legitimately build a wrapper script."""
        f = tmp_path / "todo"
        f.write_text("#!/bin/sh\nexec ./todo-bin \"$@\"\n")
        if os.name != "nt":
            f.chmod(0o755)
        assert {p.name for p in self.mod._executables(tmp_path)} == {"todo"}

    @pytest.mark.parametrize("magic,name", [
        (b"MZ\x90\x00", "PE (Windows)"),
        (b"\xcf\xfa\xed\xfe", "Mach-O 64"),
        (b"\xca\xfe\xba\xbe", "Mach-O universal"),
    ])
    def test_other_platforms_binaries_count(self, tmp_path, magic, name):
        (tmp_path / self._exe_name("todo")).write_bytes(magic + b"rest")
        assert self.mod._executables(tmp_path), name

    def test_a_built_command_is_found(self, tmp_path):
        self._make_executable(tmp_path / self._exe_name("todo"))
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert self._exe_name("todo") in found

    @pytest.mark.parametrize("name", ["todo.o", "todo.a", "libtodo.so",
                                      "todo.obj", "todo.d", "notes.md"])
    def test_object_files_and_libraries_are_not_commands(self, tmp_path, name):
        """The extension check is what excludes these, and it has to:
        they are given real ELF headers here because a `.so` genuinely IS
        an ELF image, so magic bytes alone would admit it."""
        (tmp_path / name).write_bytes(b"\x7fELF fake")
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert name not in found

    def test_a_plain_command_still_counts(self, tmp_path):
        """The other half: the exclusion above must not be so broad that
        nothing is ever a command."""
        (tmp_path / "todo").write_bytes(b"\x7fELF fake")
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert "todo" in found

    def test_sources_and_headers_are_not_commands(self, tmp_path):
        for name in ("main.c", "todo.h", "app.cpp", "util.hpp"):
            (tmp_path / name).write_text("int main(void){return 0;}")
        assert self.mod._executables(tmp_path) == {}

    def test_a_source_file_that_starts_with_a_shebang_is_still_a_source(
            self, tmp_path):
        """Where the extension check and the magic check overlap.

        Ordinary sources are excluded by their bytes alone — no `.c` file
        carries an ELF header — so the extension check only earns its place
        on the one input that fools the magic check. A mutation run made
        that concrete: deleting the extension check was NOT caught until
        this case existed.
        """
        (tmp_path / "generated.c").write_text("#!/bin/sh\nint main(){}\n")
        (tmp_path / "gen.h").write_text("#!/bin/sh\n")
        assert self.mod._executables(tmp_path) == {}

    def test_cmake_probe_binaries_are_not_the_project(self, tmp_path):
        """CMake writes `cmTC_*` probes that compile AND run successfully.
        Counting one would mean reporting a project as having produced a
        command when its own build produced nothing."""
        internals = tmp_path / "build" / "CMakeFiles" / "CMakeTmp"
        internals.mkdir(parents=True)
        self._make_executable(internals / self._exe_name("cmTC_abc12"))
        assert self.mod._executables(tmp_path) == {}

    def test_a_cmtc_probe_is_skipped_even_outside_cmakefiles(self, tmp_path):
        self._make_executable(tmp_path / self._exe_name("cmTC_dead01"))
        assert self.mod._executables(tmp_path) == {}

    def test_the_projects_own_git_directory_is_skipped(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        self._make_executable(git / self._exe_name("hook"))
        assert self.mod._executables(tmp_path) == {}


class TestWhereTheProjectIs:

    def setup_method(self):
        self.mod = _contract()

    def test_the_root_when_it_holds_the_source(self, tmp_path, monkeypatch):
        (tmp_path / "main.c").write_text("int main(void){return 0;}")
        monkeypatch.setattr(self.mod, "ROOT", tmp_path)
        assert self.mod._project_dir() == tmp_path

    def test_one_level_down_when_a_scaffold_made_a_directory(self, tmp_path,
                                                            monkeypatch):
        app = tmp_path / "app"
        app.mkdir()
        (app / "CMakeLists.txt").write_text("project(x)")
        monkeypatch.setattr(self.mod, "ROOT", tmp_path)
        assert self.mod._project_dir() == app

    def test_nothing_when_there_is_no_c_anywhere(self, tmp_path, monkeypatch):
        (tmp_path / "README.md").write_text("hello")
        monkeypatch.setattr(self.mod, "ROOT", tmp_path)
        assert self.mod._project_dir() is None

    def test_a_dependency_tree_is_not_the_project(self, tmp_path, monkeypatch):
        """`vendor/` holding C is somebody else's code, and picking it would
        judge a dependency instead of the artifact."""
        vendor = tmp_path / "vendor"
        vendor.mkdir()
        (vendor / "zlib.c").write_text("int f(void){return 0;}")
        (tmp_path / "README.md").write_text("hello")
        monkeypatch.setattr(self.mod, "ROOT", tmp_path)
        assert self.mod._project_dir() is None


class TestTheStubCheck:
    """The one assertion in the TestCase itself that needs no compiler.

    A mutation run found it had NO unit coverage: replacing the 400
    character threshold with -1 left all 54 tests green, so the check was
    validated only by container fixtures. Run here through unittest so the
    real method executes, rather than by reading the source for a number.
    """

    def _run_stub_test(self, project):
        import unittest

        mod = _contract()
        mod.ROOT = project
        suite = unittest.TestSuite()
        suite.addTest(mod.CBuildContract(
            "test_it_is_a_program_rather_than_a_stub"))
        result = unittest.TestResult()
        suite.run(result)
        return result

    def test_a_stub_is_rejected(self, tmp_path):
        (tmp_path / "main.c").write_text("int main(void){return 0;}")
        (tmp_path / "Makefile").write_text("todo: main.c\n\tcc -o todo main.c\n")
        result = self._run_stub_test(tmp_path)
        assert len(result.failures) == 1, "a 25-character program must fail"
        assert "too little to be a program" in result.failures[0][1]

    def test_a_real_program_is_accepted(self, tmp_path):
        body = "\n".join(f"int helper_{i}(int x) {{ return x * {i} + 1; }}"
                         for i in range(40))
        (tmp_path / "main.c").write_text(
            f"#include <stdio.h>\n{body}\nint main(void){{return 0;}}\n")
        (tmp_path / "Makefile").write_text("todo: main.c\n\tcc -o todo main.c\n")
        result = self._run_stub_test(tmp_path)
        assert result.failures == [], result.failures
        assert result.errors == [], result.errors

    def test_headers_count_toward_the_total(self, tmp_path):
        """A project can legitimately put most of its code in headers."""
        (tmp_path / "main.c").write_text("int main(void){return 0;}")
        (tmp_path / "big.h").write_text("/* " + "x" * 500 + " */\n")
        (tmp_path / "Makefile").write_text("todo: main.c\n\tcc -o todo main.c\n")
        assert self._run_stub_test(tmp_path).failures == []


class TestTheLoopsFallbackVerifyCommand:
    """`verify_cmd_for_language` had no C branch, so a C step whose plan
    declared no gate fell back to accepting the model's own summary.

    C has no default test command — there is no runner to guess at — so
    this trusts a declared `test:` rule and nothing else, exactly as the
    JavaScript branch only trusts `npm test` when package.json defines it.
    A wrong verify command is worse than none: the loop would chase
    failures in the verifier instead of the code.
    """

    def _verify(self, root):
        from agentchanti.orchestrator.agent_loop import verify_cmd_for_language
        return verify_cmd_for_language("c", str(root))

    def test_a_declared_test_target_is_used(self, tmp_path):
        (tmp_path / "Makefile").write_text(
            "todo: main.c\n\tcc -o todo main.c\n\ntest: todo\n\t./tests.sh\n")
        assert self._verify(tmp_path) == "make test"

    def test_no_makefile_means_no_command(self, tmp_path):
        assert self._verify(tmp_path) is None

    def test_a_makefile_without_a_test_target_means_no_command(self, tmp_path):
        """Guessing `make test` here would fail on every step, for a
        project that is perfectly fine."""
        (tmp_path / "Makefile").write_text("todo: main.c\n\tcc -o todo main.c\n")
        assert self._verify(tmp_path) is None

    def test_phony_alone_is_not_a_target(self, tmp_path):
        """`.PHONY: test` declares something ABOUT the target, and sits on
        its own line where a naive scan would match it."""
        (tmp_path / "Makefile").write_text(
            "todo: main.c\n\tcc -o todo main.c\n\n.PHONY: test\n")
        assert self._verify(tmp_path) is None

    def test_a_variable_named_test_is_not_a_target(self, tmp_path):
        (tmp_path / "Makefile").write_text(
            "test := ./run.sh\ntodo: main.c\n\tcc -o todo main.c\n")
        assert self._verify(tmp_path) is None

    def test_test_mentioned_inside_a_recipe_is_not_a_target(self, tmp_path):
        """Recipe lines are tab-indented, so they never start at column 0."""
        (tmp_path / "Makefile").write_text(
            "check: todo\n\ttest -x ./todo && echo ok\n")
        assert self._verify(tmp_path) is None

    def test_lowercase_makefile_is_found(self, tmp_path):
        (tmp_path / "makefile").write_text("test:\n\t./tests.sh\n")
        assert self._verify(tmp_path) == "make test"

    def test_cpp_gets_the_same_treatment(self, tmp_path):
        from agentchanti.orchestrator.agent_loop import verify_cmd_for_language
        (tmp_path / "Makefile").write_text("test:\n\t./tests.sh\n")
        assert verify_cmd_for_language("cpp", str(tmp_path)) == "make test"

    def test_other_languages_are_unaffected(self, tmp_path):
        from agentchanti.orchestrator.agent_loop import verify_cmd_for_language
        assert verify_cmd_for_language("go", str(tmp_path)) == "go test ./..."
        assert verify_cmd_for_language("python", str(tmp_path)) == \
            "python -m pytest -q"
        assert verify_cmd_for_language("rust", str(tmp_path)) is None


class TestWhichCommandGetsRun:
    """The order decides which executable the crash check actually runs,
    and the first cut sorted by path — which is arbitrary.

    Measured 2026-09-30 on a tree holding `tests.sh` beside `todo`:
    `tests.sh` sorted first, so the check ran the TEST SUITE rather than
    the program, and passed because the suite exits 0. A fixture had been
    green on exactly that basis. A plan declaring `target:
    tests/test_todo.sh` makes it worse, since that script sorts ahead too.
    """

    def setup_method(self):
        self.mod = _contract()

    def _binary(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELF fake")

    def _script(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\necho hi\n")

    def _order(self, project):
        """Relative POSIX paths, not names.

        Names alone made `test_the_root_outranks_a_subdirectory` vacuous:
        `build/todo` and `todo` are both named "todo", so the assertion
        held whichever won. A mutation run caught it — deleting the depth
        key left every test green.
        """
        cls = self.mod.CBuildContract
        cls.project = project
        cls._before = {}
        return [p.relative_to(project).as_posix() for p in
                cls("test_the_command_starts_without_crashing")._commands()]

    def test_a_test_directory_is_never_the_command(self, tmp_path):
        self._script(tmp_path / "tests" / "test_todo.sh")
        self._binary(tmp_path / "todo")
        assert self._order(tmp_path) == ["todo"]

    @pytest.mark.parametrize("d", ["tests", "test", "spec", "__tests__",
                                   "testing"])
    def test_every_test_directory_name(self, tmp_path, d):
        self._script(tmp_path / d / "run.sh")
        assert self.mod._executables(tmp_path) == {}

    def test_a_compiled_binary_outranks_a_script(self, tmp_path):
        """`tests.sh` at the ROOT is not in a test directory, so only the
        binary-over-script rule saves it."""
        self._script(tmp_path / "tests.sh")
        self._binary(tmp_path / "todo")
        assert self._order(tmp_path) == ["todo", "tests.sh"]

    def test_a_script_is_still_used_when_there_is_no_binary(self, tmp_path):
        """The launcher-wrapper shape: a build may produce only a script."""
        self._script(tmp_path / "todo")
        assert self._order(tmp_path) == ["todo"]

    def test_the_root_outranks_a_subdirectory(self, tmp_path):
        self._binary(tmp_path / "build" / "todo")
        self._binary(tmp_path / "todo")
        assert self._order(tmp_path) == ["todo", "build/todo"]

    def test_a_freshly_built_command_outranks_a_stale_one(self, tmp_path):
        """Rule 1 still wins over the rest: what this build produced beats
        whatever was lying there."""
        import os
        self._binary(tmp_path / "aaa_old")
        self._binary(tmp_path / "zzz_new")
        cls = self.mod.CBuildContract
        cls.project = tmp_path
        old = tmp_path / "aaa_old"
        os.utime(old, (1_000_000, 1_000_000))
        cls._before = {old: old.stat().st_mtime}
        names = [p.name for p in
                 cls("test_the_command_starts_without_crashing")._commands()]
        assert names == ["zzz_new", "aaa_old"], names

    def test_program_kind_distinguishes_them(self, tmp_path):
        (tmp_path / "b").write_bytes(b"\x7fELF")
        (tmp_path / "s").write_text("#!/bin/sh\n")
        (tmp_path / "t").write_text("hello")
        assert self.mod._program_kind(tmp_path / "b") == "binary"
        assert self.mod._program_kind(tmp_path / "s") == "script"
        assert self.mod._program_kind(tmp_path / "t") is None


class TestALibraryIsNotACommand:
    """A C project with no `main` is a library, and demanding an
    executable of it would fail a perfectly good one.

    The Rust contract draws this line with `src/main.rs`; C has no such
    marker, so the source is what says it.
    """

    def _declares_main(self, project):
        mod = _contract()
        cls = mod.CBuildContract
        cls.project = project
        return cls("test_it_produces_an_executable")._declares_a_main()

    @pytest.mark.parametrize("body", [
        "int main(void) { return 0; }",
        "int main(int argc, char **argv) { return 0; }",
        "void main() {}",
        "  int  main ( void ) { return 0; }",
    ])
    def test_a_program_is_recognised(self, tmp_path, body):
        (tmp_path / "main.c").write_text(body)
        assert self._declares_main(tmp_path) is True

    def test_a_library_has_no_main(self, tmp_path):
        (tmp_path / "todo.c").write_text(
            "int todo_add(const char *t) { return 0; }\n")
        (tmp_path / "todo.h").write_text("int todo_add(const char *t);\n")
        assert self._declares_main(tmp_path) is False

    def test_a_main_in_a_test_directory_does_not_count(self, tmp_path):
        """A test harness often has its own main; that does not make the
        project a command."""
        (tmp_path / "todo.c").write_text("int todo_add(void) { return 0; }\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "run.c").write_text("int main(void) { return 0; }\n")
        assert self._declares_main(tmp_path) is False

    def test_a_call_to_main_is_not_a_definition(self, tmp_path):
        """`return main(argc, argv);` mid-line is a call, not a definition
        — only a line that STARTS with the return type declares one."""
        (tmp_path / "todo.c").write_text(
            "int wrapper(int a, char **b) { return main(a, b); }\n")
        assert self._declares_main(tmp_path) is False


class TestTheLibrarySkipActuallyHappens:
    """Testing `_declares_a_main` is not testing what it guards.

    A mutation run said so: deleting the `if not self._declares_a_main()`
    branch left every test green, because they all called the helper
    directly and none ran the check it protects. This runs the real test
    method with the build stubbed out, so no compiler is needed.
    """

    def _run(self, project):
        import unittest

        mod = _contract()
        mod.ROOT = project
        cls = mod.CBuildContract
        # The build is not what is under test here, and stubbing it keeps
        # this runnable on a machine with no toolchain (CI included).
        cls._build = lambda self: (0, "")
        cls.label = "make"
        suite = unittest.TestSuite()
        suite.addTest(cls("test_it_produces_an_executable"))
        result = unittest.TestResult()
        suite.run(result)
        return result

    def test_a_library_skips_rather_than_fails(self, tmp_path):
        (tmp_path / "Makefile").write_text("lib:\n\tar rcs libtodo.a todo.o\n")
        (tmp_path / "todo.c").write_text("int todo_add(void){return 0;}\n")
        result = self._run(tmp_path)
        assert result.failures == [], result.failures
        assert len(result.skipped) == 1
        assert "library, not a command" in result.skipped[0][1]

    def test_a_program_that_built_nothing_still_fails(self, tmp_path):
        """The other half: the skip must not swallow a real miss."""
        (tmp_path / "Makefile").write_text("todo:\n\t@echo nothing\n")
        (tmp_path / "main.c").write_text("int main(void){return 0;}\n")
        result = self._run(tmp_path)
        assert result.skipped == [], result.skipped
        assert len(result.failures) == 1
        assert "no executable" in result.failures[0][1]

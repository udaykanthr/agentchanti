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
        path.write_bytes(b"\x7fELF fake")
        if os.name != "nt":
            path.chmod(0o755)

    def _exe_name(self, stem):
        return f"{stem}.exe" if os.name == "nt" else stem

    def test_a_built_command_is_found(self, tmp_path):
        self._make_executable(tmp_path / self._exe_name("todo"))
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert self._exe_name("todo") in found

    @pytest.mark.parametrize("name", ["todo.o", "todo.a", "libtodo.so",
                                      "todo.obj", "todo.d", "notes.md"])
    def test_object_files_and_libraries_are_not_commands(self, tmp_path,
                                                        monkeypatch, name):
        """A `.so` has the execute bit set on many distributions and is
        emphatically not a command.

        Forced onto the POSIX path deliberately. On Windows the `.exe`
        filter excludes all of these before `_NOT_A_COMMAND` is consulted,
        so the test would pass over a contract that had lost that check
        entirely — which a mutation run confirmed: deleting the check was
        caught on Linux and not here. A guard that is only tested on one
        platform is half-tested.
        """
        monkeypatch.setattr(self.mod.os, "name", "posix")
        monkeypatch.setattr(self.mod.os, "access", lambda *a, **k: True)
        (tmp_path / name).write_bytes(b"\x7fELF fake")
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert name not in found

    def test_a_plain_command_still_counts_on_the_posix_path(self, tmp_path,
                                                           monkeypatch):
        """The other half: the exclusion above must not be so broad that
        nothing is ever a command."""
        monkeypatch.setattr(self.mod.os, "name", "posix")
        monkeypatch.setattr(self.mod.os, "access", lambda *a, **k: True)
        (tmp_path / "todo").write_bytes(b"\x7fELF fake")
        found = {p.name for p in self.mod._executables(tmp_path)}
        assert "todo" in found

    def test_sources_and_headers_are_not_commands(self, tmp_path, monkeypatch):
        """Forced onto the POSIX path for the reason above: on Windows the
        `.exe` filter would hide the loss of this check."""
        monkeypatch.setattr(self.mod.os, "name", "posix")
        monkeypatch.setattr(self.mod.os, "access", lambda *a, **k: True)
        for name in ("main.c", "todo.h", "app.cpp", "util.hpp"):
            (tmp_path / name).write_text("int main(void){return 0;}")
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

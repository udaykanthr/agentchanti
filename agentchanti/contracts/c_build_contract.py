"""A deterministic acceptance check for a C or C++ project.

WHY THIS FILE EXISTS

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a contract this pipeline
seeds — and until now the seeder could write one for Python, JavaScript,
Go, Rust and Java. A greenfield C or C++ build therefore had none of the
three no matter how well it went. This is the sixth instance of one
structural gap: evidence seeding is per language, and every new language
starts with none.

WHY IT IS SHIPPED RATHER THAN GENERATED

The JavaScript path learned this the expensive way: four consecutive
model-written contracts each failed a correct project, in four different
ways, because the space of ways to observe a build wrongly is larger than
a prompt can enumerate. So the contract for a C project is written once,
by people who can run it.

It is still independent in the sense that matters: nothing in it was
authored by the model whose work it judges, and the hash check withdraws
it as evidence if the run edits it.

WHY IT IS PYTHON AND NOT C

For the reason Rust's and Java's contracts are Python, in its strongest
form yet. A `.c` file dropped into the project is swept up by
`make`, by a CMake `GLOB`, or by `cc *.c` — so a contract written in C
would be *compiled and linked into the artifact it judges*, and a second
`main` would break the very build it exists to measure. Worse, a link
error in the instrument is indistinguishable, in the build log, from a
defect in the code. Python stays outside the translation units.

WHAT MAKES C DIFFERENT FROM EVERY LANGUAGE BEFORE IT

Three things, and each one shapes a check below.

**There is no manifest.** Go has `go.mod`, Rust `Cargo.toml`, Java
`pom.xml`; every one of those contracts opens by asking the manifest what
the project is called. C has nothing of the kind. The closest thing is a
build system — a Makefile, a CMakeLists.txt — which describes how to
build but never declares "this is a project named X". So the first check
asks whether a build system exists at all, and a project with none is
**skipped with a precise reason** rather than judged by a build this
contract invented for it. Inventing the build is how you end up reporting
a missing `-lm` as a defect in the code.

**The binary's name is unpredictable.** `cargo build` puts it in
`target/debug/`, `mvn package` writes a jar named from the manifest. A
Makefile emits whatever its author chose. So this contract does not guess:
it records the executables present before the build and looks for what
the build *added or refreshed*. Asking the tree rather than guessing a
name is the same principle that keeps the other five honest.

**A crash is a signal, not a string.** Rust prints `thread panicked`,
Java prints `Exception in thread "main"`, Go prints `goroutine 1
[running]`. C prints nothing at all — it dies on SIGSEGV or SIGABRT and
the shell reports 139 or 134. Python's `subprocess` surfaces that as a
*negative* returncode on POSIX and as a large NTSTATUS on Windows, so
that is what gets checked. The refusal every earlier contract carries
applies here too and matters more: **a non-zero exit is not a crash.** A
CLI invoked with no arguments is supposed to refuse, and four consecutive
generated contracts had to be corrected for calling that a failure.

WHAT IT DOES AND DOES NOT CLAIM

A floor, not a ceiling: a build system exists, the build succeeds, it
produces an executable, that executable starts without dying on a signal,
and there is more than a stub's worth of source. It knows nothing about
the task.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("AGENTCHANTI_PROJECT_ROOT", ".")).resolve()
SKIP = {".git", ".agentchanti", "node_modules", "vendor", "__pycache__",
        ".vscode", ".idea"}
TIMEOUT = 600

SOURCE_EXT = {".c", ".cc", ".cpp", ".cxx", ".c++"}
HEADER_EXT = {".h", ".hh", ".hpp", ".hxx"}

# Not executables, whatever the filesystem's execute bit says. A `.so`
# has one set on many distributions and is not a command.
_NOT_A_COMMAND = {".o", ".obj", ".a", ".lib", ".so", ".dylib", ".dll",
                  ".d", ".mk", ".cmake", ".txt", ".json", ".md", ".log",
                  ".gcda", ".gcno", ".pdb", ".ilk", ".exp"}

# Directories a build fills with its own machinery. CMake in particular
# writes probe executables (`cmTC_*`) that compile and run successfully
# and are emphatically not the project's command.
_BUILD_INTERNALS = {"CMakeFiles", "CMakeTmp", ".cmake", "Testing"}

# A build that failed because it could not fetch something is the
# instrument being unavailable, never the code being wrong. Rarer in C
# than in Rust, but CMake's FetchContent makes it possible.
_NETWORK_RE = re.compile(
    r"could not resolve host|connection (?:refused|timed out)"
    r"|failed to (?:fetch|download|connect)|network is unreachable"
    r"|temporary failure in name resolution|SSL|certificate"
    r"|server certificate verification failed",
    re.IGNORECASE)

# The compiler could not be found — the toolchain is missing, which is not
# a statement about the source.
_NO_COMPILER_RE = re.compile(
    r"(?:cc|gcc|g\+\+|clang|clang\+\+|cl)\s*:?\s*(?:command )?not found"
    r"|no such file or directory:\s*[\"']?(?:cc|gcc|g\+\+|clang)"
    r"|CMAKE_C(?:XX)?_COMPILER not set"
    r"|No CMAKE_C(?:XX)?_COMPILER could be found",
    re.IGNORECASE)


def _project_dir():
    """The directory holding the build system, or the source.

    Mirrors `_crate_dir` in the Rust contract: usually the root, sometimes
    one level down when a scaffold made a subdirectory.
    """
    def _has_build(d):
        return any((d / n).is_file() for n in
                   ("Makefile", "makefile", "GNUmakefile", "CMakeLists.txt",
                    "meson.build"))

    def _has_source(d):
        return any(p.suffix.lower() in SOURCE_EXT
                   for p in d.iterdir() if p.is_file())

    if _has_build(ROOT) or _has_source(ROOT):
        return ROOT
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and entry.name not in SKIP:
            try:
                if _has_build(entry) or _has_source(entry):
                    return entry
            except OSError:
                continue
    return None


def _build_plan(project):
    """How to build this project, from what it actually contains.

    Returns ``(argv_list, label)`` or ``(None, reason)``. Order matters:
    CMake before Make, because a CMake project's generated Makefile lives
    in a build directory and running bare `make` at the top would fail
    over a project that is perfectly fine.
    """
    if (project / "CMakeLists.txt").is_file():
        if shutil.which("cmake") is None:
            return None, ("the project uses CMake and cmake is not "
                          "installed, so this proves nothing about the code")
        return ([["cmake", "-S", ".", "-B", "build"],
                 ["cmake", "--build", "build"]], "cmake")
    for name in ("Makefile", "makefile", "GNUmakefile"):
        if (project / name).is_file():
            make = shutil.which("make") or shutil.which("gmake")
            if make is None:
                return None, ("the project uses make and make is not "
                              "installed, so this proves nothing about the "
                              "code")
            return ([[make]], "make")
    if (project / "meson.build").is_file():
        if shutil.which("meson") is None:
            return None, ("the project uses meson and meson is not "
                          "installed, so this proves nothing about the code")
        return ([["meson", "setup", "build"],
                 ["meson", "compile", "-C", "build"]], "meson")

    # No build system. Compiling by hand is only unambiguous for a single
    # translation unit: the moment there are two, the link order, the
    # include paths and the libraries are the build system's decisions and
    # not this contract's to invent. Guessing them would mean reporting a
    # missing `-lm` as a defect in the code.
    sources = [p for p in sorted(project.iterdir())
               if p.is_file() and p.suffix.lower() in SOURCE_EXT]
    if len(sources) == 1:
        src = sources[0]
        cxx = src.suffix.lower() != ".c"
        cc = ((shutil.which("g++") or shutil.which("clang++")) if cxx
              else (shutil.which("cc") or shutil.which("gcc")
                    or shutil.which("clang")))
        if cc is None:
            return None, ("no C compiler is installed, so this proves "
                          "nothing about the code")
        out = "a.out" if os.name != "nt" else "a.exe"
        return ([[cc, src.name, "-o", out]], "direct")
    if not sources:
        return None, ("no C or C++ source found — this contract does not "
                      "apply")
    return None, (f"{len(sources)} source files and no Makefile, "
                  f"CMakeLists.txt or meson.build — how to link them is the "
                  f"build system's decision, and guessing it would judge "
                  f"this contract's guess rather than the project")


def _looks_like_a_program(path):
    """Whether the file's own bytes say it is executable.

    The execute bit is NOT evidence, which is the whole reason this exists.
    Measured 2026-09-30: on a Windows bind mount every file reports mode
    777, so `Makefile`, `main.c` and `store.h` were all "executable" — and
    `Makefile` sorts before `todo`, so the crash check ran the Makefile,
    got an exec-format error, and read that as "did not crash". A
    segfaulting program passed. The same is true of any FAT, exFAT or CIFS
    mount and of a checkout made with an odd umask.

    So the question becomes what the file IS: an ELF, PE or Mach-O image,
    or a script declaring its interpreter. That is decisive and needs no
    cooperation from the filesystem — the same reason this contract reads
    the tree instead of guessing the binary's name.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return head[:4] in (b"\x7fELF",                      # ELF
                        b"\xcf\xfa\xed\xfe",             # Mach-O 64 LE
                        b"\xce\xfa\xed\xfe",             # Mach-O 32 LE
                        b"\xca\xfe\xba\xbe",             # Mach-O universal
                        ) or head[:2] in (b"MZ",         # PE / COFF
                                          b"#!")         # a script


def _executables(project):
    """Every plausible command in the tree, by path -> mtime.

    Deliberately asks the filesystem rather than guessing a name: a
    Makefile emits whatever its author chose. Build machinery and
    non-commands are excluded, most importantly CMake's own `cmTC_*` probe
    binaries, which build and run fine and are not the project.
    """
    found = {}
    for path in project.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & SKIP or parts & _BUILD_INTERNALS:
            continue
        suffix = path.suffix.lower()
        if suffix in _NOT_A_COMMAND:
            continue
        if suffix in SOURCE_EXT or suffix in HEADER_EXT:
            continue
        if path.name.startswith("cmTC_"):
            continue
        if not _looks_like_a_program(path):
            continue
        try:
            found[path] = path.stat().st_mtime
        except OSError:
            continue
    return found


def _died_on_a_signal(returncode):
    """Whether the process was killed rather than having exited.

    C's spelling of a crash, and the one place this contract cannot reuse
    a regex from its siblings: a segfaulting program prints nothing.

    POSIX: `subprocess` reports a signal death as a negative returncode.
    Windows: there are no signals, and an access violation surfaces as an
    NTSTATUS with the high bit set (0xC0000005 and friends), which arrives
    here as a very large positive integer.

    A plain non-zero exit is NOT this. A CLI given no arguments is
    supposed to refuse, and calling that a crash is the mistake four
    generated contracts had to be corrected for.
    """
    if returncode is None:
        return False
    if returncode < 0:
        return True
    return returncode >= 0xC0000000


def _run(args, cwd):
    try:
        p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                           errors="replace", timeout=TIMEOUT)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return None, f"timed out after {TIMEOUT}s"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


class CBuildContract(unittest.TestCase):
    """The floor every C or C++ project has to clear."""

    @classmethod
    def setUpClass(cls):
        cls.project = _project_dir()
        if cls.project is None:
            raise unittest.SkipTest(
                "no C or C++ source found — this contract does not apply")
        cls.plan, cls.label = _build_plan(cls.project)
        cls.tmp = tempfile.mkdtemp(prefix="c-contract-")
        cls._built = None
        cls._before = _executables(cls.project)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def _build(self):
        """Build once, shared across tests.

        Skips — never fails — when the toolchain is absent or a fetch
        failed: an unavailable instrument must never convict the code.
        """
        if self.plan is None:
            self.skipTest(self.label)
        cls = type(self)
        if cls._built is None:
            code, out = 0, ""
            for argv in self.plan:
                code, chunk = _run(argv, self.project)
                out += chunk
                if code != 0:
                    break
            cls._built = (code, out)
        code, out = cls._built
        if code != 0 and _NETWORK_RE.search(out):
            self.skipTest(f"the build could not fetch what it needed, so "
                          f"this proves nothing about the code:\n{out[-500:]}")
        if code != 0 and _NO_COMPILER_RE.search(out):
            self.skipTest(f"no working compiler was found, so this proves "
                          f"nothing about the code:\n{out[-500:]}")
        return code, out

    def _commands(self):
        """Executables present after a successful build, freshest first.

        NOT "executables new since before the build", which is what this
        asked at first and is wrong. `make` is incremental: a project whose
        binary is already up to date rebuilds nothing, so the newness test
        reported "the build succeeded but left no new executable" over a
        perfectly good project. That is the normal case rather than an edge
        one — agentchanti runs the build itself during the run, so the
        binary almost always exists before the contract looks.

        Caught 2026-09-30 by a fixture that had been built once already;
        `make-good` passed only because its tree happened to be clean.

        Existence is sound here because `test_the_project_builds` runs
        first and everything downstream skips when it fails — so by this
        point the build has succeeded, and an executable in the tree is its
        output. Fresh ones sort first so the just-built command is the one
        actually run.
        """
        after = _executables(self.project)
        before = type(self)._before
        return sorted(after, key=lambda p: (p in before
                                            and after[p] <= before[p],
                                            str(p)))

    def test_it_has_a_build_system(self):
        """The closest thing C has to declaring itself.

        Every sibling contract opens by reading a manifest. C has none, so
        the question becomes whether the project says how to build itself —
        and a single source file says it unambiguously enough.
        """
        if self.plan is None:
            self.skipTest(self.label)
        self.assertTrue(self.plan, "no way to build this project was found")

    def test_the_project_builds(self):
        code, out = self._build()
        self.assertEqual(code, 0,
                         f"the {self.label} build failed:\n{out[-1500:]}")

    def test_it_produces_an_executable(self):
        code, _out = self._build()
        if code != 0:
            self.skipTest("the project does not build, already reported")
        self.assertTrue(
            self._commands(),
            f"the {self.label} build succeeded but there is no executable "
            f"anywhere under {self.project.name} — a C project that builds "
            f"nothing runnable has not been built")

    def test_the_command_starts_without_crashing(self):
        """Run it with no arguments, in a scratch directory.

        A non-zero exit is NOT a failure here — a CLI with no arguments is
        supposed to refuse, and printing usage then exiting 1 is correct
        behaviour. Dying on SIGSEGV or SIGABRT is the real signal: it
        means the program cannot start at all.
        """
        code, _out = self._build()
        if code != 0:
            self.skipTest("the project does not build, already reported")
        fresh = self._commands()
        if not fresh:
            self.skipTest("no runnable command was produced, already "
                          "reported")
        work = Path(self.tmp) / "run"
        work.mkdir(exist_ok=True)
        rc, out = _run([str(fresh[0])], work)
        self.assertFalse(
            _died_on_a_signal(rc),
            f"{fresh[0].name} died on a signal (exit {rc}) instead of "
            f"running — the program cannot start:\n{out[-800:]}")

    def test_it_is_a_program_rather_than_a_stub(self):
        total = 0
        for path in self.project.rglob("*"):
            if not path.is_file() or set(path.parts) & SKIP:
                continue
            if path.suffix.lower() not in (SOURCE_EXT | HEADER_EXT):
                continue
            try:
                total += len(path.read_text(encoding="utf-8",
                                            errors="replace").strip())
            except OSError:
                continue
        self.assertGreater(total, 400,
                           f"the project holds only {total} characters of C "
                           f"source — too little to be a program")


if __name__ == "__main__":
    print(json.dumps({"contract": "c_build_contract"}))
    unittest.main(verbosity=2)

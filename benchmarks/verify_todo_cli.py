"""Independent behavioural check for a generated todo CLI, any language.

Ground truth that does not trust the pipeline's own claim, and that
neither agent under test wrote. The task states an exact CLI contract, so
ONE probe judges every language implementing it — Python, JavaScript, Go,
Rust, Java and C/C++ so far. That is the point: a per-language probe
confounds task with language, and nothing could then say whether a
difference came from the language or the job.

Unlike a launch check, this asserts BEHAVIOUR. Each of the eleven steps
is a FRESH process, so state that does not persist to disk fails, and a
wrong exit code on an out-of-range index fails. Everything runs in a COPY
of the project, so an existing `todos.json` cannot pre-seed the result and
a failed probe cannot damage the artifact.

Two gates are reported separately, because they are different claims:

  G1  the CLI contract above — the one that matters, since neither agent
      wrote it
  G2  the project's OWN test suite — self-authored, so it is recorded
      rather than trusted

An empty collection is a verdict in NEITHER direction. Most runners exit 0
over zero tests, so G2 requires evidence that something actually ran
before calling a green exit a pass; without it a project shipping no tests
scores identically to one whose suite passes.

Every refusal here was a measured false verdict. A build gets its own
timeout, because two correct Maven projects were failed as "timed out"
against a cap meant for a CLI invocation. A `todo` wrapper the project
ships outranks the per-language guess, because one artifact's build copies
dependencies rather than shading them and no guessed invocation could run
it. C source with no Makefile is a definite FAIL rather than undecidable,
because the task requires `make`.

    python verify_todo_cli.py <project-dir>

Exit code 0 = the CLI contract PASSED, 1 = FAILED, 2 = could not be
verified. The full JSON verdict, including G2, goes to stdout. As in
`verify_dt_invariance.py`, 2 is deliberate: a refusal must never be
recorded as a failure.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT = 30

# A build is not a CLI invocation and must not share its budget. Measured
# 2026-09-30: two correct Maven projects were reported FAIL "timed out"
# against the 30s cap, and build in 36s and 38s when allowed to finish -
# a compiler resolving plugins and a `todo add milk` are different kinds
# of operation. Erring long costs the probe minutes; erring short reports
# working code as broken, which is the mistake this whole harness exists
# to avoid.
BUILD_TIMEOUT = 300


_MANIFEST_SOURCES = {
    "pom.xml": ("*.java", "Java"),
    "build.gradle": ("*.java", "Java"),
    "Cargo.toml": ("*.rs", "Rust"),
    "go.mod": ("*.go", "Go"),
    "Makefile": ("*.c", "C"),
    "CMakeLists.txt": ("*.c", "C"),
}


def _manifest_without_sources(root: Path):
    """A build descriptor with no code to build, or None.

    Decidable, so it is a verdict rather than a refusal: a project that
    declares how to build itself and contains nothing to compile cannot
    satisfy any contract step.
    """
    for manifest, (pattern, language) in _MANIFEST_SOURCES.items():
        if not (root / manifest).is_file():
            continue
        found = [p for p in root.rglob(pattern)
                 if not any(part in ("target", "build", ".git", "node_modules")
                            for part in p.relative_to(root).parts)]
        if not found:
            return (f"{manifest} declares a project but there is not one "
                    f"{language} source file in it — nothing was built")
    return None


def _build(root: Path):
    """Compile, for languages that need it. Returns an error or None.

    A build failure is a real verdict, not a probe problem: a todo
    manager that does not compile does not satisfy any contract step.
    """
    if (root / "go.mod").is_file():
        code, out = run(["go", "build", "./..."], root, BUILD_TIMEOUT)
        return None if code == 0 else f"go build failed: {out.strip()[-200:]}"
    if (root / "Cargo.toml").is_file():
        code, out = run(["cargo", "build", "--quiet"], root, BUILD_TIMEOUT)
        return None if code == 0 else f"cargo build failed: {out.strip()[-200:]}"
    if (root / "pom.xml").is_file():
        code, out = run(["mvn", "-q", "-B", "-DskipTests", "package"], root, BUILD_TIMEOUT)
        return None if code == 0 else f"mvn package failed: {out.strip()[-200:]}"
    # C: CMake before make, because a CMake project's generated Makefile
    # lives in a build directory and bare `make` at the top would fail over
    # a project that is fine.
    if (root / "CMakeLists.txt").is_file():
        code, out = run(["cmake", "-S", ".", "-B", "build"], root, BUILD_TIMEOUT)
        if code != 0:
            return f"cmake configure failed: {out.strip()[-200:]}"
        code, out = run(["cmake", "--build", "build"], root, BUILD_TIMEOUT)
        return None if code == 0 else f"cmake build failed: {out.strip()[-200:]}"
    if any((root / n).is_file() for n in ("Makefile", "makefile",
                                          "GNUmakefile")):
        code, out = run(["make"], root, BUILD_TIMEOUT)
        return None if code == 0 else f"make failed: {out.strip()[-200:]}"
    return None


def _task_named_wrapper(root: Path):
    """The invocation the task asked for, when the project supplied it.

    The task says the program MUST be runnable as `todo <command>`, and a
    project that ships a `todo` launcher has answered that literally. The
    per-language guesses below are inferences about how to start the thing;
    a wrapper is the project's own statement, so it outranks them.

    Measured 2026-09-30 on java-aider-3: its wrapper runs
    `java -cp target/todo.jar:target/lib/*`, because the build copies
    dependencies rather than shading them. The `java -jar` guess cannot
    work there and reported a correct artifact as FAIL on all 11 steps.

    Deliberately not a shell invocation: the file is read and its
    interpreter taken from the shebang, so nothing is handed to `sh -c`
    and a wrapper is never a way to smuggle a command past the probe.
    """
    for name in ("todo", "todo.sh", "todo.cmd", "todo.bat"):
        cand = root / name
        if not cand.is_file():
            continue
        if cand.suffix in (".cmd", ".bat"):
            return [str(cand)], "wrapper"
        try:
            # Only the shebang line, never the whole file. In C the
            # task's `todo` IS the compiled binary, so a full read would
            # slurp an entire executable to look at its first two bytes.
            with open(cand, "rb") as fh:
                head = fh.read(256)
            first = head.decode("utf-8", "replace").splitlines()[:1]
        except (OSError, IndexError):
            continue
        if not first or not first[0].startswith("#!"):
            continue
        # `#!/usr/bin/env sh` -> ["sh"]; `#!/bin/bash` -> ["/bin/bash"].
        shebang = first[0][2:].strip().split()
        if not shebang:
            continue
        argv = shebang[1:] if shebang[0].endswith("env") else [shebang[0]]
        if not argv:
            continue
        return argv + [str(cand)], "wrapper"
    return None


def _entry(root: Path):
    """How to run this project, from what the task told it to produce.

    One probe, many languages: the 11-step contract below is identical
    everywhere, because the task states an exact CLI contract rather than
    an implementation. Only the invocation differs.
    """
    wrapper = _task_named_wrapper(root)
    if wrapper is not None:
        return wrapper
    if (root / "main.py").is_file():
        venv = root / "venv" / "Scripts" / "python.exe"
        py = str(venv) if venv.is_file() else sys.executable
        return [py, "main.py"], "python"
    if (root / "index.js").is_file():
        return ["node", "index.js"], "node"
    if (root / "index.mjs").is_file():
        return ["node", "index.mjs"], "node"
    if (root / "go.mod").is_file():
        # Prefer a built binary; `go run .` recompiles on every command,
        # which would make eleven fresh processes needlessly slow.
        for name in ("todo", "todo.exe", "main", "main.exe"):
            exe = root / name
            if exe.is_file():
                return [str(exe)], "go"
        return ["go", "run", "."], "go"
    if (root / "Cargo.toml").is_file():
        for rel in ("target/debug/todo", "target/debug/todo.exe",
                    "target/release/todo", "target/release/todo.exe"):
            exe = root / rel
            if exe.is_file():
                return [str(exe)], "rust"
        return ["cargo", "run", "--quiet", "--"], "rust"
    if any((root / n).is_file() for n in ("Makefile", "makefile",
                                          "GNUmakefile", "CMakeLists.txt")):
        # The task names `./todo`, so that is the contract. A CMake build
        # puts it under build/ by default.
        for rel in ("todo", "todo.exe", "build/todo", "build/todo.exe",
                    "bin/todo", "build/bin/todo"):
            exe = root / rel
            if exe.is_file():
                return [str(exe)], "c"
        # Any single executable the build produced, rather than guessing a
        # name — a Makefile emits whatever its author chose. Sources,
        # objects and CMake's own cmTC_* probes are excluded.
        skip_ext = {".c", ".h", ".cc", ".cpp", ".hpp", ".o", ".obj", ".a",
                    ".so", ".d", ".txt", ".json", ".md", ".cmake", ".mk"}
        cands = [p for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix.lower() not in skip_ext
                 and not p.name.startswith("cmTC_")
                 and "CMakeFiles" not in p.parts
                 and os.access(p, os.X_OK)]
        if cands:
            return [str(cands[0])], "c"
        return None, None
    if (root / "pom.xml").is_file():
        # An executable jar if the build wrote a Main-Class, else the class
        # itself on the compiled classpath. Java is the only language here
        # where "the runnable thing" depends on build configuration rather
        # than existing as a binary.
        import zipfile
        for jar in sorted((root / "target").glob("*.jar")
                          if (root / "target").is_dir() else []):
            try:
                with zipfile.ZipFile(jar) as zf:
                    mf = zf.read("META-INF/MANIFEST.MF").decode(
                        "utf-8", "replace")
                if re.search(r"(?mi)^Main-Class:\s*\S+", mf):
                    return ["java", "-jar", str(jar)], "java"
            except (OSError, KeyError, zipfile.BadZipFile):
                continue
        classes = root / "target" / "classes"
        main_re = re.compile(
            r"public\s+static\s+void\s+main\s*\(\s*(?:final\s+)?String")
        for src in sorted(root.rglob("*.java")):
            if "test" in src.parts:
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not main_re.search(text):
                continue
            # The class name comes from the package declaration, not the
            # path: `Main.java` in `package todo` is `todo.Main`, and
            # running the stem throws "Could not find or load main class".
            pkg = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", text)
            fqcn = f"{pkg.group(1)}.{src.stem}" if pkg else src.stem
            if classes.is_dir():
                return ["java", "-cp", str(classes), fqcn], "java"
        return None, None
    return None, None


def run(cmd, cwd, timeout=TIMEOUT):
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return None, "timed out"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _c_sources_without_a_build_file(root: Path):
    """C code that nothing can build, or None.

    The mirror of `_manifest_without_sources`: that catches a build file
    with no code, this catches code with no build file. Both are definite
    failures rather than undecidable ones, because the task states
    outright that the project MUST build with a single `make` command — so
    a tree with no Makefile has not met the spec, and reporting UNKNOWN
    would hide a real miss behind "could not judge".

    Scoped to C deliberately. Every other language here declares itself
    through a manifest, so this question only arises where there is none.
    """
    if any((root / n).is_file() for n in ("Makefile", "makefile",
                                          "GNUmakefile", "CMakeLists.txt",
                                          "meson.build")):
        return None
    if any((root / m).is_file() for m in _MANIFEST_SOURCES):
        return None
    sources = [p for p in root.rglob("*")
               if p.is_file() and p.suffix.lower() in (".c", ".cc", ".cpp",
                                                       ".cxx")
               and ".git" not in p.parts]
    if not sources:
        return None
    return (f"{len(sources)} C source file(s) and no Makefile, "
            f"CMakeLists.txt or meson.build — the task requires the project "
            f"to build with a single `make`")


def check_contract(root: Path):
    """G1 - the CLI contract, exercised as a user would.

    Runs in a COPY so the project's own todos.json cannot pre-seed the
    result, and so a failed probe leaves the artifact untouched.
    """
    empty = _manifest_without_sources(root)
    if empty:
        # A manifest with no code is a DEFINITE failure, not an undecidable
        # one. Measured 2026-09-30: all three aider Java runs wrote a valid
        # pom.xml and not one .java file, in ~40s. Reporting UNKNOWN would
        # have hidden that behind "could not judge" — the same mistake as
        # calling a build that never terminates undecidable.
        return "FAIL", empty, {"steps_passed": 0, "steps_total": 11}
    # A compiled language may have no runnable entry until it is BUILT, and
    # the build happens below. Go and Rust hide this behind `go run .` and
    # `cargo run --`, which work from source; Java has no equivalent, so a
    # Maven project with sources but no target/ looked like "no entry point"
    # and three correct artifacts were reported UNKNOWN. A manifest is
    # enough to proceed — the post-build check at the bottom is the one
    # that decides.
    unbuildable = _c_sources_without_a_build_file(root)
    if unbuildable:
        return "FAIL", unbuildable, {"steps_passed": 0, "steps_total": 11}
    entry, _kind = _entry(root)
    if entry is None and not any((root / m).is_file() for m in
                                 _MANIFEST_SOURCES):
        return "UNKNOWN", ("no entry point found (main.py / index.js / "
                           "go.mod / Cargo.toml / pom.xml)"), {}

    tmp = Path(tempfile.mkdtemp(prefix="todo-probe-"))
    work = tmp / "app"
    try:
        shutil.copytree(root, work, ignore=shutil.ignore_patterns(
            "node_modules", ".git", "venv", ".venv", "__pycache__",
            ".agentchanti", "todos.json",
            # A CMakeCache.txt records the ABSOLUTE source path it was
            # configured for, so carrying one into the probe's temp copy
            # makes cmake refuse outright: "The source ... does not match
            # the source used to generate cache". Measured 2026-09-30 on a
            # correct CMake project, reported as `cmake configure failed`.
            # Dropping the cache makes it reconfigure, which is what a
            # fresh checkout would do anyway.
            "CMakeCache.txt", "CMakeFiles"))
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return "UNKNOWN", f"could not copy the project: {exc}", {}

    build_error = _build(work)
    if build_error:
        shutil.rmtree(tmp, ignore_errors=True)
        return "FAIL", build_error, {"steps_passed": 0, "steps_total": 11}
    # A compiled language may only have produced its binary just now.
    entry, _kind = _entry(work)
    if entry is None:
        shutil.rmtree(tmp, ignore_errors=True)
        return "UNKNOWN", "no entry point after building", {}

    steps, failures = [], []

    def step(name, args, want_code=0, want_in=None, want_not_in=None):
        code, out = run(entry + args, work)
        ok = True
        why = ""
        if code is None:
            ok, why = False, out
        elif want_code is not None and code != want_code:
            ok, why = False, f"exit {code}, expected {want_code}"
        elif want_in and not all(w.lower() in out.lower() for w in want_in):
            missing = [w for w in want_in if w.lower() not in out.lower()]
            ok, why = False, f"output missing {missing}: {out.strip()[:90]!r}"
        elif want_not_in and any(w.lower() in out.lower() for w in want_not_in):
            ok, why = False, f"output should not contain {want_not_in}"
        steps.append((name, ok))
        if not ok:
            failures.append(f"{name}: {why}")
        return out

    # Every call is a NEW PROCESS: state must reach disk to survive.
    step("empty list", ["list"], 0, want_in=["no tasks"])
    step("add first", ["add", "Buy milk"], 0)
    step("add second", ["add", "Write tests"], 0)
    step("list shows both", ["list"], 0, want_in=["buy milk", "write tests"])
    step("complete one", ["done", "1"], 0)
    out = step("completion persisted", ["list"], 0, want_in=["buy milk"])
    if "[x]" not in out.lower():
        steps.append(("done marks [x]", False))
        failures.append(f"done marks [x]: no [x] in {out.strip()[:90]!r}")
    else:
        steps.append(("done marks [x]", True))
    step("remove", ["remove", "2"], 0)
    step("removal persisted", ["list"], 0, want_not_in=["write tests"])
    step("bad index exits 1", ["done", "99"], 1)

    persisted = (work / "todos.json").is_file()
    steps.append(("persists to todos.json", persisted))
    if not persisted:
        failures.append("persists to todos.json: file was never written")

    shutil.rmtree(tmp, ignore_errors=True)
    passed = sum(1 for _n, ok in steps if ok)
    info = {"steps_passed": passed, "steps_total": len(steps)}
    if passed == len(steps):
        return "PASS", f"all {len(steps)} contract steps passed", info
    return "FAIL", f"{passed}/{len(steps)} steps — " + "; ".join(
        failures[:3]), info


def _suite_kind(root: Path):
    """Which test runner owns this project.

    Deliberately NOT `_entry`'s kind. How a program is launched and which
    runner collects its tests are different questions, and a project that
    ships a `todo` wrapper answers only the first: measured 2026-09-30,
    taking the kind from `_entry` turned all six Java artifacts into
    `unknown project kind` the moment the wrapper was preferred. The
    manifest is what names the runner, so ask the manifest.
    """
    if (root / "pom.xml").is_file() or (root / "build.gradle").is_file():
        return "java"
    if any((root / n).is_file() for n in ("Makefile", "makefile",
                                          "GNUmakefile", "CMakeLists.txt")):
        return "c"
    if (root / "Cargo.toml").is_file():
        return "rust"
    if (root / "go.mod").is_file():
        return "go"
    if (root / "package.json").is_file():
        return "node"
    # C source with no build file at all — the project is broken, but it is
    # still a C project, and answering "python" here reports the wrong
    # instrument as unavailable instead of the real defect.
    if any(root.rglob("*.c")) or any(root.rglob("*.cpp")):
        return "c"
    # The seeded acceptance contract is a .py the HARNESS wrote, not the
    # project's own code, so it cannot make a project Python. Measured
    # 2026-09-30: a C artifact with no Makefile held exactly one .py —
    # `test_acceptance_contract.py` — and was reported as
    # `G2=UNKNOWN pytest unavailable`, which says nothing true about it.
    own = [p for p in root.glob("*.py")
           if p.name != "test_acceptance_contract.py"]
    if (root / "main.py").is_file() or own:
        return "python"
    return None


def check_own_tests(root: Path):
    """G2 - the project's own suite. Self-authored, reported separately."""
    kind = _suite_kind(root)
    if kind == "python":
        venv = root / "venv" / "Scripts" / "python.exe"
        py = str(venv) if venv.is_file() else sys.executable
        if not any(root.rglob("test*.py")):
            return "UNKNOWN", "no tests shipped"
        code, out = run([py, "-m", "pytest", "-q", "--no-header"], root, BUILD_TIMEOUT)
        if code is None or "No module named pytest" in out:
            return "UNKNOWN", "pytest unavailable"
        counts = re.findall(r"(\d+) (passed|failed|error[s]?)", out)
        if not counts:
            return "UNKNOWN", (out.strip().splitlines() or ["no output"])[-1][:90]
        bad = sum(int(n) for n, k in counts if k != "passed")
        good = sum(int(n) for n, k in counts if k == "passed")
        summary = ", ".join(f"{n} {k}" for n, k in counts)
        return ("PASS", summary) if bad == 0 and good else ("FAIL", summary)
    if kind == "node":
        pkg = root / "package.json"
        if not pkg.is_file():
            return "UNKNOWN", "no package.json"
        try:
            scripts = json.loads(pkg.read_text(encoding="utf-8",
                                               errors="replace")).get("scripts", {})
        except ValueError:
            return "UNKNOWN", "package.json will not parse"
        if "test" not in scripts:
            return "UNKNOWN", "no test script"
        npm = "npm.cmd" if os.name == "nt" else "npm"
        code, out = run([npm, "test", "--silent"], root, BUILD_TIMEOUT)
        if code is None:
            return "UNKNOWN", out
        if "no test" in out.lower() and code == 0:
            return "UNKNOWN", "the test script ran nothing"
        # A green exit is not self-describing: `--silent` hides the runner's
        # summary, and `node --test` over zero files exits 0 too. Require
        # some evidence that a test actually ran before calling it a pass -
        # the same rule the Maven branch needed.
        if code == 0 and not re.search(
                r"(?i)\b(?:pass(?:ed|ing)?|ok|tests?\s+\d+|✓|√)\b", out):
            return "UNKNOWN", "test script exited 0 with no sign a test ran"
        return ("PASS" if code == 0 else "FAIL",
                (out.strip().splitlines() or ["no output"])[-1][:90])
    if kind == "go":
        if not any(root.rglob("*_test.go")):
            return "UNKNOWN", "no tests shipped"
        code, out = run(["go", "test", "./..."], root, BUILD_TIMEOUT)
        if code is None:
            return "UNKNOWN", out
        # A package with no tests prints `? pkg [no test files]` and exits
        # 0. If EVERY package says that, a _test.go file exists somewhere
        # that go did not collect (wrong package clause, build tag), and a
        # green exit means nothing was checked.
        ran = [ln for ln in out.splitlines() if ln.strip().startswith("ok")]
        if not ran and "[no test files]" in out:
            return "UNKNOWN", "go collected no tests in any package"
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:]
        return ("PASS" if code == 0 else "FAIL",
                (tail[0] if tail else f"exit {code}")[:90])
    if kind == "rust":
        # `cargo test` compiles the crate, so a build failure surfaces here
        # too — but that is G1's verdict, not this one's.
        code, out = run(["cargo", "test", "--quiet"], root, BUILD_TIMEOUT)
        if code is None:
            return "UNKNOWN", out
        # EVERY target's result line, summed. cargo runs lib, bin and
        # doc-tests separately, and the trailing ones legitimately print
        # `running 0 tests`. Matching "0 tests" anywhere read those as the
        # whole verdict and discarded a real 7-passed result — the same
        # mistake as taking one empty collection for the suite.
        import re as _re
        results = _re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed",
                              out)
        passed = sum(int(p) for p, _f in results)
        failed = sum(int(f) for _p, f in results)
        if failed:
            return "FAIL", f"{passed} passed, {failed} failed"
        if passed:
            return "PASS", f"{passed} passed across {len(results)} target(s)"
        if not results:
            tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:]
            return "UNKNOWN", (tail[0] if tail else f"exit {code}")[:90]
        return "UNKNOWN", "no tests collected in any target"
    if kind == "java":
        if not (root / "pom.xml").is_file():
            return "UNKNOWN", "no pom.xml to run tests with"
        # NOT -q: quiet suppresses the INFO lines carrying "Tests run:"
        # and "No tests to run", so the output arrives empty and a project
        # with no tests at all falls through to its exit code and reads as
        # a PASS. Measured 2026-09-30 on java-agentchanti-3, which ships
        # zero test files and was graded PASS "no output" - the empty-suite
        # mistake agentchanti's own evidence layer refuses to make.
        code, out = run(["mvn", "-B", "test"], root, BUILD_TIMEOUT)
        if code is None:
            return "UNKNOWN", out
        rows = re.findall(
            r"Tests run: (\d+), Failures: (\d+), Errors: (\d+)", out)
        if rows:
            # Surefire prints a line per class AND a total; taking the last
            # is the `cargo test` per-target mistake in another dialect, so
            # use the largest run count and its row.
            ran, fail, err = max(((int(a), int(b), int(c)) for a, b, c in rows),
                                 key=lambda r: r[0])
            if not ran:
                return "UNKNOWN", "no tests collected"
            return ("PASS" if fail == 0 and err == 0 else "FAIL",
                    f"{ran} run, {fail} failures, {err} errors")
        if "No tests to run" in out or "Tests are skipped" in out:
            return "UNKNOWN", "no tests collected"
        return ("PASS" if code == 0 else "FAIL",
                (out.strip().splitlines() or ["no output"])[-1][:90])
    if kind == "c":
        mk = next((n for n in ("Makefile", "makefile", "GNUmakefile")
                   if (root / n).is_file()), None)
        if mk is None:
            return "UNKNOWN", "no Makefile to run `make test` with"
        text = (root / mk).read_text(encoding="utf-8", errors="replace")
        if not re.search(r"(?m)^test\s*:", text):
            return "UNKNOWN", "the Makefile declares no `test` target"
        code, out = run(["make", "test"], root, BUILD_TIMEOUT)
        if code is None:
            return "UNKNOWN", out
        # An empty collection is a verdict in neither direction, and C has
        # no standard runner to ask — so require some sign a test ran
        # before calling a green exit a pass. The Maven branch needed the
        # same rule after `-q` hid its counts.
        if code == 0 and not re.search(
                r"(?i)\b(?:pass(?:ed|ing)?|ok|assert|tests?\s+\d+"
                r"|\d+\s+tests?)\b", out):
            return "UNKNOWN", "`make test` exited 0 with no sign a test ran"
        return ("PASS" if code == 0 else "FAIL",
                (out.strip().splitlines() or ["no output"])[-1][:90])
    return "UNKNOWN", "unknown project kind"


def main():
    root = Path(sys.argv[1]).resolve()
    out = {"project": str(root)}
    entry, kind = _entry(root)
    out["entry"] = " ".join(entry) if entry else None
    out["kind"] = kind
    for name, fn in (("G1_cli_contract", lambda: check_contract(root)),
                     ("G2_own_tests", lambda: check_own_tests(root))):
        try:
            res = fn()
            out[name] = {"verdict": res[0], "detail": res[1]}
            if len(res) > 2:
                out[name].update(res[2])
        except Exception as exc:                     # never a false FAIL
            out[name] = {"verdict": "UNKNOWN",
                         "detail": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out, indent=2))
    # Exit on G1 alone. G2 is the project's own suite, which it wrote, so
    # it is reported and never allowed to decide the verdict — the same
    # demotion `evidence.py` applies to a seeded contract.
    return {"PASS": 0, "FAIL": 1}.get(out["G1_cli_contract"]["verdict"], 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-4].strip())
        sys.exit(2)
    sys.exit(main())

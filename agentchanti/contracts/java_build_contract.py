"""A deterministic acceptance check for a Java project.

WHY THIS FILE EXISTS

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a contract this pipeline
seeds — and the seeder could write one for Python, JavaScript, Go and
Rust. A greenfield Java build had none of the three no matter how well it
went. This is the fifth instance of one structural gap: evidence seeding
is per language, and every new language starts with none.

WHY IT IS SHIPPED RATHER THAN GENERATED

The JavaScript path learned this the expensive way: four consecutive
model-written contracts each failed a correct project, in four different
ways, because the space of ways to observe a build wrongly is larger than
a prompt can enumerate.

WHY IT IS PYTHON AND NOT JAVA

For Go the reason was that a Go file joins the module; for Rust, that a
Rust file is compiled by the very build being measured. Java is worse
still: a `.java` file under `src/` is compiled by `mvn package`, a file
under `src/test/java` is compiled AND executed by Surefire, and a
compile error in either fails the build the contract is supposed to
observe. Python is present wherever agentchanti runs, so the contract
stays outside the project.

WHAT MAKES JAVA HARDER THAN GO OR RUST

There is no single "the build". Go has `go build`, Rust has `cargo
build`; Java has Maven, Gradle or bare `javac`, and they are not
interchangeable. And "a runnable command" is ambiguous — Go and Rust
emit a binary, while Java emits classes or a jar that is only executable
if the build was configured to write a `Main-Class` manifest entry. So
the floor here is "a main method exists and the program starts", which is
weaker than for the other two and honest about it.

WHAT IT DOES AND DOES NOT CLAIM

A floor, not a ceiling: the project declares a build, it compiles, a main
class is reachable, running it does not throw on startup, and there is
more than a stub's worth of Java. It knows nothing about the task.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("AGENTCHANTI_PROJECT_ROOT", ".")).resolve()
SKIP = {"target", "build", ".git", ".agentchanti", "node_modules", ".gradle"}
TIMEOUT = 900

# Java's spelling of a crash. A non-zero exit is NOT one of these: a CLI
# invoked with no arguments is supposed to refuse, and treating that as a
# crash is the mistake the smoke test made when it rewrote three working
# programs.
_CRASH_RE = re.compile(
    r"Exception in thread \"main\"|Could not find or load main class"
    r"|java\.lang\.NoClassDefFoundError|A fatal error has been detected")

# A build that failed because it could not reach Maven Central is the
# instrument being unavailable, never the code being wrong. Java's tree is
# larger than Rust's and there is no equivalent of `cargo fetch` for
# whatever dependency the model happened to choose.
_NETWORK_RE = re.compile(
    r"Could not resolve dependencies|Could not transfer artifact"
    r"|Connection (?:timed out|refused)|Network is unreachable"
    r"|Could not find artifact|repository .* was cached in the local"
    r"|PKIX|SSLHandshake|UnknownHost",
    re.IGNORECASE)

_MAIN_RE = re.compile(
    r"public\s+static\s+void\s+main\s*\(\s*(?:final\s+)?String")


def _project_dir():
    """The directory owning the build, or holding the sources."""
    for probe in (lambda d: (d / "pom.xml").is_file(),
                  lambda d: (d / "build.gradle").is_file()
                  or (d / "build.gradle.kts").is_file(),
                  lambda d: any(d.rglob("*.java"))):
        if probe(ROOT):
            return ROOT
        for entry in sorted(ROOT.iterdir()):
            if entry.is_dir() and entry.name not in SKIP and probe(entry):
                return entry
    return None


def _run(args, cwd):
    try:
        p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                           errors="replace", timeout=TIMEOUT)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return None, f"timed out after {TIMEOUT}s"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _java_sources(project):
    out = []
    for path in project.rglob("*.java"):
        if any(part in SKIP for part in path.relative_to(project).parts):
            continue
        out.append(path)
    return out


class JavaBuildContract(unittest.TestCase):
    """The floor every Java project has to clear."""

    @classmethod
    def setUpClass(cls):
        cls.project = _project_dir()
        if cls.project is None:
            raise unittest.SkipTest(
                "no pom.xml, build.gradle or .java file found — not a Java "
                "project, so this contract does not apply")
        if shutil.which("java") is None:
            raise unittest.SkipTest("no JDK is installed")
        cls.maven = (cls.project / "pom.xml").is_file()
        cls.gradle = ((cls.project / "build.gradle").is_file()
                      or (cls.project / "build.gradle.kts").is_file())
        cls.tmp = tempfile.mkdtemp(prefix="java-contract-")
        cls._built = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def _build(self):
        """Build once, shared across tests. Skips on a registry failure."""
        cls = type(self)
        if cls._built is None:
            if cls.maven:
                if shutil.which("mvn") is None:
                    cls._built = (None, "__skip__ maven is not installed")
                else:
                    cls._built = _run(
                        ["mvn", "-q", "-B", "-DskipTests", "package"],
                        self.project)
            elif cls.gradle:
                if shutil.which("gradle") is None:
                    cls._built = (None, "__skip__ this project uses Gradle "
                                        "and gradle is not installed, so the "
                                        "build cannot be observed here")
                else:
                    cls._built = _run(["gradle", "-q", "build", "-x", "test"],
                                      self.project)
            else:
                out_dir = Path(self.tmp) / "classes"
                out_dir.mkdir(exist_ok=True)
                srcs = [str(p) for p in _java_sources(self.project)]
                cls._built = _run(["javac", "-d", str(out_dir)] + srcs,
                                  self.project) if srcs else (
                    None, "__skip__ no .java sources to compile")
        code, out = cls._built
        if code is None and out.startswith("__skip__"):
            self.skipTest(out[len("__skip__"):].strip())
        if code != 0 and _NETWORK_RE.search(out or ""):
            self.skipTest(f"the build could not reach its dependency "
                          f"repository, so this proves nothing about the "
                          f"code:\n{(out or '')[-500:]}")
        return code, out

    def test_the_project_declares_a_build(self):
        if self.maven:
            text = (self.project / "pom.xml").read_text(encoding="utf-8",
                                                        errors="replace")
            self.assertIn("<artifactId>", text,
                          "pom.xml declares no artifactId")
        elif self.gradle:
            self.assertTrue(True, "a Gradle build script is present")
        else:
            self.assertTrue(_java_sources(self.project),
                            "no build descriptor and no .java sources")

    def test_it_compiles(self):
        code, out = self._build()
        self.assertEqual(code, 0,
                         f"the build failed:\n{(out or '')[-1500:]}")

    def _main_class(self):
        """A class with a main method, preferring an executable jar."""
        for jar_dir in ("target", "build/libs"):
            for jar in sorted((self.project / jar_dir).glob("*.jar")
                              if (self.project / jar_dir).is_dir() else []):
                try:
                    with zipfile.ZipFile(jar) as zf:
                        mf = zf.read("META-INF/MANIFEST.MF").decode(
                            "utf-8", "replace")
                    m = re.search(r"(?mi)^Main-Class:\s*(\S+)", mf)
                    if m:
                        return ("jar", jar, m.group(1))
                except (OSError, KeyError, zipfile.BadZipFile):
                    continue
        for path in _java_sources(self.project):
            try:
                if _MAIN_RE.search(path.read_text(encoding="utf-8",
                                                  errors="replace")):
                    return ("source", path, None)
            except OSError:
                continue
        return None

    def test_a_main_method_is_reachable(self):
        code, _out = self._build()
        if code != 0:
            self.skipTest("the project does not build, already reported")
        found = self._main_class()
        self.assertIsNotNone(
            found,
            "no main method found: no jar declares a Main-Class and no "
            "source declares `public static void main(String…)`")

    def test_the_program_starts_without_throwing(self):
        """Run it with no arguments, in a scratch directory.

        A non-zero exit is NOT a failure here — a CLI invoked with no
        arguments is supposed to refuse. An uncaught exception is the real
        signal: it means the program cannot start.
        """
        code, _out = self._build()
        if code != 0:
            self.skipTest("the project does not build, already reported")
        found = self._main_class()
        if found is None:
            self.skipTest("no main method to start, already reported")
        kind, where, main_class = found
        work = Path(self.tmp) / "run"
        work.mkdir(exist_ok=True)
        if kind == "jar":
            _rc, out = _run(["java", "-jar", str(where)], work)
        else:
            classes = self.project / "target" / "classes"
            if not classes.is_dir():
                classes = Path(self.tmp) / "classes"
            if not classes.is_dir():
                self.skipTest("no compiled classes to run")
            # Derive the class name from its package declaration, because a
            # file path is not a class name once packages are involved.
            text = where.read_text(encoding="utf-8", errors="replace")
            pkg = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", text)
            fqcn = f"{pkg.group(1)}.{where.stem}" if pkg else where.stem
            _rc, out = _run(["java", "-cp", str(classes), fqcn], work)
        self.assertNotRegex(out, _CRASH_RE,
                            f"the program threw on startup:\n{out[-800:]}")

    def test_it_is_a_program_rather_than_a_stub(self):
        total = 0
        for path in _java_sources(self.project):
            if "test" in path.parts or path.name.endswith("Test.java"):
                continue
            try:
                total += len(path.read_text(encoding="utf-8",
                                            errors="replace").strip())
            except OSError:
                continue
        self.assertGreater(total, 400,
                           f"the project holds only {total} characters of "
                           f"non-test Java — too little to be a program")


if __name__ == "__main__":
    print(json.dumps({"contract": "java_build_contract"}))
    unittest.main(verbosity=2)

"""A deterministic acceptance check for a Rust project.

WHY THIS FILE EXISTS

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a contract this pipeline
seeds — and until now the seeder could write one for Python, JavaScript
and Go. A greenfield Rust build therefore had none of the three no matter
how well it went, which is the same defect recorded for JavaScript before
`_seed_js_builtin` and for Go before `_seed_go_builtin`. Evidence seeding
is per language, and every new language starts with none.

WHY IT IS SHIPPED RATHER THAN GENERATED

The JavaScript path learned this the expensive way: four consecutive
model-written contracts each failed a correct project, in four different
ways, because the space of ways to observe a build wrongly is larger than
a prompt can enumerate. So the contract for a Rust project is written
once, by people who can run it.

It is still independent in the sense that matters: nothing in it was
authored by the model whose work it judges, and the hash check withdraws
it as evidence if the run edits it.

WHY IT IS PYTHON AND NOT RUST

More strongly than for Go. A Rust file placed in the crate is *compiled*
— by `cargo build` if it lands in `src/`, by `cargo test` if it lands in
`tests/` — so a contract written in Rust can break the very compilation
it exists to measure, and a compile error in the instrument would be
reported as a defect in the artifact. Python is already present wherever
agentchanti runs, so the contract stays outside the crate and observes it
from there.

WHAT IT DOES AND DOES NOT CLAIM

A floor, not a ceiling: the crate declares itself, it builds, a runnable
command comes out of it, that command starts without panicking, and there
is more than a stub's worth of source. It knows nothing about the task.
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
SKIP = {"target", ".git", ".agentchanti", "node_modules", "vendor"}
TIMEOUT = 600

# Rust's spelling of a crash. A non-zero exit is NOT one of these: a CLI
# invoked with no arguments is supposed to refuse, and treating that as a
# crash is the mistake the smoke test made when it rewrote three working
# programs.
_PANIC_RE = re.compile(r"thread '[^']*' panicked at|note: run with `RUST_BACKTRACE",
                       re.IGNORECASE)

# A build that failed because it could not reach crates.io is the
# instrument being unavailable, never the code being wrong. Rust makes
# this likelier than Go, which needs no registry for a JSON CLI.
_NETWORK_RE = re.compile(
    r"failed to (?:fetch|download|get)|could not connect|network failure"
    r"|error sending request|spurious network error|timed out"
    r"|failed to resolve|registry index|SSL|certificate",
    re.IGNORECASE)


def _crate_dir():
    """The directory owning Cargo.toml — often the root, sometimes below."""
    if (ROOT / "Cargo.toml").is_file():
        return ROOT
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and entry.name not in SKIP:
            if (entry / "Cargo.toml").is_file():
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


class RustBuildContract(unittest.TestCase):
    """The floor every Rust project has to clear."""

    @classmethod
    def setUpClass(cls):
        cls.crate = _crate_dir()
        if cls.crate is None:
            raise unittest.SkipTest(
                "no Cargo.toml found — not a Rust crate, so this contract "
                "does not apply")
        if shutil.which("cargo") is None:
            raise unittest.SkipTest("the cargo toolchain is not installed")
        cls.tmp = tempfile.mkdtemp(prefix="rust-contract-")
        cls._built = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def _build(self):
        """Build once, shared across tests. Skips on a registry failure."""
        cls = type(self)
        if cls._built is None:
            code, out = _run(["cargo", "build", "--quiet"], self.crate)
            cls._built = (code, out)
        code, out = cls._built
        if code != 0 and _NETWORK_RE.search(out):
            self.skipTest(f"cargo could not reach the registry, so this "
                          f"proves nothing about the code:\n{out[-500:]}")
        return code, out

    def test_crate_declares_itself(self):
        text = (self.crate / "Cargo.toml").read_text(encoding="utf-8",
                                                     errors="replace")
        self.assertRegex(text, r"(?m)^\s*\[package\]",
                         "Cargo.toml declares no [package] section")
        self.assertRegex(text, r"(?m)^\s*name\s*=",
                         "Cargo.toml declares no package name")

    def test_the_crate_builds(self):
        code, out = self._build()
        self.assertEqual(code, 0, f"`cargo build` failed:\n{out[-1500:]}")

    def test_it_produces_a_runnable_command(self):
        """A library crate is fine; a project asked for a CLI must build one."""
        code, _out = self._build()
        if code != 0:
            self.skipTest("the crate does not build, already reported")
        binaries = [p for p in (self.crate / "target" / "debug").glob("*")
                    if p.is_file() and os.access(p, os.X_OK)
                    and p.suffix.lower() in ("", ".exe")
                    and not p.name.startswith(".")]
        if not binaries and not (self.crate / "src" / "main.rs").is_file():
            self.skipTest("no src/main.rs — this crate is a library, not a "
                          "command")
        self.assertTrue(binaries,
                        "src/main.rs exists but `cargo build` produced no "
                        "executable in target/debug")

    def test_the_command_starts_without_panicking(self):
        """Run it with no arguments, in a scratch directory.

        A non-zero exit is NOT a failure here — a CLI with no arguments is
        supposed to refuse. A panic is the real signal: it means the
        binary cannot start at all.
        """
        code, _out = self._build()
        if code != 0:
            self.skipTest("the crate does not build, already reported")
        binaries = [p for p in (self.crate / "target" / "debug").glob("*")
                    if p.is_file() and os.access(p, os.X_OK)
                    and p.suffix.lower() in ("", ".exe")
                    and not p.name.startswith(".")]
        if not binaries:
            self.skipTest("no runnable command to start")
        work = Path(self.tmp) / "run"
        work.mkdir(exist_ok=True)
        _rc, out = _run([str(binaries[0])], work)
        self.assertNotRegex(out, _PANIC_RE,
                            f"the command panicked on startup:\n{out[-800:]}")

    def test_it_is_a_program_rather_than_a_stub(self):
        total = 0
        src = self.crate / "src"
        for path in (src.rglob("*.rs") if src.is_dir() else []):
            try:
                total += len(path.read_text(encoding="utf-8",
                                            errors="replace").strip())
            except OSError:
                continue
        self.assertGreater(total, 400,
                           f"src/ holds only {total} characters of Rust "
                           f"source — too little to be a program")


if __name__ == "__main__":
    print(json.dumps({"contract": "rust_build_contract"}))
    unittest.main(verbosity=2)

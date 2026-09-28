"""A deterministic acceptance check for a Go project.

WHY THIS FILE EXISTS

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a contract this pipeline
seeds — and until now the seeder could write one only for Python and
JavaScript. A greenfield Go build therefore had none of the three no
matter how well it went. Measured 2026-09-28: a containerised Go run
produced a todo manager that passes an external 11-step behavioural
probe, and exited 1 on the last line with "nothing outside this run's own
output verified it". That is the same defect already recorded for
JavaScript before `_seed_js_builtin` existed.

WHY IT IS SHIPPED RATHER THAN GENERATED

The JavaScript path learned this the expensive way: four consecutive
model-written contracts each failed a correct project, in four different
ways, because the space of ways to observe a build wrongly is larger than
a prompt can enumerate. So the contract for a Go project is written once,
by people who can run it.

It is still independent in the sense that matters: nothing in it was
authored by the model whose work it judges, and the hash check withdraws
it as evidence if the run edits it.

WHAT IT DOES AND DOES NOT CLAIM

A floor, not a ceiling. It proves the module builds, that a runnable
command comes out, and that running that command does not panic. It
knows nothing about the task.

It is written in Python, not Go, on purpose. A Go file would be part of
the module under test — collected by `go test ./...`, built by
`go build ./...`, and able to break the very compilation it is meant to
measure. Python is already present wherever agentchanti runs, so the
contract stays outside the artifact and observes it from the outside.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("AGENTCHANTI_PROJECT_ROOT", ".")).resolve()
SKIP = {"vendor", ".git", ".agentchanti", "node_modules", "testdata"}
TIMEOUT = 300


def _go_module_dir():
    """The directory owning go.mod — often the root, sometimes below it."""
    if (ROOT / "go.mod").is_file():
        return ROOT
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and entry.name not in SKIP:
            if (entry / "go.mod").is_file():
                return entry
    return None


def _run(args, cwd, env=None):
    try:
        p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                           errors="replace", timeout=TIMEOUT, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return None, f"timed out after {TIMEOUT}s"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


class GoBuildContract(unittest.TestCase):
    """The floor every Go project has to clear."""

    @classmethod
    def setUpClass(cls):
        cls.module = _go_module_dir()
        if cls.module is None:
            raise unittest.SkipTest(
                "no go.mod found — not a Go module, so this contract does "
                "not apply")
        if shutil.which("go") is None:
            # An absent toolchain is the instrument being unavailable and
            # must never convict the code.
            raise unittest.SkipTest("the go toolchain is not installed")
        cls.tmp = tempfile.mkdtemp(prefix="go-contract-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_module_declares_itself(self):
        text = (self.module / "go.mod").read_text(encoding="utf-8",
                                                  errors="replace")
        self.assertRegex(text, r"(?m)^module\s+\S+",
                         "go.mod declares no module path")

    def test_every_package_compiles(self):
        code, out = _run(["go", "build", "./..."], self.module)
        self.assertEqual(code, 0, f"`go build ./...` failed:\n{out[-1200:]}")

    def test_it_produces_a_runnable_command(self):
        """A library is fine, but a project asked for a CLI must build one."""
        out_path = Path(self.tmp) / "artifact"
        code, out = _run(["go", "build", "-o", str(out_path), "."],
                         self.module)
        if code != 0 and "no Go files" in out:
            self.skipTest("the root package is not a main package — this "
                          "project is a library, not a command")
        self.assertEqual(code, 0, f"`go build -o … .` failed:\n{out[-1200:]}")
        self.assertTrue(out_path.is_file() or
                        out_path.with_suffix(".exe").is_file(),
                        "the build reported success but produced no binary")

    def test_the_command_starts_without_panicking(self):
        """Run it with no arguments, in a scratch directory.

        A non-zero exit is NOT a failure here: a CLI invoked with no
        arguments is supposed to refuse, and treating that as a crash is
        a mistake this project has already made once, in the smoke test,
        where it rewrote three working programs. A panic is the real
        signal — it means the binary cannot start at all.
        """
        out_path = Path(self.tmp) / "artifact"
        if not out_path.is_file():
            code, out = _run(["go", "build", "-o", str(out_path), "."],
                             self.module)
            if code != 0:
                self.skipTest("no runnable command to start")
        work = Path(self.tmp) / "run"
        work.mkdir(exist_ok=True)
        _code, out = _run([str(out_path)], work)
        self.assertNotIn("panic:", out,
                         f"the command panicked on startup:\n{out[-800:]}")
        self.assertNotIn("goroutine 1 [running]", out,
                         f"the command panicked on startup:\n{out[-800:]}")

    def test_it_is_a_program_rather_than_a_stub(self):
        """Some real Go source, so an empty module cannot pass."""
        total = 0
        for path in self.module.rglob("*.go"):
            if any(part in SKIP for part in path.relative_to(self.module).parts):
                continue
            if path.name.endswith("_test.go"):
                continue
            try:
                total += len(path.read_text(encoding="utf-8",
                                            errors="replace").strip())
            except OSError:
                continue
        self.assertGreater(total, 400,
                           f"the module holds only {total} characters of "
                           f"non-test Go source — too little to be a program")


if __name__ == "__main__":
    print(json.dumps({"contract": "go_build_contract"}))
    unittest.main(verbosity=2)

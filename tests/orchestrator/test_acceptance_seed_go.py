"""A greenfield Go build can earn independent evidence.

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a contract this pipeline
seeds — and the seeder could write one only for Python and JavaScript.
So a Go project had none of the three however well it went.

Measured 2026-09-28: a containerised Go run produced a todo manager that
passes an external 11-step behavioural probe, and exited 1 on the last
line with "nothing outside this run's own output verified it". That is
the third instance of one structural gap — evidence seeding is per
language, and every new language starts with none.

The contract is shipped rather than generated, for the reason
`_seed_js_builtin` records: four consecutive model-written JS contracts
each failed a correct project, in four different ways.
"""
from pathlib import Path

from agentchanti.orchestrator.acceptance_seed import (BUILTIN_GO_CONTRACT,
                                                      SEED_BASENAME,
                                                      seed_acceptance_tests)


class _Client:
    """Seeding Go must cost nothing — if this is called, it is a bug."""

    def __init__(self):
        self.calls = 0

    def generate_response(self, _prompt):
        self.calls += 1
        raise AssertionError("the built-in Go contract must not call an LLM")


class TestSeedingGo:

    def test_a_go_project_gets_a_contract(self, tmp_path):
        client = _Client()
        path = seed_acceptance_tests("Build a todo CLI.", str(tmp_path),
                                     client, language="go")
        assert path is not None
        assert Path(path).name == SEED_BASENAME
        assert client.calls == 0, "seeding Go must spend no tokens"

    def test_golang_is_the_same_language(self, tmp_path):
        assert seed_acceptance_tests("t", str(tmp_path), _Client(),
                                     language="golang") is not None

    def test_the_written_file_is_the_shipped_contract(self, tmp_path):
        path = seed_acceptance_tests("t", str(tmp_path), _Client(),
                                     language="go")
        written = Path(path).read_text(encoding="utf-8")
        shipped = Path(BUILTIN_GO_CONTRACT).read_text(encoding="utf-8")
        assert shipped in written, "the body must be the shipped contract"

    def test_it_carries_a_seed_header(self, tmp_path):
        """Without the stamp it is indistinguishable from a user's suite,
        and the demotion rule could not apply to it."""
        path = seed_acceptance_tests("t", str(tmp_path), _Client(),
                                     language="go")
        assert "agentchanti:acceptance-seed" in Path(path).read_text(
            encoding="utf-8").splitlines()[0]


class TestTheContractItself:
    """Validated against four real trees in a container: a working todo
    manager passes all five checks; a module that does not compile, one
    that panics on startup, and an empty stub each fail."""

    def test_it_is_python_not_go(self):
        """A Go file would be part of the module under test — collected by
        `go test ./...` and able to break the compilation it measures."""
        assert BUILTIN_GO_CONTRACT.endswith(".py")

    def test_it_parses(self):
        import ast
        ast.parse(Path(BUILTIN_GO_CONTRACT).read_text(encoding="utf-8"))

    def test_a_non_zero_exit_is_not_treated_as_a_crash(self):
        """A CLI run with no arguments is supposed to refuse. Treating that
        as a crash is a mistake this project already made once, in the
        smoke test, where it rewrote three working programs."""
        body = Path(BUILTIN_GO_CONTRACT).read_text(encoding="utf-8")
        assert "panic:" in body
        assert "A non-zero exit is NOT a failure here" in body

    def test_a_missing_toolchain_skips_rather_than_fails(self):
        """An absent toolchain is the instrument being unavailable and
        must never convict the code."""
        body = Path(BUILTIN_GO_CONTRACT).read_text(encoding="utf-8")
        assert 'shutil.which("go") is None' in body
        assert "SkipTest" in body

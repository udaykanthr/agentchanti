"""A chain of inline scripts must bypass cmd.exe one script at a time.

Measured 2026-09-16, the A/B's round 3. The pipeline conjoined two correct
gates with `&&`:

    python -c "...assert 'pygame>=2.5,<3' in t..." && python -c "...'requires-python = \\">=3.10\\"' in text..."

`_shell_free_argv` bypasses cmd.exe only for a single inline script, so the
chain went through cmd.exe, whose quote tracking broke on `\\"` and read
`<3` as input redirection. Every one of 20+ verifications returned the same
44 characters — `The system cannot find the path specified.` — while the
first script alone passed. gate_integrity called the gate STALLED,
escalation failed, and a correct step ended the run.
"""
import os
import sys
import textwrap

import pytest

from agentchanti.executor import Executor

pytestmark = pytest.mark.skipif(os.name != "nt",
                                reason="cmd.exe quoting is Windows-only")

# The measured gate, verbatim.
GATE = (
    'python -c "from pathlib import Path; t=Path(\'pyproject.toml\').read_text'
    '(encoding=\'utf-8\'); assert \'pygame>=2.5,<3\' in t and \'pytest>=8,<9\' '
    'in t and \'>=3.10\' in t and Path(\'README.md\').is_file() and '
    'Path(\'.gitignore\').is_file()" && python -c "from pathlib import Path; '
    'text=Path(\'pyproject.toml\').read_text(encoding=\'utf-8\'); assert '
    '\'pygame>=2.5,<3\' in text and \'pytest>=8,<9\' in text and '
    '\'requires-python = \\">=3.10\\"\' in text and \'snake_game\' in text"'
)

# The relevant lines of the pyproject.toml that run produced.
PYPROJECT = textwrap.dedent('''
    [project]
    name = "two-player-snake"
    requires-python = ">=3.10"
    dependencies = [
        "pygame>=2.5,<3",
    ]

    [project.optional-dependencies]
    dev = [
        "pytest>=8,<9",
    ]

    [tool.setuptools.packages.find]
    include = ["snake_game*"]
''')


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (tmp_path / "README.md").write_text("# Snake\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("venv/\n", encoding="utf-8")
    return tmp_path


def test_the_measured_gate_passes_against_the_artifact_it_judged(project):
    ok, out = Executor().run_command(GATE, cwd=str(project))
    assert ok, out


def test_without_the_chain_split_it_is_the_measured_failure(project,
                                                            monkeypatch):
    """States the defect, so the fix is not just a claim."""
    monkeypatch.setattr(Executor, "_inline_script_chain",
                        staticmethod(lambda cmd: None))
    ok, out = Executor().run_command(GATE, cwd=str(project))
    assert not ok
    assert "cannot find the path" in out.lower()


def test_a_failing_script_stops_the_chain(tmp_path):
    """`&&` semantics: nothing after a failure runs."""
    marker = tmp_path / "second_ran.txt"
    cmd = ('python -c "import sys; sys.exit(0 if 2 < 1 else 3)" && '
           f'python -c "open(r\'{marker}\', \'w\').write(\'x\')"')
    ok, _ = Executor().run_command(cmd, cwd=str(tmp_path))
    assert not ok
    assert not marker.exists()


def test_both_outputs_are_kept(tmp_path):
    cmd = ('python -c "print(\'first\' if 1 < 2 else \'\')" && '
           'python -c "print(\'second\' if 2 > 1 else \'\')"')
    ok, out = Executor().run_command(cmd, cwd=str(tmp_path))
    assert ok, out
    assert "first" in out and "second" in out


class TestWhatIsAChain:

    def test_the_measured_gate_splits_into_its_two_scripts(self):
        chain = Executor._inline_script_chain(GATE)
        assert chain is not None and len(chain) == 2
        for segment in chain:
            argv = Executor._win_split(segment)
            assert len(argv) == 3 and argv[1] == "-c"

    def test_each_segment_reparses_to_the_same_script(self):
        whole = Executor._win_split(GATE)
        chain = Executor._inline_script_chain(GATE)
        assert [Executor._win_split(s)[2] for s in chain] == [whole[2], whole[6]]

    @pytest.mark.parametrize("cmd", [
        'npm test && python -c "assert 1 < 2"',          # not all inline
        'python -c "assert 1 < 2" && python -m pytest',  # not all inline
        'python -c "print(1)" && python -c "print(2)"',  # no metacharacter
        'python -c "assert 1 < 2"',                      # no chain at all
        'python -c "a = 1 && 2"',                        # && inside quotes
    ])
    def test_everything_else_keeps_the_shell(self, cmd):
        assert Executor._inline_script_chain(cmd) is None

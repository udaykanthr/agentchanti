"""The same command dressed differently is still the same command.

Observed live (loop mode, Pac-Man task, 2026-08-05, second run). Told not
to re-run a failing command, the model did not stop re-running it — it
re-ran it wearing a different hat:

    cd /d %CD% && python -m unittest test_pacman -v 2>&1 | head -100
    cd /d %CD% && python -m unittest test_pacman -v 2>&1 | head -150
    python -m unittest test_pacman -v

One piece of work, three literal strings, so the exact-match repeat guard
saw three distinct commands and never fired.

The `| head -N` half is its own defect: head does not exist on Windows, so
those two commands failed in the shell and the model never saw the test
output it was asking for. The tool drops the pipe and runs the real command
instead — the output is length-capped by the tool anyway, which is all the
pipe was ever for.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from agentchanti.agent_tools import AgentTools
from agentchanti.orchestrator.agent_loop import normalize_command


OBSERVED = [
    r'cd /d %CD% && python -m unittest test_pacman -v 2>&1 | head -100',
    r'cd /d %CD% && python -m unittest test_pacman -v 2>&1 | head -150',
    r'python -m unittest test_pacman -v',
]


def test_the_three_observed_spellings_collapse_to_one():
    assert len({normalize_command(c) for c in OBSERVED}) == 1
    assert normalize_command(OBSERVED[0]) == "python -m unittest test_pacman -v"


@pytest.mark.parametrize("cmd,expected", [
    ('cd . && python -m pytest', 'python -m pytest'),
    ('cd /d "C:\\Temp\\x y" && python app.py', 'python app.py'),
    ('cd a && cd b && python app.py', 'python app.py'),
    ('python app.py | tail -5', 'python app.py'),
    ('python app.py 2>&1 | more', 'python app.py'),
    ('python   app.py', 'python app.py'),
])
def test_wrappers_are_stripped(cmd, expected):
    assert normalize_command(cmd) == expected


@pytest.mark.parametrize("cmd", [
    # A pipe that transforms rather than truncates changes the exit status,
    # so it is a genuinely different command and must not be collapsed.
    'python app.py | findstr /i error',
    'python app.py > out.txt',
    'pytest tests/a.py',
    'pytest tests/b.py',
])
def test_meaningful_differences_survive(cmd):
    assert normalize_command(cmd) == " ".join(cmd.split())


def test_distinct_commands_stay_distinct():
    assert (normalize_command('pytest tests/a.py')
            != normalize_command('pytest tests/b.py'))


def test_empty_and_whitespace_are_safe():
    assert normalize_command("") == ""
    assert normalize_command("   ") == ""
    assert normalize_command(None) == ""


@unittest.skipUnless(os.name == "nt", "Windows-only shell behaviour")
class WindowsOutputPipe(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pipe_")
        self.executor = MagicMock()
        self.executor.run_command.return_value = (True, "3 tests passed")
        self.executor.last_exit_code = 0
        self.tools = AgentTools(project_root=self.root,
                                executor=self.executor)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_head_pipe_is_dropped_and_the_real_command_runs(self):
        out = self.tools._tool_run_command(
            "python -m unittest test_pacman -v 2>&1 | head -100")
        ran = self.executor.run_command.call_args[0][0]
        self.assertEqual(ran, "python -m unittest test_pacman -v 2>&1")
        # The model saw the output it was actually after...
        self.assertIn("3 tests passed", out)
        # ...and is told why its pipe vanished, so it stops adding them.
        self.assertIn("head/tail/more do not exist on Windows", out)

    def test_a_command_without_an_output_pipe_is_untouched(self):
        self.tools._tool_run_command("python -m unittest test_pacman -v")
        ran = self.executor.run_command.call_args[0][0]
        self.assertEqual(ran, "python -m unittest test_pacman -v")

    def test_a_filtering_pipe_is_left_alone(self):
        # findstr exists on Windows and changes the exit status — dropping
        # it would silently change what the gate means.
        self.tools._tool_run_command('python -c "import main" | findstr error')
        ran = self.executor.run_command.call_args[0][0]
        self.assertIn("findstr error", ran)

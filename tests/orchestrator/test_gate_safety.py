"""Tests for the destructive-gate check (orchestrator/gate_safety.py).

The module exists because of one measured incident, so the first test is
that incident verbatim. The rest divide into two halves that matter
equally: it must catch the shapes that end a run, and it must stay quiet
over the ordinary gates this project generates by the hundred — a check
that flags real gates would be turned off, and then the next taskkill
lands.
"""

import pytest

from agentchanti.orchestrator.gate_safety import (
    check_gate_safety,
    destructive_reason,
    neutralize_destructive_gates,
    sanitize_gate,
    split_shell_segments,
)
from agentchanti.orchestrator.plan_step import PlanStep


# The gate that killed the pipeline at 2026-08-17 23:42, verbatim.
INCIDENT_GATE = (
    'python -c "from main import CubeCollectorGame; g = CubeCollectorGame(); '
    "assert hasattr(g, 'MOVE_INTERVAL') and g.MOVE_INTERVAL == 0.18 and "
    "hasattr(g, 'score') and g.score == 0 and callable(g.run)\" && "
    "python main.py & timeout /t 2 /nobreak & taskkill /im python.exe /f "
    "2>nul || exit /b 0"
)


class TestTheIncident:
    def test_the_gate_that_killed_the_run_is_caught(self):
        reason = destructive_reason(INCIDENT_GATE)
        assert reason is not None
        assert "taskkill" in reason

    def test_the_assertion_survives_and_the_kill_does_not(self):
        safe, reason = sanitize_gate(INCIDENT_GATE)
        assert reason is not None
        assert "taskkill" not in safe
        # The head is the check the planner actually meant.
        assert "MOVE_INTERVAL == 0.18" in safe
        assert safe.startswith("python -c")
        # And what survives is still a runnable command, not a fragment
        # ending in a dangling separator.
        assert not safe.rstrip().endswith(("&", "&&", "|", "||", ";"))

    def test_sanitized_gate_is_itself_safe(self):
        safe, _ = sanitize_gate(INCIDENT_GATE)
        assert destructive_reason(safe) is None


class TestDestructiveShapes:
    @pytest.mark.parametrize("cmd", [
        "taskkill /im python.exe /f",
        "taskkill /f /im node.exe",
        "pkill -f python",
        "killall python",
        "kill -9 12345",
        "Stop-Process -Name python -Force",
        "rm -rf build",
        "rm -fr /tmp/x",
        "rm -Rf dist",
        "rmdir /s /q venv",
        "rd /s /q node_modules",
        "del /s *.py",
        "Remove-Item -Recurse -Force venv",
        'python -c "import shutil; shutil.rmtree(\'build\')"',
        "git reset --hard",
        "git clean -fdx",
        "shutdown /s /t 0",
        "Restart-Computer",
        "diskpart",
        "format c:",
        "dd if=/dev/zero of=/dev/sda",
        "docker system prune -f",
    ])
    def test_flagged(self, cmd):
        assert destructive_reason(cmd) is not None, cmd

    def test_a_kill_scoped_to_one_pid_is_not_machine_wide(self):
        # The distinction the incident turned on: /im names an image,
        # /pid names one process the gate itself started.
        assert destructive_reason("taskkill /pid 1234 /f") is None
        assert destructive_reason("Stop-Process -Id 1234") is None

    def test_it_is_caught_wherever_it_sits_in_the_chain(self):
        assert destructive_reason("echo hi && rm -rf build") is not None
        assert destructive_reason("rm -rf build && echo hi") is not None
        assert destructive_reason("echo a | pkill -f x") is not None


class TestQuietOverRealGates:
    """Gates taken from this project's own runs. None may be flagged."""

    @pytest.mark.parametrize("cmd", [
        "python -m unittest",
        "python -m pytest -q",
        "npm test",
        "go test ./...",
        "python main.py --headless --frames 3",
        'python -c "from pathlib import Path; assert '
        "Path('requirements.txt').read_text(encoding='utf-8').strip() == "
        "'pygame'\"",
        'python -c "from game import Game; g=Game(0); assert '
        'g.pellets_remaining()>0; assert all(not g.map.is_wall(*e.tile) '
        'for e in g.entities())"',
        'python -c "from panda3d.core import loadPrcFileData; '
        "loadPrcFileData('', 'window-type none'); from main import "
        'CubeCollectorGame; g=CubeCollectorGame(); g._move_snake()"',
        "cd react-home && npm test -- --run",
        "python -m unittest -v tests/test_game.py",
    ])
    def test_not_flagged(self, cmd):
        assert destructive_reason(cmd) is None, cmd

    def test_a_safe_gate_is_returned_byte_identical(self):
        cmd = "python -m unittest"
        safe, reason = sanitize_gate(cmd)
        assert reason is None
        assert safe == cmd

    def test_formatting_a_string_is_not_formatting_a_drive(self):
        # `format` only counts against a drive letter — the word appears
        # in ordinary Python.
        assert destructive_reason(
            'python -c "assert \'{}\'.format(1) == \'1\'"') is None

    def test_deleting_one_named_file_is_not_a_wildcard_delete(self):
        assert destructive_reason("del build.log") is None


class TestSegmentation:
    def test_separators_inside_a_payload_do_not_split_it(self):
        segs = split_shell_segments('python -c "a and b; c && d" && echo ok')
        assert len(segs) == 2
        assert segs[0][0] == 'python -c "a and b; c && d"'
        assert segs[0][1] == "&&"
        assert segs[1][0] == "echo ok"

    def test_a_quoted_destructive_string_still_counts(self):
        # Documented bias: the check is deliberately not quote-aware for
        # matching, because a false positive costs one gate and a false
        # negative costs the machine.
        assert destructive_reason('python -c "print(\'rm -rf /\')"') is not None


class TestNeutralizeInPlace:
    def _steps(self):
        return [
            PlanStep(id="1.1", step_type="CODE", verify_cmd=INCIDENT_GATE),
            PlanStep(id="2.1", step_type="CODE",
                     verify_cmd="python -m unittest"),
            PlanStep(id="3.1", step_type="CODE", verify_cmd=""),
        ]

    def test_check_reports_only_the_unsafe_step(self):
        gaps = check_gate_safety(self._steps())
        assert [sid for sid, _ in gaps] == ["1.1"]

    def test_neutralize_rewrites_only_the_unsafe_step(self):
        steps = self._steps()
        changed = neutralize_destructive_gates(steps)
        assert [sid for sid, _, _ in changed] == ["1.1"]
        assert destructive_reason(steps[0].verify_cmd) is None
        assert "MOVE_INTERVAL" in steps[0].verify_cmd
        assert steps[1].verify_cmd == "python -m unittest"
        assert steps[2].verify_cmd == ""

    def test_it_is_idempotent(self):
        steps = self._steps()
        neutralize_destructive_gates(steps)
        after_first = steps[0].verify_cmd
        assert neutralize_destructive_gates(steps) == []
        assert steps[0].verify_cmd == after_first


class TestLedgerRefusesToRecord:
    def test_a_destructive_gate_never_enters_the_ledger(self):
        # The ledger re-runs every recorded gate after every later wave,
        # so this is where one destructive command becomes many.
        from agentchanti.orchestrator.wave_snapshots import GateLedger

        ledger = GateLedger()
        ledger.record(INCIDENT_GATE, "1.1")
        ledger.record("python -m unittest", "2.1")
        assert list(ledger.gates()) == ["python -m unittest"]

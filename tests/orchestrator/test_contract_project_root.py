"""The contract lives in the project root, and must not look above it.

Measured 2026-09-15, a 7-step Snake run: every gate green, the smoke test
launched the game, ghost 0 violated — and exit 1, because the seeded
contract began::

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    GAME_SCRIPT = PROJECT_ROOT / "snake_game.py"

That is the conventional line for a file in `tests/`. The contract is
written to the project ROOT, so `parents[1]` is the directory above the
project: `snake_game.py` was reported missing while it sat beside the
contract, and the game was launched with cwd outside the project and
exited 2. The seeding prompt never said where the file lives. Both symptoms
were assertion FAILURES, which the runnability repair rightly never
rewrites — so nothing caught it.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, _PROMPT, _header, _platform_note, _should_seed,
    platform_signal_reason, root_escape_reason, seed_acceptance_tests,
    seed_state,
)
from tests.orchestrator.test_contract_runnability import ScriptedClient

ESCAPING = textwrap.dedent('''
    import unittest
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]


    class Contract(unittest.TestCase):
        def test_game(self):
            import game
            g = game.Game()
            self.assertEqual(g.advance(), "moved")
            self.assertEqual(g.score, 0)
            self.assertTrue((PROJECT_ROOT / "game.py").is_file())
''').strip()

FIXED = ESCAPING.replace(".parents[1]", ".parent")

GAME = textwrap.dedent('''
    class Game:
        score = 0

        def advance(self):
            return "moved"
''')


class TestDetection:

    def test_the_measured_line(self):
        assert root_escape_reason(ESCAPING) == "parents[1]"

    @pytest.mark.parametrize("line, marker", [
        ("ROOT = Path(__file__).resolve().parent.parent", ".parent.parent"),
        ("ROOT = Path(__file__).parents[2]", "parents[2]"),
        ("ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
         "os.path.dirname(os.path.dirname(__file__))"),
        ("HERE = Path(__file__).resolve()\nROOT = HERE.parents[1]",
         "parents[1]"),
    ])
    def test_every_spelling(self, line, marker):
        src = "import os\nfrom pathlib import Path\n" + line + "\n"
        assert root_escape_reason(src) == marker

    @pytest.mark.parametrize("line", [
        "ROOT = Path(__file__).resolve().parent",
        "ROOT = Path(__file__).resolve().parents[0]",
        "ROOT = os.path.dirname(os.path.abspath(__file__))",
        "ROOT = Path.cwd()",
        "DATA = Path('assets').parent.parent",      # not derived from __file__
    ])
    def test_the_correct_ways_are_left_alone(self, line):
        src = "import os\nfrom pathlib import Path\n" + line + "\n"
        assert root_escape_reason(src) is None

    def test_the_prompt_says_where_the_file_lives(self):
        assert "IN THE\nPROJECT ROOT" in _PROMPT
        assert "Path(__file__).resolve().parent" in _PROMPT
        assert "parents[1]" in _PROMPT


def test_the_difference_is_real(tmp_path):
    """Run both against one project: the measured line fails, the fix passes."""
    (tmp_path / "game.py").write_text(GAME)
    results = {}
    for name, src in (("escaping", ESCAPING), ("fixed", FIXED)):
        (tmp_path / SEED_BASENAME).write_text(src + "\n")
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "test_acceptance_contract"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
        results[name] = proc.returncode
    assert results == {"escaping": 1, "fixed": 0}


class TestSeeding:

    def test_it_is_repaired_before_it_is_written(self, tmp_path):
        client = ScriptedClient(ESCAPING, FIXED)
        seed_acceptance_tests("a game", str(tmp_path), client,
                              language="python")
        written = seed_state(os.path.join(str(tmp_path), SEED_BASENAME))[2]
        assert root_escape_reason(written) is None
        assert len(client.prompts) == 2
        assert "CANNOT JUDGE THIS PROJECT" in client.prompts[1]
        assert "parents[1]" in client.prompts[1], "the draft was not shown"

    def test_refused_when_every_retry_still_escapes(self, tmp_path):
        client = ScriptedClient(ESCAPING, ESCAPING, ESCAPING)
        path = seed_acceptance_tests("a game", str(tmp_path), client,
                                     language="python")
        assert path is None
        assert not (tmp_path / SEED_BASENAME).exists()


SIGINT_CONTRACT = textwrap.dedent('''
    import signal
    import subprocess
    import sys
    import unittest
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent


    class Contract(unittest.TestCase):
        def test_game(self):
            import game
            g = game.Game()
            self.assertEqual(g.advance(), "moved")
            self.assertEqual(g.score, 0)
            p = subprocess.Popen([sys.executable, "game.py"], cwd=PROJECT_ROOT)
            p.send_signal(signal.SIGINT)
            p.wait()
''').strip()

SIGINT_FIXED = SIGINT_CONTRACT.replace("p.send_signal(signal.SIGINT)",
                                       "p.terminate()")


class TestPlatform:
    """The measured contract's second defect, under the first."""

    def test_the_measured_call_is_rejected_on_windows(self):
        assert (platform_signal_reason(SIGINT_CONTRACT, platform="win32")
                == "send_signal(signal.SIGINT)")

    def test_it_is_correct_posix_and_left_alone_there(self):
        assert platform_signal_reason(SIGINT_CONTRACT, platform="linux") is None

    @pytest.mark.parametrize("line, marker", [
        ("p.send_signal(signal.SIGKILL)", "send_signal(signal.SIGKILL)"),
        ("signal.alarm(3)", "signal.alarm"),
        ("os.killpg(os.getpgid(p.pid), signal.SIGTERM)", "os.killpg"),
        ("subprocess.Popen(['x'], preexec_fn=os.setsid)", "preexec_fn="),
    ])
    def test_other_posix_only_constructs(self, line, marker):
        src = "import os, signal, subprocess\np = None\n" + line + "\n"
        assert platform_signal_reason(src, platform="win32") == marker

    @pytest.mark.parametrize("line", [
        "p.send_signal(signal.SIGTERM)",
        "p.send_signal(signal.CTRL_BREAK_EVENT)",
        "p.terminate()",
        "signal.signal(signal.SIGINT, handler)",   # SIGINT exists on Windows
    ])
    def test_what_windows_accepts(self, line):
        src = "import signal\np = None\nhandler = None\n" + line + "\n"
        assert platform_signal_reason(src, platform="win32") is None

    def test_the_prompt_names_the_platform(self):
        assert "Windows" in _platform_note("win32")
        assert "SIGINT" in _platform_note("win32")
        assert _platform_note("linux") == "PLATFORM: linux."

    def test_seeding_repairs_it_on_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        client = ScriptedClient(SIGINT_CONTRACT, SIGINT_FIXED)
        seed_acceptance_tests("a game", str(tmp_path), client,
                              language="python")
        written = seed_state(os.path.join(str(tmp_path), SEED_BASENAME))[2]
        assert "send_signal(signal.SIGINT)" not in written
        assert len(client.prompts) == 2
        assert "PLATFORM: Windows" in client.prompts[1]


class TestAnExistingBrokenContract:
    """Rerunning the same prompt must not reuse a contract that cannot work."""

    def _write(self, root, body, task="a game", edited=False):
        body = "\n" + body + "\n"
        header = _header(task, body)
        if edited:
            body += "# someone edited this\n"
        path = os.path.join(str(root), SEED_BASENAME)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + body)
        return path

    def test_same_task_escaping_contract_is_reseeded(self, tmp_path):
        path = self._write(tmp_path, ESCAPING)
        assert _should_seed("a game", str(tmp_path), path) is True

    def test_same_task_correct_contract_is_kept(self, tmp_path):
        path = self._write(tmp_path, FIXED)
        assert _should_seed("a game", str(tmp_path), path) is False

    def test_an_edited_contract_belongs_to_its_editor(self, tmp_path):
        path = self._write(tmp_path, ESCAPING, edited=True)
        assert _should_seed("a game", str(tmp_path), path) is False

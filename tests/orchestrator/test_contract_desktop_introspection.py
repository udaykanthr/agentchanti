"""A contract must judge the program, not the desktop it happens to run on.

Measured 2026-09-15, a Snake run with every earlier contract fix in place:
every gate green, the smoke test launched the game, and exit 1 on

    snake_game.py did not open a visible pygame window within 8.0 seconds.

The contract launched the game with `sys.executable`, then walked the
desktop with `ctypes.WinDLL("user32").EnumWindows` for a visible window
owned by `process.pid`, and meant to scrape its pixels through gdi32.
Measured by hand the window existed, titled 'Two Player Snake', owned by
pid 22824 (`C:\\Python313\\python.exe`) — whose parent was pid 13136, the
`venv\\Scripts\\python.exe` launcher `Popen` returned. A Windows venv
interpreter re-executes the base one, so the check could never match.
"""
import os
import textwrap

import pytest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, _PROMPT, _STRUCTURAL_NOTE, _header, _should_seed,
    desktop_introspection_reason, seed_acceptance_tests, seed_state,
    structural_defect_reason,
)
from agentchanti.orchestrator.plan_step import (
    _produced_paths, parse_structured_plan,
)
from tests.orchestrator.test_contract_runnability import ScriptedClient

# Reduced from the measured contract, construct for construct.
MEASURED = textwrap.dedent('''
    import ctypes
    from ctypes import wintypes
    import subprocess
    import sys
    import unittest


    class AcceptanceContractTests(unittest.TestCase):
        def test_window(self):
            import snake_game
            self.assertEqual(snake_game.WINDOW_WIDTH, 600)
            self.assertEqual(snake_game.WINDOW_HEIGHT, 600)
            process = subprocess.Popen([sys.executable, "snake_game.py"])
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(None, ctypes.byref(owner))
            self.assertEqual(owner.value, process.pid)
''').strip()

HEADLESS = textwrap.dedent('''
    import os
    import unittest


    class AcceptanceContractTests(unittest.TestCase):
        def test_window(self):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            import snake_game
            self.assertEqual(snake_game.WINDOW_WIDTH, 600)
            self.assertEqual(snake_game.WINDOW_HEIGHT, 600)
''').strip()


class TestDetection:

    def test_the_measured_contract(self):
        assert desktop_introspection_reason(MEASURED) == 'ctypes.WinDLL("user32")'
        assert "inspects the desktop" in structural_defect_reason(MEASURED)

    @pytest.mark.parametrize("src, marker", [
        ("import win32gui", "import win32gui"),
        ("from PIL import ImageGrab", "PIL.ImageGrab"),
        ("import PIL.ImageGrab", "import PIL.ImageGrab"),
        ("import mss", "import mss"),
        ("import ctypes\ngdi = ctypes.WinDLL('gdi32.dll')",
         'ctypes.WinDLL("gdi32.dll")'),
        ("import ctypes\nu = ctypes.windll.user32", "windll.user32"),
        ("from pywinauto import Application", "from pywinauto import ..."),
    ])
    def test_every_way_in(self, src, marker):
        assert desktop_introspection_reason(src) == marker

    @pytest.mark.parametrize("src", [
        HEADLESS,
        "import ctypes\nk = ctypes.WinDLL('kernel32')",     # not the desktop
        "import pygame\nsurface = pygame.display.get_surface()",
        "import PIL.Image",
    ])
    def test_the_program_and_other_dlls_are_left_alone(self, src):
        assert desktop_introspection_reason(src) is None

    def test_the_repair_note_says_what_to_do_instead(self):
        """Sent only to a contract that made the mistake, not every prompt."""
        assert "Never inspect the desktop" in _STRUCTURAL_NOTE
        assert "SDL_VIDEODRIVER=dummy" in _STRUCTURAL_NOTE
        assert "inspect the desktop" not in _PROMPT


class TestSeeding:

    def test_it_is_repaired_before_it_is_written(self, tmp_path):
        client = ScriptedClient(MEASURED, HEADLESS)
        seed_acceptance_tests("a snake game", str(tmp_path), client,
                              language="python")
        written = seed_state(os.path.join(str(tmp_path), SEED_BASENAME))[2]
        assert desktop_introspection_reason(written) is None
        assert len(client.prompts) == 2
        assert "inspects the desktop" in client.prompts[1]

    def test_an_existing_one_is_reseeded_on_rerun(self, tmp_path):
        body = "\n" + MEASURED + "\n"
        path = os.path.join(str(tmp_path), SEED_BASENAME)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_header("a snake game", body) + body)
        assert _should_seed("a snake game", str(tmp_path), path) is True


class TestProducesIsNotAFileList:
    """The same run's ghost reported `violated-exists: <3`."""

    def test_the_measured_line(self):
        assert _produced_paths("installed pygame>=2.5,<3") == []

    def test_real_files_and_globs_survive(self):
        assert _produced_paths("venv\\, app/src/*, .env.example") == [
            "venv\\", "app/src/*", ".env.example"]

    def test_through_the_parser(self):
        plan = ("==PLAN==\n\n--STEP 2.1 [CMD] depends:none\n"
                "Install dependencies.\n> python -m pip install -r requirements.txt\n"
                "produces: installed pygame>=2.5,<3\n")
        steps = parse_structured_plan(plan)
        assert "<3" not in steps[0].target_files

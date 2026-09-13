"""A contract that needs a person cannot pass an unattended run.

`unrunnable_gate_reason` already refuses a gate this platform's shell
cannot execute, because no output of the step could change the verdict.
An interactive contract is the same category one layer over: no behaviour
of the code can make it finish.

Measured 2026-09-10, a 3D pinball run. The seeded contract opened a
Tkinter window and asked a human to watch the game and type scores in::

    score_before = self.integer("Score before the final scoring hit")
    self.action("Press Esc in the game. Do not close it with the window
                 manager.")
    print("\\nThe game is still open; press Esc in its window when finished.")

Every gate was green, the suite passed, and the game demonstrably worked —
confirmed afterwards by screenshot, ~1090 fps, and a full-power launch that
cleared the lane into the arena. The run still exited non-zero, because
nobody was there to answer. It is also self-concealing: it leaves a real
application window parked on the desktop, which reads as a hung program and
sends the reader debugging a rendering bug that does not exist.
"""
import textwrap
import unittest

from agentchanti.orchestrator.acceptance_seed import (
    _PROMPT, interactive_reason, mocking_reason,
)

# Reduced from the measured contract, including its lazy import — the real
# one did `    import tkinter as tk` inside a method, so anything anchored
# to the start of a line would have missed the one construct that mattered.
MEASURED = textwrap.dedent('''
    import subprocess
    import unittest


    class _Observer:
        def __init__(self):
            import tkinter as tk
            self.window = tk.Tk()

        def integer(self, prompt):
            ...


    class PinballAcceptance(unittest.TestCase):
        def test_session(self):
            game = subprocess.Popen(["python", "-m", "pinball"])
            score_before = self.integer("Score before the final scoring hit")
            self.assertGreater(self.integer("Score after"), score_before)
            self.action("Press Esc in the game.")
''')

PROGRAMMATIC = textwrap.dedent('''
    import unittest

    from pinball.state import GameState


    class PinballAcceptance(unittest.TestCase):
        def test_scoring(self):
            g = GameState()
            self.assertTrue(g.launch())
            self.assertEqual(g.award_hit("bumper"), 100)
            self.assertEqual(g.score, 100)
''')


class InteractiveContractsAreRefused(unittest.TestCase):

    def test_the_measured_contract_is_caught(self):
        self.assertEqual(interactive_reason(MEASURED), "tkinter")

    def test_a_lazy_import_inside_a_method_is_caught(self):
        """The real one imported tkinter inside `_Observer.__init__`."""
        self.assertIn("    import tkinter as tk", MEASURED)
        self.assertIsNotNone(interactive_reason(MEASURED))

    def test_every_way_of_asking_a_person(self):
        for src, marker in (
            ("answer = input('score? ')", "input("),
            ("import tkinter", "tkinter"),
            ("from getpass import getpass", "getpass"),
            ("import msvcrt; msvcrt.getch()", "msvcrt.getch"),
            ("line = sys.stdin.readline()", "sys.stdin.read"),
        ):
            with self.subTest(marker=marker):
                self.assertEqual(interactive_reason(src), marker)

    def test_a_programmatic_contract_is_left_alone(self):
        """Or this becomes a machine for refusing good contracts."""
        self.assertIsNone(interactive_reason(PROGRAMMATIC))
        self.assertIsNone(mocking_reason(PROGRAMMATIC))

    def test_ordinary_words_are_not_accused(self):
        """Matching prose would fire on legitimate assertions."""
        for src in (
            'self.assertEqual(hud["controls_text"], "Press Space to launch")',
            'self.assertIn("observe", doc)',
            "# the user presses the flipper key\nassert g.flip()",
        ):
            with self.subTest(src=src[:40]):
                self.assertIsNone(interactive_reason(src))

    def test_the_prompt_forbids_it_up_front(self):
        """Cheaper to never generate one than to detect and regenerate."""
        self.assertIn("NEVER require a human", _PROMPT)
        self.assertIn("input()", _PROMPT)
        self.assertIn("tkinter", _PROMPT)


if __name__ == "__main__":
    unittest.main()

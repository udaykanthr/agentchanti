"""Sampling a window the instant it exists measures the blank buffer.

Measured 2026-09-16 in `test1`, a clean-slate pinball run from a folder
holding only `.agentchanti.yaml` and `prompt.txt`. The seeded contract
polled until `pygame.display.get_surface()` returned non-None — which
happens the moment the game calls `set_mode`, before its first frame —
and read the pixels back in the very next statement:

    rendered_pixels = pygame.image.tostring(surface, "RGB")
    colors = {rendered_pixels[i:i + 3] for i in range(0, len(...), 3)}
    self.assertGreater(len(colors), 1,
                       "The opened playfield must render visible content.")

A fresh surface is uniformly black, so `len(colors) == 1`. Measured on
that run's own artifact, with the contract's own code:

    immediately (what the contract does)  ->    1 distinct colour
    after 0.25s                           ->  364 distinct colours
    after 1.0s                            ->  364 distinct colours

The game was entirely correct. Policy B held — a seeded contract cannot
convict the code — so the run exited 0 and claimed nothing it had not
proven, but the verdict was lost to a race.
"""
import textwrap

from agentchanti.orchestrator.acceptance_seed import (
    render_race_reason,
    structural_defect_reason,
)

# The measured contract's shape, reduced to the statements that matter.
MEASURED = textwrap.dedent('''
    import pygame, runpy, threading, time, unittest


    class T(unittest.TestCase):
        def test_it(self):
            threading.Thread(target=runpy.run_path, args=("pinball.py",),
                             daemon=True).start()
            surface = None
            deadline = time.monotonic() + 5.0
            try:
                while time.monotonic() < deadline:
                    if pygame.display.get_init():
                        surface = pygame.display.get_surface()
                        if surface is not None:
                            break
                    time.sleep(0.01)

                self.assertIsNotNone(surface, "must open a window")
                self.assertEqual(surface.get_size(), (400, 600))

                rendered = pygame.image.tostring(surface, "RGB")
                colors = {rendered[i:i + 3] for i in range(0, len(rendered), 3)}
                self.assertGreater(len(colors), 1, "must render content")
            finally:
                pygame.quit()
''')


class TestTheMeasuredIncident:

    def test_it_is_named(self):
        reason = render_race_reason(MEASURED)
        assert reason and "tostring" in reason

    def test_the_polling_sleep_does_not_excuse_it(self):
        """`time.sleep(0.01)` inside the acquire loop is part of ACQUIRING.

        It is lexically between the two, which is exactly why the check
        cannot work on line ranges.
        """
        assert "time.sleep(0.01)" in MEASURED
        assert render_race_reason(MEASURED)

    def test_it_reaches_the_structural_screen(self):
        reason = structural_defect_reason(MEASURED)
        assert reason and "blank until" in reason

    def test_the_reason_says_what_to_do(self):
        reason = structural_defect_reason(MEASURED)
        assert "retry in a" in reason or "wait for a frame" in reason
        assert "as strict" in reason, "a repair must not check less"


def _fn(body: str) -> str:
    return ("import pygame, time, unittest\n\n\n"
            "class T(unittest.TestCase):\n    def test_it(self):\n"
            + textwrap.indent(textwrap.dedent(body), "        "))


class TestWhatMakesItSafe:

    def test_a_sleep_between_the_two_is_enough(self):
        src = _fn('''
            surface = pygame.display.get_surface()
            time.sleep(0.5)
            px = pygame.image.tostring(surface, "RGB")
            self.assertGreater(len({px[i:i+3] for i in range(0, len(px), 3)}), 1)
        ''')
        assert render_race_reason(src) is None

    def test_a_frame_wait_is_enough(self):
        src = _fn('''
            surface = pygame.display.get_surface()
            pygame.time.wait(300)
            px = pygame.image.tostring(surface, "RGB")
            self.assertGreater(len(px), 0)
        ''')
        assert render_race_reason(src) is None

    def test_joining_the_game_thread_is_enough(self):
        src = _fn('''
            surface = pygame.display.get_surface()
            game_thread.join(timeout=2.0)
            px = pygame.image.tostring(surface, "RGB")
            self.assertGreater(len(px), 0)
        ''')
        assert render_race_reason(src) is None

    def test_retrying_in_a_loop_is_enough(self):
        """The honest fix: sample until it settles, fail if it never does."""
        src = _fn('''
            surface = pygame.display.get_surface()
            colors = set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                px = pygame.image.tostring(surface, "RGB")
                colors = {px[i:i + 3] for i in range(0, len(px), 3)}
                if len(colors) > 1:
                    break
            self.assertGreater(len(colors), 1, "must render content")
        ''')
        assert render_race_reason(src) is None


class TestWhatItMustNotAccuse:

    def test_a_contract_that_never_reads_pixels(self):
        src = _fn('''
            surface = pygame.display.get_surface()
            self.assertIsNotNone(surface)
            self.assertEqual(surface.get_size(), (400, 600))
        ''')
        assert render_race_reason(src) is None

    def test_a_surface_the_contract_drew_on_itself(self):
        """It owns the surface, so there is no other thread to wait for."""
        src = _fn('''
            surf = pygame.Surface((10, 10))
            surf.fill((255, 0, 0))
            self.assertEqual(surf.get_at((0, 0))[:3], (255, 0, 0))
        ''')
        assert render_race_reason(src) is None

    def test_a_contract_with_no_pygame_in_it_at_all(self):
        src = _fn('''
            from snake_game import Game
            g = Game(seed=1)
            g.advance(0.5)
            self.assertGreater(g.score, 0)
        ''')
        assert render_race_reason(src) is None

    def test_an_unparseable_contract_is_no_opinion(self):
        assert render_race_reason("def broken( :\n") is None

    def test_an_empty_source_is_no_opinion(self):
        assert render_race_reason("") is None

    def test_a_read_before_any_acquisition_is_not_this_defect(self):
        """Nothing was acquired, so this check has nothing to say."""
        src = _fn('''
            px = pygame.image.tostring(self.own_surface, "RGB")
            self.assertGreater(len(px), 0)
        ''')
        assert render_race_reason(src) is None

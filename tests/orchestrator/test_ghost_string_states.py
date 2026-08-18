"""`degenerate-long-run` must read a state vocabulary spelled as strings.

`_parse_state_guard` read only `Name` and `Attribute`, so

    if self._state in ("win", "game_over"):
        return

parsed to no guard at all. No guard meant no guarded advance method, which
meant the suite was never a candidate, which disarmed the check entirely
for any project whose states are plain strings.

Measured 2026-08-17, glm-5.2 on the Pac-Man task: `Game.advance` opened
with exactly that guard and `guarded_advance_methods` returned {}. The
task prompt *mandates* string states -- `state -> "start" | "playing" |
"win" | "game_over"` -- so on that task the check could never have fired.

The pin side had the identical blindness, and the fix has to be symmetric:
teaching only the guard side to read strings would make every suite that
pins with `assertEqual(game.state, "playing")` look unpinned, and flag runs
it genuinely protects.
"""

import textwrap

import pytest

from agentchanti.orchestrator.ghost import (
    degenerate_long_runs, guarded_advance_methods,
)

# A guard spelled with string literals, as the measured artifact spelled it.
SIM = textwrap.dedent('''
    class Game:
        def __init__(self):
            self.state = "playing"
            self.frames = 0

        def advance(self, dt):
            if self.state in ("win", "game_over"):
                return
            self.frames += 1
            if self.frames > 50:
                self.state = "game_over"
''')

LONG_RUN = textwrap.dedent('''
    import unittest
    from sim import Game

    class T(unittest.TestCase):
        def test_long_run(self):
            g = Game()
            for _ in range(2000):
                g.advance(1 / 60.0)
                self.assertGreaterEqual(g.frames, 0)
''')

PINNED = textwrap.dedent('''
    import unittest
    from sim import Game

    class T(unittest.TestCase):
        def test_long_run(self):
            g = Game()
            for _ in range(2000):
                g.advance(1 / 60.0)
                self.assertEqual(g.state, "playing")
''')

ASSERT_PINNED = textwrap.dedent('''
    import unittest
    from sim import Game

    class T(unittest.TestCase):
        def test_long_run(self):
            g = Game()
            for _ in range(2000):
                g.advance(1 / 60.0)
                assert g.state == "playing"
''')

TAUTOLOGY = textwrap.dedent('''
    import unittest
    from sim import Game

    class T(unittest.TestCase):
        def test_long_run(self):
            g = Game()
            for _ in range(2000):
                g.advance(1 / 60.0)
                self.assertIn(g.state, ("playing", "win", "game_over"))
''')


@pytest.fixture
def project(tmp_path):
    def build(test_source):
        (tmp_path / "sim.py").write_text(SIM, encoding="utf-8")
        (tmp_path / "test_sim.py").write_text(test_source, encoding="utf-8")
        return str(tmp_path)
    return build


def test_a_string_literal_guard_is_recognised():
    guards = guarded_advance_methods(SIM)
    assert "Game.advance" in guards
    guard = guards["Game.advance"]
    assert guard.attr == "state"
    assert guard.names == frozenset({"win", "game_over"})
    assert guard.halts("game_over")
    assert guard.proceeds("playing")


def test_the_degenerate_string_state_run_is_flagged(project):
    findings = degenerate_long_runs(project(LONG_RUN), ["sim.py", "test_sim.py"])
    assert len(findings or []) == 1


@pytest.mark.parametrize("source", [PINNED, ASSERT_PINNED],
                         ids=["assertEqual", "bare-assert"])
def test_a_string_pin_still_silences_the_check(source, project):
    """The symmetric half. Without it the fix trades a false negative for
    a false positive on every suite that does protect its run."""
    findings = degenerate_long_runs(project(source), ["sim.py", "test_sim.py"])
    assert not findings


def test_a_string_tautology_is_still_not_a_pin(project):
    """`assertIn(state, ("playing", "win", "game_over"))` admits the very
    terminal states it was meant to exclude — spelled as strings now, but
    no more of a guard than when spelled as constants."""
    findings = degenerate_long_runs(project(TAUTOLOGY), ["sim.py", "test_sim.py"])
    assert len(findings or []) == 1


def test_non_string_constants_are_not_state_values():
    """`if self.done is True: return` is not a state vocabulary, and
    admitting it would invent guards on ordinary boolean flags."""
    source = textwrap.dedent('''
        class Game:
            def advance(self, dt):
                if self.done is True:
                    return
                self.frames += 1
    ''')
    assert "Game.advance" not in guarded_advance_methods(source)

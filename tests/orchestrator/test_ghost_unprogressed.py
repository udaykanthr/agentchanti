"""`unprogressed-long-run`: an endurance loop that never asserts movement.

The third defect in the family, and the one the other two cannot see.
`degenerate-long-run` asks whether the loop stopped doing work;
`varied-input-ignored` asks whether the work read the input at all.
Neither covers a loop whose object reads its input, does work on every
iteration, and still goes nowhere.

Measured 2026-08-17, glm-5.2 on the Pac-Man task with agent_loop on.
`Entity._move` stepped toward the current tile centre before stepping
forward, so once past the centre the correction moved it backward and at
small dt consumed the whole travel budget:

    advance(0.01) -> x=1.04 -> x=1.0 -> x=1.04 ...

Net displacement zero at any dt below ~0.2. The suite drove 20000 frames
at 1/60 and passed, because its only in-loop assertion was
`assertLessEqual(cur, prev)` on a pellet count that never moved. All 19 of
its own tests were green while external probes of dt-scaling and of
`press()` both failed.
"""

import textwrap

import pytest

from agentchanti.orchestrator.ghost import (
    _asserts_progress, unprogressed_long_runs,
)

SIM = textwrap.dedent('''
    class Game:
        def __init__(self):
            self.state = "playing"
            self.pellets = 100
            self.x = 1.0

        def advance(self, dt):
            if self.state in ("win", "game_over"):
                return
            # Work happens every call; it just nets to nothing.
            self.x += self.speed_step(dt)
            self.x -= self.speed_step(dt)

        def speed_step(self, dt):
            return 4.0 * dt
''')


def _suite(body):
    return textwrap.dedent('''
        import unittest
        from sim import Game

        class T(unittest.TestCase):
            def test_endurance(self):
                g = Game()
                start = g.x
                prev = g.pellets
    ''') + textwrap.indent(textwrap.dedent(body), " " * 8)


# The measured shape: monotonicity over a quantity that never moves.
MONOTONE = _suite('''
    for _ in range(20000):
        g.advance(1 / 60.0)
        cur = g.pellets
        self.assertLessEqual(cur, prev)
        prev = cur
''')

# The other measured shape: a pure invariant, true of a still board.
INVARIANT = _suite('''
    for _ in range(600):
        g.advance(1 / 60.0)
        self.assertFalse(g.x < 0)
''')

# What the fix looks like: one assertion that something moved.
PROGRESS_IN_LOOP = _suite('''
    for _ in range(20000):
        g.advance(1 / 60.0)
        self.assertNotEqual(g.x, start)
''')

PROGRESS_AFTER_LOOP = _suite('''
    for _ in range(20000):
        g.advance(1 / 60.0)
    self.assertLess(g.pellets, prev)
''')

BARE_ASSERT_PROGRESS = _suite('''
    for _ in range(20000):
        g.advance(1 / 60.0)
    assert g.x != start
''')


@pytest.fixture
def project(tmp_path):
    def build(test_source):
        (tmp_path / "sim.py").write_text(SIM, encoding="utf-8")
        (tmp_path / "test_sim.py").write_text(test_source, encoding="utf-8")
        return str(tmp_path)
    return build


@pytest.mark.parametrize("source", [MONOTONE, INVARIANT],
                         ids=["monotone-assertion", "invariant-assertion"])
def test_unprotected_endurance_claims_are_reported(source, project):
    findings = unprogressed_long_runs(project(source), ["sim.py", "test_sim.py"])
    assert len(findings or []) == 1


@pytest.mark.parametrize(
    "source", [PROGRESS_IN_LOOP, PROGRESS_AFTER_LOOP, BARE_ASSERT_PROGRESS],
    ids=["in-loop", "after-loop", "bare-assert"])
def test_a_single_progress_assertion_silences_it(source, project):
    findings = unprogressed_long_runs(project(source), ["sim.py", "test_sim.py"])
    assert not findings


def test_a_sampling_loop_is_not_an_endurance_claim(project):
    """200 frames sampling `state` into a set, then checking the set's
    members are legal, came from a run whose artifact passed all nine
    external probes. Asserting progress is not that test's job."""
    source = _suite('''
        observed = set()
        for _ in range(200):
            g.advance(1 / 60.0)
            observed.add(g.state)
        for s in observed:
            self.assertIn(s, ("start", "playing", "win", "game_over"))
    ''')
    assert not unprogressed_long_runs(project(source), ["sim.py", "test_sim.py"])


# ── the predicate that decides it ────────────────────────────────────────

def _stmts(src):
    import ast
    return ast.parse(textwrap.dedent(src)).body


@pytest.mark.parametrize("src", [
    "self.assertNotEqual(g.tile, before)",
    "self.assertLess(counts[-1], initial)",
    "self.assertGreater(moved, baseline)",
    "assert g.x != start",
    "self.assertTrue(g.x != start)",
    "self.assertFalse(g.tile == before)",
])
def test_strict_comparisons_between_varying_things_are_progress(src):
    assert _asserts_progress(_stmts(src))


@pytest.mark.parametrize("src", [
    "self.assertLessEqual(cur, prev)",          # monotone admits equality
    "self.assertGreaterEqual(cur, prev)",
    "self.assertEqual(g.state, 'playing')",
    "self.assertGreater(g.pellets_remaining(), 0)",   # strict, but vs a literal
    "self.assertNotEqual(g.state, 'game_over')",      # strict, but vs a literal
    "self.assertFalse(g.map.is_wall(x, y))",          # a plain invariant
    "self.assertIn(g.state, VALID)",
])
def test_these_all_hold_on_a_frozen_world(src):
    assert not _asserts_progress(_stmts(src))

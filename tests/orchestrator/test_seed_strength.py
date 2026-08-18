"""Can the seeded contract tell a working artifact from a stub?

The three contracts this benchmark actually produced, from one prompt on
two models, are the fixtures: 23 mocked tests, 2 substantive ones, and
one asserting only that a process had not exited. The third earned
`Evidence: independent (pre-existing-tests)` honestly — it ran, it
passed, and it would pass over any program that starts.
"""

import pytest

from agentchanti.orchestrator.seed_strength import weak_contract_reason


LIVENESS_ONLY = '''
import subprocess
import sys
import time
import unittest


class Contract(unittest.TestCase):
    def test_game_starts_and_remains_running(self):
        p = subprocess.Popen([sys.executable, "snake_game.py"])
        time.sleep(2)
        self.assertIsNone(p.poll(), "the game should still be running")
        p.terminate()
'''

EXISTENCE_ONLY = '''
import unittest


class Contract(unittest.TestCase):
    def test_api_exists(self):
        import snake_game
        game = snake_game.Game()
        self.assertIsNotNone(game)
        self.assertTrue(hasattr(game, "move"))
        self.assertIsInstance(game.positions, list)
'''

SUBSTANTIVE = '''
import unittest


class Contract(unittest.TestCase):
    def test_snake_moves_one_cell(self):
        import snake_game
        game = snake_game.Game()
        before = game.positions[0]
        game.move()
        self.assertNotEqual(game.positions[0], before)

    def test_snake_starts_with_three_segments(self):
        import snake_game
        game = snake_game.Game()
        self.assertEqual(len(game.positions), 3)
'''


class TestWeakContractsAreCaught:
    def test_liveness_only_is_weak(self):
        reason = weak_contract_reason(LIVENESS_ONLY)
        assert reason is not None
        assert "stub" in reason

    def test_existence_only_is_weak(self):
        assert weak_contract_reason(EXISTENCE_ONLY) is not None

    def test_a_tautology_is_weak(self):
        src = ('import unittest\n\n\nclass C(unittest.TestCase):\n'
               '    def test_x(self):\n        self.assertTrue(True)\n')
        assert weak_contract_reason(src) is not None


class TestRealContractsAreNotRejected:
    def test_two_comparing_assertions_are_enough(self):
        assert weak_contract_reason(SUBSTANTIVE) is None

    def test_a_bare_assert_on_observed_state_counts(self):
        src = ('import unittest\n\n\nclass C(unittest.TestCase):\n'
               '    def test_x(self):\n'
               '        import game\n'
               '        g = game.Game()\n'
               '        assert g.score == 0\n'
               '        assert len(g.cells) == 3\n')
        assert weak_contract_reason(src) is None

    def test_membership_in_an_observed_collection_counts(self):
        src = ('import unittest\n\n\nclass C(unittest.TestCase):\n'
               '    def test_x(self):\n'
               '        import game\n'
               '        g = game.Game()\n'
               '        self.assertIn(g.food, g.free_cells())\n'
               '        self.assertIn(g.head, g.occupied())\n')
        assert weak_contract_reason(src) is None

    def test_membership_in_a_literal_set_does_not_count(self):
        # `assertIn(state, ("a", "b", "c"))` admits every value the code
        # can produce — the tautology `degenerate-long-run` also refuses.
        src = ('import unittest\n\n\nclass C(unittest.TestCase):\n'
               '    def test_x(self):\n'
               '        import game\n'
               '        g = game.Game()\n'
               '        self.assertIn(g.state, ("start", "playing", "over"))\n')
        assert weak_contract_reason(src) is not None


class TestSilences:
    def test_unparseable_source_is_another_checks_problem(self):
        assert weak_contract_reason("def broken(:\n") is None

    def test_a_module_with_no_tests_is_another_checks_problem(self):
        assert weak_contract_reason("import unittest\n") is None

    def test_empty_is_named(self):
        assert weak_contract_reason("") == "empty"


class TestTheSeedRetriesOnce:
    def _client(self, *responses):
        from unittest.mock import MagicMock
        c = MagicMock()
        c.generate_response.side_effect = [
            "```python\n" + r + "```" for r in responses]
        return c

    def test_a_weak_first_draft_is_replaced(self, tmp_path):
        from agentchanti.orchestrator.acceptance_seed import (
            SEED_BASENAME, seed_acceptance_tests,
        )
        client = self._client(LIVENESS_ONLY, SUBSTANTIVE)
        path = seed_acceptance_tests("Build a snake game.", str(tmp_path),
                                     client, language="python")
        assert path is not None
        assert client.generate_response.call_count == 2
        written = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")
        assert "assertNotEqual" in written
        assert "poll()" not in written

    def test_a_strong_first_draft_costs_no_second_call(self, tmp_path):
        from agentchanti.orchestrator.acceptance_seed import seed_acceptance_tests

        client = self._client(SUBSTANTIVE)
        assert seed_acceptance_tests("Build a snake game.", str(tmp_path),
                                     client, language="python") is not None
        assert client.generate_response.call_count == 1

    def test_a_still_weak_retry_keeps_the_contract_anyway(self, tmp_path):
        """A shallow check that runs still catches a crashing artifact.

        Refusing would trade a weak instrument for none at all — so it is
        kept, and the log says plainly what it can and cannot catch.
        """
        from agentchanti.orchestrator.acceptance_seed import (
            SEED_BASENAME, seed_acceptance_tests,
        )
        client = self._client(LIVENESS_ONLY, EXISTENCE_ONLY)
        path = seed_acceptance_tests("Build a snake game.", str(tmp_path),
                                     client, language="python")
        assert path is not None
        assert (tmp_path / SEED_BASENAME).exists()

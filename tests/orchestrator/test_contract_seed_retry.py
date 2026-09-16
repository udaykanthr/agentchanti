"""An unusable response is not a contract that was judged and refused.

Measured 2026-09-16, benchmark task `django-webapp`:

    22:54:08 [OpenAI] Streaming ~1426 est. tokens
    22:56:51 [OpenAI] Streamed usage: prompt=1769 (cached=0) completion=9679
    22:56:51 [OpenAI] Response:
             9679
    22:56:51 [AcceptanceSeed] response was not a usable test module —
             no independent check seeded

9,679 completion tokens over 2m43s of reasoning, and nothing usable came
back. The run went on to build a Django app that passed all four
ground-truth commands — `manage.py check`, `manage.py test`, `/` → 200,
`/dashboard/` → 302 — and still exited 1, because with no contract there
was nothing independent left to verify it.

Every other rejection in the seeder retries `_REPAIR_ATTEMPTS` times:
mocking, interactive, structural, documentation, weakness. The FIRST
generation — the one that decides whether a contract exists at all — did
not, so a single flaky response cost the whole run its evidence.
"""
from agentchanti.orchestrator import acceptance_seed
from agentchanti.orchestrator.acceptance_seed import seed_acceptance_tests

TASK = "create a snake game in python"

GOOD = '''
import unittest

from snake_game import Game


class C(unittest.TestCase):
    def test_moves(self):
        g = Game(seed=1)
        before = g.head
        g.advance(0.5)
        self.assertNotEqual(g.head, before)

    def test_scores(self):
        g = Game(seed=1)
        g.eat()
        self.assertEqual(g.score, 10)
'''

# What the measured call actually returned: reasoning, no module.
BURN = "9679"


class _Client:
    """Returns each scripted response in turn."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_response(self, prompt):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return BURN


def _seed(tmp_path, client):
    return seed_acceptance_tests(TASK, str(tmp_path), client)


class TestTheMeasuredIncident:

    def test_a_burned_response_is_retried(self, tmp_path):
        client = _Client(BURN, GOOD)

        path = _seed(tmp_path, client)

        assert path is not None, "the second attempt returned a real contract"
        assert client.calls == 2

    def test_the_contract_is_written_to_disk(self, tmp_path):
        _seed(tmp_path, _Client(BURN, GOOD))
        seeded = tmp_path / acceptance_seed.SEED_BASENAME
        assert seeded.is_file()
        assert "class C" in seeded.read_text(encoding="utf-8")

    def test_before_the_fix_one_burn_was_the_end_of_it(self, tmp_path):
        """Pin what changed: the first response used to decide everything."""
        client = _Client(BURN, GOOD)
        assert acceptance_seed._generate(client, TASK) is None
        assert client.calls == 1


class TestTheBounds:

    def test_it_gives_up_after_the_budget(self, tmp_path):
        """Not unbounded: a model that cannot produce one never will."""
        client = _Client(BURN, BURN, BURN, BURN)

        path = _seed(tmp_path, client)

        assert path is None
        assert client.calls == acceptance_seed._REPAIR_ATTEMPTS
        assert not (tmp_path / acceptance_seed.SEED_BASENAME).exists()

    def test_a_good_first_response_costs_no_extra_call(self, tmp_path):
        client = _Client(GOOD)

        path = _seed(tmp_path, client)

        assert path is not None
        assert client.calls == 1, "a working draft must not be re-asked"

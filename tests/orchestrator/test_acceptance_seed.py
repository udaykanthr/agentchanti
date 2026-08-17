"""Acceptance tests written from the task, before any code exists.

`evidence.py` only counts two things as independent: user-supplied
`acceptance_cmds`, or a pre-existing test file the run left byte-identical.
A greenfield build has neither, so it is judged by a suite it wrote itself
— and three measured runs shipped exit 0 over artifacts that failed every
external probe while their own tests were green.

The point is *when*, not who. A test written after game.py exists is
written by an agent that has just read game.py. Seeding one from the task
text before the first step runs is the only moment a check can be written
that the code cannot have shaped, and it then flows through the existing
snapshot/independence machinery unchanged.
"""

import os

from unittest.mock import MagicMock

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, seed_acceptance_tests,
)

GOOD = '''```python
import unittest


class AcceptanceContract(unittest.TestCase):
    def test_player_moves(self):
        from game import Game
        g = Game(seed=0)
        g.start()
        before = g.entities()[0].tile
        g.press("left")
        for _ in range(500):
            g.advance(0.001)
        self.assertNotEqual(g.entities()[0].tile, before)
```'''


def _client(response):
    c = MagicMock()
    c.generate_response.return_value = response
    return c


def test_a_usable_suite_is_written(tmp_path):
    path = seed_acceptance_tests("Build a game.", str(tmp_path),
                                 _client(GOOD), language="python")
    assert path is not None
    written = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")
    assert "import unittest" in written
    assert "assertNotEqual" in written


def test_an_existing_suite_is_never_overwritten(tmp_path):
    """A test that already predates the run is stronger evidence than
    anything generated here, and clobbering it would destroy the very
    independence this exists to create."""
    (tmp_path / "test_mine.py").write_text("import unittest\n",
                                           encoding="utf-8")
    path = seed_acceptance_tests("Build a game.", str(tmp_path),
                                 _client(GOOD), language="python")
    assert path is None
    assert not (tmp_path / SEED_BASENAME).exists()


def test_an_unusable_response_writes_nothing(tmp_path):
    """No file is an honest "nothing independent verified this". A broken
    or empty suite would be worse than none — it would fail the run for a
    reason that has nothing to do with the code."""
    for junk in ("I could not determine any testable behaviour.",
                 "```python\nthis is not valid python(\n```",
                 "```python\nimport unittest\n```",     # no test method
                 ""):
        assert seed_acceptance_tests("Build a game.", str(tmp_path),
                                     _client(junk), language="python") is None
        assert not (tmp_path / SEED_BASENAME).exists()


def test_a_non_python_project_is_skipped(tmp_path):
    assert seed_acceptance_tests("Build an app.", str(tmp_path),
                                 _client(GOOD), language="javascript") is None


def test_a_generation_failure_never_raises(tmp_path):
    client = MagicMock()
    client.generate_response.side_effect = RuntimeError("provider down")
    assert seed_acceptance_tests("Build a game.", str(tmp_path),
                                 client, language="python") is None


def test_the_seed_is_a_test_file_the_snapshot_will_record(tmp_path):
    """The whole mechanism depends on `_is_test_file` recognising it and
    `snapshot_test_files` hashing it a moment later."""
    from agentchanti.orchestrator.evidence import snapshot_test_files
    from agentchanti.orchestrator.pipeline import _is_test_file

    assert _is_test_file(SEED_BASENAME)
    seed_acceptance_tests("Build a game.", str(tmp_path),
                          _client(GOOD), language="python")
    snap = snapshot_test_files(str(tmp_path))
    assert SEED_BASENAME in snap


def test_rewriting_the_seed_forfeits_independence(tmp_path):
    """The rule that makes seeding safe: if a later step edits it, the
    hash changes and the run says so rather than claiming verification."""
    from agentchanti.orchestrator.evidence import (
        snapshot_test_files, surviving_pre_existing_tests,
    )
    seed_acceptance_tests("Build a game.", str(tmp_path),
                          _client(GOOD), language="python")
    snap = snapshot_test_files(str(tmp_path))
    assert surviving_pre_existing_tests(str(tmp_path), snap) == [SEED_BASENAME]

    (tmp_path / SEED_BASENAME).write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        pass\n", encoding="utf-8")
    assert surviving_pre_existing_tests(str(tmp_path), snap) == []


def test_the_seed_file_is_protected_from_overwrite():
    """Executor refuses to clobber it the way it refuses requirements.txt."""
    from agentchanti.executor import Executor
    assert SEED_BASENAME in Executor._PROTECTED_FILENAMES


def test_it_is_written_before_any_step_could_have_shaped_it(tmp_path):
    """Regression guard on the prompt itself: it must ask for assertions
    from the TASK, never from files on disk."""
    client = _client(GOOD)
    seed_acceptance_tests("Build a Pac-Man clone.", str(tmp_path),
                          client, language="python")
    prompt = client.generate_response.call_args.args[0]
    assert "BEFORE any code exists" in prompt
    assert "Build a Pac-Man clone." in prompt
    # The rules that came out of the measured failures.
    assert "assertLessEqual" in prompt        # monotonicity is not progress
    assert "hundreds of" in prompt            # a range needs exercising
    assert "RANGE" in prompt

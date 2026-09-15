"""A contract that needs a file a later step will write is ahead, not broken.

Measured 2026-09-15, a 12-step Snake run on 0.8.1. After wave 6 the seeded
contract reported ``FAILED (failures=3, errors=1)``. The one error was::

    Path("README.md").stat().st_size
    FileNotFoundError: [WinError 2] The system cannot find the file specified: 'README.md'

README.md was the declared target of step 5.2, which ran in wave 7. The
error was read as a broken instrument; two repair attempts spent 26k tokens
and blocked the run for three minutes, both were rejected, and the
unmodified contract passed 5/5 at the end of the run.
"""
import inspect
import os
import textwrap

import pytest

from agentchanti.orchestrator import cli
from agentchanti.orchestrator.acceptance_seed import (
    _names_pending_target, reset_contract_repairs, verify_contract_runs,
)
from tests.orchestrator.test_contract_runnability import (
    PROJECT, RealExecutor, ScriptedClient, _write_contract,
)

# The measured shape: an ERROR on a planned file that does not exist yet.
NEEDS_README = """
    import unittest
    from pathlib import Path

    import project


    class Contract(unittest.TestCase):
        def test_docs_and_rules(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            self.assertGreater(Path("README.md").stat().st_size, 0)
            self.assertEqual(game.score, 0)
"""

# A real instrument defect, next to a message that names a pending file.
GENUINE_CRASH = """
    import unittest
    from pathlib import Path

    import project


    class Contract(unittest.TestCase):
        def test_docs(self):
            self.assertTrue(Path("README.md").is_file(),
                            "Required file is missing: README.md")

        def test_rules(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            self.assertEqual(sum(game.colour[:3]), 3)
"""


@pytest.fixture
def project(tmp_path):
    reset_contract_repairs()
    with open(os.path.join(str(tmp_path), "project.py"), "w",
              encoding="utf-8") as fh:
        fh.write(textwrap.dedent(PROJECT))
    yield tmp_path
    reset_contract_repairs()


def test_a_file_a_later_step_writes_defers_the_repair(project):
    _write_contract(project, NEEDS_README)
    client = ScriptedClient()                  # any generation would raise
    result = verify_contract_runs(RealExecutor(project), str(project), client,
                                  "a game", pending_targets=["README.md"])
    assert result is None
    assert not client.prompts, "a file not written yet was 'repaired'"


def test_without_the_plan_it_is_still_repaired(project):
    """States the measured cost the plan now avoids."""
    _write_contract(project, NEEDS_README)
    broken = textwrap.dedent(NEEDS_README).strip()
    client = ScriptedClient(broken, broken)
    verify_contract_runs(RealExecutor(project), str(project), client, "a game")
    assert client.prompts, "the pre-fix behaviour this test documents changed"


def test_a_deferral_is_not_a_verdict(project):
    """The contract must be looked at again once the file exists."""
    _write_contract(project, NEEDS_README)
    client = ScriptedClient()
    ex = RealExecutor(project)
    verify_contract_runs(ex, str(project), client, "a game",
                         pending_targets=["README.md"])
    runs = len(ex.commands)
    (project / "README.md").write_text("# Game\n")
    verify_contract_runs(ex, str(project), client, "a game", pending_targets=[])
    assert len(ex.commands) > runs


def test_the_final_check_never_defers_on_the_plan(project):
    """There is no later step at the end; a missing file is then real."""
    _write_contract(project, NEEDS_README)
    broken = textwrap.dedent(NEEDS_README).strip()
    client = ScriptedClient(broken, broken)
    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a game", final=True, pending_targets=["README.md"])
    assert client.prompts


def test_a_genuine_crash_beside_a_pending_file_is_still_repaired(project):
    """Judged per line: an assertion MESSAGE naming the pending file does not
    excuse a TypeError elsewhere."""
    _write_contract(project, GENUINE_CRASH)
    broken = textwrap.dedent(GENUINE_CRASH).strip()
    client = ScriptedClient(broken, broken)
    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a game", pending_targets=["README.md"])
    assert client.prompts, "a real instrument crash was excused"


class TestMatching:

    @pytest.mark.parametrize("line, target", [
        ("FileNotFoundError: [WinError 2] The system cannot find the file "
         "specified: 'README.md'", "README.md"),
        ("FileNotFoundError: [Errno 2] No such file or directory: "
         "'C:\\\\proj\\\\tests\\\\test_game.py'", "tests/test_game.py"),
        ("python: can't open file '/proj/tests/test_storage.py': [Errno 2]",
         "./tests/test_storage.py"),
    ])
    def test_the_shapes_that_occur(self, line, target):
        assert _names_pending_target(line, [target])

    def test_a_different_missing_file_is_not_pending(self):
        line = "FileNotFoundError: No such file or directory: 'config.json'"
        assert _names_pending_target(line, ["README.md"]) is None

    def test_a_mention_without_a_missing_file_error_is_not_enough(self):
        line = "AssertionError: Required file is missing: README.md"
        assert _names_pending_target(line, ["README.md"]) is None

    def test_no_plan_means_no_deferral(self):
        line = "FileNotFoundError: 'README.md'"
        assert _names_pending_target(line, None) is None
        assert _names_pending_target(line, []) is None


def test_the_pipeline_passes_the_later_waves_targets():
    src = inspect.getsource(cli._main_impl)
    assert "waves[wave_idx + 1:]" in src
    assert "pending_targets=_pending_targets" in src

"""A contract the run wrote cannot, on its own, fail the run.

`require_independent_evidence` turned "nothing independent verified this"
into exit 1. When the only instrument that could have verified it was a
contract this pipeline wrote before any code existed, that was the model
grading the model. Measured 2026-09-13..15 on one prompt: four consecutive
runs exited 1 over working games, each on a different contract mistake —
README wording, `parents[1]` from the project root, a Win32 window lookup
that could never match a venv child process, and the exact text
`>=2.5`/`<3.0` in requirements.txt.

The rule is narrow on purpose: every other way the flag fails a run stays.
"""
import inspect
import os

from agentchanti import api
from agentchanti.orchestrator import cli
from agentchanti.orchestrator.acceptance_seed import SEED_BASENAME, _header
from agentchanti.orchestrator.evidence import (
    _digest, seeded_contract_was_the_only_witness,
)

BODY = "\nimport unittest\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(1, 2)\n"


def _seed(root):
    path = os.path.join(str(root), SEED_BASENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_header("a game", BODY) + BODY)
    return {SEED_BASENAME: _digest(path)}


def _user_suite(root, name="test_user.py"):
    path = os.path.join(str(root), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(BODY.lstrip())
    return {name: _digest(path)}


def test_the_measured_case_does_not_fail_the_run(tmp_path):
    snap = _seed(tmp_path)
    assert seeded_contract_was_the_only_witness(str(tmp_path), snap) == \
        SEED_BASENAME


def test_user_acceptance_cmds_keep_the_failure(tmp_path):
    snap = _seed(tmp_path)
    assert seeded_contract_was_the_only_witness(
        str(tmp_path), snap, ["python acceptance.py"]) is None


def test_a_user_suite_keeps_the_failure(tmp_path):
    snap = {**_seed(tmp_path), **_user_suite(tmp_path)}
    assert seeded_contract_was_the_only_witness(str(tmp_path), snap) is None


def test_a_contract_the_agent_edited_keeps_the_failure(tmp_path):
    """Tampering forfeits the contract; nothing survives, so no excuse."""
    snap = _seed(tmp_path)
    with open(os.path.join(str(tmp_path), SEED_BASENAME), "a",
              encoding="utf-8") as fh:
        fh.write("# edited to pass\n")
    assert seeded_contract_was_the_only_witness(str(tmp_path), snap) is None


def test_no_evidence_at_all_keeps_the_failure(tmp_path):
    assert seeded_contract_was_the_only_witness(str(tmp_path), {}) is None


def test_both_entry_points_apply_the_same_rule():
    # run_task delegates to an inner function; the module is the seam.
    for src in (inspect.getsource(cli._main_impl), inspect.getsource(api)):
        assert "seeded_contract_was_the_only_witness" in src
        assert "require_independent_evidence is set" in src

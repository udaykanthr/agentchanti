"""A retry that is not told why the last attempt failed is a blind re-roll.

Measured live 2026-09-09: both repair attempts came back weaker than the
original (0 then 1 substantive assertion against 2), because the second
prompt was byte-identical to the first — nothing said the first had been
refused, or why. The still-crashing branch already fed its error back;
these tests pin that every other rejection now does too.
"""
import subprocess
import sys
import textwrap

import pytest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, reset_contract_repairs, verify_contract_runs,
)
from tests.orchestrator.test_contract_runnability import (  # noqa: F401
    CRASHING, GUTTED, PROJECT, REPAIRED, STILL_CRASHING, RealExecutor,
    ScriptedClient, _write_contract, project,
)


@pytest.fixture(autouse=True)
def _clean_state():
    reset_contract_repairs()
    yield
    reset_contract_repairs()


def test_a_weak_repair_is_told_it_was_weak(project):
    """The exact live failure: two blind asks, two weak answers."""
    _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(GUTTED).strip(),
                            textwrap.dedent(REPAIRED).strip())

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is not None, "the second attempt did not recover"
    assert len(client.prompts) == 2
    first, second = client.prompts
    assert "PREVIOUS ATTEMPT WAS REJECTED" not in first, \
        "the first ask cannot cite a rejection that has not happened"
    assert "PREVIOUS ATTEMPT WAS REJECTED" in second, \
        "the retry never said the last attempt had been refused"
    assert "meaningful assertion" in second, \
        "the retry did not say WHY it was refused"
    proc = subprocess.run([sys.executable, "-m", "unittest", SEED_BASENAME],
                          cwd=str(project), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_still_crashing_repair_is_told_what_it_broke(project):
    _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(STILL_CRASHING).strip(),
                            textwrap.dedent(REPAIRED).strip())

    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a snake game")

    second = client.prompts[-1]
    assert "PREVIOUS ATTEMPT WAS REJECTED" in second
    assert "still" in second and "crashed" in second
    assert "palette" in second, "the new error itself was not carried"


def test_a_mocking_repair_is_told_it_mocked(project):
    mocked = textwrap.dedent(REPAIRED).strip().replace(
        "import unittest",
        "import unittest\nfrom unittest.mock import patch  # noqa: F401", 1)
    _write_contract(project, CRASHING)
    client = ScriptedClient(mocked, textwrap.dedent(REPAIRED).strip())

    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a snake game")

    assert len(client.prompts) == 2
    assert "PREVIOUS ATTEMPT WAS REJECTED" in client.prompts[-1]
    assert "stub" in client.prompts[-1] or "mock" in client.prompts[-1]


def test_the_feedback_never_leaks_the_artifact(project):
    """Independence again: the rejection reason must not carry the code."""
    _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(GUTTED).strip(),
                            textwrap.dedent(REPAIRED).strip())

    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a snake game")

    source = (project / "project.py").read_text(encoding="utf-8")
    body = source.split("class Game:", 1)[1].strip()
    for prompt in client.prompts:
        assert body not in prompt, "the artifact's source leaked into a retry"

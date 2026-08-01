"""Shared test isolation for orchestrator tests."""

import pytest

from agentchanti.orchestrator.agent_loop import reset_attempt_journal


@pytest.fixture(autouse=True)
def _isolate_attempt_journal():
    """Clear the agent-loop attempt journal around every test.

    The journal is process-global and keyed by step index, and almost
    every test here uses step_idx=0. Without this a test sees the
    attempts recorded by whichever tests ran before it — observed as an
    escalation-context assertion failing against "attempt 28" from
    unrelated cases.
    """
    reset_attempt_journal()
    yield
    reset_attempt_journal()

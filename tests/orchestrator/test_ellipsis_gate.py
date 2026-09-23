"""`...` left in a gate is a blank the plan never filled in.

Measured 2026-09-23, glm-5.3:cloud. The step's entire verify command was

    cd my-app && ...

The shell answers "'...' is not recognized", identically over three
versions of the file; `gate STALLED` fired, escalation was suppressed, and
the run was reported failed over an app whose contract passes 7/7 by hand.
`_PLACEHOLDER_RE` only ever matched `<angle placeholders>`.
"""
import pytest

from agentchanti.orchestrator.plan_step import unrunnable_gate_reason


@pytest.mark.parametrize("cmd", [
    "cd my-app && ...",
    "cd my-app && npm run build && ...",
    "npm test && … ",                      # the single-character ellipsis
    "...",
])
def test_an_ellipsis_command_is_refused(cmd):
    why = unrunnable_gate_reason(cmd)
    assert why and "..." in why


@pytest.mark.parametrize("cmd", [
    # Python's Ellipsis inside a payload is valid code, not a blank.
    'python -c "from typing import Any; x: Any = ...; assert x is Ellipsis"',
    'node -e "console.log(\'a...b\')"',
    "cd my-app && npm run build",
    "python -m pytest tests/test_a.py",
])
def test_real_commands_are_untouched(cmd):
    assert unrunnable_gate_reason(cmd) is None


def test_a_filename_with_dots_is_not_a_placeholder():
    assert unrunnable_gate_reason("python ...hidden/run.py") is None

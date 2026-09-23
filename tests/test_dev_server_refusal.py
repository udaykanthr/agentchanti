"""The agent's run_command refuses to start a server that never exits.

Measured 2026-09-22: five dev-server attempts in one step, each burning the
full 120s timeout, the last of which hung the pipeline for four hours.
"""
import pytest

from agentchanti.agent_tools import dev_server_reason


@pytest.mark.parametrize("cmd", [
    "cd my-app && npm run dev",
    "cd my-app && npm run dev &",
    'cd my-app && start /b npm run dev && timeout /t 6 /nobreak >nul && '
    'node -e "fetch(\'http://localhost:3000\')"',
    "npx next dev",
    "npx vite",
    "python manage.py runserver",
])
def test_measured_and_sibling_servers_are_refused(cmd):
    assert "build" in dev_server_reason(cmd)


@pytest.mark.parametrize("cmd", [
    "npm --prefix my-app run build",
    "cd my-app && npx next build",
    "npx vite build",
    "npm test",
    "npm start",                      # a CLI's start script exits
    "node -e \"console.log('npm run dev is refused')\"".replace("npm run dev", "dev"),
])
def test_commands_that_exit_are_left_alone(cmd):
    assert dev_server_reason(cmd) is None

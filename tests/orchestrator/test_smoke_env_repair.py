"""A launch crash for a missing package is an environment finding.

Measured 2026-08-18 on both benchmark paths of the same prompt: the
project venv held nothing but pip, `python main.py` crashed on
`import pygame`, and the fix the model returned rewrote the graphical
entry point into a silent fallback to headless mode. The relaunch then
succeeded and every gate stayed green over a Pac-Man that never opens a
window. Asked to stop a missing import from crashing the app, a model
removes the import — so the environment has to be repaired before the
crash is ever shown to one.
"""

import os

import pytest

from agentchanti.orchestrator import smoke_test as st


PYGAME_TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "C:\\proj\\main.py", line 104, in run_game\n'
    '    import pygame\n'
    "ModuleNotFoundError: No module named 'pygame'\n"
)

# The graceful spelling: the app caught its own ImportError and advised.
PYGAME_ADVICE = (
    "Pygame is required for graphical mode. "
    "Install it with: python -m pip install pygame\n"
)


class TestDependencyNamedBy:
    def test_reads_a_module_not_found_traceback(self):
        assert st._dependency_named_by(PYGAME_TRACEBACK, {}) == "pygame"

    def test_reads_the_apps_own_install_advice(self):
        # No traceback at all — the shape a bare ModuleNotFoundError
        # match misses, and the one the classic run actually produced.
        assert st._dependency_named_by(PYGAME_ADVICE, {}) == "pygame"

    def test_refuses_a_module_the_project_defines(self):
        out = "ModuleNotFoundError: No module named 'game'"
        assert st._dependency_named_by(out, {"game.py": "..."}) is None

    def test_refuses_a_standard_library_name(self):
        # Installing this fetches whatever squats on PyPI under the name.
        out = "ModuleNotFoundError: No module named 'json'"
        assert st._dependency_named_by(out, {}) is None

    def test_silent_on_an_unrelated_crash(self):
        out = "TypeError: advance() missing 1 required positional argument"
        assert st._dependency_named_by(out, {}) is None


class _FakeExecutor:
    def __init__(self, importable=False, install_ok=True):
        self.importable = importable
        self.install_ok = install_ok
        self.commands = []

    def run_command(self, cmd, **kw):
        self.commands.append(cmd)
        if " -c " in cmd:
            return self.importable, ""
        return self.install_ok, "" if self.install_ok else "no matching dist"


class TestRepairMissingDependency:
    def test_installs_into_the_project_venv(self, tmp_path, monkeypatch):
        from agentchanti.orchestrator import agent_loop
        py = os.path.join(str(tmp_path), "venv", "bin", "python")
        monkeypatch.setattr(agent_loop, "_venv_python", lambda root: py)
        monkeypatch.chdir(tmp_path)
        ex = _FakeExecutor()
        assert st._repair_missing_dependency("pygame", ex) is True
        assert any(py in c and "-m pip install pygame" in c
                   for c in ex.commands)

    def test_refuses_when_the_project_has_no_venv(self, tmp_path, monkeypatch):
        # Installing into the ambient interpreter would be the pipeline
        # writing to the user's machine on a guess.
        from agentchanti.orchestrator import agent_loop
        monkeypatch.setattr(agent_loop, "_venv_python", lambda root: None)
        monkeypatch.chdir(tmp_path)
        ex = _FakeExecutor()
        assert st._repair_missing_dependency("pygame", ex) is False
        assert ex.commands == []

    def test_does_not_install_something_already_importable(
            self, tmp_path, monkeypatch):
        from agentchanti.orchestrator import agent_loop
        py = os.path.join(str(tmp_path), "venv", "bin", "python")
        monkeypatch.setattr(agent_loop, "_venv_python", lambda root: py)
        monkeypatch.chdir(tmp_path)
        ex = _FakeExecutor(importable=True)
        assert st._repair_missing_dependency("pygame", ex) is False
        assert not any("pip install" in c for c in ex.commands)

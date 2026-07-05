"""Tests for project-venv interpreter resolution in the Executor.

A pipeline step creates a venv and installs packages into it, but every
command runs in a fresh shell, so `activate` never persists.  The Executor
must prepend the venv's Scripts/bin dir to PATH so bare `python`/`pip`/
`pytest` resolve to the venv interpreter.
"""

import os

import pytest

from agentchanti.executor import Executor


def _make_fake_venv(root, name="venv"):
    """Create the minimal on-disk structure _venv_bin_dir looks for."""
    if os.name == 'nt':
        bin_dir = os.path.join(str(root), name, "Scripts")
        py = "python.exe"
    else:
        bin_dir = os.path.join(str(root), name, "bin")
        py = "python"
    os.makedirs(bin_dir)
    py_path = os.path.join(bin_dir, py)
    with open(py_path, "w") as f:
        f.write("")
    if os.name != 'nt':
        os.chmod(py_path, 0o755)  # `which` skips non-executable files
    return bin_dir


class TestVenvBinDir:
    def test_no_venv_returns_none(self, tmp_path):
        assert Executor._venv_bin_dir(str(tmp_path)) is None

    def test_detects_venv(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path, "venv")
        assert Executor._venv_bin_dir(str(tmp_path)) == os.path.abspath(bin_dir)

    def test_detects_dot_venv(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path, ".venv")
        assert Executor._venv_bin_dir(str(tmp_path)) == os.path.abspath(bin_dir)

    def test_venv_preferred_over_dot_venv(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path, "venv")
        _make_fake_venv(tmp_path, ".venv")
        assert Executor._venv_bin_dir(str(tmp_path)) == os.path.abspath(bin_dir)

    def test_ignores_venv_dir_without_python(self, tmp_path):
        # A half-created or foreign 'venv' folder must not hijack PATH
        sub = "Scripts" if os.name == 'nt' else "bin"
        os.makedirs(os.path.join(str(tmp_path), "venv", sub))
        assert Executor._venv_bin_dir(str(tmp_path)) is None

    def test_defaults_to_cwd(self, tmp_path, monkeypatch):
        bin_dir = _make_fake_venv(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert Executor._venv_bin_dir() == os.path.abspath(bin_dir)


class TestInjectVenvPath:
    def test_prepends_to_path(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path)
        env = {"PATH": "/usr/bin"}
        Executor._inject_venv_path(env, str(tmp_path))
        assert env["PATH"].split(os.pathsep)[0] == os.path.abspath(bin_dir)
        assert "/usr/bin" in env["PATH"]

    def test_sets_virtual_env(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path)
        env = {"PATH": ""}
        Executor._inject_venv_path(env, str(tmp_path))
        assert env["VIRTUAL_ENV"] == os.path.dirname(os.path.abspath(bin_dir))

    def test_noop_without_venv(self, tmp_path):
        env = {"PATH": "/usr/bin"}
        Executor._inject_venv_path(env, str(tmp_path))
        assert env == {"PATH": "/usr/bin"}

    def test_reuses_existing_path_key_casing(self, tmp_path):
        # Windows env blocks often carry 'Path' — a second 'PATH' key would
        # produce a duplicate in the child environment
        bin_dir = _make_fake_venv(tmp_path)
        env = {"Path": "C:\\Windows"}
        Executor._inject_venv_path(env, str(tmp_path))
        assert "PATH" not in env
        assert env["Path"].startswith(os.path.abspath(bin_dir))

    def test_idempotent(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path)
        env = {"PATH": "/usr/bin"}
        Executor._inject_venv_path(env, str(tmp_path))
        once = env["PATH"]
        Executor._inject_venv_path(env, str(tmp_path))
        assert env["PATH"] == once

    def test_handles_missing_path_key(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path)
        env = {}
        Executor._inject_venv_path(env, str(tmp_path))
        assert env["PATH"] == os.path.abspath(bin_dir)


@pytest.mark.skipif(os.name != 'nt' and not os.path.exists('/bin/sh'),
                    reason="needs a shell")
class TestRunCommandUsesVenv:
    def test_run_command_resolves_venv_python_first(self, tmp_path):
        """`python` on PATH must point into the project venv."""
        bin_dir = _make_fake_venv(tmp_path)
        executor = Executor()
        if os.name == 'nt':
            probe = "where python"
        else:
            probe = "which -a python"
        ok, out = executor.run_command(probe, cwd=str(tmp_path))
        assert ok
        assert out.splitlines()[0].strip().startswith(os.path.abspath(bin_dir))

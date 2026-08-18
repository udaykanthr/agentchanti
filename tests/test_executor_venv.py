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

    def test_finds_single_subproject_venv(self, tmp_path):
        # Scaffolded project one level down: probes run from the pipeline
        # root but the venv lives in user_portal/venv.
        sub = tmp_path / "user_portal"
        sub.mkdir()
        bin_dir = _make_fake_venv(sub, "venv")
        assert Executor._venv_bin_dir(str(tmp_path)) == os.path.abspath(bin_dir)

    def test_direct_venv_wins_over_subproject(self, tmp_path):
        bin_dir = _make_fake_venv(tmp_path, "venv")
        sub = tmp_path / "app"
        sub.mkdir()
        _make_fake_venv(sub, "venv")
        assert Executor._venv_bin_dir(str(tmp_path)) == os.path.abspath(bin_dir)

    def test_ambiguous_subproject_venvs_return_none(self, tmp_path):
        for name in ("app_a", "app_b"):
            sub = tmp_path / name
            sub.mkdir()
            _make_fake_venv(sub, "venv")
        assert Executor._venv_bin_dir(str(tmp_path)) is None

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


class TestRewriteVenvInstall:
    """The install must land in the interpreter the rest of the run uses.

    Measured 2026-08-18 on both benchmark paths: a planner step spelled
    ``python -m venv venv && python3 -m pip install -U pygame`` left the
    venv holding nothing but pip, because a Windows venv has no
    ``python3.exe`` and the name fell through to the ambient interpreter.
    Every later command ran under the empty venv.
    """

    def _venv_py(self, root, name="venv"):
        return Executor._venv_python_at(os.path.abspath(str(root)), name)

    def test_redirects_python3_install_to_a_venv_the_command_creates(
            self, tmp_path):
        # The venv does not exist yet — it is built by the first segment,
        # so PATH injection (computed before the command ran) cannot help.
        cmd = "python -m venv venv && python3 -m pip install -U pygame"
        out = Executor._rewrite_venv_install(cmd, str(tmp_path))
        assert self._venv_py(tmp_path) in out
        assert "python3 -m pip" not in out

    def test_drops_user_flag_when_redirecting(self, tmp_path):
        # pip refuses `--user` inside a venv, so carrying the flag over
        # would turn a wrong-target install into a failing one.
        cmd = "python -m venv venv && python3 -m pip install -U pygame --user"
        out = Executor._rewrite_venv_install(cmd, str(tmp_path))
        assert "--user" not in out
        assert self._venv_py(tmp_path) in out

    def test_bare_pip_install_uses_the_existing_venv(self, tmp_path):
        _make_fake_venv(tmp_path)
        out = Executor._rewrite_venv_install("pip install pygame",
                                             str(tmp_path))
        assert out.startswith('"' + self._venv_py(tmp_path) + '" -m pip install')

    def test_silent_without_a_project_venv(self, tmp_path):
        # A project on the ambient interpreter must never be redirected
        # into a venv that does not exist.
        cmd = "pip install requests"
        assert Executor._rewrite_venv_install(cmd, str(tmp_path)) == cmd

    def test_leaves_non_install_commands_alone(self, tmp_path):
        _make_fake_venv(tmp_path)
        for cmd in ("python main.py", "npm install", "python -m venv venv"):
            assert Executor._rewrite_venv_install(cmd, str(tmp_path)) == cmd

    def test_stops_at_a_directory_change(self, tmp_path):
        # After `cd`, "the project venv" is no longer the one we resolved.
        _make_fake_venv(tmp_path)
        cmd = "cd sub && pip install pygame"
        assert Executor._rewrite_venv_install(cmd, str(tmp_path)) == cmd

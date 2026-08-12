"""Inline scripts must not be parsed by cmd.exe.

`python -c "...assert n > 0..."` is ordinary Python, but under
``shell=True`` cmd.exe reads that `>` as redirection: the command's real
stdout lands in a file literally named `0` and the caller is handed an
empty string. Found in a benchmark run where two agent-loop verification
commands returned nothing, leaving the model unable to see why its code
"failed" — the step burned all 8 turns and escalated.

The contract these tests pin: divert ONLY single inline-script
invocations, never anything using genuine shell syntax.
"""

import os
import subprocess

import pytest

from agentchanti.executor import Executor

windows_only = pytest.mark.skipif(
    os.name != 'nt', reason="cmd.exe parsing is Windows-specific")


# The exact shape that broke: escaped inner quotes plus a `>` operator.
ESCAPED = ('python -c "exec(\\"import random\\nassert 1 > 0\\n'
           'print(\'ok\')\\")"')
SIMPLE = 'python -c "x=1; assert x > 0; print(\'ok\')"'


class TestShellFreeArgv:
    @windows_only
    @pytest.mark.parametrize("cmd", [ESCAPED, SIMPLE])
    def test_diverts_inline_script_containing_an_operator(self, cmd):
        argv = Executor._shell_free_argv(cmd)
        assert argv is not None
        assert len(argv) == 3
        assert argv[1] == '-c'
        # The script must survive parsing intact — a mangled script would
        # be a different bug wearing the fix's clothes.
        assert 'assert' in argv[2] and '>' in argv[2]

    @windows_only
    @pytest.mark.parametrize("cmd", [
        # Genuine redirection: `>` and its target are separate argv
        # entries, so this is real shell work and must stay on the shell.
        'python -m pytest > out.txt',
        'python -c "print(1)" | more',
        'call venv\\Scripts\\activate && python -m pytest',
        'python script.py && echo done',
    ])
    def test_keeps_real_shell_syntax_on_the_shell(self, cmd):
        assert Executor._shell_free_argv(cmd) is None

    @windows_only
    def test_leaves_operator_free_scripts_alone(self):
        # Nothing to fix — keep the existing path to bound the change.
        assert Executor._shell_free_argv('python -c "print(1)"') is None

    @windows_only
    @pytest.mark.parametrize("cmd", [
        'pytest -c "a > b"',            # not an inline-script interpreter
        'python -m pytest',             # not an inline-script flag
        'python script.py',
    ])
    def test_ignores_non_inline_script_commands(self, cmd):
        assert Executor._shell_free_argv(cmd) is None

    def test_no_op_off_windows(self):
        if os.name == 'nt':
            pytest.skip("POSIX shells do not have this defect")
        assert Executor._shell_free_argv(SIMPLE) is None


class TestEndToEnd:
    @windows_only
    @pytest.mark.parametrize("cmd", [ESCAPED, SIMPLE])
    def test_output_is_captured_and_no_stray_file_appears(self, cmd, tmp_path):
        ex = Executor()
        ok, out = ex.run_command(cmd, cwd=str(tmp_path))
        assert ok, out
        assert 'ok' in out
        # The redirection targets cmd.exe would have created.
        strays = [p.name for p in tmp_path.iterdir() if p.name in ('0', '1')]
        assert not strays, f"stdout was redirected into {strays}"

    @windows_only
    def test_shell_commands_still_work(self, tmp_path):
        ex = Executor()
        ok, out = ex.run_command('echo hello && echo world',
                                 cwd=str(tmp_path))
        assert ok
        assert 'hello' in out and 'world' in out

    @windows_only
    def test_genuine_redirection_still_redirects(self, tmp_path):
        ex = Executor()
        ok, _ = ex.run_command('python -c "print(1)" > captured.txt',
                               cwd=str(tmp_path))
        assert ok
        assert (tmp_path / "captured.txt").exists()

    @windows_only
    def test_exit_code_still_propagates(self, tmp_path):
        ex = Executor()
        ok, _ = ex.run_command('python -c "assert 1 > 2"', cwd=str(tmp_path))
        assert not ok
        assert ex.last_exit_code != 0


class TestVenvInterpreterIsPreserved:
    """The bypass must not silently move scripts off the project venv.

    subprocess does not honour env's PATH when resolving an executable on
    Windows (CreateProcess searches the PARENT's PATH), so a naive
    ``Popen(['python', ...], env=...)`` launches the system interpreter and
    reports the project's own packages missing.
    """

    @windows_only
    def test_inline_script_uses_the_venv_interpreter(self, tmp_path):
        import sys
        # A venv whose "interpreter" is really the running one, so the
        # script can execute and report which path launched it.
        bin_dir = tmp_path / "venv" / "Scripts"
        bin_dir.mkdir(parents=True)
        shim = bin_dir / "python.exe"
        shim.write_bytes(open(sys.executable, 'rb').read())
        # A copied exe with no pyvenv.cfg beside it fails to start with
        # "failed to locate pyvenv.cfg" whenever the SUITE itself is run
        # from a venv — the copy then has no base installation to resolve
        # against. Write a real one so the fixture works under both.
        (bin_dir.parent / "pyvenv.cfg").write_text(
            f"home = {os.path.dirname(sys._base_executable or sys.executable)}\n"
            f"include-system-site-packages = false\n"
            f"version = {'.'.join(map(str, sys.version_info[:3]))}\n",
            encoding="utf-8")

        ex = Executor()
        ok, out = ex.run_command(
            'python -c "import sys; assert 1 > 0; print(sys.executable)"',
            cwd=str(tmp_path))
        assert ok, out
        assert str(bin_dir).lower() in out.lower(), (
            f"inline script ran under {out.strip()}, not the project venv")

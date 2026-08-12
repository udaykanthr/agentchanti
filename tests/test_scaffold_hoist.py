"""Hoisting a scaffold into the project root must not leave a copy behind.

`move dir\\*` on Windows moves FILES ONLY — subdirectories stay put. The
standard scaffold hoist an agent writes is therefore half-complete::

    npm create vite@latest scaffold -- --template react
    move scaffold\\* . && type scaffold\\.gitignore >> .gitignore && rmdir scaffold

`src\\` and `public\\` remain, the `rmdir` fails with "The directory is not
empty" (exit 1), and the run continues with TWO copies of every component.

Observed twice in one afternoon — a leftover `vite-react-scaffold\\` and a
nested `home_page\\home_page\\`. Both were then indexed by the KB, so
semantic search served later steps a stale duplicate of the very file they
were editing.
"""

import os

import pytest

from agentchanti.executor import Executor

windows_only = pytest.mark.skipif(
    os.name != 'nt', reason="`move dir\\*` semantics are Windows-specific")


class TestRewrite:
    def test_adds_a_subdirectory_pass(self):
        out = Executor._rewrite_single_unix_cmd('move scaffold\\* .')
        assert out.startswith('(move scaffold\\* .')
        assert 'for /d' in out
        assert 'scaffold\\*' in out

    def test_is_grouped_so_a_caller_s_chain_applies_to_the_whole(self):
        # Ungrouped, a trailing `&& rmdir` binds to the FOR body and is
        # skipped entirely when the directory has no subdirectories.
        out = Executor._rewrite_single_unix_cmd('move scaffold\\* .')
        assert out.startswith('(') and out.endswith(')')

    def test_preserves_flags(self):
        out = Executor._rewrite_single_unix_cmd('move /Y scaffold\\* .')
        assert '/Y' in out.split('&')[0]
        assert '/Y' in out.split('&', 1)[1]

    def test_handles_forward_slashes_and_nested_paths(self):
        out = Executor._rewrite_single_unix_cmd('move build/out/* dist')
        assert 'for /d %i in (build/out\\*)' in out
        assert out.rstrip(')').endswith('dist')

    @pytest.mark.parametrize("cmd", [
        'move a.txt b.txt',           # ordinary file move
        'move scaffold .',            # whole directory, already correct
        'npm create vite@latest x',
    ])
    def test_leaves_other_commands_alone(self, cmd):
        assert Executor._rewrite_single_unix_cmd(cmd) == cmd

    @windows_only
    def test_wired_into_the_windows_rewrite_pass(self):
        # The chained form is split on && before rewriting, so the repair
        # has to survive reassembly.
        out = Executor._rewrite_unix_cmd_for_windows(
            'move scaffold\\* . && rmdir scaffold')
        assert 'for /d' in out
        assert out.rstrip().endswith('rmdir scaffold')


@windows_only
class TestEndToEnd:
    """Drives the real cmd.exe, because the bug IS cmd.exe's semantics."""

    def _scaffold(self, root):
        s = root / "scaffold"
        (s / "src").mkdir(parents=True)
        (s / "public").mkdir()
        (s / "package.json").write_text("{}", encoding="utf-8")
        (s / "src" / "App.jsx").write_text("export default 1\n",
                                           encoding="utf-8")
        return s

    def test_the_full_hoist_now_succeeds_and_leaves_nothing_behind(
            self, tmp_path):
        self._scaffold(tmp_path)
        ex = Executor()
        ok, out = ex.run_command('move scaffold\\* . && rmdir scaffold',
                                 cwd=str(tmp_path))
        assert ok, out
        # Everything arrived...
        assert (tmp_path / "package.json").is_file()
        assert (tmp_path / "src" / "App.jsx").is_file()
        assert (tmp_path / "public").is_dir()
        # ...and no duplicate tree survives to be indexed.
        assert not (tmp_path / "scaffold").exists()

    def test_unrepaired_form_is_what_broke(self, tmp_path):
        """Pins the defect: without the subdirectory pass this fails."""
        import subprocess
        self._scaffold(tmp_path)
        p = subprocess.Popen('move scaffold\\* . && rmdir scaffold',
                             shell=True, cwd=str(tmp_path),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        p.communicate()
        assert p.returncode != 0
        # The duplicate that then gets indexed.
        assert (tmp_path / "scaffold" / "src" / "App.jsx").is_file()

    def test_works_when_there_are_no_subdirectories(self, tmp_path):
        flat = tmp_path / "flat"
        flat.mkdir()
        (flat / "a.txt").write_text("a", encoding="utf-8")
        ex = Executor()
        ok, out = ex.run_command('move flat\\* . && rmdir flat',
                                 cwd=str(tmp_path))
        assert ok, out
        assert (tmp_path / "a.txt").is_file()
        assert not (tmp_path / "flat").exists()

"""findstr reads a `/` in an argument as a switch, so a path with forward
slashes never reaches it.

Measured 2026-09-23, gpt-oss:20b-cloud. Four component steps were gated on

    findstr /c:"export default function NavBar" my-app/components/NavBar.tsx

which answers `FINDSTR: Cannot open NavBar.tsx` and exits 1 over a file
containing exactly that line. 30 loop turns, an escalation and 136k tokens,
and all four components were correct. The grep->findstr translation already
converted separators; nothing covered a findstr the planner wrote itself.
"""
import os
import subprocess

import pytest

from agentchanti.orchestrator.gate_integrity import (
    findstr_path_variant, platform_equivalent_variants)

MEASURED = ('findstr /c:"export default function NavBar" '
            'my-app/components/NavBar.tsx')

windows_only = pytest.mark.skipif(os.name != "nt", reason="findstr is Windows")


class TestTheRewrite:

    def test_the_measured_gate_gets_backslashes(self):
        assert findstr_path_variant(MEASURED) == (
            'findstr /c:"export default function NavBar" '
            'my-app\\components\\NavBar.tsx')

    def test_switches_are_untouched(self):
        out = findstr_path_variant('findstr /r /c:"a b" src/app/page.tsx')
        assert "/r" in out and "/c:" in out and "src\\app\\page.tsx" in out

    def test_each_file_in_a_chain_is_rewritten(self):
        out = findstr_path_variant(
            'findstr /c:"a" x/one.tsx && findstr /c:"b" y/two.tsx')
        assert "x\\one.tsx" in out and "y\\two.tsx" in out

    def test_only_findstr_segments_change(self):
        out = findstr_path_variant(
            'node -e "require(\'./a/b.js\')" && findstr /c:"x" c/d.tsx')
        assert "./a/b.js" in out and "c\\d.tsx" in out

    @pytest.mark.parametrize("cmd", [
        'findstr /c:"x" page.tsx',                  # no directory, nothing to do
        'findstr /c:"x" my-app\\page.tsx',          # already backslashed
        'cd my-app && npm run build',               # no findstr at all
    ])
    def test_nothing_to_rewrite_returns_none(self, cmd):
        assert findstr_path_variant(cmd) is None


@windows_only
class TestAgainstTheRealShell:

    def _project(self, tmp_path):
        comp = tmp_path / "my-app" / "components"
        comp.mkdir(parents=True)
        (comp / "NavBar.tsx").write_text(
            "export default function NavBar(): JSX.Element {\n  return null\n}\n",
            encoding="utf-8")
        return str(tmp_path)

    def _run(self, cmd, cwd):
        return subprocess.run(cmd, shell=True, cwd=cwd,
                              capture_output=True).returncode

    def test_the_original_fails_and_the_variant_passes(self, tmp_path):
        cwd = self._project(tmp_path)
        assert self._run(MEASURED, cwd) == 1
        variant = dict(platform_equivalent_variants(MEASURED))[
            "findstr-path-separator"]
        assert self._run(variant, cwd) == 0

    def test_the_variant_still_fails_on_a_wrong_file(self, tmp_path):
        """A translation that passes everything is not a translation."""
        cwd = self._project(tmp_path)
        wrong = MEASURED.replace("NavBar\"", "Sidebar\"")
        variant = findstr_path_variant(wrong)
        assert self._run(variant, cwd) == 1

"""A gate naming a program this machine does not have can never pass.

Measured 2026-09-22 on Windows. A plan gated a component step on Node
`require()`-ing a .tsx file; `_node_jsx_error` rightly refused it, and the
in-place repair replaced it with

    grep -q "export.*function Header\\|export const Header\\|export { Header }"
      my-app/components/Header.tsx && grep -q "lucide-react" ... && exit 0 || exit 1

There is no grep on the machine. 10 turns and an escalation over a correct
Header.tsx, then STALLED — and the auto-resume brought the same gate back
from the checkpoint and stalled again.
"""
import os
import subprocess

import pytest

from agentchanti.orchestrator import gate_integrity
from agentchanti.orchestrator import plan_step
from agentchanti.orchestrator.gate_integrity import (
    grep_to_findstr, missing_tool_reason, platform_equivalent_variants)
from agentchanti.orchestrator.plan_step import (
    PlanStep, repair_verify_commands, unrunnable_gate_reason)

MEASURED = ('grep -q "export.*function Header\\\\|export const Header\\\\|'
            'export { Header }" my-app/components/Header.tsx && grep -q '
            '"lucide-react" my-app/components/Header.tsx && exit 0 || exit 1')

windows_only = pytest.mark.skipif(os.name != "nt", reason="findstr is Windows")


@pytest.fixture
def no_grep(monkeypatch):
    real = gate_integrity.shutil.which
    monkeypatch.setattr(
        gate_integrity.shutil, "which",
        lambda name, *a, **k: None if name in gate_integrity._POSIX_TOOLS
        else real(name, *a, **k))


class TestDetection:

    def test_the_measured_gate_is_unrunnable_without_grep(self, no_grep):
        assert "grep" in missing_tool_reason(MEASURED)
        assert unrunnable_gate_reason(MEASURED)

    def test_silent_where_the_tool_exists(self, monkeypatch):
        monkeypatch.setattr(gate_integrity.shutil, "which",
                            lambda *a, **k: "/usr/bin/grep")
        assert missing_tool_reason(MEASURED) is None

    def test_a_tool_a_later_step_installs_is_not_judged(self, no_grep):
        """pytest in a venv that does not exist yet is not a missing tool."""
        assert missing_tool_reason("venv\\Scripts\\pytest -q") is None
        assert missing_tool_reason("npx vitest run") is None

    def test_grep_inside_a_quoted_payload_is_not_a_command(self, no_grep):
        assert missing_tool_reason(
            'node -e "console.log(\'use grep | sed\')"') is None

    def test_a_later_segment_is_found(self, no_grep):
        assert missing_tool_reason("cd my-app && cat package.json")


@windows_only
class TestTranslation:

    def _project(self, tmp_path, body):
        comp = tmp_path / "my-app" / "components"
        comp.mkdir(parents=True)
        (comp / "Header.tsx").write_text(body, encoding="utf-8")
        return str(tmp_path)

    def _run(self, cmd, cwd):
        return subprocess.run(cmd, shell=True, cwd=cwd,
                              capture_output=True).returncode

    def test_the_measured_gate_passes_over_a_correct_header(self, tmp_path,
                                                            no_grep):
        cwd = self._project(tmp_path,
                            "import { Menu } from 'lucide-react'\n"
                            "export default function Header() {}\n")
        variant = dict(platform_equivalent_variants(MEASURED))["grep-to-findstr"]
        assert unrunnable_gate_reason(variant) is None
        assert self._run(variant, cwd) == 0

    def test_and_still_fails_a_wrong_one(self, tmp_path, no_grep):
        """A translation that passes everything is not a translation."""
        cwd = self._project(tmp_path, "export default function Header() {}\n")
        variant = dict(platform_equivalent_variants(MEASURED))["grep-to-findstr"]
        assert self._run(variant, cwd) == 1

    def test_paths_use_backslashes(self):
        """findstr reads `/` in an argument as a switch."""
        out = grep_to_findstr('grep -q "x" my-app/a.tsx')
        assert "my-app\\a.tsx" in out and "my-app/a.tsx" not in out

    def test_extended_regex_is_declined_not_approximated(self):
        assert grep_to_findstr('grep -E "a+" f.txt') is None
        assert grep_to_findstr('grep -q "a\\s+b" f.txt') is None


class TestRepair:

    class _LLM:
        def __init__(self, reply):
            self.reply = reply
            self.prompt = ""

        def generate_response(self, prompt):
            self.prompt = prompt
            return self.reply

    def _step(self):
        return PlanStep(id="1.2", description="Create Header", step_type="CODE",
                        target_files=["my-app/components/Header.tsx"],
                        verify_cmd="node -e \"require('./my-app/components/Header.tsx')\"")

    @windows_only
    def test_the_prompt_names_the_shell(self, no_grep):
        llm = self._LLM("")
        repair_verify_commands([self._step()], [("1.2", "tsx")], llm)
        assert "no grep" in llm.prompt.lower()

    @windows_only
    def test_a_grep_reply_is_translated_rather_than_accepted(self, no_grep):
        step = self._step()
        llm = self._LLM("1.2: " + MEASURED)
        repaired = repair_verify_commands([step], [("1.2", "tsx")], llm)
        assert repaired == ["1.2"]
        assert "grep" not in step.verify_cmd and "findstr" in step.verify_cmd

    def test_an_untranslatable_reply_is_still_refused(self, no_grep):
        step = self._step()
        before = step.verify_cmd
        llm = self._LLM('1.2: sed -n "/x/p" my-app/components/Header.tsx')
        assert repair_verify_commands([step], [("1.2", "tsx")], llm) == []
        assert step.verify_cmd == before

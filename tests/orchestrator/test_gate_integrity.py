"""A gate can be the defect, rather than the code it judges.

Pins the real incident: a planner-declared `node -e` gate whose regex was
double-escaped. Under a POSIX shell the quoting collapses `\\\\` to `\\`
and the regex means "any character"; under cmd.exe it survives and means
"a literal backslash, s, or S", so the gate could never pass on Windows.
The CSS edit was correct on turn 1; the loop, the escalation and the
recovery loop then burned 24 turns and ~182k tokens failing against it.
"""

import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentchanti.agent_tools import AgentTools
from agentchanti.executor import Executor
from agentchanti.llm.chat_types import ChatResponse, ToolCall
from agentchanti.orchestrator.agent_loop import run_agent_loop
from agentchanti.orchestrator.gate_integrity import (
    collapse_posix_escapes,
    effective_gate,
    platform_equivalent_variants,
    record_gate_repair,
    repaired_gate,
    reset_repairs,
)

windows_only = pytest.mark.skipif(
    os.name != 'nt', reason="the dialect gap is Windows-specific")


# Verbatim from the failing run's plan (step 1.1), abridged only in the
# selector list — the escaping, which is what this guards, is untouched.
FOOTER_GATE = (
    'node -e "const s=require(\'fs\').readFileSync(\'src/App.css\',\'utf8\');'
    "if(!['.site-footer__grid','.site-footer__bottom'].every(x=>s.includes(x))"
    '||!/@media \\\\(max-width: 48rem\\\\)[\\\\s\\\\S]*\\\\.site-footer__grid'
    "[\\\\s\\\\S]*grid-template-columns:\\\\s*1fr/.test(s))process.exit(1)\""
)


@pytest.fixture(autouse=True)
def _clean_repairs():
    reset_repairs()
    yield
    reset_repairs()


class TestCollapse:
    def test_collapses_the_real_footer_gate(self):
        out, changed = collapse_posix_escapes(FOOTER_GATE)
        assert changed
        # What node would have received under a POSIX shell.
        assert '[\\s\\S]' in out
        assert '\\(max-width: 48rem\\)' in out
        assert '[\\\\s\\\\S]' not in out

    def test_leaves_single_backslashes_alone(self):
        cmd = 'node -e "if(!/[\\s\\S]/.test(s))process.exit(1)"'
        out, changed = collapse_posix_escapes(cmd)
        assert not changed
        assert out == cmd

    def test_ignores_backslashes_outside_quotes(self):
        # On Windows these are path separators, not escapes. Rewriting
        # here would corrupt a working command to fix an unrelated bug.
        cmd = r'call venv\\Scripts\\activate && python -m pytest'
        out, changed = collapse_posix_escapes(cmd)
        assert not changed
        assert out == cmd

    def test_escaped_quote_is_preserved_and_does_not_end_the_string(self):
        # `\"` means a literal quote on BOTH platforms, so there is
        # nothing to reconcile — and mistaking it for a closing quote
        # would mis-scan everything after it.
        cmd = 'python -c "print(\\"a\\"); x = 1 \\\\ 2"'
        out, _ = collapse_posix_escapes(cmd)
        assert '\\"a\\"' in out

    def test_four_backslashes_collapse_to_two(self):
        out, changed = collapse_posix_escapes('node -e "x=/\\\\\\\\d/"')
        assert changed
        assert '/\\\\d/' in out

    def test_empty_and_unquoted_commands_are_untouched(self):
        assert collapse_posix_escapes('') == ('', False)
        assert collapse_posix_escapes('npm test') == ('npm test', False)


class TestVariants:
    @windows_only
    def test_offers_a_variant_for_the_footer_gate(self):
        variants = platform_equivalent_variants(FOOTER_GATE)
        assert len(variants) == 1
        reason, cmd = variants[0]
        assert reason == "posix-backslash-collapse"
        assert '[\\s\\S]' in cmd

    @windows_only
    def test_no_variant_when_nothing_would_change(self):
        assert platform_equivalent_variants('npm test') == []
        assert platform_equivalent_variants(
            'node -e "if(!/[\\s\\S]/.test(s))process.exit(1)"') == []

    def test_no_variants_on_posix(self):
        if os.name == 'nt':
            pytest.skip("POSIX shells already collapsed it")
        assert platform_equivalent_variants(FOOTER_GATE) == []


class TestLedgerSeesTheRepair:
    """A repaired gate must reach the monotonic ledger, or the run dies anyway.

    Observed: a plan wrote `&& npm --prefix react-home run build` INSIDE
    the `node -e "..."` string, making the payload a JavaScript syntax
    error that no correct code could ever satisfy. The loop's flag-variant
    escape hatch recovered the step correctly — and then the ledger
    rechecked the ORIGINAL, saw it fail exactly as it always had, called
    it a REGRESSION, rolled the wave back and failed the run. The step
    passed; the run still lost the work.
    """

    MALFORMED = (
        'node -e "const s=require(\'fs\').readFileSync(\'a.jsx\',\'utf8\');'
        'if(!s.includes(\'x\'))process.exit(1) && npm run build"')
    WORKING = (
        'node -e "const s=require(\'fs\').readFileSync(\'a.jsx\',\'utf8\');'
        'if(!s.includes(\'x\'))process.exit(1)" && npm run build')

    def test_record_passed_gate_stores_the_repaired_form(self):
        from agentchanti.orchestrator.step_handlers import _record_passed_gate
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger

        record_gate_repair(self.MALFORMED, self.WORKING, "flag-variant")

        step = SimpleNamespace(id="2.1", _verified_gate_cmd=self.MALFORMED)
        recorded = []
        ledger = get_gate_ledger()
        with patch.object(ledger, "record",
                          side_effect=lambda c, i: recorded.append(c)):
            _record_passed_gate(True, step, MagicMock(), task="t")

        assert recorded == [self.WORKING], recorded

    def test_an_unrepaired_gate_is_recorded_verbatim(self):
        from agentchanti.orchestrator.step_handlers import _record_passed_gate
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger

        step = SimpleNamespace(id="1.1", _verified_gate_cmd="npm test")
        recorded = []
        ledger = get_gate_ledger()
        with patch.object(ledger, "record",
                          side_effect=lambda c, i: recorded.append(c)):
            _record_passed_gate(True, step, MagicMock(), task="t")

        assert recorded == ["npm test"]

    def test_a_failed_step_records_nothing(self):
        from agentchanti.orchestrator.step_handlers import _record_passed_gate
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger

        step = SimpleNamespace(id="1.1", _verified_gate_cmd="npm test")
        ledger = get_gate_ledger()
        with patch.object(ledger, "record") as rec:
            _record_passed_gate(False, step, MagicMock(), task="t")
        rec.assert_not_called()


class TestRepairRegistry:
    def test_records_and_resolves_a_repair(self):
        record_gate_repair("orig", "fixed", "posix-backslash-collapse")
        assert repaired_gate("orig") == "fixed"
        assert effective_gate("orig") == "fixed"

    def test_unrepaired_command_passes_through(self):
        assert repaired_gate("untouched") is None
        assert effective_gate("untouched") == "untouched"

    def test_ignores_a_no_op_repair(self):
        record_gate_repair("same", "same", "r")
        assert repaired_gate("same") is None


# ---------------------------------------------------------------------------
# End to end: the incident itself, driven through the real loop
# ---------------------------------------------------------------------------

_HAS_NODE = shutil.which("node") is not None

# Satisfies the gate's INTENDED regex: the media query, then the grid
# selector, then the one-column rule.
GOOD_CSS = """\
.site-footer__grid { display: grid; }
.site-footer__bottom { display: flex; }

@media (max-width: 48rem) {
  .site-footer__grid { grid-template-columns: 1fr; gap: 2rem; }
  .site-footer__bottom { flex-direction: column; }
}
"""


@pytest.mark.skipif(not _HAS_NODE, reason="needs node to run the real gate")
@windows_only
class TestLoopRecoversFromAnUnsatisfiableGate:
    """The whole point: correct code must not fail on a broken instrument.

    Drives the REAL loop against the REAL gate command with a real node
    subprocess — no stubbing of the thing under test.
    """

    def _project(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "App.css").write_text(GOOD_CSS, encoding="utf-8")
        return AgentTools(project_root=str(tmp_path), executor=Executor())

    def _llm(self):
        """Edits a file (so the gate is asked), then claims done — forever.

        Deliberately not a fixed response list: when the gate stays red the
        loop feeds the failure back and asks again, and a list that runs
        out raises StopIteration, which would look like a product failure.
        """
        state = {"calls": 0}

        def _chat(_messages, tools=None):
            state["calls"] += 1
            if state["calls"] == 1 and tools:
                return ChatResponse(
                    tool_calls=[ToolCall(name="write_file",
                                         arguments={"path": "src/notes.txt",
                                                    "content": "done\n"},
                                         id="c1")],
                    stop_reason="tool_calls")
            return ChatResponse(text="Footer styling complete.",
                                stop_reason="stop")

        llm = MagicMock()
        llm.chat.side_effect = _chat
        return llm

    def test_correct_code_passes_despite_the_broken_gate(self, tmp_path):
        tools = self._project(tmp_path)
        success, info = run_agent_loop(
            self._llm(), tools, "Update the footer styling",
            "build the site", max_turns=4, verify_cmd=FOOTER_GATE)

        assert success, info
        # And the repair is recorded, so the monotonic ledger re-checks the
        # form that can actually pass rather than the one that cannot.
        assert repaired_gate(FOOTER_GATE) is not None
        assert '[\\s\\S]' in effective_gate(FOOTER_GATE)

    def test_without_the_variant_check_the_same_run_fails(self, tmp_path):
        """Pins the pre-fix behaviour — otherwise the test above proves nothing."""
        tools = self._project(tmp_path)
        with patch("agentchanti.orchestrator.agent_loop."
                   "platform_equivalent_variants", return_value=[]):
            success, _info = run_agent_loop(
                self._llm(), tools, "Update the footer styling",
                "build the site", max_turns=4, verify_cmd=FOOTER_GATE)
        assert not success

    def test_a_genuinely_failing_gate_still_fails(self, tmp_path):
        """The escape hatch must not become a way around real failures.

        Same mis-escaped gate, but CSS that does NOT satisfy its intended
        meaning either — no reading of the command can pass, so the step
        must stay red.
        """
        tools = self._project(tmp_path)
        (tmp_path / "src" / "App.css").write_text(
            ".unrelated { color: red; }\n", encoding="utf-8")
        success, _info = run_agent_loop(
            self._llm(), tools, "Update the footer styling",
            "build the site", max_turns=4, verify_cmd=FOOTER_GATE)
        assert not success
        assert repaired_gate(FOOTER_GATE) is None

"""Regression tests for the post-step status footer in CLIDisplay.

Before the fix, ``show_status`` only rendered when ``_build_panels`` was in
the pre-step branch (``has_steps == False``). Once any step was loaded, the
status_message was set but never displayed — long post-pipeline phases like
wiring verification appeared frozen for 60-90 seconds.

These tests pin the post-step rendering so the freeze can never regress.

See: bugfix branch — cli_display.py::_build_panels post-step STATUS panel.
"""
import unittest
from unittest.mock import patch

# Force headless mode so the Live screen never actually takes over the
# terminal during the test run.
with patch("agentchanti.cli_display._RICH_AVAILABLE", False):
    from agentchanti.cli_display import CLIDisplay


def _make_headless_display() -> CLIDisplay:
    """Build a CLIDisplay that does not own the terminal."""
    with patch("agentchanti.cli_display._RICH_AVAILABLE", False):
        d = CLIDisplay("test task")
    # Belt-and-braces: ensure no Live handle exists.
    d._live = None
    return d


class TestPostStepStatusFooter(unittest.TestCase):
    """show_status must work in BOTH pre-step and post-step phases."""

    def test_show_status_sets_message_pre_step(self):
        """Pre-step (no steps loaded): status message is set."""
        d = _make_headless_display()
        d.show_status("Planning...")
        self.assertEqual(d.status_message, "Planning...")

    def test_show_status_works_after_steps_loaded(self):
        """Post-step (steps loaded): status message is still set.

        This is the core regression: previously the message was set but
        no panel rendered it because the if/elif chain in _build_panels
        only reached has_status when has_steps was False.
        """
        d = _make_headless_display()
        d.set_steps(["Step A", "Step B"])
        d.complete_step(0, "done")
        d.complete_step(1, "done")

        d.show_status("Verifying cross-file wiring...")
        self.assertEqual(d.status_message, "Verifying cross-file wiring...")

    def test_build_panels_includes_status_panel_post_step(self):
        """The post-step STATUS panel must appear in _build_panels output."""
        d = _make_headless_display()
        d.set_steps(["Step A"])
        d.complete_step(0, "done")

        baseline_panels = d._build_panels()
        d.show_status("Verifying cross-file wiring...")
        with_status_panels = d._build_panels()

        self.assertEqual(
            len(with_status_panels), len(baseline_panels) + 1,
            "show_status post-step must add exactly one panel "
            "(the STATUS footer)",
        )

    def test_clear_status_via_empty_string(self):
        """show_status('') must clear the message and remove the panel."""
        d = _make_headless_display()
        d.set_steps(["Step A"])
        d.complete_step(0, "done")
        d.show_status("Verifying cross-file wiring...")

        baseline_with_status = len(d._build_panels())
        d.show_status("")
        cleared_panels = len(d._build_panels())

        self.assertEqual(d.status_message, "")
        self.assertEqual(
            cleared_panels, baseline_with_status - 1,
            "Clearing status must remove the STATUS footer panel",
        )

    def test_planning_section_uses_custom_title(self):
        """Post-step status panel must use STATUS title, not PLANNING."""
        d = _make_headless_display()
        d.show_status("Verifying cross-file wiring...")
        # _build_planning_section is the renderer; pass title_text="STATUS"
        # to confirm it accepts the override (the call in _build_panels
        # uses the same kwarg).
        panel = d._build_planning_section(title_text="STATUS")
        # Rich Panel.title is a Text object — render to plain string.
        title_str = str(panel.title)
        self.assertIn("STATUS", title_str)
        self.assertNotIn("PLANNING", title_str)


class TestExecutionClearsPlannerStatus(unittest.TestCase):
    """Both entry points must clear the planning status before executing waves.

    Regression: the planner sets ``show_status('Requesting steps from
    planner...')`` and nothing inside ``_execute_step`` ever touches the
    status, so the message used to stay pinned to the STATUS panel for the
    entire pipeline run — even after every step finished and tests passed.
    The fix clears the status right before the wave execution loop in both
    ``orchestrator/cli.py`` and ``api.py``. These tests pin that contract by
    asserting the source files contain a ``show_status("")`` call adjacent
    to the wave loop.
    """

    def _read(self, relpath: str) -> str:
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        return (root / relpath).read_text(encoding="utf-8")

    def test_cli_clears_status_before_wave_loop(self):
        src = self._read("agentchanti/orchestrator/cli.py")
        # Find the wave loop and walk backwards a few lines.
        marker = "for wave_idx, wave in enumerate(waves):"
        idx = src.find(marker)
        self.assertGreater(idx, 0, "wave loop not found in cli.py")
        preface = src[max(0, idx - 600):idx]
        self.assertIn(
            'display.show_status("")', preface,
            "cli.py must clear status before wave execution starts "
            "(see TestExecutionClearsPlannerStatus docstring)",
        )

    def test_api_clears_status_before_wave_loop(self):
        src = self._read("agentchanti/api.py")
        marker = "for wave_idx, wave in enumerate(waves):"
        idx = src.find(marker)
        self.assertGreater(idx, 0, "wave loop not found in api.py")
        preface = src[max(0, idx - 600):idx]
        self.assertIn(
            'display.show_status("")', preface,
            "api.py must clear status before wave execution starts "
            "(see TestExecutionClearsPlannerStatus docstring)",
        )


class TestSetStatusHelper(unittest.TestCase):
    """``set_status`` must never be the thing that fails a run."""

    def test_sets_message_on_a_real_display(self):
        from agentchanti.cli_display import set_status
        d = _make_headless_display()
        set_status(d, "Running the full test suite...")
        self.assertEqual(d.status_message, "Running the full test suite...")

    def test_tolerates_none_and_statusless_displays(self):
        from agentchanti.cli_display import set_status
        set_status(None, "x")
        set_status(object(), "x")          # no show_status attribute

    def test_tolerates_a_raising_display(self):
        from agentchanti.cli_display import set_status

        class Broken:
            def show_status(self, _m):
                raise RuntimeError("renderer died")

        set_status(Broken(), "x")


class TestStatusOnlyProxy(unittest.TestCase):
    """Post-wave loops must not write into a finished step's row.

    Regression: the smoke-test repair loops are not plan steps, but they
    call the agent loop with ``step_idx=0``, so their per-turn progress
    overwrote the row of step 1 — which had finished minutes earlier.
    Observed: step 1 ("Create the virtual environment", long green)
    reading "Agent loop 3/8: edit_file".
    """

    def test_step_info_becomes_a_status_message(self):
        from agentchanti.cli_display import status_only
        d = _make_headless_display()
        d.set_steps(["step one", "step two"])
        proxy = status_only(d, "Smoke test")
        proxy.step_info(0, "Agent loop 3/8: edit_file")
        self.assertEqual(d.status_message,
                         "Smoke test: Agent loop 3/8: edit_file")
        self.assertNotIn("Agent loop", d.steps[0].get("info", "") or "")

    def test_other_attributes_pass_through(self):
        from agentchanti.cli_display import status_only
        d = _make_headless_display()
        proxy = status_only(d, "Smoke test")
        self.assertIs(proxy.show_status.__self__, d)

    def test_none_display_stays_none(self):
        from agentchanti.cli_display import status_only
        self.assertIsNone(status_only(None, "Smoke test"))


class TestPostWaveStagesReportProgress(unittest.TestCase):
    """Every stage after the wave loop must drive the STATUS footer.

    Regression: the footer plumbing existed (see TestPostStepStatusFooter)
    but nothing called it once the waves ended. Bulk test, smoke test, the
    gate rechecks and learning extraction ran for a minute or more while
    the UI showed a finished step list and only the clock and token
    counters moved — indistinguishable from a hang.
    """

    def _read(self, relpath: str) -> str:
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        return (root / relpath).read_text(encoding="utf-8")

    def test_cli_sets_status_for_each_post_wave_stage(self):
        src = self._read("agentchanti/orchestrator/cli.py")
        tail = src[src.find("# ── 13.5. Bulk test execution"):]
        self.assertGreater(len(tail), 0, "post-wave section not found")
        for fragment in ("Running the full test suite",
                         "Launching the app to check it starts",
                         "Extracting learnings from this run"):
            self.assertIn(fragment, tail,
                          f"no status message for stage: {fragment}")

    def test_gate_recheck_reports_and_clears(self):
        from unittest.mock import MagicMock

        from agentchanti.orchestrator.cli import _enforce_monotonic_gates
        from agentchanti.orchestrator.wave_snapshots import get_gate_ledger

        get_gate_ledger().reset()
        get_gate_ledger().record("python -m pytest -q", "1.1")
        seen = []
        display = MagicMock()
        display.show_status.side_effect = seen.append
        snapshots = MagicMock()
        snapshots.managed = True
        executor = MagicMock()
        executor.run_command.return_value = (True, "")

        self.assertTrue(_enforce_monotonic_gates(
            snapshots, executor, "bulk-test fixes", display=display))
        self.assertTrue(any("Re-checking" in m for m in seen),
                        f"no recheck progress message: {seen}")
        self.assertEqual(seen[-1], "",
                         "footer must be cleared once gates are green")
        get_gate_ledger().reset()


if __name__ == "__main__":
    unittest.main()

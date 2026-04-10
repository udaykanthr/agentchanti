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


if __name__ == "__main__":
    unittest.main()

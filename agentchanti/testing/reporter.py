"""
Reporter — turns Validator results into a pass/fail summary for humans and CI.

Produces both a console-friendly summary (total passed / failed, per-step
diagnostics with LLM-written reasoning on failures) and a machine-readable
JSON artifact suitable for CI gating.
"""

from __future__ import annotations

from pathlib import Path


class Reporter:
    """Render Validator assertion results as a pass/fail report."""

    def render_console(self, assertion_results: list[dict]) -> str:
        """Return a human-readable summary string."""
        raise NotImplementedError("Reporter.render_console is not implemented yet")

    def render_json(self, assertion_results: list[dict], output_path: str | Path) -> Path:
        """Write a machine-readable JSON report for CI. Return the written path."""
        raise NotImplementedError("Reporter.render_json is not implemented yet")

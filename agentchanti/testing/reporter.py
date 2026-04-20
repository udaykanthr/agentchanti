"""
Reporter — renders Validator output as console + JSON reports.

Two surfaces:
  * ``render_console`` — compact summary for terminals, with per-assertion
    detail on failures so the cause of a red run is visible without
    re-running in verbose mode.
  * ``render_json``    — machine-readable artifact for CI. Stable key
    names so dashboards + merge-blocking can parse it directly.

Report exit semantics (for the caller, not enforced here):
  * any ``passed=False`` and ``skipped=False``          → test failed, exit 1
  * all ``passed=True`` (skipped entries allowed)        → test passed, exit 0
  * any unhandled exception during replay/validate       → exit 2
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .validator import AssertionResult


class Reporter:
    """Render a list of AssertionResult as console text or JSON."""

    def render_console(self, results: list[AssertionResult]) -> str:
        if not results:
            return "no assertions evaluated (empty spec?)"
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)
        lines: list[str] = []
        lines.append(f"{passed} passed, {failed} failed, {skipped} skipped "
                     f"({len(results)} total)")
        lines.append("")
        for r in results:
            mark = _mark(r)
            lines.append(f"  {mark} [{r.kind}] {r.id}")
            # Always show detail for non-pass; show for pass only when
            # detail carries non-trivial info (rare, e.g. glob matches).
            if not r.passed or (r.detail and r.detail not in ("ok", "")):
                lines.append(f"      {r.detail}")
        lines.append("")
        return "\n".join(lines)

    def render_json(
        self,
        results: list[AssertionResult],
        output_path: str | Path,
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "total":   len(results),
                "passed":  sum(1 for r in results if r.passed),
                "failed":  sum(1 for r in results if not r.passed and not r.skipped),
                "skipped": sum(1 for r in results if r.skipped),
            },
            "assertions": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        return out


def _mark(r: AssertionResult) -> str:
    if r.skipped:
        return "SKIP"
    return "PASS" if r.passed else "FAIL"

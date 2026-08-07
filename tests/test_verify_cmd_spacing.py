"""Tests that sanitising a verify gate does not corrupt `set VAR=value`.

`_declared_verify_cmd` split a gate on `&&`, stripped each segment, and
rejoined with `" && "` unconditionally — rewriting whitespace even when it
dropped nothing. On cmd.exe that is not cosmetic: `set VAR=dummy && next`
assigns `"dummy "`, trailing space included.

Observed 2026-08-07, classic iteration 3. The planner wrote

    set SDL_VIDEODRIVER=dummy&& set SDL_AUDIODRIVER=dummy&& python -c "..."

deliberately without spaces. The pipeline ran it with spaces, SDL looked for
a display driver literally named "dummy ", the gate exited 1, and the
diagnosis round "fixed" it by adding a `_normalize_sdl_driver_environment()`
function to the GENERATED project's main.py — harness damage shipped in the
delivered artifact.
"""

import unittest

from agentchanti.orchestrator.step_handlers import _strip_space_before_amp


GATE = ('set SDL_VIDEODRIVER=dummy&& set SDL_AUDIODRIVER=dummy&& '
        'python -c "import main"')


class StripSpaceBeforeAmpTest(unittest.TestCase):
    # ── the regression ────────────────────────────────────────────────
    def test_space_before_amp_is_closed_after_set(self):
        spaced = GATE.replace("&& ", " && ")
        self.assertEqual(_strip_space_before_amp(spaced), GATE)

    def test_every_set_in_a_chain_is_repaired(self):
        out = _strip_space_before_amp("set A=1 && set B=2 && set C=3 && go")
        self.assertNotIn(" &&", out)
        self.assertIn("set C=3&& go", out)

    def test_export_is_repaired_too(self):
        self.assertEqual(_strip_space_before_amp("export A=1 && echo hi"),
                         "export A=1&& echo hi")

    def test_or_separator_is_repaired(self):
        self.assertEqual(_strip_space_before_amp("set A=1 || echo no"),
                         "set A=1|| echo no")

    # ── everything else keeps its spacing ─────────────────────────────
    def test_non_assignment_segments_are_untouched(self):
        for cmd in ("cd app && npm test",
                    "python -m pytest && echo ok",
                    "python -m unittest -v"):
            self.assertEqual(_strip_space_before_amp(cmd), cmd)

    def test_already_correct_input_is_unchanged(self):
        self.assertEqual(_strip_space_before_amp(GATE), GATE)

    def test_idempotent(self):
        once = _strip_space_before_amp(GATE.replace("&& ", " && "))
        self.assertEqual(_strip_space_before_amp(once), once)


class DeclaredVerifyCmdTest(unittest.TestCase):
    """End to end through the gate sanitiser itself."""

    class _Step:
        def __init__(self, verify_cmd):
            self.verify_cmd = verify_cmd
            self.description = "run the game headless"

    def _sanitise(self, cmd):
        from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
        from agentchanti.orchestrator.memory import FileMemory
        return _declared_verify_cmd(self._Step(cmd), FileMemory(), task="t")

    def test_set_chain_survives_sanitising(self):
        out = self._sanitise(GATE)
        self.assertIsNotNone(out, "gate was rejected outright")
        self.assertNotIn("dummy &&", out,
                         "a space before && re-entered the set assignment")

    def test_plain_gate_is_preserved(self):
        out = self._sanitise("python -m unittest -v")
        self.assertEqual(out, "python -m unittest -v")


if __name__ == "__main__":
    unittest.main()

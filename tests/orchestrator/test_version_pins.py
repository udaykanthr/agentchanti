"""An exact pin nobody asked for, and a stdlib name nobody should install.

Measured 2026-09-13, "create a 2 snake game in python include production
ready features". The IntentAgent's REQUIREMENTS_SPEC listed
``Packages/versions: - `pygame==2.6.0``` — the user never wrote a version.
The planner and the seeded contract both inherited it; pygame 2.6.0 has no
wheel for the venv's Python 3.13, so the install tried a source build and
died on ``No module named 'distutils.msvccompiler'``.

Then two things went wrong at once. The env self-heal ran
``pip install distutils`` — a standard-library name, fetched from PyPI. And
the install step spent 16 turns and an escalation (~123k prompt tokens,
over half the run) searching winget for Visual Studio Build Tools before
recreating the project venv on Python 3.11 to make the pin installable.
pygame 2.6.1 had the wheel all along.
"""
import unittest

from agentchanti.agents.planner import PlannerAgent
from agentchanti.orchestrator.pipeline import (
    _is_stdlib_module, _missing_third_party_module,
)
from agentchanti.orchestrator.version_pins import relax_unrequested_pins

RAW_TASK = "create a 2 snake game in python include production ready features"

# The measured spec line and the measured plan gate.
SPEC = "Packages/versions:\n- `pygame==2.6.0`\nKB topics: Pygame"
GATE = ("verify: python -c \"from pathlib import Path; assert Path("
        "'requirements.txt').read_text(encoding='utf-8').strip() == "
        "'pygame==2.6.0'\"")


class TheMeasuredPin(unittest.TestCase):

    def test_the_spec_pin_is_relaxed(self):
        new, changes = relax_unrequested_pins(SPEC, RAW_TASK)
        self.assertIn("pygame~=2.6", new)
        self.assertNotIn("==2.6.0", new)
        self.assertEqual(changes, [("pygame==2.6.0", "pygame~=2.6")])

    def test_the_relaxed_pin_admits_the_wheel_that_existed(self):
        try:
            from packaging.specifiers import SpecifierSet
        except ImportError:
            self.skipTest("packaging not installed")
        spec = SpecifierSet("~=2.6")
        self.assertIn("2.6.1", spec, "the version with a 3.13 wheel")
        self.assertIn("2.6.0", spec, "the version the model meant")
        self.assertNotIn("3.0.0", spec, "a new major is not admitted")

    def test_a_gate_and_its_manifest_stay_consistent(self):
        """Every occurrence is rewritten the same way, or the gate that
        asserts the manifest's exact text fails a correct manifest."""
        plan = GATE + "\n\n```requirements.txt\npygame==2.6.0\n```"
        new, _ = relax_unrequested_pins(plan, RAW_TASK)
        self.assertEqual(new.count("pygame~=2.6"), 2)
        self.assertNotIn("pygame==2.6.0", new)

    def test_the_replacement_is_safe_unquoted_in_a_shell(self):
        """`pip install pygame>=2.6,<3` unquoted is a redirect."""
        new, _ = relax_unrequested_pins("> pip install pygame==2.6.0", RAW_TASK)
        self.assertEqual(new, "> pip install pygame~=2.6")
        self.assertNotIn("<", new.split("install", 1)[1])


class WhatItMustNotTouch(unittest.TestCase):

    def test_python_comparisons_are_not_pins(self):
        for code in (
            "assert g.speed==1.5",
            "assert speed==1.5",
            "assert x == 1.2",
            "assert platform.python_version()=='3.13.0'",
            "assert sys.version_info >= (3, 10)",
            "if score==100: win()",
        ):
            with self.subTest(code=code):
                self.assertEqual(relax_unrequested_pins(code, RAW_TASK)[0],
                                 code)

    def test_pins_the_user_asked_for_are_kept(self):
        for task in (
            "build a snake game with pygame==2.6.0",
            "build a snake game using pygame 2.6.0",
            "build a snake game; pin all dependency versions",
            "use exact versions for every dependency",
        ):
            with self.subTest(task=task):
                self.assertEqual(relax_unrequested_pins(SPEC, task)[0], SPEC)

    def test_another_packages_version_in_the_task_does_not_keep_this_pin(self):
        task = "use numpy 2.6.0 for the maths"
        self.assertIn("pygame~=2.6", relax_unrequested_pins(SPEC, task)[0])

    def test_pre_releases_are_deliberate(self):
        text = "django==5.1.0rc1"
        self.assertEqual(relax_unrequested_pins(text, RAW_TASK)[0], text)


class Shapes(unittest.TestCase):

    def test_zero_major_keeps_the_minor(self):
        """Below 1.0 a minor bump is a breaking change."""
        self.assertEqual(relax_unrequested_pins("foo==0.4.2", "")[0],
                         "foo~=0.4.2")

    def test_extras_survive(self):
        self.assertEqual(
            relax_unrequested_pins("uvicorn[standard]==1.30.1", "")[0],
            "uvicorn[standard]~=1.30")

    def test_a_pin_ending_a_sentence(self):
        self.assertEqual(
            relax_unrequested_pins("Install requests==2.32.3.", "")[0],
            "Install requests~=2.32.")

    def test_dotted_and_dashed_names(self):
        self.assertEqual(
            relax_unrequested_pins("zope.interface==6.4.0 and "
                                   "typing-extensions==4.12.2", "")[0],
            "zope.interface~=6.4 and typing-extensions~=4.12")


class _Stub:
    max_output_tokens = 16384

    def __init__(self, plan):
        self.plan = plan

    def generate_response(self, prompt):
        return self.plan


class PlannerWiring(unittest.TestCase):
    """Through the planner's own methods — the seam is the wiring."""

    def test_a_plan_the_planner_returns_is_relaxed(self):
        planner = PlannerAgent("Planner", "Planner", "plan", _Stub(GATE))
        plan = planner.process(RAW_TASK, language="python")
        self.assertIn("'pygame~=2.6'", plan)

    def test_the_users_raw_words_decide_not_the_enriched_task(self):
        """process() receives the ENRICHED task, which contains the pin —
        judging against it would keep every pin the spec invented."""
        planner = PlannerAgent("Planner", "Planner", "plan", _Stub(GATE))
        planner._user_task = RAW_TASK
        plan = planner.process(RAW_TASK + "\n" + SPEC, language="python")
        self.assertIn("'pygame~=2.6'", plan)

    def test_a_plan_honours_a_pin_the_user_requested(self):
        planner = PlannerAgent("Planner", "Planner", "plan", _Stub(GATE))
        planner._user_task = "snake game with pygame==2.6.0"
        plan = planner.process("snake game", language="python")
        self.assertIn("'pygame==2.6.0'", plan)

    def test_the_enriched_spec_is_relaxed_before_anyone_reads_it(self):
        class _Intent:
            def analyze_intent(self, task, **kw):
                return task + "\n\nREQUIREMENTS_SPEC:\n" + SPEC

        planner = PlannerAgent("Planner", "Planner", "plan",
                               _Stub("1. step"))
        try:
            planner.pre_analyze(RAW_TASK, intent_agent=_Intent(),
                                language="python")
        except Exception:
            pass    # later pre-analysis stages are out of scope here
        enriched = getattr(planner, "_enriched_task", None)
        self.assertIsNotNone(enriched, "pre_analyze never reached the spec")
        self.assertIn("pygame~=2.6", enriched)
        self.assertNotIn("pygame==2.6.0", enriched)


class StdlibIsNeverInstalled(unittest.TestCase):
    FILES = ["snake_game.py", "tests/test_game.py"]

    def test_the_measured_output(self):
        out = ("ModuleNotFoundError: No module named "
               "'distutils.msvccompiler'")
        self.assertIsNone(_missing_third_party_module(out, self.FILES))

    def test_removed_modules_count_even_on_an_interpreter_without_them(self):
        """3.12+ `sys.stdlib_module_names` no longer lists distutils."""
        for mod in ("distutils", "imp", "asyncore", "imghdr", "telnetlib",
                    "lib2to3"):
            with self.subTest(mod=mod):
                self.assertTrue(_is_stdlib_module(mod))

    def test_present_stdlib_modules(self):
        for mod in ("tkinter", "sqlite3", "json"):
            with self.subTest(mod=mod):
                self.assertIsNone(_missing_third_party_module(
                    f"No module named '{mod}'", self.FILES))

    def test_third_party_names_still_heal(self):
        """The guard must not cost the healer its whole purpose."""
        for mod in ("pygame", "yaml", "requests"):
            with self.subTest(mod=mod):
                self.assertEqual(_missing_third_party_module(
                    f"No module named '{mod}'", self.FILES), mod)


if __name__ == "__main__":
    unittest.main()

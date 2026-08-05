"""The plan's own gate names the runner; the language default is a guess.

Observed live (classic mode, Pac-Man task, 2026-08-05). The brief's
acceptance criterion was `python -m unittest -v`, and the plan gated its
TEST step on `python -m unittest test_pacman... -v`. The handler ignored
both and took the Python default:

  17:06:54  python -c "import pytest"        -> exit 1
  17:06:54  Auto-installing: pip install pytest
  17:06:57  Performing pre-execution baseline test analysis via python -m pytest
  17:06:57  Baseline test run success=False   (exit 5, no tests collected)

so the run installed a runner the project does not use, took its baseline
through it, and then briefed the tester with pytest conventions — bare
`def test_x()` functions and `pytest.raises`, none of which
`python -m unittest` collects. unittest ships with CPython, so preferring
it costs nothing and cannot fail to be present.

Also pinned here: the pip self-upgrade that broke the same run's first
step (`pip install --upgrade pip` cannot replace the running pip.exe on
Windows, and `&&` carried its exit 1 to the rest of the chain).
"""

from __future__ import annotations

import pytest

from agentchanti.orchestrator.plan_step import PlanStep
from agentchanti.orchestrator.step_handlers import (
    _fix_pip_self_upgrade,
    _plan_declared_test_runner,
)


# ── runner comes from the plan's gate ────────────────────────────────

def test_unittest_gate_selects_unittest():
    step = PlanStep(
        id="3.1", step_type="TEST",
        verify_cmd="python -m unittest test_pacman.TestWall.test_dt -v")
    assert _plan_declared_test_runner(step, "python") == \
        "python -m unittest discover -v"


def test_pytest_gate_leaves_the_default_alone():
    step = PlanStep(id="3.1", step_type="TEST",
                    verify_cmd="python -m pytest tests/ -q")
    assert _plan_declared_test_runner(step, "python") is None


def test_no_gate_leaves_the_default_alone():
    assert _plan_declared_test_runner(
        PlanStep(id="3.1", step_type="TEST"), "python") is None
    assert _plan_declared_test_runner(None, "python") is None


def test_non_python_languages_are_untouched():
    step = PlanStep(id="3.1", step_type="TEST",
                    verify_cmd="python -m unittest discover")
    assert _plan_declared_test_runner(step, "javascript") is None


def test_tester_briefed_for_unittest_forbids_pytest_conventions():
    from agentchanti.agents.tester import TesterAgent

    rules = TesterAgent._python_test_rules(unittest_runner=True)
    assert "unittest.TestCase" in rules
    assert "assertRaises" in rules
    assert "Do NOT import pytest" in rules

    default = TesterAgent._python_test_rules()
    assert "pytest.raises" in default


# ── pip cannot upgrade itself, and `&&` spreads the damage ───────────

def test_self_upgrade_segment_is_routed_through_python_m_pip():
    cmd = (r"python -m venv venv && call venv\Scripts\activate.bat "
           r"&& pip install --upgrade pip "
           r"&& call venv\Scripts\activate.bat && pip install pygame")
    out = _fix_pip_self_upgrade(cmd)
    assert out is not None
    assert "python -m pip install --upgrade pip" in out
    # The install the step actually exists for survives untouched, still
    # preceded by the activation that makes `pip` mean the venv's pip.
    assert out.endswith(r"call venv\Scripts\activate.bat && pip install pygame")


def test_packages_riding_along_with_pip_are_preserved():
    # The exact shape that broke the loop run: dropping this segment would
    # have lost the setuptools and wheel upgrades too.
    cmd = "pip install --upgrade pip setuptools wheel && pip install pygame"
    out = _fix_pip_self_upgrade(cmd)
    assert out == ("python -m pip install --upgrade pip setuptools wheel "
                   "&& pip install pygame")


def test_a_bare_self_upgrade_is_rewritten_not_removed():
    assert _fix_pip_self_upgrade("pip install --upgrade pip") == \
        "python -m pip install --upgrade pip"
    assert _fix_pip_self_upgrade("pip install -U pip") == \
        "python -m pip install -U pip"


def test_python_m_pip_is_already_correct():
    assert _fix_pip_self_upgrade("python -m pip install -U pip") is None
    assert _fix_pip_self_upgrade(
        "python -m pip install --upgrade pip setuptools") is None


def test_an_ordinary_install_is_left_alone():
    assert _fix_pip_self_upgrade("pip install pygame") is None
    assert _fix_pip_self_upgrade("pip install -r requirements.txt") is None
    assert _fix_pip_self_upgrade("pip install --upgrade pygame") is None


# ── scoping a unittest run to specific files ─────────────────────────

def test_unittest_scoping_uses_dotted_modules_not_paths():
    """`unittest discover <path>` reads the path as a START DIRECTORY.

    Regression from the runner-choice fix above. Once the plan's unittest
    gate was honoured, the per-file scoping still appended file paths as
    though it were pytest, producing

        python -m unittest discover -v tests/test_game_states.py

    which fails with "Start directory is not importable" — so every TEST
    step reported 0/2 files passed while nothing was wrong with the tests.
    Scoping to named modules means discovery is not wanted: drop it.
    """
    from agentchanti.orchestrator.step_handlers import _build_scoped_test_cmd

    files = {"tests/test_game_states.py": "", "tests/test_invariants.py": ""}
    cmd = _build_scoped_test_cmd("python -m unittest discover -v", files)
    assert cmd == ("python -m unittest -v "
                   "tests.test_game_states tests.test_invariants")
    assert "discover" not in cmd
    assert ".py" not in cmd


def test_unittest_scoping_drops_discover_flags_too():
    from agentchanti.orchestrator.step_handlers import _build_scoped_test_cmd

    cmd = _build_scoped_test_cmd(
        'python -m unittest discover -s . -p "test_*.py" -v',
        {"tests/test_x.py": ""})
    assert cmd == "python -m unittest -v tests.test_x"


def test_unittest_scoping_keeps_a_venv_interpreter():
    from agentchanti.orchestrator.step_handlers import _build_scoped_test_cmd

    cmd = _build_scoped_test_cmd(
        r"venv/Scripts/python.exe -m unittest discover", {"tests/test_x.py": ""})
    assert cmd == "venv/Scripts/python.exe -m unittest tests.test_x"


def test_pytest_scoping_still_uses_file_paths():
    from agentchanti.orchestrator.step_handlers import _build_scoped_test_cmd

    cmd = _build_scoped_test_cmd("python -m pytest", {"tests/test_x.py": ""})
    assert cmd == "python -m pytest tests/test_x.py"


# ── a compound command must never be token-rewritten ─────────────────

def test_a_compound_pip_command_is_left_intact():
    """Token-level trimming is only valid for a standalone install.

    Observed live (loop mode, sonnet-5, 2026-08-06). The plan's step 1 said

        pip install -U pygame && pip freeze > requirements.txt

    The idempotency pass read every token after `install` as a flag or a
    package, so `&&`, `pip`, `freeze` and `>` were classified as packages
    and `requirements.txt` as a flag. What ran was

        pip install -U requirements.txt && freeze

    which cannot succeed — and it became the step's gate. The agent loop
    burned its whole budget on it (it read pip's own source trying to work
    out why `requirements.txt` was not a package) while the real goal,
    installing pygame, had already succeeded. The run died at step 1 of 9
    with a correct plan.
    """
    from unittest.mock import MagicMock
    from agentchanti.orchestrator.step_handlers import _make_cmd_idempotent

    ex = MagicMock()
    ex.run_command.side_effect = lambda c, **k: (True, "")  # all installed
    cmd = "pip install -U pygame && pip freeze > requirements.txt"
    out, _ = _make_cmd_idempotent(cmd, ex)
    assert out == cmd, "a compound command must pass through unchanged"


@pytest.mark.parametrize("cmd", [
    "pip install -r requirements.txt > log.txt",
    "pip install pygame; echo done",
    "pip install pygame | tee out.txt",
    "pip install pygame || echo failed",
    "pip install pygame < input.txt",
])
def test_every_shell_operator_disables_rewriting(cmd):
    from unittest.mock import MagicMock
    from agentchanti.orchestrator.step_handlers import _make_cmd_idempotent

    ex = MagicMock()
    ex.run_command.side_effect = lambda c, **k: (True, "")
    out, _ = _make_cmd_idempotent(cmd, ex)
    assert out == cmd


def test_a_standalone_install_is_still_trimmed():
    """The optimisation itself must survive — only compounds are exempt."""
    from unittest.mock import MagicMock
    from agentchanti.orchestrator.step_handlers import _make_cmd_idempotent

    ex = MagicMock()
    # pygame present, numpy absent
    ex.run_command.side_effect = lambda c, **k: ("pygame" in c, "")
    out, why = _make_cmd_idempotent("pip install pygame numpy", ex)
    assert out == "pip install numpy"
    assert "pygame" in why

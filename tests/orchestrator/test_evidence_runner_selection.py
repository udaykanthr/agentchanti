"""A survivor must be run by a runner that can COLLECT it.

Every pre-existing test file used to be executed as `python -m unittest`
whatever framework wrote it, and unittest collects only TestCase
subclasses. So a user's pytest-style suite collected zero tests,
`empty_suite_reason` correctly reported that it proved nothing, the
verdict fell through to self-authored, and `require_independent_evidence`
failed the run.

Measured 2026-09-16, benchmark task `bugfix` on 0.8.3:

    10:22:42  python -m pytest -q          green
    ... six green pytest runs, the pipeline's own ...
    10:22:51  python -m unittest test_calc.py     collected 0 tests
    10:22:51  Pipeline failed: require_independent_evidence is set and
              nothing outside this run's own output verified it

over a correct one-line fix, a suite the run left byte-identical, which
passes 2/2 under pytest. The defect only ever reached a USER's suite --
the strongest evidence class there is -- because a seeded contract is
written unittest-style and was always collected fine.
"""
import os

from agentchanti.orchestrator.evidence import (
    _runner_command,
    run_pre_existing_tests,
)

# The benchmark's own seed file, verbatim.
MEASURED_PYTEST_STYLE = (
    "from calc import add, multiply\n\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n\n\n"
    "def test_multiply():\n"
    "    assert multiply(2, 3) == 6\n"
)

TESTCASE_STYLE = (
    "import unittest\n\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        self.assertEqual(2 + 3, 5)\n"
)

UNITTEST_EMPTY = "\nRan 0 tests in 0.000s\n\nNO TESTS RAN\n"
PYTEST_EMPTY = ("collected 0 items" + chr(10) + chr(10)
                + "no tests ran in 0.01s" + chr(10))
PYTEST_GREEN = "..                                        [100%]\n2 passed in 0.05s\n"


def _write(root, rel, text):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class _Runners:
    """Behaves like the two real runners against a pytest-style file."""

    def __init__(self, pytest_installed=True):
        self.commands = []
        self.pytest_installed = pytest_installed

    def run_command(self, cmd, timeout=None, **kw):
        self.commands.append(cmd)
        if "pytest" in cmd:
            if not self.pytest_installed:
                return False, ("py" + chr(92) + "python.exe: "
                               "No module named pytest")
            return True, PYTEST_GREEN
        return False, UNITTEST_EMPTY          # unittest collects nothing


class TestRunnerChoice:

    def test_a_pytest_style_file_gets_pytest(self, tmp_path):
        _write(str(tmp_path), "test_calc.py", MEASURED_PYTEST_STYLE)
        cmd = _runner_command(str(tmp_path), "test_calc.py")
        assert "pytest" in cmd and "test_calc.py" in cmd

    def test_a_testcase_file_still_gets_unittest(self, tmp_path):
        """unittest needs nothing installed -- the default must not move."""
        _write(str(tmp_path), "test_c.py", TESTCASE_STYLE)
        assert _runner_command(str(tmp_path), "test_c.py").startswith(
            "python -m unittest")

    def test_an_unparseable_file_keeps_unittest(self, tmp_path):
        _write(str(tmp_path), "test_c.py", "def broken( :\n")
        assert "unittest" in _runner_command(str(tmp_path), "test_c.py")

    def test_a_file_that_cannot_be_read_keeps_unittest(self, tmp_path):
        assert "unittest" in _runner_command(str(tmp_path), "gone.py")

    def test_a_mixed_file_keeps_unittest(self, tmp_path):
        """A TestCase alongside bare functions is collected by unittest."""
        _write(str(tmp_path), "test_c.py",
               TESTCASE_STYLE + "\n\ndef test_loose():\n    assert True\n")
        assert "unittest" in _runner_command(str(tmp_path), "test_c.py")


class TestTheMeasuredIncident:

    def test_the_evidence_is_no_longer_lost(self, tmp_path):
        root = str(tmp_path)
        _write(root, "test_calc.py", MEASURED_PYTEST_STYLE)
        ex = _Runners()

        ok, detail = run_pre_existing_tests(ex, root, ["test_calc.py"])

        assert ok is True, detail
        assert "passed" in detail
        assert any("pytest" in c for c in ex.commands)
        assert not any("unittest" in c for c in ex.commands), \
            "running it under unittest is what collected nothing"

    def test_before_the_fix_this_produced_no_verdict(self, tmp_path):
        """Pin the old behaviour as the thing being corrected."""
        root = str(tmp_path)
        _write(root, "test_calc.py", MEASURED_PYTEST_STYLE)

        class _UnittestOnly:
            def run_command(self, cmd, timeout=None, **kw):
                return False, UNITTEST_EMPTY

        ok, detail = run_pre_existing_tests(
            _UnittestOnly(), root, ["test_calc.py"])
        assert ok is None and "collected no tests" in detail


class TestTheInstrumentBeingMissing:

    def test_absent_pytest_is_inconclusive_not_a_failure(self, tmp_path):
        """A missing runner says nothing about the code."""
        root = str(tmp_path)
        _write(root, "test_calc.py", MEASURED_PYTEST_STYLE)

        ok, detail = run_pre_existing_tests(
            _Runners(pytest_installed=False), root, ["test_calc.py"])

        assert ok is not False, "a missing runner must never convict the code"
        assert "pytest is not installed" in detail


class TestOneObject:

    def test_the_collection_question_has_one_answer(self):
        """The layer that REPORTS a file uncollectable and the layer that
        RUNS it must consult the same function, or they can disagree about
        the same file -- the resolution `EMPTY_SUITE_RE` already uses."""
        from agentchanti.orchestrator import evidence, ghost

        src = evidence._runner_command.__doc__ or ""
        assert "needs_pytest_runner" in src
        assert ghost.needs_pytest_runner(MEASURED_PYTEST_STYLE) is True
        assert ghost.needs_pytest_runner(TESTCASE_STYLE) is False


class TestTheRunnerMissingFromTheProjectVenv:
    """Measured on the retest: the plan made a venv and never installed
    pytest, so the correctly-chosen runner could not start. `Executor`
    puts the venv's bin dir on PATH, so bare `python` IS the project's
    interpreter -- right for everything that imports the project."""

    def _tree(self, tmp_path):
        _write(str(tmp_path), "test_calc.py", MEASURED_PYTEST_STYLE)
        return str(tmp_path)

    def test_a_pass_on_the_host_interpreter_is_adopted(self, tmp_path):
        root = self._tree(tmp_path)

        class _Ex:
            def __init__(self):
                self.commands = []

            def run_command(self, cmd, timeout=None, **kw):
                self.commands.append(cmd)
                if cmd.startswith("python -m pytest"):
                    return False, "No module named pytest"
                return True, PYTEST_GREEN        # absolute-path interpreter

        ex = _Ex()
        ok, detail = run_pre_existing_tests(ex, root, ["test_calc.py"])

        assert ok is True, detail
        assert len(ex.commands) == 2
        assert ex.commands[1].startswith('"') and "-m pytest" in ex.commands[1]

    def test_a_failure_on_the_host_interpreter_is_discarded(self, tmp_path):
        """It has the project's dependencies only by accident."""
        root = self._tree(tmp_path)

        class _Ex:
            def run_command(self, cmd, timeout=None, **kw):
                if cmd.startswith("python -m pytest"):
                    return False, "No module named pytest"
                return False, "ModuleNotFoundError: No module named 'pygame'"

        ok, detail = run_pre_existing_tests(_Ex(), root, ["test_calc.py"])

        assert ok is not False, (
            "an environment difference must never convict the code")
        assert "pytest is not installed" in detail

    def test_an_empty_fallback_run_is_not_a_pass(self, tmp_path):
        root = self._tree(tmp_path)

        class _Ex:
            def run_command(self, cmd, timeout=None, **kw):
                if cmd.startswith("python -m pytest"):
                    return False, "No module named pytest"
                return True, PYTEST_EMPTY

        ok, detail = run_pre_existing_tests(_Ex(), root, ["test_calc.py"])
        assert ok is not True, "an empty collection proves nothing"

    def test_nothing_is_installed_by_the_evidence_layer(self, tmp_path):
        root = self._tree(tmp_path)

        class _Ex:
            def __init__(self):
                self.commands = []

            def run_command(self, cmd, timeout=None, **kw):
                self.commands.append(cmd)
                if cmd.startswith("python -m pytest"):
                    return False, "No module named pytest"
                return True, PYTEST_GREEN

        ex = _Ex()
        run_pre_existing_tests(ex, root, ["test_calc.py"])
        assert not any("install" in c for c in ex.commands)

"""Environment self-healing for test runs.

Observed live: a fresh venv installed only the runtime package, then every
test run failed with ``No module named pytest`` while the BulkTest fix loop
— which may only edit test files — burned three LLM rounds on a problem
whose fix was one pip install.
"""

from agentchanti.orchestrator.pipeline import (
    _ensure_pytest_available,
    _missing_third_party_module,
    _plan_declared_suite_cmd,
)


class _Step:
    def __init__(self, target_files, verify_cmd=None):
        self.target_files = target_files
        self.verify_cmd = verify_cmd


class TestPlanDeclaredSuiteCmd:
    def test_single_command_covering_all_files(self):
        steps = [_Step(["src/test_game.py"],
                       'python -m unittest discover -s src -p "test_*.py" -v')]
        cmd = _plan_declared_suite_cmd(steps, {"src/test_game.py": ""})
        assert cmd == 'python -m unittest discover -s src -p "test_*.py" -v'

    def test_normalizes_whitespace(self):
        steps = [_Step(["t/test_x.py"], "python  -m   unittest  discover")]
        assert _plan_declared_suite_cmd(
            steps, {"t/test_x.py": ""}) == "python -m unittest discover"

    def test_uncovered_file_yields_none(self):
        # A declared command that only covers one of two collected files must
        # not be used as the suite command (it would skip the other).
        steps = [_Step(["t/test_a.py"], "pytest t/test_a.py")]
        assert _plan_declared_suite_cmd(
            steps, {"t/test_a.py": "", "t/test_b.py": ""}) is None

    def test_conflicting_commands_yield_none(self):
        steps = [
            _Step(["t/test_a.py"], "cmd-one"),
            _Step(["t/test_b.py"], "cmd-two"),
        ]
        assert _plan_declared_suite_cmd(
            steps, {"t/test_a.py": "", "t/test_b.py": ""}) is None

    def test_no_declared_command_yields_none(self):
        steps = [_Step(["t/test_a.py"], None)]
        assert _plan_declared_suite_cmd(steps, {"t/test_a.py": ""}) is None
        assert _plan_declared_suite_cmd(None, {"t/test_a.py": ""}) is None

    def test_matches_across_path_separators(self):
        steps = [_Step(["src\\test_game.py"], "unittest-cmd")]
        assert _plan_declared_suite_cmd(
            steps, {"src/test_game.py": ""}) == "unittest-cmd"


class _StubExecutor:
    """Records commands; scripted result for the --version check."""

    def __init__(self, version_ok: bool):
        self.version_ok = version_ok
        self.commands: list[str] = []

    def run_command(self, cmd, cwd=None, timeout=None, **_kw):
        self.commands.append(cmd)
        if "--version" in cmd:
            return self.version_ok, "pytest 9.0.0" if self.version_ok else ""
        return True, "installed"


class TestEnsurePytest:
    def test_present_runner_not_reinstalled(self):
        ex = _StubExecutor(version_ok=True)
        _ensure_pytest_available(ex)
        assert len(ex.commands) == 1
        assert "--version" in ex.commands[0]

    def test_missing_runner_installed(self):
        ex = _StubExecutor(version_ok=False)
        _ensure_pytest_available(ex, cwd="proj")
        assert any("pip install pytest" in c for c in ex.commands)


class TestMissingThirdPartyModule:
    FILES = ["src/snake_game/game.py", "tests/test_entities.py", "utils.py"]

    def test_detects_missing_runner(self):
        out = (r"C:\proj\venv\Scripts\python.exe: No module named pytest")
        assert _missing_third_party_module(out, self.FILES) == "pytest"

    def test_detects_quoted_dotted_module(self):
        out = "ModuleNotFoundError: No module named 'yaml.loader'"
        assert _missing_third_party_module(out, self.FILES) == "yaml"

    def test_project_local_package_not_installable(self):
        # `snake_game` lives under src/ — failing to import it is a
        # sys.path problem; pip-installing the name would be wrong (and
        # a dependency-confusion risk)
        out = "ModuleNotFoundError: No module named 'snake_game'"
        assert _missing_third_party_module(out, self.FILES) is None

    def test_top_level_local_module_not_installable(self):
        out = "ModuleNotFoundError: No module named 'utils'"
        assert _missing_third_party_module(out, self.FILES) is None

    def test_bare_sibling_module_file_not_installable(self):
        # `import game` resolving to `src/game.py` is a sys.path problem;
        # pip-installing the (real, unrelated) PyPI `game` package would
        # clobber the project's own module — a dependency-confusion hazard.
        files = ["src/game.py", "src/player.py", "src/test_game.py"]
        out = "ModuleNotFoundError: No module named 'game'"
        assert _missing_third_party_module(out, files) is None

    def test_no_match_returns_none(self):
        assert _missing_third_party_module("2 failed, 1 passed", self.FILES) is None
        assert _missing_third_party_module("", self.FILES) is None


class _CrashOnceExecutor:
    """Plan gate dies with an NTSTATUS code, then succeeds on retry."""

    def __init__(self, exit_codes):
        self._codes = list(exit_codes)
        self.commands: list[str] = []
        self.last_exit_code = 0

    def run_command(self, cmd, cwd=None, timeout=None, **_kw):
        self.commands.append(cmd)
        if "--version" in cmd:
            self.last_exit_code = 0
            return True, "pytest 9.1.1"
        self.last_exit_code = self._codes.pop(0) if self._codes else 0
        return self.last_exit_code == 0, "output"


class TestPlanGateAbnormalExitRetry:
    """A crashed gate process is not evidence the suite fails.

    Observed: `python -m unittest -v` passed four times in a row, then
    access-violated (0xC0000005) on the BulkTest gate. BulkTest believed
    it, fell back to `python -m pytest` -- a different command with
    different import roots -- and that run reported two failures, starting
    a fix cascade that ended with the run reporting failure.
    """

    def _run(self, executor):
        from agentchanti.orchestrator.pipeline import (
            run_bulk_test_execution_and_fix,
        )
        from unittest.mock import MagicMock

        memory = MagicMock()
        memory.all_files.return_value = {"tests/test_x.py": "def test_a(): pass"}
        memory.as_dict.return_value = {"tests/test_x.py": "def test_a(): pass"}
        return run_bulk_test_execution_and_fix(
            memory=memory, executor=executor, coder=MagicMock(),
            display=MagicMock(), language="python", task="t",
            cfg=MagicMock(), all_plan_steps=[
                _Step(["tests/test_x.py"], "python -m unittest -v")],
        )

    def test_crashed_gate_is_retried_before_falling_back(self):
        # 0xC0000005 on the first gate run, clean pass on the retry.
        ex = _CrashOnceExecutor([3221225477, 0])
        ok, _err = self._run(ex)
        gate_runs = [c for c in ex.commands if c == "python -m unittest -v"]
        assert len(gate_runs) == 2, ex.commands
        assert ok
        # The framework runner must never have been reached.
        assert not any(c.startswith("python -m pytest")
                       and "--version" not in c for c in ex.commands)

    def test_a_repeat_crash_still_falls_back(self):
        ex = _CrashOnceExecutor([3221225477, 3221225477, 0])
        self._run(ex)
        gate_runs = [c for c in ex.commands if c == "python -m unittest -v"]
        assert len(gate_runs) == 2
        assert any(c.startswith("python -m pytest")
                   and "--version" not in c for c in ex.commands)

    def test_an_ordinary_failure_is_not_retried(self):
        # exit 1 is a real result; retrying it would just waste a suite run.
        ex = _CrashOnceExecutor([1, 0])
        self._run(ex)
        gate_runs = [c for c in ex.commands if c == "python -m unittest -v"]
        assert len(gate_runs) == 1, ex.commands

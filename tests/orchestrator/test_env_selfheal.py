"""Environment self-healing for test runs.

Observed live: a fresh venv installed only the runtime package, then every
test run failed with ``No module named pytest`` while the BulkTest fix loop
— which may only edit test files — burned three LLM rounds on a problem
whose fix was one pip install.
"""

from agentchanti.orchestrator.pipeline import (
    _ensure_pytest_available,
    _missing_third_party_module,
)


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

    def test_no_match_returns_none(self):
        assert _missing_third_party_module("2 failed, 1 passed", self.FILES) is None
        assert _missing_third_party_module("", self.FILES) is None

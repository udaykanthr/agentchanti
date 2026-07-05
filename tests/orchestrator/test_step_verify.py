"""Per-step execution verification — registry + Python import verifier.

The checks run real subprocesses against the current interpreter: a
module that imports a nonexistent package must fail to load; a clean
module must pass.
"""

from agentchanti.executor import Executor
from agentchanti.orchestrator.step_verify import (
    _python_import_target,
    _ts_verify_cmd,
    verify_step_files,
)


class TestPythonImportTarget:
    def test_flat_module(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _python_import_target("main.py") == ("main", ".")

    def test_src_layout_without_init(self, tmp_path, monkeypatch):
        # src/ has no __init__.py → src is the sys.path root
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert _python_import_target("src/pkg/mod.py") == ("pkg.mod", "src")

    def test_flat_package_with_init(self, tmp_path, monkeypatch):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert _python_import_target("pkg/mod.py") == ("pkg.mod", ".")

    def test_init_file_imports_its_package(self, tmp_path, monkeypatch):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert _python_import_target("pkg/__init__.py") == ("pkg", ".")

    def test_non_python_returns_none(self):
        assert _python_import_target("README.md") is None

    def test_invalid_identifier_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _python_import_target("my-script.py") is None


class TestVerifyStepFiles:
    def test_good_module_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ok_mod.py").write_text("X = 1\n")
        assert verify_step_files(
            {"ok_mod.py": "X = 1\n"}, "python", Executor()) == []

    def test_missing_import_caught(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bad_mod.py").write_text("import surely_not_real_xyz\n")
        errs = verify_step_files({"bad_mod.py": "..."}, "python", Executor())
        assert len(errs) == 1
        assert "bad_mod.py" in errs[0]
        assert "surely_not_real_xyz" in errs[0]

    def test_broken_package_wiring_caught(self, tmp_path, monkeypatch):
        # A package module importing a sibling that does not exist —
        # the integration failure class that previously surfaced only
        # at the end-of-pipeline smoke test.
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
        (tmp_path / "src" / "pkg" / "game.py").write_text(
            "from .entities import Snake\n")
        monkeypatch.chdir(tmp_path)
        errs = verify_step_files(
            {"src/pkg/game.py": "..."}, "python", Executor())
        assert len(errs) == 1
        assert "src/pkg/game.py" in errs[0]

    def test_unknown_language_skips_silently(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert verify_step_files(
            {"a.go": "package main"}, "go", Executor()) == []

    def test_test_files_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "test_x.py").write_text("import surely_not_real_xyz\n")
        assert verify_step_files(
            {"test_x.py": "..."}, "python", Executor()) == []


class TestTypeScriptVerifier:
    def test_requires_tsconfig(self, tmp_path):
        assert _ts_verify_cmd("src/app.ts", str(tmp_path)) is None

    def test_command_with_tsconfig(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        assert _ts_verify_cmd(
            "src/app.ts", str(tmp_path)) == "npx tsc --noEmit"

    def test_non_ts_file_skipped(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        assert _ts_verify_cmd("src/app.css", str(tmp_path)) is None

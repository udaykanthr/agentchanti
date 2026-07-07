"""Tests for verification hygiene: wiring-context sanitizer, JS build
smoke check, and language-aware package grounding."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agentchanti.orchestrator.pipeline import (
    _WIRING_MAX_FILE_CHARS,
    _missing_js_packages,
    _sanitize_wiring_context,
)
from agentchanti.orchestrator.api_grounding import (
    get_installed_package_versions,
)
from agentchanti.orchestrator.smoke_test import (
    _find_js_project_dir,
    run_smoke_verification,
)


class TestSanitizeWiringContext(unittest.TestCase):

    def test_dedupes_mixed_separators(self):
        ctx = {
            "app/src/App.jsx": "const a = 1;",
            "app\\src\\App.jsx": "const a = 1;",
        }
        clean = _sanitize_wiring_context(ctx)
        self.assertEqual(list(clean.keys()), ["app/src/App.jsx"])

    def test_drops_node_modules_and_artifacts(self):
        ctx = {
            "app/node_modules/bootstrap/dist/css/bootstrap.min.css": "x" * 999,
            "app/dist/bundle.js": "bundled",
            "app/package-lock.json": "{}",
            "app/src/theme.min.css": "minified",
            "app/src/App.jsx": "const a = 1;",
        }
        clean = _sanitize_wiring_context(ctx)
        self.assertEqual(list(clean.keys()), ["app/src/App.jsx"])

    def test_caps_oversized_files(self):
        ctx = {"big.css": "x" * (_WIRING_MAX_FILE_CHARS + 500)}
        clean = _sanitize_wiring_context(ctx)
        self.assertLess(len(clean["big.css"]),
                        _WIRING_MAX_FILE_CHARS + 100)
        self.assertIn("truncated for verification", clean["big.css"])

    def test_source_dir_named_build_files_kept(self):
        # 'build' as a *directory* is vendor output; a file named
        # build.py must survive.
        ctx = {"scripts/build.py": "print(1)", "build/out.js": "x"}
        clean = _sanitize_wiring_context(ctx)
        self.assertEqual(list(clean.keys()), ["scripts/build.py"])


class TestMissingJsPackages(unittest.TestCase):

    VITEST_OUTPUT = """
 FAIL  src/__tests__/App.test.jsx [ src/__tests__/App.test.jsx ]
Error: Cannot find package '@testing-library/react' imported from C:/x/src/__tests__/App.test.jsx
 FAIL  src/components/__tests__/Header.test.jsx
Error: Cannot find package '@testing-library/react' imported from C:/x/Header.test.jsx
Error: Cannot find module 'axios'
"""

    def test_scoped_and_plain_packages_extracted_deduped(self):
        self.assertEqual(
            _missing_js_packages(self.VITEST_OUTPUT),
            ["@testing-library/react", "axios"])

    def test_relative_imports_ignored(self):
        out = "Error: Cannot find module './Header' imported from x"
        self.assertEqual(_missing_js_packages(out), [])

    def test_no_matches(self):
        self.assertEqual(_missing_js_packages("AssertionError: 1 != 2"), [])


class TestNpmGrounding(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="npmg_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_pkg(self):
        with open(os.path.join(self.root, "package.json"), "w") as f:
            f.write("{}")

    def test_js_uses_npm_ls(self):
        self._write_pkg()
        executor = MagicMock()
        executor.run_command.return_value = (True, json.dumps({
            "dependencies": {
                "react": {"version": "19.1.0"},
                "bootstrap": {"version": "5.3.7"},
            }
        }))
        versions = get_installed_package_versions(
            cwd=self.root, executor=executor, language="javascript")
        self.assertEqual(versions,
                         {"react": "19.1.0", "bootstrap": "5.3.7"})
        cmd = executor.run_command.call_args[0][0]
        self.assertIn("npm ls", cmd)
        self.assertNotIn("pip", cmd)

    def test_js_without_package_json_skips_probe(self):
        executor = MagicMock()
        versions = get_installed_package_versions(
            cwd=self.root, executor=executor, language="typescript")
        self.assertEqual(versions, {})
        executor.run_command.assert_not_called()

    def test_npm_ls_nonzero_exit_still_parsed(self):
        self._write_pkg()
        executor = MagicMock()
        executor.run_command.return_value = (
            False, '{"dependencies": {"react": {"version": "19.1.0"}}}')
        versions = get_installed_package_versions(
            cwd=self.root, executor=executor, language="javascript")
        self.assertEqual(versions, {"react": "19.1.0"})

    def test_python_path_unchanged(self):
        executor = MagicMock()
        executor.run_command.return_value = (
            True, '[{"name": "pytest", "version": "9.1.1"}]')
        versions = get_installed_package_versions(
            executor=executor, language="python")
        self.assertEqual(versions, {"pytest": "9.1.1"})
        self.assertIn("pip list", executor.run_command.call_args[0][0])


class TestJsBuildVerification(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="jsbuild_")
        self.prev_cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.prev_cwd)
        shutil.rmtree(self.root, ignore_errors=True)

    def _seed_project(self, with_build=True):
        os.makedirs("my-app/src", exist_ok=True)
        scripts = {"build": "vite build"} if with_build else {}
        with open("my-app/package.json", "w") as f:
            json.dump({"scripts": scripts}, f)

    def _memory(self):
        memory = MagicMock()
        memory.all_files.return_value = {
            "my-app/src/App.jsx": "code", "my-app/src/main.jsx": "code"}
        return memory

    def _run(self, executor, cfg=None, coder=None):
        return run_smoke_verification(
            memory=self._memory(), executor=executor,
            coder=coder or MagicMock(), display=MagicMock(),
            task="build a homepage", language="javascript", cfg=cfg)

    def test_find_js_project_dir(self):
        self._seed_project()
        self.assertEqual(
            _find_js_project_dir({"my-app/src/App.jsx": "x"}), "my-app")
        self.assertIsNone(_find_js_project_dir({"other/x.js": "x"}))

    def test_build_pass(self):
        self._seed_project()
        executor = MagicMock()
        executor.run_command.return_value = (True, "built in 1.2s")
        ok, err = self._run(executor)
        self.assertTrue(ok)
        executor.run_command.assert_called_once_with(
            "npm run build", timeout=300, cwd="my-app")

    def test_no_build_script_skips(self):
        self._seed_project(with_build=False)
        executor = MagicMock()
        ok, _ = self._run(executor)
        self.assertTrue(ok)
        executor.run_command.assert_not_called()

    def test_build_failure_without_loop_fails(self):
        self._seed_project()
        executor = MagicMock()
        executor.run_command.return_value = (False, "Rollup error: x")
        cfg = MagicMock()
        cfg.AGENT_LOOP = False
        cfg.SMOKE_TEST_ENABLED = True
        ok, err = self._run(executor, cfg=cfg)
        self.assertFalse(ok)
        self.assertIn("Production build failed", err)

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop")
    def test_build_failure_recovered_by_loop(self, mock_rec):
        self._seed_project()
        executor = MagicMock()
        executor.run_command.return_value = (False, "Rollup error: x")
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.SMOKE_TEST_ENABLED = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        coder = MagicMock()
        coder.llm_client.supports_tools.return_value = True
        mock_rec.return_value = (True, "fixed the import")
        ok, _ = self._run(executor, cfg=cfg, coder=coder)
        self.assertTrue(ok)
        mock_rec.assert_called_once()
        self.assertEqual(mock_rec.call_args[1]["verify_cmd"],
                         "npm run build")


if __name__ == "__main__":
    unittest.main()

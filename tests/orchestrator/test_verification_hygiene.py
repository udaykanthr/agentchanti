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
    _resolve_existing_by_basename,
    _sanitize_wiring_context,
)
from agentchanti.orchestrator.api_grounding import (
    get_installed_package_versions,
)
from agentchanti.orchestrator.smoke_test import (
    _django_settings_module,
    _django_template_checks,
    _find_django_project_dir,
    _find_js_project_dir,
    _run_django_verification,
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


class TestResolveExistingByBasename(unittest.TestCase):
    """Phantom plan paths must retarget to the real file, or refuse."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rebb_")
        self.prev = os.getcwd()
        os.chdir(self.root)
        self.memory = MagicMock()
        self.memory.all_files.return_value = {}

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, rel):
        os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
        with open(rel, "w") as f:
            f.write("x = 1\n")

    def test_django_dot_layout_retargets(self):
        # Plan said project/project/urls.py; reality is project/urls.py
        self._write("project/urls.py")
        self.assertEqual(
            _resolve_existing_by_basename(r"project\project\urls.py",
                                          self.memory),
            "project/urls.py")

    def test_memory_and_disk_agree_on_single_file(self):
        self._write("app/settings.py")
        self.memory.all_files.return_value = {"app\\settings.py": "..."}
        self.assertEqual(
            _resolve_existing_by_basename("wrong/place/settings.py",
                                          self.memory),
            "app/settings.py")

    def test_memory_file_plus_scaffold_file_is_ambiguous(self):
        # Session memory knows only accounts/urls.py; the scaffold's real
        # root urls.py exists on disk. Retargeting must refuse — guessing
        # here previously wired the root URLconf edit into the app file.
        self._write("proj/accounts/urls.py")
        self._write("proj/proj/urls.py")
        self.memory.all_files.return_value = {
            "proj/accounts/urls.py": "..."}
        self.assertIsNone(
            _resolve_existing_by_basename("proj/urls.py", self.memory))

    def test_ambiguous_returns_none(self):
        self._write("a/urls.py")
        self._write("b/urls.py")
        self.assertIsNone(
            _resolve_existing_by_basename("wrong/urls.py", self.memory))

    def test_missing_returns_none(self):
        self.assertIsNone(
            _resolve_existing_by_basename("no/such/file.py", self.memory))

    def test_vendor_dirs_ignored(self):
        self._write("venv/Lib/urls.py")
        self.assertIsNone(
            _resolve_existing_by_basename("wrong/urls.py", self.memory))


class TestProbeInconclusiveExceptions(unittest.TestCase):
    """Framework modules that need app settings must not be reported missing."""

    def _run_probe(self, checks, module_src):
        import subprocess
        import sys
        from agentchanti.orchestrator.api_grounding import _PROBE_TEMPLATE
        root = tempfile.mkdtemp(prefix="probe_")
        try:
            with open(os.path.join(root, "fakefw.py"), "w",
                      encoding="utf-8") as f:
                f.write(module_src)
            with open(os.path.join(root, "probe.py"), "w",
                      encoding="utf-8") as f:
                f.write(_PROBE_TEMPLATE)
            with open(os.path.join(root, "checks.json"), "w",
                      encoding="utf-8") as f:
                json.dump(checks, f)
            proc = subprocess.run(
                [sys.executable, "probe.py", "checks.json"],
                cwd=root, capture_output=True, text=True, timeout=60)
            return proc.stdout
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_improperly_configured_is_not_missing(self):
        out = self._run_probe(
            [["fakefw", None]],
            "class ImproperlyConfigured(Exception):\n    pass\n"
            "raise ImproperlyConfigured('settings required')\n")
        self.assertIn("PROBE_DONE", out)
        self.assertNotIn("MODULE_MISSING", out)

    def test_genuinely_missing_module_still_reported(self):
        out = self._run_probe([["no_such_module_xyz", None]], "")
        self.assertIn("MODULE_MISSING::no_such_module_xyz", out)


class TestBulkTestLoopFirstFix(unittest.TestCase):
    """A grounded loop attempt runs before the per-file fix machinery."""

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "fixed the import root"))
    def test_loop_recovery_short_circuits_bulk_fixing(self, mock_rec):
        from agentchanti.orchestrator.pipeline import (
            run_bulk_test_execution_and_fix,
        )
        memory = MagicMock()
        memory._scaffolded_subproject = None
        memory.all_files.return_value = {
            "tests/test_views.py": "from app import views\n"}
        executor = MagicMock()
        executor.run_command.return_value = (False, "No module named 'app'")
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        coder = MagicMock()
        coder.llm_client.supports_tools.return_value = True
        ok, err = run_bulk_test_execution_and_fix(
            memory=memory, executor=executor, coder=coder,
            display=MagicMock(), language="python",
            task="build the app", cfg=cfg)
        self.assertTrue(ok)
        mock_rec.assert_called_once()
        self.assertIn("No module named", mock_rec.call_args[1]["error_info"])
        self.assertTrue(
            mock_rec.call_args[1]["verify_cmd"].endswith("pytest"))


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


class TestDjangoVerification(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="djv_")
        self.prev = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.root, ignore_errors=True)

    def _seed_django(self, sub="user_portal"):
        os.makedirs(f"{sub}/config", exist_ok=True)
        open(f"{sub}/manage.py", "w").write("#!/usr/bin/env python\n")
        open(f"{sub}/config/settings.py", "w").write("DEBUG = True\n")
        return sub

    def test_find_django_project_dir(self):
        sub = self._seed_django()
        files = {f"{sub}/accounts/views.py": "x"}
        self.assertEqual(_find_django_project_dir(files), sub)
        self.assertIsNone(_find_django_project_dir({"other/x.py": "x"}))

    def test_settings_module_discovery(self):
        sub = self._seed_django()
        self.assertEqual(_django_settings_module(sub), "config.settings")
        self.assertIsNone(_django_settings_module("."))

    def test_template_checks_derive_loader_names(self):
        sub = self._seed_django()
        paths = [
            f"{sub}/templates/accounts/partials/header.html",
            f"{sub}/accounts/templates/accounts/home.html",
        ]
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write("<html/>")
        files = {p: "<html/>" for p in paths}
        files[f"{sub}/static/css/home.css"] = "css"  # not a template
        checks = dict(_django_template_checks(files, sub))
        self.assertEqual(set(checks), {"accounts/partials/header.html",
                                       "accounts/home.html"})
        self.assertTrue(checks["accounts/home.html"].endswith("home.html"))

    @patch("agentchanti.orchestrator.agent_loop.run_recovery_loop",
           return_value=(True, "moved templates into app dir"))
    def test_probe_failure_triggers_recovery(self, mock_rec):
        sub = self._seed_django()
        memory = MagicMock()
        memory.all_files.return_value = {}
        executor = MagicMock()
        executor.run_command.return_value = (
            False, "SHADOWED: template 'accounts/home.html' resolves to "
                   "old path\nDJANGO_PROBE_DONE")
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        coder = MagicMock()
        coder.llm_client.supports_tools.return_value = True
        ok, _ = _run_django_verification(
            memory, executor, coder, MagicMock(), "task", "python", cfg, sub)
        self.assertTrue(ok)
        mock_rec.assert_called_once()
        self.assertIn("SHADOWED", mock_rec.call_args[1]["error_info"])
        self.assertIn(f"cd {sub} &&", mock_rec.call_args[1]["verify_cmd"])

    def test_probe_pass_returns_ok(self):
        sub = self._seed_django()
        memory = MagicMock()
        memory.all_files.return_value = {}
        executor = MagicMock()
        executor.run_command.return_value = (True, "DJANGO_PROBE_DONE")
        ok, err = _run_django_verification(
            memory, executor, MagicMock(), MagicMock(), "task", "python",
            MagicMock(), sub)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_incomplete_probe_is_nonblocking(self):
        sub = self._seed_django()
        memory = MagicMock()
        memory.all_files.return_value = {}
        executor = MagicMock()
        executor.run_command.return_value = (False, "ImportError: django")
        ok, _ = _run_django_verification(
            memory, executor, MagicMock(), MagicMock(), "task", "python",
            MagicMock(), sub)
        self.assertTrue(ok)


class TestParserThreadSafety(unittest.TestCase):
    """tree_sitter.Parser is stateful native code — one instance per thread."""

    def test_each_thread_gets_its_own_parser(self):
        import threading
        from agentchanti.kb.local.parser import _get_ts_parser

        results = {}

        def grab(tid):
            # Keep strong references: without them a finished thread's
            # parser is freed and a later allocation can reuse its id.
            p1 = _get_ts_parser("python")
            p2 = _get_ts_parser("python")
            results[tid] = (p1, p2)

        threads = [threading.Thread(target=grab, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        if any(p1 is None for p1, _ in results.values()):
            self.skipTest("tree-sitter python grammar not available")
        # Cached within a thread…
        for p1, p2 in results.values():
            self.assertIs(p1, p2)
        # …but never shared across threads.
        distinct = {id(p1) for p1, _ in results.values()}
        self.assertEqual(len(distinct), len(results))


class TestStepVerifyDjangoInconclusive(unittest.TestCase):
    """Django app modules that need app context are not load failures."""

    def _run(self, output):
        from agentchanti.orchestrator.step_verify import verify_step_files
        executor = MagicMock()
        executor.run_command.return_value = (False, output)
        return verify_step_files(
            {"accounts/views.py": "from django.shortcuts import render"},
            "python", executor)

    def test_app_registry_not_ready_is_inconclusive(self):
        self.assertEqual(
            self._run("raise AppRegistryNotReady(...)\n"
                      "django.core.exceptions.AppRegistryNotReady: Apps "
                      "aren't loaded yet."), [])

    def test_improperly_configured_is_inconclusive(self):
        self.assertEqual(
            self._run("ImproperlyConfigured: Requested setting "
                      "INSTALLED_APPS, but settings are not configured."), [])

    def test_real_import_error_still_reported(self):
        errors = self._run("ModuleNotFoundError: No module named 'flask'")
        self.assertEqual(len(errors), 1)
        self.assertIn("fails to load", errors[0])


class TestDjangoTestsPyRecognized(unittest.TestCase):

    def test_tests_py_is_a_test_file(self):
        from agentchanti.orchestrator.pipeline import _is_test_file
        self.assertTrue(_is_test_file("user_portal/accounts/tests.py"))
        self.assertTrue(_is_test_file("accounts\\tests.py"))
        self.assertFalse(_is_test_file("accounts/forms.py"))
        # 'contests.py' must not match
        self.assertFalse(_is_test_file("app/contests.py"))


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

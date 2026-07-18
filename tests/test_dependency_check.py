"""Tests for the post-step dependency validation system."""

from unittest.mock import patch

import pytest

from agentchanti.orchestrator.dependency_check import (
    FileDeps,
    IntegrationGap,
    DependencySnapshot,
    extract_file_deps,
    build_snapshot,
    find_gaps,
    build_fix_prompt,
    run_dependency_check,
    _normalize_import_path,
    _file_matches_import,
    _is_external_import,
    _is_test_file,
    _guess_parent_file,
    _extract_component_props,
    _find_signature_gaps,
    _scan_balanced_destructure,
    _split_top_level_params,
)


# ── extract_file_deps ────────────────────────────────────────────


class TestExtractFileDeps:
    def test_python_imports(self):
        content = (
            "from app.models import User\n"
            "import os\n"
            "from .utils import helper\n"
        )
        deps = extract_file_deps("app/views.py", content)
        assert "app.models" in deps.imports
        assert "os" in deps.imports
        assert ".utils" in deps.imports

    def test_python_exports(self):
        content = (
            "class UserService:\n"
            "    pass\n\n"
            "def get_users():\n"
            "    pass\n\n"
            "_private_func = lambda: None\n"
        )
        deps = extract_file_deps("app/services.py", content)
        assert "UserService" in deps.exports
        assert "get_users" in deps.exports
        # Private symbols excluded
        assert "_private_func" not in deps.exports

    def test_python_dunder_all(self):
        content = '__all__ = ["Foo", "Bar"]\n'
        deps = extract_file_deps("pkg/__init__.py", content)
        assert "Foo" in deps.exports
        assert "Bar" in deps.exports

    def test_js_esm_imports(self):
        content = (
            "import React from 'react';\n"
            "import { useState } from 'react';\n"
            "import Header from './components/Header';\n"
            "import './styles.css';\n"
        )
        deps = extract_file_deps("src/App.jsx", content)
        assert "react" in deps.imports
        assert "./components/Header" in deps.imports
        assert "./styles.css" in deps.imports

    def test_js_cjs_imports(self):
        content = (
            "const express = require('express');\n"
            "const { Router } = require('./router');\n"
        )
        deps = extract_file_deps("server.js", content)
        assert "express" in deps.imports
        assert "./router" in deps.imports

    def test_js_exports(self):
        content = (
            "export default function Header() { return <h1>Hi</h1>; }\n"
            "export const API_URL = 'http://localhost';\n"
            "export function fetchData() {}\n"
        )
        deps = extract_file_deps("src/Header.jsx", content)
        assert "Header" in deps.exports
        assert "API_URL" in deps.exports
        assert "fetchData" in deps.exports

    def test_js_has_default_export_true(self):
        content = "export default function App() { return <div/>; }"
        deps = extract_file_deps("src/App.jsx", content)
        assert deps.has_default_export is True

    def test_js_has_default_export_false(self):
        content = "function App() { return <div/>; }"
        deps = extract_file_deps("src/App.jsx", content)
        assert deps.has_default_export is False

    def test_js_module_exports_counts_as_default(self):
        content = "module.exports = App;"
        deps = extract_file_deps("src/App.js", content)
        assert deps.has_default_export is True

    def test_js_default_imports_detected(self):
        content = (
            "import React from 'react';\n"
            "import Header from './components/Header';\n"
            "import { useState } from 'react';\n"
        )
        deps = extract_file_deps("src/App.jsx", content)
        # Only local default imports tracked (not external like 'react')
        assert "./components/Header" in deps.default_imports

    def test_named_import_not_in_default_imports(self):
        content = "import { Header } from './components/Header';\n"
        deps = extract_file_deps("src/App.jsx", content)
        assert deps.default_imports == []

    def test_js_module_exports(self):
        content = "module.exports = { add, subtract };\n"
        deps = extract_file_deps("math.js", content)
        assert "add" in deps.exports
        assert "subtract" in deps.exports

    def test_ts_imports_exports(self):
        content = (
            "import { Component } from '@angular/core';\n"
            "import { UserService } from './services/user.service';\n"
            "export class AppComponent {}\n"
            "export interface AppConfig {}\n"
        )
        deps = extract_file_deps("src/app.component.ts", content)
        assert "@angular/core" in deps.imports
        assert "./services/user.service" in deps.imports
        assert "AppComponent" in deps.exports
        assert "AppConfig" in deps.exports

    def test_go_imports_exports(self):
        content = (
            'import "fmt"\n'
            'import "myapp/models"\n\n'
            "func GetUser() {}\n"
            "func privateHelper() {}\n"
            "type UserStore struct{}\n"
        )
        deps = extract_file_deps("handlers.go", content)
        assert "fmt" in deps.imports
        assert "myapp/models" in deps.imports
        assert "GetUser" in deps.exports
        assert "UserStore" in deps.exports
        # Private (lowercase) functions not exported in Go
        assert "privateHelper" not in deps.exports

    def test_unknown_extension_returns_empty(self):
        deps = extract_file_deps("README.md", "# Hello")
        assert deps.imports == []
        assert deps.exports == []

    def test_java_imports_exports(self):
        content = (
            "import java.util.List;\n"
            "import com.myapp.models.User;\n\n"
            "public class UserController {\n"
            "}\n"
        )
        deps = extract_file_deps("UserController.java", content)
        assert "java.util.List" in deps.imports
        assert "com.myapp.models.User" in deps.imports
        assert "UserController" in deps.exports

    def test_rust_imports_exports(self):
        content = (
            "use std::collections::HashMap;\n"
            "use crate::models::User;\n\n"
            "pub fn get_users() -> Vec<User> { vec![] }\n"
            "pub struct UserStore {}\n"
        )
        deps = extract_file_deps("handlers.rs", content)
        assert "std::collections::HashMap" in deps.imports
        assert "crate::models::User" in deps.imports
        assert "get_users" in deps.exports
        assert "UserStore" in deps.exports


# ── build_snapshot ────────────────────────────────────────────────


class TestBuildSnapshot:
    def test_builds_snapshot_from_memory(self):
        files = {
            "src/App.tsx": "import Header from './Header';\nexport default function App() {}",
            "src/Header.tsx": "export default function Header() {}",
        }
        snap = build_snapshot(files)
        assert "src/App.tsx" in snap.file_deps
        assert "src/Header.tsx" in snap.file_deps

    def test_skips_non_code_files(self):
        files = {
            "README.md": "# Hello",
            "package.json": '{"name": "test"}',
            "src/App.tsx": "export default function App() {}",
        }
        snap = build_snapshot(files)
        assert "README.md" not in snap.file_deps
        assert "package.json" not in snap.file_deps
        assert "src/App.tsx" in snap.file_deps

    def test_includes_test_files(self):
        # build_snapshot intentionally includes test files so find_gaps can
        # detect exports that are only consumed by tests (vs real callers).
        # find_gaps itself skips test files when reporting orphaned exports.
        files = {
            "src/App.tsx": "export default function App() {}",
            "src/App.test.tsx": "import App from './App';\ntest('renders', () => {});",
            "tests/test_views.py": "from app.views import index\ndef test_index(): pass",
        }
        snap = build_snapshot(files)
        assert "src/App.tsx" in snap.file_deps
        assert "src/App.test.tsx" in snap.file_deps
        assert "tests/test_views.py" in snap.file_deps


# ── _is_test_file ────────────────────────────────────────────────


class TestIsTestFile:
    def test_python_test_file(self):
        assert _is_test_file("test_views.py")
        assert _is_test_file("tests/test_models.py")

    def test_js_test_file(self):
        assert _is_test_file("App.test.tsx")
        assert _is_test_file("utils.spec.js")
        assert _is_test_file("Header.test.jsx")

    def test_go_test_file(self):
        assert _is_test_file("handlers_test.go")

    def test_not_test_file(self):
        assert not _is_test_file("App.tsx")
        assert not _is_test_file("views.py")
        assert not _is_test_file("main.go")


# ── _normalize_import_path ────────────────────────────────────────


class TestNormalizeImportPath:
    def test_relative_js_import(self):
        result = _normalize_import_path("./components/Header", "src/App.tsx")
        assert result == "src/components/Header"

    def test_parent_relative_js_import(self):
        result = _normalize_import_path("../utils/format", "src/components/Header.tsx")
        assert result == "src/utils/format"

    def test_absolute_import_unchanged(self):
        result = _normalize_import_path("react", "src/App.tsx")
        assert result == "react"

    def test_python_relative_import(self):
        result = _normalize_import_path(".models", "app/views.py")
        assert "models" in result


# ── _file_matches_import ─────────────────────────────────────────


class TestFileMatchesImport:
    def test_exact_match_without_extension(self):
        assert _file_matches_import("src/components/Header.tsx", "src/components/Header")

    def test_suffix_match(self):
        assert _file_matches_import("my-app/src/App.tsx", "src/App")

    def test_python_dotted_path(self):
        assert _file_matches_import("app/models.py", "app.models")

    def test_index_file_match(self):
        assert _file_matches_import("src/components/index.ts", "src/components")

    def test_no_match(self):
        assert not _file_matches_import("src/utils/format.ts", "src/components/Header")


# ── _is_external_import ──────────────────────────────────────────


class TestIsExternalImport:
    def test_react_is_external(self):
        assert _is_external_import("react", "src/App.tsx")

    def test_express_is_external(self):
        assert _is_external_import("express", "server.js")

    def test_relative_is_not_external(self):
        assert not _is_external_import("./components/Header", "src/App.tsx")
        assert not _is_external_import("../utils", "src/components/Header.tsx")

    def test_python_stdlib_is_external(self):
        assert _is_external_import("os", "app/views.py")
        assert _is_external_import("json", "app/views.py")
        assert _is_external_import("sys", "app/main.py")

    def test_python_relative_is_not_external(self):
        assert not _is_external_import(".models", "app/views.py")

    def test_python_project_module_is_not_external(self):
        assert not _is_external_import("app.models", "app/views.py")

    def test_alias_import_is_not_external(self):
        assert not _is_external_import("@/components/Header", "src/App.tsx")
        assert not _is_external_import("~/utils", "src/App.tsx")

    def test_java_stdlib_is_external(self):
        assert _is_external_import("java.util.List", "MyClass.java")
        assert _is_external_import("javax.swing.JFrame", "MyClass.java")

    def test_java_project_is_not_external(self):
        assert not _is_external_import("com.myapp.models.User", "MyController.java")

    def test_rust_stdlib_is_external(self):
        assert _is_external_import("std::collections::HashMap", "main.rs")

    def test_rust_crate_is_not_external(self):
        assert not _is_external_import("crate::models::User", "main.rs")

    def test_python_project_venv_package_is_external(self, tmp_path, monkeypatch):
        """A package installed only in the TARGET project's venv (not in
        agentchanti's own environment) must still be recognized as
        external.

        Regression: ``_is_external_import`` used to rely solely on
        ``importlib.util.find_spec`` against agentchanti's own
        interpreter, which has no visibility into a venv a pipeline step
        just created for the project being built. Packages like
        ``arcade``/``pygame`` (which agentchanti itself doesn't depend
        on) were misclassified as broken local imports, causing DepCheck
        to have the LLM generate a bogus local stub file
        (e.g. ``src/arcade.py``) that shadowed the real package.
        """
        monkeypatch.chdir(tmp_path)
        site_packages = tmp_path / "venv" / "Lib" / "site-packages"
        (site_packages / "arcade").mkdir(parents=True)
        (site_packages / "arcade" / "__init__.py").write_text("")

        assert _is_external_import("arcade", "src/game_window.py")

    def test_python_venv_package_not_present_is_not_external(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        site_packages = tmp_path / "venv" / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)

        assert not _is_external_import("totally_made_up_pkg_xyz", "src/main.py")


# ── _guess_parent_file ────────────────────────────────────────────


class TestGuessParentFile:
    def test_step_text_mentions_both(self):
        memory = {
            "src/components/Header.tsx": "export default function Header() {}",
            "src/App.tsx": "export default function App() {}",
        }
        result = _guess_parent_file(
            "src/components/Header.tsx",
            "Create a Header component and add it to App",
            memory,
        )
        assert result == "src/App.tsx"

    def test_index_file_in_same_directory(self):
        memory = {
            "src/components/Header.tsx": "...",
            "src/components/index.ts": "...",
        }
        result = _guess_parent_file(
            "src/components/Header.tsx",
            "Create Header component",
            memory,
        )
        assert result == "src/components/index.ts"

    def test_common_root_file(self):
        memory = {
            "src/components/Header.tsx": "...",
            "src/App.tsx": "...",
        }
        result = _guess_parent_file(
            "src/components/Header.tsx",
            "Create Header component",
            memory,
        )
        assert result == "src/App.tsx"


# ── find_gaps ─────────────────────────────────────────────────────


class TestFindGaps:
    def test_detects_orphaned_export(self):
        before = DependencySnapshot()
        after_files = {
            "src/App.tsx": "export default function App() { return <div/>; }",
            "src/components/Header.tsx": "export default function Header() { return <h1/>; }",
        }
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/components/Header.tsx"],
            step_text="Create Header component",
            memory_files=after_files,
        )
        assert len(gaps) >= 1
        orphan = [g for g in gaps if g.gap_type == "orphaned_export"]
        assert len(orphan) == 1
        assert orphan[0].source_file == "src/components/Header.tsx"

    def test_no_false_positive_when_imported(self):
        after_files = {
            "src/App.tsx": (
                "import Header from './components/Header';\n"
                "export default function App() { return <Header/>; }"
            ),
            "src/components/Header.tsx": "export default function Header() { return <h1/>; }",
        }
        before = DependencySnapshot()
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/components/Header.tsx"],
            step_text="Create Header component",
            memory_files=after_files,
        )
        orphan = [g for g in gaps if g.gap_type == "orphaned_export"]
        assert len(orphan) == 0

    def test_django_app_module_not_flagged_as_orphan(self):
        """A Django app's forms.py is wired by the framework (settings/URLconf),
        not imported by sibling files. Flagging it as an orphaned export led the
        auto-fix to re-export it from __init__.py, which crashed manage.py check
        with 'populate() isn't reentrant'. It must never be flagged."""
        after_files = {
            "spacious_site/settings.py": "INSTALLED_APPS = ['core']",
            "core/apps.py": "from django.apps import AppConfig\nclass CoreConfig(AppConfig): pass",
            "core/forms.py": (
                "from django.contrib.auth.forms import UserCreationForm\n"
                "class SignUpForm(UserCreationForm): pass"
            ),
        }
        before = DependencySnapshot()
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["core/forms.py"],
            step_text="Create core/forms.py",
            memory_files=after_files,
        )
        orphan = [g for g in gaps if g.gap_type == "orphaned_export"]
        assert orphan == []

    def test_plain_python_forms_still_flagged_outside_django(self):
        """The Django skip must be Django-specific: a forms.py in a non-Django
        project (no manage.py/settings.py, no apps.py) is still a normal module."""
        after_files = {
            "pkg/main.py": "print('hi')",
            "pkg/forms.py": "class SignUpForm: pass",
        }
        before = DependencySnapshot()
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["pkg/forms.py"],
            step_text="Create forms",
            memory_files=after_files,
        )
        orphan = [g for g in gaps if g.gap_type == "orphaned_export"]
        assert len(orphan) == 1

    def test_no_false_positive_when_same_basename_imported_from_different_path(self):
        """App.jsx imports './components/Homepage' — creating pages/Homepage.jsx
        should NOT be flagged as orphaned since the same basename is already wired."""
        before_files = {
            "src/App.jsx": (
                "import Homepage from './components/Homepage';\n"
                "export default function App() { return <Homepage/>; }"
            ),
            "src/components/Homepage.jsx": "export default function Homepage() {}",
        }
        before = build_snapshot(before_files)
        after_files = dict(before_files)
        after_files["src/pages/Homepage.jsx"] = (
            "import Header from '../components/Header';\n"
            "export default function Homepage() { return <div><Header/></div>; }"
        )
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/pages/Homepage.jsx"],
            step_text="Implement the Homepage page",
            memory_files=after_files,
        )
        orphan = [g for g in gaps if g.gap_type == "orphaned_export"]
        assert len(orphan) == 0

    def test_detects_broken_import(self):
        before = DependencySnapshot()
        after_files = {
            "src/App.tsx": (
                "import Header from './components/Header';\n"
                "export default function App() { return <Header/>; }"
            ),
        }
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/App.tsx"],
            step_text="Create App component",
            memory_files=after_files,
        )
        broken = [g for g in gaps if g.gap_type == "broken_import"]
        assert len(broken) == 1
        assert broken[0].symbol == "./components/Header"

    def test_package_self_import_not_broken(self):
        # `from . import views` in a Django app's urls.py — the package
        # is the importer's own directory; flagging it burned an LLM
        # fix call on every Django run.
        before = DependencySnapshot()
        after_files = {
            "spacious_site/sitepages/urls.py": (
                "from django.urls import path\n"
                "from . import views\n"
                "urlpatterns = [path('', views.home, name='home')]\n"
            ),
            "spacious_site/sitepages/views.py": (
                "def home(request):\n    return None\n"
            ),
        }
        after = build_snapshot(after_files)
        # In production `django` resolves as external via the target
        # project's own venv (a Django task pip-installs it there). This
        # test must not depend on Django being installed in agentchanti's
        # env — CI runs a clean interpreter where `django.urls` would
        # otherwise be misflagged as a broken local import. Patch the
        # venv check to reflect the production guarantee.
        with patch(
            "agentchanti.orchestrator.dependency_check."
            "_project_has_installed_package",
            side_effect=lambda mod: mod == "django",
        ):
            gaps = find_gaps(
                before, after,
                new_files=["spacious_site/sitepages/urls.py"],
                step_text="Create app URL router",
                memory_files=after_files,
            )
        broken = [g for g in gaps if g.gap_type == "broken_import"]
        assert broken == []

    def test_ignores_external_imports(self):
        before = DependencySnapshot()
        after_files = {
            "src/App.tsx": (
                "import React from 'react';\n"
                "export default function App() { return <div/>; }"
            ),
        }
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/App.tsx"],
            step_text="Create App",
            memory_files=after_files,
        )
        broken = [g for g in gaps if g.gap_type == "broken_import"]
        assert len(broken) == 0

    def test_ignores_pre_existing_broken_imports(self):
        before_files = {
            "src/App.tsx": (
                "import Missing from './missing';\n"
                "export default function App() {}"
            ),
        }
        before = build_snapshot(before_files)
        # After step: same broken import still there, not newly introduced
        after_files = dict(before_files)
        after_files["src/utils.ts"] = "export function helper() {}"
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/utils.ts"],
            step_text="Add utils",
            memory_files=after_files,
        )
        broken = [g for g in gaps if g.gap_type == "broken_import"]
        assert len(broken) == 0

    def test_detects_missing_connection_from_step_text(self):
        after_files = {
            "src/Header.tsx": "export default function Header() {}",
            "src/App.tsx": "export default function App() { return <div/>; }",
        }
        before = DependencySnapshot()
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/Header.tsx", "src/App.tsx"],
            step_text="Add Header to App",
            memory_files=after_files,
        )
        # Should detect either orphaned_export or missing_connection
        assert len(gaps) >= 1

    def test_detects_missing_default_export_when_default_imported(self):
        """A new component file lacks export default but is default-imported."""
        before_files = {
            "src/App.tsx": (
                "import Dashboard from './pages/Dashboard';\n"
                "export default function App() { return <Dashboard/>; }"
            ),
        }
        before = build_snapshot(before_files)
        after_files = dict(before_files)
        # Dashboard.jsx created WITHOUT export default
        after_files["src/pages/Dashboard.tsx"] = (
            "function Dashboard() { return <div>Dashboard</div>; }"
        )
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/pages/Dashboard.tsx"],
            step_text="Create Dashboard page",
            memory_files=after_files,
        )
        missing_def = [g for g in gaps if g.gap_type == "missing_default_export"]
        assert len(missing_def) == 1
        assert missing_def[0].source_file == "src/pages/Dashboard.tsx"
        assert "Dashboard" in missing_def[0].symbol

    def test_detects_lost_default_export_after_edit(self):
        """A file that previously had export default loses it after editing."""
        before_files = {
            "src/App.tsx": "export default function App() { return <div/>; }",
            "src/Header.tsx": "export default function Header() { return <h1/>; }",
        }
        before = build_snapshot(before_files)
        after_files = {
            "src/App.tsx": before_files["src/App.tsx"],
            # Header was edited and lost its export default
            "src/Header.tsx": "function Header() { return <h1>Updated</h1>; }",
        }
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/Header.tsx"],
            step_text="Update Header component",
            memory_files=after_files,
        )
        missing_def = [g for g in gaps if g.gap_type == "missing_default_export"]
        assert len(missing_def) == 1
        assert "was removed during editing" in missing_def[0].description

    def test_no_missing_default_export_when_present(self):
        """No gap when the component file has export default."""
        before = DependencySnapshot()
        after_files = {
            "src/App.tsx": (
                "import Header from './components/Header';\n"
                "export default function App() { return <Header/>; }"
            ),
            "src/components/Header.tsx": "export default function Header() { return <h1/>; }",
        }
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/components/Header.tsx"],
            step_text="Create Header",
            memory_files=after_files,
        )
        missing_def = [g for g in gaps if g.gap_type == "missing_default_export"]
        assert len(missing_def) == 0

    def test_no_gaps_when_everything_connected(self):
        before_files = {
            "src/App.tsx": "export default function App() {}",
        }
        before = build_snapshot(before_files)
        after_files = {
            "src/App.tsx": (
                "import Header from './components/Header';\n"
                "export default function App() { return <Header/>; }"
            ),
            "src/components/Header.tsx": "export default function Header() { return <h1/>; }",
        }
        after = build_snapshot(after_files)
        gaps = find_gaps(
            before, after,
            new_files=["src/components/Header.tsx"],
            step_text="Create Header",
            memory_files=after_files,
        )
        assert len(gaps) == 0


# ── build_fix_prompt ──────────────────────────────────────────────


class TestBuildFixPrompt:
    def test_formats_gaps_and_files(self):
        gaps = [
            IntegrationGap(
                gap_type="orphaned_export",
                source_file="src/Header.tsx",
                target_file="src/App.tsx",
                symbol="Header",
                description="File 'src/Header.tsx' exports [Header] but no file imports it.",
            ),
        ]
        memory = {
            "src/Header.tsx": "export default function Header() {}",
            "src/App.tsx": "export default function App() {}",
        }
        prompt = build_fix_prompt(gaps, memory, "Create Header", "typescript")
        assert "ORPHANED EXPORT" in prompt
        assert "src/Header.tsx" in prompt
        assert "src/App.tsx" in prompt

    def test_detects_esm_module_system(self):
        memory = {
            "src/App.tsx": "import X from './X';\nexport default function App() {}",
        }
        prompt = build_fix_prompt([], memory, "test", "typescript")
        assert "ES Modules" in prompt

    def test_detects_cjs_module_system(self):
        memory = {
            "server.js": "const express = require('express');\nmodule.exports = {};",
        }
        prompt = build_fix_prompt([], memory, "test", "javascript")
        assert "CommonJS" in prompt


# ── run_dependency_check ──────────────────────────────────────────


class _FakeMemory:
    def __init__(self, files):
        self._files = dict(files)

    def all_files(self):
        return dict(self._files)

    def update(self, files):
        self._files.update(files)


class _FakeDisplay:
    def __init__(self):
        self.messages = []

    def step_info(self, step_idx, msg):
        self.messages.append(msg)


class _FakeLLMClient:
    def __init__(self, response=""):
        self.response = response
        self.called = False

    def generate_response(self, prompt):
        self.called = True
        return self.response


class _FakeExecutor:
    def parse_code_blocks(self, response):
        return {}

    def parse_code_blocks_fuzzy(self, response):
        return {}

    def write_files(self, files):
        return list(files.keys())


class _FakeConfig:
    DEPENDENCY_CHECK_ENABLED = True


class TestRunDependencyCheck:
    def test_returns_empty_when_disabled(self):
        cfg = _FakeConfig()
        cfg.DEPENDENCY_CHECK_ENABLED = False
        result = run_dependency_check(
            0, "test", ["a.py"],
            DependencySnapshot(), DependencySnapshot(),
            _FakeMemory({}), _FakeLLMClient(), _FakeExecutor(),
            _FakeDisplay(), "python", cfg,
        )
        assert result == {}

    def test_returns_empty_when_no_gaps(self):
        files = {
            "src/App.tsx": "import Header from './Header';\nexport default function App() {}",
            "src/Header.tsx": "export default function Header() {}",
        }
        before = build_snapshot({"src/App.tsx": files["src/App.tsx"]})
        after = build_snapshot(files)
        llm = _FakeLLMClient()
        result = run_dependency_check(
            0, "Add Header", ["src/Header.tsx"],
            before, after,
            _FakeMemory(files), llm, _FakeExecutor(),
            _FakeDisplay(), "typescript",
        )
        assert result == {}
        assert not llm.called  # No LLM call when no gaps

    def test_returns_empty_when_too_few_files(self):
        files = {"src/App.tsx": "export default function App() {}"}
        result = run_dependency_check(
            0, "test", ["src/App.tsx"],
            DependencySnapshot(), build_snapshot(files),
            _FakeMemory(files), _FakeLLMClient(), _FakeExecutor(),
            _FakeDisplay(), "typescript",
        )
        assert result == {}

    def test_calls_llm_when_gaps_found(self):
        files = {
            "src/App.tsx": "export default function App() { return <div/>; }",
            "src/Header.tsx": "export default function Header() { return <h1/>; }",
        }
        before = DependencySnapshot()
        after = build_snapshot(files)
        llm = _FakeLLMClient("#### [FILE]: src/App.tsx\n```\nimport Header from './Header';\nexport default function App() { return <Header/>; }\n```")

        class ParseExecutor(_FakeExecutor):
            def parse_code_blocks(self, response):
                if "#### [FILE]:" in response:
                    return {
                        "src/App.tsx": "import Header from './Header';\nexport default function App() { return <Header/>; }",
                    }
                return {}

        result = run_dependency_check(
            0, "Create Header", ["src/Header.tsx"],
            before, after,
            _FakeMemory(files), llm, ParseExecutor(),
            _FakeDisplay(), "typescript",
        )
        assert llm.called
        assert "src/App.tsx" in result


# ── _scan_balanced_destructure / _split_top_level_params ─────────


class TestBalancedScanner:
    def test_simple_destructure(self):
        s = "({a, b})"
        inner, end = _scan_balanced_destructure(s, 1)
        assert inner == "a, b"
        assert s[end - 1] == "}"

    def test_nested_object_default(self):
        s = "({a = { x: 1, y: 2 }, b})"
        inner, _ = _scan_balanced_destructure(s, 1)
        assert inner == "a = { x: 1, y: 2 }, b"

    def test_brace_inside_string_does_not_close(self):
        s = "({a = 'has } brace', b})"
        inner, _ = _scan_balanced_destructure(s, 1)
        assert inner == "a = 'has } brace', b"

    def test_brace_inside_template_literal(self):
        s = "({a = `tpl } brace`, b})"
        inner, _ = _scan_balanced_destructure(s, 1)
        assert inner == "a = `tpl } brace`, b"

    def test_escaped_quote_in_string(self):
        s = "({a = 'it\\'s fine, really', b})"
        inner, _ = _scan_balanced_destructure(s, 1)
        assert inner == "a = 'it\\'s fine, really', b"

    def test_unclosed_destructure_returns_none(self):
        s = "({a, b"
        assert _scan_balanced_destructure(s, 1) is None

    def test_split_respects_string_commas(self):
        parts = _split_top_level_params("a = 'x, y, z', b")
        assert parts == ["a = 'x, y, z'", " b"]

    def test_split_respects_object_commas(self):
        parts = _split_top_level_params("a = { p: 1, q: 2 }, b")
        assert parts == ["a = { p: 1, q: 2 }", " b"]

    def test_split_respects_array_commas(self):
        parts = _split_top_level_params("a = [1, 2, 3], b")
        assert parts == ["a = [1, 2, 3]", " b"]

    def test_split_respects_call_commas(self):
        parts = _split_top_level_params("a = f(1, 2), b")
        assert parts == ["a = f(1, 2)", " b"]


# ── _extract_component_props ─────────────────────────────────────


class TestExtractComponentProps:
    def test_required_props_no_defaults(self):
        content = "function Foo({ a, b, c }) { return null; }"
        assert _extract_component_props(content) == {"Foo": {"a", "b", "c"}}

    def test_props_with_defaults_are_optional(self):
        content = "function Foo({ a = 1, b = 2 }) { return null; }"
        assert _extract_component_props(content) == {}

    def test_mixed_required_and_optional(self):
        content = "function Foo({ a, b = 2, c }) { return null; }"
        assert _extract_component_props(content) == {"Foo": {"a", "c"}}

    def test_string_default_with_commas_does_not_create_fake_props(self):
        """Regression: prose commas in default strings used to be parsed as params.

        See dependency_check.py: a description like
        'Build apps with React, Tailwind, and joy' would split into fake
        required props 'Tailwind' and 'and joy'.
        """
        content = (
            "function Hero({\n"
            "  title = 'Build apps faster',\n"
            "  description = 'Modern homepage with header, animated banner, and layout.',\n"
            "}) { return null; }"
        )
        assert _extract_component_props(content) == {}

    def test_object_default_with_commas_does_not_create_fake_props(self):
        """Regression: nested object defaults used to leak inner keys as fake props."""
        content = (
            "function Hero({\n"
            "  primaryCta = { label: 'Get started', to: '/signup' },\n"
            "  secondaryCta = { label: 'Learn more', to: '/features' },\n"
            "}) { return null; }"
        )
        assert _extract_component_props(content) == {}

    def test_real_world_hero_banner(self):
        """Regression: the exact shape from the agentchanti demo bug report."""
        content = (
            "function HeroBanner({\n"
            "  eyebrow = 'New feature',\n"
            "  title = 'Build responsive apps faster with React and Tailwind CSS',\n"
            "  description = 'A modern single-page homepage with a responsive header, animated hero banner, and a polished mobile-friendly layout.',\n"
            "  primaryCta = { label: 'Get started', to: '/signup' },\n"
            "  secondaryCta = { label: 'Learn more', to: '/features' },\n"
            "}) {\n"
            "  return null;\n"
            "}"
        )
        # All five props have defaults → none required → no entry.
        assert _extract_component_props(content) == {}

    def test_arrow_component(self):
        content = "const Foo = ({ a, b }) => null;"
        assert _extract_component_props(content) == {"Foo": {"a", "b"}}

    def test_export_default_arrow(self):
        content = "export default function Foo({ a, b }) { return null; }"
        assert _extract_component_props(content) == {"Foo": {"a", "b"}}

    def test_rest_prop_ignored(self):
        content = "function Foo({ a, ...rest }) { return null; }"
        assert _extract_component_props(content) == {"Foo": {"a"}}

    def test_lowercase_function_not_a_component(self):
        content = "function helper({ a, b }) { return null; }"
        assert _extract_component_props(content) == {}

    def test_zero_arg_function_not_destructured(self):
        content = "function Foo() { return null; }"
        assert _extract_component_props(content) == {}

    def test_positional_props_function_not_destructured(self):
        content = "function Foo(props) { return null; }"
        assert _extract_component_props(content) == {}

    def test_two_components_in_one_file(self):
        content = (
            "function Foo({ a, b }) { return null; }\n"
            "function Bar({ x = 1, y }) { return null; }"
        )
        assert _extract_component_props(content) == {
            "Foo": {"a", "b"},
            "Bar": {"y"},
        }


# ── _find_signature_gaps ─────────────────────────────────────────


class TestFindSignatureGaps:
    def test_caller_missing_required_props_triggers_gap(self):
        """Positive control: real signature mismatch must still be detected."""
        files = {
            "src/Foo.jsx": "export default function Foo({ a, b, c }) { return null; }",
            "src/App.jsx": "import Foo from './Foo';\nfunction App() { return <Foo />; }",
        }
        gaps = _find_signature_gaps(["src/Foo.jsx"], files)
        stale = [g for g in gaps if g.gap_type == "stale_caller"]
        assert len(stale) == 1
        assert stale[0].source_file == "src/Foo.jsx"
        assert stale[0].target_file == "src/App.jsx"
        assert stale[0].symbol == "Foo"

    def test_caller_with_correct_props_no_gap(self):
        files = {
            "src/Foo.jsx": "export default function Foo({ a, b }) { return null; }",
            "src/App.jsx": "function App() { return <Foo a={1} b={2} />; }",
        }
        gaps = _find_signature_gaps(["src/Foo.jsx"], files)
        assert [g for g in gaps if g.gap_type == "stale_caller"] == []

    def test_no_gap_when_all_props_have_defaults(self):
        """Regression: the HeroBanner case must not produce a stale_caller gap."""
        files = {
            "src/HeroBanner.jsx": (
                "function HeroBanner({\n"
                "  title = 'Build apps, fast',\n"
                "  description = 'Header, banner, and layout.',\n"
                "  primaryCta = { label: 'Go', to: '/x' },\n"
                "}) { return null; }\n"
                "export default HeroBanner;"
            ),
            "src/App.jsx": "function App() { return <HeroBanner />; }",
        }
        gaps = _find_signature_gaps(["src/HeroBanner.jsx"], files)
        assert [g for g in gaps if g.gap_type == "stale_caller"] == []

    def test_spread_caller_skipped(self):
        """Spread props in JSX call site must suppress the gap (caller may pass them)."""
        files = {
            "src/Foo.jsx": "export default function Foo({ a, b, c }) { return null; }",
            "src/App.jsx": "function App({ p }) { return <Foo {...p} />; }",
        }
        gaps = _find_signature_gaps(["src/Foo.jsx"], files)
        assert [g for g in gaps if g.gap_type == "stale_caller"] == []


# ── Orphaned-export suppressions ──────────────────────────────────


class TestPlanDeclaresImport:
    def test_module_dot_notation(self):
        from agentchanti.orchestrator.dependency_check import _plan_declares_import
        assert _plan_declares_import("src/snake.py", {"src.snake"})

    def test_windows_path_vs_module_notation(self):
        from agentchanti.orchestrator.dependency_check import _plan_declares_import
        assert _plan_declares_import("src\\snake.py", {"src.snake"})

    def test_file_path_notation(self):
        from agentchanti.orchestrator.dependency_check import _plan_declares_import
        assert _plan_declares_import("src/snake.py", {"src/snake.py"})

    def test_bare_module_name(self):
        from agentchanti.orchestrator.dependency_check import _plan_declares_import
        assert _plan_declares_import("src/snake.py", {"snake"})

    def test_no_match(self):
        from agentchanti.orchestrator.dependency_check import _plan_declares_import
        assert not _plan_declares_import("src/snake.py", {"src.food"})

    def test_no_partial_stem_match(self):
        from agentchanti.orchestrator.dependency_check import _plan_declares_import
        assert not _plan_declares_import("src/rattlesnake.py", {"snake"})


class TestOrphanSuppressions:
    def test_entrypoint_with_main_guard_not_orphaned(self):
        """A module executed via `if __name__ == "__main__"` is an entry
        point — no importer is expected, so it must not be flagged."""
        after_files = {
            "src/game.py": (
                "class SnakeGame:\n    pass\n\n"
                "def main():\n    pass\n\n"
                'if __name__ == "__main__":\n    main()\n'
            ),
            "src/other.py": "def helper():\n    pass\n",
        }
        gaps = find_gaps(
            DependencySnapshot(), build_snapshot(after_files),
            new_files=["src/game.py"],
            step_text="Create the game entrypoint",
            memory_files=after_files,
        )
        assert [g for g in gaps if g.gap_type == "orphaned_export"] == []

    def test_pending_plan_import_suppresses_orphan(self):
        """A pending TEST step declares `imports: src.snake:Snake` — the
        wiring belongs to that future step, so snake.py is not an orphan."""
        after_files = {
            "src/snake.py": "class Snake:\n    pass\n",
            "src/constants.py": "GRID_WIDTH = 20\n",
        }
        gaps = find_gaps(
            DependencySnapshot(), build_snapshot(after_files),
            new_files=["src/snake.py"],
            step_text="Create Snake game logic class",
            memory_files=after_files,
            plan_declared_imports={"src.snake"},
        )
        assert [g for g in gaps if g.gap_type == "orphaned_export"] == []

    def test_orphan_still_detected_without_declared_import(self):
        """Control: same scenario without a plan declaration still flags."""
        after_files = {
            "src/snake.py": "class Snake:\n    pass\n",
            "src/constants.py": "GRID_WIDTH = 20\n",
        }
        gaps = find_gaps(
            DependencySnapshot(), build_snapshot(after_files),
            new_files=["src/snake.py"],
            step_text="Create Snake game logic class",
            memory_files=after_files,
            plan_declared_imports={"src.food"},
        )
        orphan = [g for g in gaps if g.gap_type == "orphaned_export"]
        assert len(orphan) == 1
        assert orphan[0].source_file == "src/snake.py"

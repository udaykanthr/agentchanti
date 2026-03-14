"""Tests for the post-step dependency validation system."""

import pytest

from multi_agent_coder.orchestrator.dependency_check import (
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

    def test_skips_test_files(self):
        files = {
            "src/App.tsx": "export default function App() {}",
            "src/App.test.tsx": "import App from './App';\ntest('renders', () => {});",
            "tests/test_views.py": "from app.views import index\ndef test_index(): pass",
        }
        snap = build_snapshot(files)
        assert "src/App.tsx" in snap.file_deps
        assert "src/App.test.tsx" not in snap.file_deps
        assert "tests/test_views.py" not in snap.file_deps


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

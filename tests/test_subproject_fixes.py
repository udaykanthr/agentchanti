"""Tests for _prefix_subproject_paths and coder prompt file extensions.

Covers:
- _prefix_subproject_paths: prefixing new files with sub-project root
- CoderAgent prompt example using correct file extensions
- CMD-output-based sub-project detection
- _NON_SUBPROJECT_DIRS blocklist
"""

import os
import re
from unittest.mock import patch

from agentchanti.orchestrator.step_handlers import (
    _prefix_subproject_paths,
    _detect_subproject_root,
)
from agentchanti.orchestrator.memory import FileMemory
from agentchanti.agents.coder import _LANG_TO_EXT


# ─── Helpers ────────────────────────────────────────────────────────


def _make_memory(files: dict[str, str]) -> FileMemory:
    """Create a FileMemory with given files (no embeddings)."""
    mem = FileMemory(embedding_store=None)
    mem.update(files)
    return mem


# ─── _prefix_subproject_paths ───────────────────────────────────────


class TestPrefixSubprojectPaths:
    def test_prefixes_new_files(self):
        """New files without sub-project prefix get prefixed."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
            "my-app/package.json": "{}",
        })

        files = {
            "components/Header.tsx": "header code",
            "components/Footer.tsx": "footer code",
        }
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "my-app/components/Header.tsx" in result
        assert "my-app/components/Footer.tsx" in result
        assert "components/Header.tsx" not in result
        assert "components/Footer.tsx" not in result

    def test_does_not_double_prefix(self):
        """Files already prefixed with sub-project root are unchanged."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {"my-app/src/App.js": "updated A"}
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "my-app/src/App.js" in result
        assert len(result) == 1

    def test_skips_internal_files(self):
        """Internal tracking files (_cmd_output/) are not prefixed."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {
            "_cmd_output/step_1.txt": "output",
            "src/NewFile.js": "new code",
        }
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "_cmd_output/step_1.txt" in result
        assert "my-app/src/NewFile.js" in result
        assert "my-app/_cmd_output/step_1.txt" not in result

    def test_skips_known_memory_files(self):
        """Files that already exist in memory by exact path are untouched."""
        memory = _make_memory({
            "src/utils.js": "utility code",
            "my-app/src/App.js": "A",
        })

        files = {"src/utils.js": "updated utility"}
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "src/utils.js" in result
        assert "my-app/src/utils.js" not in result

    def test_no_subproject_returns_unchanged(self):
        """When subproject is empty string, files are unchanged."""
        memory = _make_memory({})
        files = {"src/App.js": "code"}
        result = _prefix_subproject_paths(files, "", memory)

        assert result == files

    def test_handles_trailing_slash(self):
        """Sub-project root with trailing slash doesn't double-slash."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {"src/index.js": "new code"}
        result = _prefix_subproject_paths(files, "my-app/", memory)

        assert "my-app/src/index.js" in result
        assert "my-app//src/index.js" not in result

    def test_prefixes_dunder_test_dirs(self):
        """__tests__/ directories MUST be prefixed (not treated as internal)."""
        memory = _make_memory({
            "responsive-spa/src/App.jsx": "A",
        })

        files = {
            "__tests__/Header.test.js": "test code",
            "__mocks__/fileMock.js": "mock",
        }
        result = _prefix_subproject_paths(files, "responsive-spa", memory)

        assert "responsive-spa/__tests__/Header.test.js" in result
        assert "responsive-spa/__mocks__/fileMock.js" in result
        assert "__tests__/Header.test.js" not in result
        assert "__mocks__/fileMock.js" not in result

    def test_skips_all_internal_tracking_paths(self):
        """All known internal paths (_cmd_output, _fix_output, _search_context) are skipped."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {
            "_cmd_output/step_1.txt": "output",
            "_fix_output/step_2.txt": "fix",
            "_search_context/step_3.txt": "search",
            "src/NewFile.js": "new code",
        }
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "_cmd_output/step_1.txt" in result
        assert "_fix_output/step_2.txt" in result
        assert "_search_context/step_3.txt" in result
        assert "my-app/src/NewFile.js" in result


class TestPlannerLeaderRename:
    """Pin the planner-leader rename branch.

    Background: a CMD-step diagnosis fix can rename the scaffold
    directory (e.g. the planner says ``cd myapp && npm create
    vite@latest .`` but the diagnosis fix runs ``npm create vite@latest
    my-app``).  Subsequent inline-code paths still carry the planner's
    name (``myapp/src/...``) but the actual subproject is ``my-app/``.
    Without this branch the prefixer would blindly produce
    ``my-app/myapp/src/...``, double-nesting the entire homepage and
    leaving Vite serving the default scaffold while tests pass against
    the orphaned tree.
    """

    def test_renames_hyphen_variant(self):
        """The exact failing case from a real run: planner emitted
        ``myapp/src/...`` but the actual scaffold is ``my-app/``."""
        memory = _make_memory({})
        files = {"myapp/src/components/Header.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/src/components/Header.jsx" in result
        assert "my-app/myapp/src/components/Header.jsx" not in result

    def test_renames_camelcase_variant(self):
        """``myApp`` is the same project as ``my-app`` after
        normalisation."""
        memory = _make_memory({})
        files = {"myApp/src/main.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/src/main.jsx" in result

    def test_renames_snake_case_variant(self):
        """``my_app/src/...`` ↔ ``my-app/`` after normalisation."""
        memory = _make_memory({})
        files = {"my_app/src/index.css": "body {}"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/src/index.css" in result

    def test_renames_pascalcase_variant(self):
        """``MyApp`` matches ``my-app``."""
        memory = _make_memory({})
        files = {"MyApp/vite.config.js": "config"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/vite.config.js" in result

    def test_does_not_strip_src_leading_dir(self):
        """``src/`` is a standard project top dir — must NOT be
        stripped, even though ``src`` could in principle be a
        project name in some other setting.  This pins that
        legitimate ``src/Foo.jsx`` paths still get the prefix
        prepended (becoming ``my-app/src/Foo.jsx``), not stripped to
        the bare ``Foo.jsx``."""
        memory = _make_memory({})
        files = {"src/components/Header.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/src/components/Header.jsx" in result

    def test_does_not_strip_lib_leading_dir(self):
        memory = _make_memory({})
        files = {"lib/utils.js": "export"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/lib/utils.js" in result

    def test_does_not_strip_components_leading_dir(self):
        """``components/`` is in the common-top-dirs list to avoid
        false positives on project names that happen to be
        ``components`` (rare but allowed)."""
        memory = _make_memory({})
        files = {"components/Header.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/components/Header.jsx" in result

    def test_does_not_strip_unrelated_project_leader(self):
        """An unrelated project name (e.g. ``other-project/``) is NOT
        a normalisation match for ``my-app/``, so it gets prefixed
        normally rather than stripped.  This is a false-negative-safe
        rule — we'd rather double-prefix than silently rewrite to
        the wrong directory."""
        memory = _make_memory({})
        files = {"other-project/src/Header.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/other-project/src/Header.jsx" in result

    def test_already_prefixed_path_left_alone(self):
        """Sanity: when the path already starts with the actual
        prefix, the planner-leader branch must not fire."""
        memory = _make_memory({})
        files = {"my-app/src/components/Header.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app", memory)
        assert "my-app/src/components/Header.jsx" in result
        # Must NOT have produced a doubly-prefixed copy.
        assert "my-app/my-app/src/components/Header.jsx" not in result

    def test_rename_works_with_trailing_slash_subproject(self):
        """Subproject argument may include a trailing slash —
        normalisation must still work."""
        memory = _make_memory({})
        files = {"myapp/src/App.jsx": "<jsx>"}
        result = _prefix_subproject_paths(files, "my-app/", memory)
        assert "my-app/src/App.jsx" in result

    def test_rename_handles_multiple_files_in_one_call(self):
        """All files in a single batch are independently rewritten."""
        memory = _make_memory({})
        files = {
            "myapp/src/components/Header.jsx": "h",
            "myapp/src/components/HeroBanner.jsx": "b",
            "myapp/src/App.jsx": "a",
            "myapp/vite.config.js": "c",
            "myapp/vitest.setup.js": "s",
        }
        result = _prefix_subproject_paths(files, "my-app", memory)
        for src in (
            "my-app/src/components/Header.jsx",
            "my-app/src/components/HeroBanner.jsx",
            "my-app/src/App.jsx",
            "my-app/vite.config.js",
            "my-app/vitest.setup.js",
        ):
            assert src in result, f"missing {src}"
        # Confirm none of the doubly-nested forms made it through.
        for bad in (
            "my-app/myapp/src/components/Header.jsx",
            "my-app/myapp/src/App.jsx",
            "my-app/myapp/vite.config.js",
        ):
            assert bad not in result, f"unexpected {bad}"


# ─── CMD-output-based sub-project detection ─────────────────────────


class TestCmdOutputSubprojectDetection:
    """Tests for _detect_subproject_root's CMD output parsing (Fallback 0).

    This is the critical fix: when only _cmd_output/ entries exist in memory
    (no source files yet), the function must still detect the sub-project
    by parsing the create command from the CMD output.
    """

    def test_detects_create_next_app(self):
        """Detect sub-project from 'npx create-next-app my-app'."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app@latest my-bootstrap-website --typescript --yes\n"
                "Creating a new Next.js app...\n"
                "Success!"
            ),
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "my-bootstrap-website"

    def test_detects_create_react_app(self):
        """Detect sub-project from 'npx create-react-app my-app'."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-react-app my-react-app\n"
                "Creating a new React app..."
            ),
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "my-react-app"

    def test_ignores_dot_slash_current_dir(self):
        """When create-next-app uses ./ (current dir), return None."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app ./ --typescript --yes\n"
                "Creating a new Next.js app..."
            ),
        })
        result = _detect_subproject_root(memory)
        assert result is None

    def test_ignores_dot_current_dir(self):
        """When create-next-app uses . (current dir), return None."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app . --typescript --yes\n"
                "Creating a new Next.js app..."
            ),
        })
        result = _detect_subproject_root(memory)
        assert result is None

    def test_requires_manifest_on_disk(self):
        """Directory must have a manifest file to be considered a sub-project."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app my-app --yes\n"
                "Creating..."
            ),
        })
        # Dir exists but no manifest file
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=False):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_detects_from_fix_output(self):
        """Also parse _fix_output/ entries (failed + retried commands)."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": "$ cd foo\n",
            "_fix_output/step_1.txt": (
                "$ npx create-next-app ./new-project --yes\n"
                "Success!"
            ),
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "new-project"


# ─── Non-subproject directory blocklist ──────────────────────────────


class TestNonSubprojectDirsBlocklist:
    """Tests that common source dirs (src, app, components, etc.) are NOT
    incorrectly identified as sub-project roots."""

    def test_src_is_not_subproject(self):
        """src/ should never be returned as a sub-project root."""
        memory = _make_memory({
            "src/components/Header.tsx": "A",
            "src/components/Footer.tsx": "B",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_components_is_not_subproject(self):
        """components/ should never be returned as a sub-project root."""
        memory = _make_memory({
            "components/Header.tsx": "A",
            "components/Footer.tsx": "B",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_pages_is_not_subproject(self):
        """pages/ should never be returned as a sub-project root."""
        memory = _make_memory({
            "pages/_app.tsx": "A",
            "pages/index.tsx": "B",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_real_project_name_is_detected(self):
        """A real project directory name (my-app) should still be detected."""
        memory = _make_memory({
            "my-app/src/App.tsx": "A",
            "my-app/package.json": "{}",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "my-app"

    def test_bare_package_dir_is_not_subproject(self):
        """A plain directory holding source with NO manifest and no own
        venv is a package under the repo-root project, not a sub-project.

        Regression guard for the brick-breaker failure: classifying
        ``brick_breaker/`` as a sub-project prepended ``cd brick_breaker``
        to its gate, which broke the root-relative venv path and the
        ``import brick_breaker.main`` probe, producing a false monotonic
        regression that rolled back working code.
        """
        memory = _make_memory({
            "brick_breaker/main.py": "import pygame",
            "brick_breaker/test_main.py": "import brick_breaker.main",
        })
        # Dir exists on disk; no manifest, no __init__.py, no own venv.
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os.path.exists", return_value=False):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_memory_manifest_marks_subproject(self):
        """A manifest tracked in memory (not yet visible on the disk check)
        is sufficient self-containment evidence to detect the sub-project."""
        memory = _make_memory({
            "svc/main.py": "x = 1",
            "svc/pyproject.toml": "[project]",
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os.path.exists", return_value=False):
            result = _detect_subproject_root(memory)
        assert result == "svc"


# ─── Coder prompt file extension ────────────────────────────────────


class TestCoderPromptExtension:
    def test_lang_to_ext_has_all_common_languages(self):
        """Mapping covers the most common languages."""
        assert _LANG_TO_EXT["python"] == ".py"
        assert _LANG_TO_EXT["javascript"] == ".js"
        assert _LANG_TO_EXT["typescript"] == ".ts"
        assert _LANG_TO_EXT["go"] == ".go"
        assert _LANG_TO_EXT["rust"] == ".rs"
        assert _LANG_TO_EXT["java"] == ".java"

    def test_no_dotpython_extension(self):
        """The mapping should NEVER produce .python as extension."""
        for lang, ext in _LANG_TO_EXT.items():
            assert ext != ".python", f"Language {lang} maps to .python"
            assert ext != ".javascript", f"Language {lang} maps to .javascript"
            assert ext != ".typescript", f"Language {lang} maps to .typescript"

    def test_all_extensions_are_short(self):
        """File extensions should be short (max 6 chars incl. dot)."""
        for lang, ext in _LANG_TO_EXT.items():
            assert ext.startswith("."), f"Extension for {lang} missing dot: {ext}"
            assert len(ext) <= 6, f"Extension for {lang} too long: {ext}"


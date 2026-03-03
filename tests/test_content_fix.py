"""
Tests for the KB-driven content-fix sanitizer.

Covers _apply_content_fixes() in step_handlers — the generic function
that reads ContentFix rules from the global KB and applies them to
LLM-generated file content before writing to disk.
"""

from __future__ import annotations

import unittest

from multi_agent_coder.kb.global_kb.error_dict import ContentFix
from multi_agent_coder.orchestrator.step_handlers import _apply_content_fixes


# The Tailwind rule exactly as seeded — used as a realistic test fixture.
_TAILWIND_FIX = ContentFix(
    name="tailwind-v4-directives",
    file_glob="*.css",
    content_pattern=r"^\s*@tailwind\s+(base|components|utilities)\s*;\s*$",
    replacement="",
    ensure_content='@import "tailwindcss";\n',
    flags="MULTILINE",
    collapse_blanks=True,
    language="all",
    description="Tailwind CSS v4 migration",
)


class TestApplyContentFixes(unittest.TestCase):
    """Tests for _apply_content_fixes."""

    def test_tailwind_v3_to_v4(self):
        """Old @tailwind directives are replaced with @import."""
        files = {
            "src/index.css": (
                "@tailwind base;\n"
                "@tailwind components;\n"
                "@tailwind utilities;\n"
                "\n"
                "body { margin: 0; }\n"
            ),
        }
        result = _apply_content_fixes(files, [_TAILWIND_FIX])
        css = result["src/index.css"]
        self.assertNotIn("@tailwind", css)
        self.assertIn('@import "tailwindcss"', css)
        self.assertIn("body { margin: 0; }", css)

    def test_no_duplicate_import(self):
        """If @import already present, don't prepend again."""
        files = {
            "src/index.css": (
                '@import "tailwindcss";\n'
                "@tailwind base;\n"
                "@tailwind components;\n"
                "@tailwind utilities;\n"
            ),
        }
        result = _apply_content_fixes(files, [_TAILWIND_FIX])
        css = result["src/index.css"]
        self.assertEqual(css.count('@import "tailwindcss"'), 1)

    def test_none_passthrough(self):
        """None or empty content_fixes returns files unchanged."""
        files = {"a.css": "body {}"}
        self.assertEqual(_apply_content_fixes(files, None), files)
        self.assertEqual(_apply_content_fixes(files, []), files)

    def test_glob_mismatch_ignored(self):
        """Non-matching files are not touched."""
        fix = ContentFix(
            name="css-only", file_glob="*.css",
            content_pattern=r"old", replacement="new",
        )
        files = {"app.js": "old stuff"}
        result = _apply_content_fixes(files, [fix])
        self.assertEqual(result["app.js"], "old stuff")

    def test_pattern_mismatch_ignored(self):
        """Files matching glob but not pattern are not touched."""
        files = {"style.css": "body { color: red; }\n"}
        result = _apply_content_fixes(files, [_TAILWIND_FIX])
        self.assertEqual(result["style.css"], files["style.css"])

    def test_collapse_blanks(self):
        """Multiple blank lines left behind are collapsed."""
        files = {
            "a.css": (
                "@tailwind base;\n"
                "\n\n\n\n"
                "body {}\n"
            ),
        }
        result = _apply_content_fixes(files, [_TAILWIND_FIX])
        # Should not have 3+ consecutive newlines
        self.assertNotIn("\n\n\n", result["a.css"])

    def test_nested_path_glob(self):
        """Glob matches basename regardless of directory depth."""
        files = {"my-app/src/styles/main.css": "@tailwind base;\n"}
        result = _apply_content_fixes(files, [_TAILWIND_FIX])
        self.assertNotIn("@tailwind", result["my-app/src/styles/main.css"])
        self.assertIn('@import "tailwindcss"', result["my-app/src/styles/main.css"])

    def test_multiple_rules(self):
        """Multiple rules can apply to different files."""
        fix_css = ContentFix(
            name="fix-css", file_glob="*.css",
            content_pattern=r"@old-css", replacement="@new-css",
        )
        fix_js = ContentFix(
            name="fix-js", file_glob="*.js",
            content_pattern=r"require\('old'\)", replacement="require('new')",
        )
        files = {
            "a.css": "@old-css\nbody {}\n",
            "b.js": "const x = require('old');\n",
        }
        result = _apply_content_fixes(files, [fix_css, fix_js])
        self.assertIn("@new-css", result["a.css"])
        self.assertIn("require('new')", result["b.js"])


if __name__ == "__main__":
    unittest.main()

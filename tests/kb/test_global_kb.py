"""
Tests for the Phase 3 — Global Knowledge Base.

Covers:
- ErrorDict: CRUD, lookup with exact/regex/fuzzy matching
- Seeder: seed errors.db and markdown files (no embed)
- GlobalKBStore: search_errors, fallback file search
- Updater: version parsing, manifest loading
- Markdown chunking
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
import unittest

import agentchanti.kb.global_kb.store
from agentchanti.kb.global_kb.store import _language_matches

# ---------------------------------------------------------------------------
# ErrorDict tests
# ---------------------------------------------------------------------------


class TestErrorDict(unittest.TestCase):
    """Tests for ``kb.global.error_dict.ErrorDict``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_errors.db")
        from agentchanti.kb.global_kb.error_dict import ErrorDict
        self.edict = ErrorDict(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_count(self):
        """add() inserts a record; count() reflects it."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="TestError",
            language="python",
            pattern=r"TestError:.*",
            cause="test cause",
            fix_template="fix it",
            severity="error",
            tags="test",
        )
        self.edict.add(ef)
        self.assertEqual(self.edict.count(), 1)
        self.assertEqual(self.edict.count(language="python"), 1)
        self.assertEqual(self.edict.count(language="java"), 0)

    def test_bulk_insert(self):
        """bulk_insert() inserts multiple records."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        errors = [
            ErrorFix(error_type=f"Err{i}", language="python",
                     fix_template=f"fix {i}")
            for i in range(10)
        ]
        self.edict.bulk_insert(errors)
        self.assertEqual(self.edict.count(), 10)

    def test_lookup_exact(self):
        """lookup() matches by error_type substring."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="NullPointerException",
            language="java",
            pattern=r"NullPointerException",
            cause="null ref",
            fix_template="check for null",
        )
        self.edict.add(ef)
        results = self.edict.lookup("java.lang.NullPointerException at line 42")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].error_type, "NullPointerException")

    def test_lookup_regex(self):
        """lookup() falls back to regex pattern matching."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="ImportError",
            language="python",
            pattern=r"No module named '\w+'",
            cause="missing module",
            fix_template="pip install the module",
        )
        self.edict.add(ef)
        results = self.edict.lookup("No module named 'requests'", language="python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].error_type, "ImportError")

    def test_lookup_fuzzy_tags(self):
        """lookup() falls back to tag-based fuzzy matching."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="MemoryLeak",
            language="all",
            pattern=r"^$",  # Won't match anything via regex
            cause="memory not freed",
            fix_template="free the memory",
            tags="memory,leak,heap",
        )
        self.edict.add(ef)
        # "memory" is in the error message and in the tags
        results = self.edict.lookup("possible memory issue detected")
        self.assertEqual(len(results), 1)

    def test_lookup_language_filter(self):
        """lookup() respects language filter, includes 'all'."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        self.edict.bulk_insert([
            ErrorFix(error_type="Err1", language="python", fix_template="fix1"),
            ErrorFix(error_type="Err2", language="java", fix_template="fix2"),
            ErrorFix(error_type="Err3", language="all", fix_template="fix3"),
        ])
        # Searching for python should get Err1 + Err3 (language=all)
        results = self.edict.lookup("Err1 occurred", language="python")
        languages = {r.language for r in results}
        self.assertIn("python", languages)
        # java should NOT be included
        self.assertNotIn("java", languages)

    def test_clear(self):
        """clear() removes all records."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        self.edict.bulk_insert([
            ErrorFix(error_type=f"E{i}", language="go", fix_template="f")
            for i in range(5)
        ])
        self.assertEqual(self.edict.count(), 5)
        self.edict.clear()
        self.assertEqual(self.edict.count(), 0)

    def test_count_by_language(self):
        """count_by_language() groups correctly."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        self.edict.bulk_insert([
            ErrorFix(error_type="E1", language="python", fix_template="f"),
            ErrorFix(error_type="E2", language="python", fix_template="f"),
            ErrorFix(error_type="E3", language="java", fix_template="f"),
        ])
        counts = self.edict.count_by_language()
        self.assertEqual(counts["python"], 2)
        self.assertEqual(counts["java"], 1)

    def test_errorfix_tag_list(self):
        """ErrorFix.tag_list() splits comma-separated tags."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(error_type="E", language="py", fix_template="f",
                      tags="a, b, c")
        self.assertEqual(ef.tag_list(), ["a", "b", "c"])

    def test_errorfix_empty_tags(self):
        """ErrorFix.tag_list() returns empty list for no tags."""
        from agentchanti.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(error_type="E", language="py", fix_template="f", tags="")
        self.assertEqual(ef.tag_list(), [])


# ---------------------------------------------------------------------------
# ContentFix tests
# ---------------------------------------------------------------------------


class TestContentFix(unittest.TestCase):
    """Tests for ContentFix CRUD in ErrorDict."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_errors.db")
        from agentchanti.kb.global_kb.error_dict import ErrorDict
        self.edict = ErrorDict(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_get(self):
        """bulk_insert + get round-trips correctly."""
        from agentchanti.kb.global_kb.error_dict import ContentFix
        fix = ContentFix(
            name="test-fix",
            file_glob="*.css",
            content_pattern=r"@old-directive",
            replacement="@new-directive",
            description="test rule",
        )
        self.edict.bulk_insert_content_fixes([fix])
        fixes = self.edict.get_content_fixes()
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0].name, "test-fix")
        self.assertEqual(fixes[0].file_glob, "*.css")
        self.assertEqual(fixes[0].content_pattern, r"@old-directive")
        self.assertEqual(fixes[0].replacement, "@new-directive")

    def test_count_and_clear(self):
        from agentchanti.kb.global_kb.error_dict import ContentFix
        fixes = [
            ContentFix(name=f"fix-{i}", file_glob="*.css",
                       content_pattern=f"pat{i}", replacement=f"rep{i}")
            for i in range(5)
        ]
        self.edict.bulk_insert_content_fixes(fixes)
        self.assertEqual(self.edict.count_content_fixes(), 5)
        self.edict.clear_content_fixes()
        self.assertEqual(self.edict.count_content_fixes(), 0)

    def test_language_filter(self):
        from agentchanti.kb.global_kb.error_dict import ContentFix
        self.edict.bulk_insert_content_fixes([
            ContentFix(name="all-fix", file_glob="*.css",
                       content_pattern="a", replacement="b", language="all"),
            ContentFix(name="py-fix", file_glob="*.py",
                       content_pattern="c", replacement="d", language="python"),
        ])
        fixes = self.edict.get_content_fixes(language="python")
        names = {f.name for f in fixes}
        self.assertIn("all-fix", names)
        self.assertIn("py-fix", names)

        fixes_js = self.edict.get_content_fixes(language="javascript")
        names_js = {f.name for f in fixes_js}
        self.assertIn("all-fix", names_js)
        self.assertNotIn("py-fix", names_js)

    def test_compiled_flags(self):
        import re
        from agentchanti.kb.global_kb.error_dict import ContentFix
        fix = ContentFix(
            name="t", file_glob="*", content_pattern="x",
            flags="MULTILINE, IGNORECASE",
        )
        self.assertEqual(fix.compiled_flags(), re.MULTILINE | re.IGNORECASE)

    def test_upsert_on_duplicate_name(self):
        """INSERT OR REPLACE updates existing rule by name."""
        from agentchanti.kb.global_kb.error_dict import ContentFix
        self.edict.bulk_insert_content_fixes([
            ContentFix(name="rule-1", file_glob="*.css",
                       content_pattern="old", replacement="v1"),
        ])
        self.edict.bulk_insert_content_fixes([
            ContentFix(name="rule-1", file_glob="*.css",
                       content_pattern="old", replacement="v2"),
        ])
        fixes = self.edict.get_content_fixes()
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0].replacement, "v2")


# ---------------------------------------------------------------------------
# Seeder tests
# ---------------------------------------------------------------------------


class TestSeeder(unittest.TestCase):
    """Tests for ``kb.global.seeder``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_seed_no_embed(self):
        """seed(embed=False) populates errors.db and writes .md files."""
        from agentchanti.kb.global_kb.seeder import seed
        from agentchanti.kb.global_kb.error_dict import ErrorDict

        summary = seed(embed=False, base_dir=self.tmpdir)

        # Check errors.db was populated
        self.assertEqual(summary["errors_seeded"], 81)
        self.assertGreaterEqual(summary["content_fixes_seeded"], 1)
        self.assertEqual(summary["docs_seeded"], 17)  # 3+0+5+5 (2 generic behavioral removed, +1 python test gen)
        self.assertEqual(summary["chunks_embedded"], 0)

        # Verify error counts per language
        db_path = os.path.join(self.tmpdir, "core", "errors.db")
        edict = ErrorDict(db_path)
        counts = edict.count_by_language()
        expected_per_lang = {
            "python": 18,  # 13 base + 2 django + 1 added by user + 2 new (IndexError deque, AttributeError function shadow)
            "javascript": 35,  # 5 base + 2 npm + 5 react-testing + 1 waitFor-assumed-content + 1 nested-router + 1 assumed-aria + 1 queryByRole-null + 1 rerender-memoryrouter + 2 vitest + 1 no-suite + 1 vite-resolve + 1 component-type-invalid + 1 suspense-fallback-sync-query + 1 empty-root + 1 css-brittle + 5 new
            "typescript": 5, "java": 5, "go": 5, "rust": 5, "csharp": 5,
        }
        for lang, expected in expected_per_lang.items():
            self.assertEqual(counts.get(lang, 0), expected,
                             f"Expected {expected} errors for {lang}")

    def test_md_files_have_frontmatter(self):
        """All seeded .md files contain valid frontmatter."""
        from agentchanti.kb.global_kb.seeder import seed

        seed(embed=False, base_dir=self.tmpdir)
        registry_dir = os.path.join(self.tmpdir, "registry")

        for dirpath, _, filenames in os.walk(registry_dir):
            for fname in filenames:
                if not fname.endswith(".md"):
                    continue
                filepath = os.path.join(dirpath, fname)
                with open(filepath, encoding="utf-8") as fh:
                    content = fh.read()
                self.assertTrue(
                    content.startswith("---"),
                    f"{filepath} missing frontmatter",
                )
                parts = content.split("---", 2)
                self.assertGreaterEqual(len(parts), 3,
                    f"{filepath} has incomplete frontmatter")
                # Check required fields
                fm = parts[1]
                for field in ("title:", "category:", "tags:", "version:"):
                    self.assertIn(field, fm,
                        f"{filepath} missing frontmatter field: {field}")

    def test_chunk_markdown(self):
        """_chunk_markdown splits correctly."""
        from agentchanti.kb.global_kb.seeder import _chunk_markdown

        text = "## Section 1\nShort.\n\n## Section 2\nAlso short."
        # Both sections are < 100 chars each, so they get merged
        chunks = _chunk_markdown(text, "Test Title", min_size=100)
        self.assertGreaterEqual(len(chunks), 1)
        # Every chunk should contain the title
        for chunk in chunks:
            self.assertIn("Test Title", chunk)

    def test_chunk_markdown_splits_large(self):
        """_chunk_markdown splits sections exceeding max_size."""
        from agentchanti.kb.global_kb.seeder import _chunk_markdown

        # Create a large section with paragraph breaks so it can be split
        paragraphs = "\n\n".join(["word " * 30 for _ in range(10)])
        large_text = "## Big Section\n\n" + paragraphs
        chunks = _chunk_markdown(large_text, "Big", min_size=50, max_size=300)
        self.assertGreater(len(chunks), 1)


# ---------------------------------------------------------------------------
# Store tests (offline mode — no Qdrant)
# ---------------------------------------------------------------------------


class TestGlobalKBStore(unittest.TestCase):
    """Tests for ``kb.global.store.GlobalKBStore``."""

    @classmethod
    def setUpClass(cls):
        """Seed the DB once for all store tests in a temp dir."""
        cls._tmpdir = tempfile.mkdtemp()
        from agentchanti.kb.global_kb.seeder import seed
        seed(embed=False, base_dir=cls._tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _store(self):
        from agentchanti.kb.global_kb.store import GlobalKBStore
        return GlobalKBStore(base_dir=self._tmpdir)

    def test_search_errors_exact(self):
        """search_errors finds NullPointerException for Java."""
        store = self._store()
        results = store.search_errors("NullPointerException", language="java")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].error_type, "NullPointerException")
        self.assertEqual(results[0].language, "java")

    def test_search_errors_regex(self):
        """search_errors finds Python AttributeError via regex."""
        store = self._store()
        results = store.search_errors(
            "AttributeError: 'NoneType' object has no attribute 'foo'",
            language="python",
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].error_type, "AttributeError")

    def test_search_errors_no_match(self):
        """search_errors returns empty for unknown error."""
        store = self._store()
        results = store.search_errors("SomeCompletelyUnknownError12345")
        self.assertEqual(len(results), 0)

    def test_fallback_file_search(self):
        """search() falls back to file search when Qdrant is unavailable."""
        store = self._store()
        # This will fail Qdrant and use fallback
        results = store.search("error handling best practices")
        # Should find the error-handling-best-practices.md doc
        self.assertGreater(len(results), 0)
        titles = [r.title for r in results]
        self.assertTrue(
            any("Error Handling" in t for t in titles),
            f"Expected 'Error Handling' in results, got: {titles}",
        )

    def test_fallback_search_category_filter(self):
        """Fallback search respects category filter."""
        store = self._store()
        results = store.search("qdrant", categories=["adr"])
        for r in results:
            self.assertEqual(r.category, "adr")

    def test_get_behavioral_instructions(self):
        """get_behavioral_instructions returns behavioral docs."""
        store = self._store()
        # Use a framework-specific query to match remaining behavioral docs
        # (generic code-review/error-analysis instructions were removed)
        results = store.get_behavioral_instructions("React component testing")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.category, "behavioral")


# ---------------------------------------------------------------------------
# Updater tests
# ---------------------------------------------------------------------------


class TestUpdater(unittest.TestCase):
    """Tests for ``kb.global.updater`` utility functions."""

    def test_parse_semver(self):
        """_parse_semver parses correctly."""
        from agentchanti.kb.global_kb.updater import _parse_semver
        self.assertEqual(_parse_semver("1.2.3"), (1, 2, 3))
        self.assertEqual(_parse_semver("v2.0.0"), (2, 0, 0))
        self.assertGreater(_parse_semver("1.1.0"), _parse_semver("1.0.9"))

    def test_load_local_manifest(self):
        """_load_local_manifest reads the core manifest."""
        from agentchanti.kb.global_kb.updater import _load_local_manifest
        manifest = _load_local_manifest()
        self.assertIn("version", manifest)
        self.assertIn("categories", manifest)

    def test_get_version(self):
        """get_version returns a version string."""
        from agentchanti.kb.global_kb.updater import get_version
        version = get_version()
        self.assertIsInstance(version, str)
        self.assertRegex(version, r"\d+\.\d+\.\d+")

    def test_get_manifest_info(self):
        """get_manifest_info returns full manifest dict."""
        from agentchanti.kb.global_kb.updater import get_manifest_info
        info = get_manifest_info()
        self.assertIn("version", info)
        self.assertIn("categories", info)
        self.assertIsInstance(info["categories"], list)

    def test_check_for_updates_no_owner(self):
        """check_for_updates gracefully handles nonexistent repo."""
        from agentchanti.kb.global_kb.updater import check_for_updates
        status = check_for_updates("nonexistent-owner-xyz", "nonexistent-repo-xyz")
        # Should not crash; update_available should be False
        self.assertFalse(status.update_available)
        self.assertIsInstance(status.current_version, str)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLIParsing(unittest.TestCase):
    """Tests that Phase 3 CLI subcommands parse correctly."""

    def _parse(self, argv: list[str]):
        from agentchanti.kb.cli import _build_parser
        parser = _build_parser()
        return parser.parse_args(argv)

    def test_seed_command(self):
        args = self._parse(["seed"])
        self.assertEqual(args.kb_cmd, "seed")
        self.assertFalse(args.no_embed)

    def test_seed_no_embed(self):
        args = self._parse(["seed", "--no-embed"])
        self.assertTrue(args.no_embed)

    def test_version_command(self):
        args = self._parse(["version"])
        self.assertEqual(args.kb_cmd, "version")

    def test_error_lookup(self):
        args = self._parse(["error-lookup", "NullPointerException"])
        self.assertEqual(args.kb_cmd, "error-lookup")
        self.assertEqual(args.message, "NullPointerException")
        self.assertIsNone(args.language)

    def test_error_lookup_with_language(self):
        args = self._parse(["error-lookup", "TypeError", "--language", "python"])
        self.assertEqual(args.message, "TypeError")
        self.assertEqual(args.language, "python")

    def test_global_search(self):
        args = self._parse(["global-search", "error handling"])
        self.assertEqual(args.kb_cmd, "global-search")
        self.assertEqual(args.query, "error handling")

    def test_global_search_with_category(self):
        args = self._parse(["global-search", "qdrant", "--category", "adr"])
        self.assertEqual(args.category, "adr")

    def test_update_check(self):
        args = self._parse(["update", "--check"])
        self.assertEqual(args.kb_cmd, "update")
        self.assertTrue(args.check)

    def test_update_category(self):
        args = self._parse(["update", "--category", "errors"])
        self.assertEqual(args.category, "errors")

    def test_clean_command(self):
        args = self._parse(["clean"])
        self.assertEqual(args.kb_cmd, "clean")
        self.assertFalse(args.force)

    def test_clean_force(self):
        args = self._parse(["clean", "--force"])
        self.assertTrue(args.force)


# ---------------------------------------------------------------------------
# Updater clean tests
# ---------------------------------------------------------------------------


class TestUpdaterClean(unittest.TestCase):
    """Tests for ``kb.global_kb.updater.clean``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_clean_removes_files(self):
        """clean() removes errors.db, global_kb.db, manifest, and registry."""
        from agentchanti.kb.global_kb.seeder import seed
        from agentchanti.kb.global_kb.updater import clean

        core_dir = os.path.join(self.tmpdir, "core")
        registry_dir = os.path.join(self.tmpdir, "registry")

        # Seed first so there's something to clean
        seed(embed=False, base_dir=self.tmpdir)
        self.assertTrue(os.path.isfile(os.path.join(core_dir, "errors.db")))
        self.assertTrue(os.path.isdir(registry_dir))

        summary = clean(base_dir=self.tmpdir)
        self.assertGreater(summary["files_removed"], 0)
        self.assertGreater(summary["dbs_removed"], 0)
        # Registry dir may still exist (with .gitignore), but no .md files
        md_count = 0
        if os.path.isdir(registry_dir):
            for _, _, fnames in os.walk(registry_dir):
                md_count += sum(1 for f in fnames if f.endswith(".md"))
        self.assertEqual(md_count, 0)
        self.assertFalse(os.path.isfile(os.path.join(core_dir, "errors.db")))

    def test_clean_idempotent(self):
        """clean() on already-clean state does not crash."""
        from agentchanti.kb.global_kb.updater import clean

        # Clean twice — second call should succeed silently
        clean(base_dir=self.tmpdir)
        summary = clean(base_dir=self.tmpdir)
        self.assertEqual(summary["files_removed"], 0)
        self.assertEqual(summary["dbs_removed"], 0)

    def test_seed_after_clean(self):
        """seed() works correctly after clean() — full round trip."""
        from agentchanti.kb.global_kb.seeder import seed
        from agentchanti.kb.global_kb.updater import clean

        clean(base_dir=self.tmpdir)
        summary = seed(embed=False, base_dir=self.tmpdir)
        self.assertEqual(summary["errors_seeded"], 81)
        self.assertGreaterEqual(summary["docs_seeded"], 1)


# ---------------------------------------------------------------------------
# Updater _apply_update returns md_files tests
# ---------------------------------------------------------------------------


class TestApplyUpdateReturnsMdFiles(unittest.TestCase):
    """Tests that _apply_update returns the list of copied md files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()       # source dir for update
        self.kb_dir = tempfile.mkdtemp()        # isolated KB base dir

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.kb_dir, ignore_errors=True)

    def test_apply_update_returns_md_files(self):
        """_apply_update returns (count, md_files) with correct metadata."""
        from agentchanti.kb.global_kb.updater import _apply_update

        # Create a fake update directory with a docs/guide.md
        docs_dir = os.path.join(self.tmpdir, "docs")
        os.makedirs(docs_dir)
        md_content = (
            "---\n"
            "title: Test Guide\n"
            "category: doc\n"
            "tags: test,guide\n"
            "---\n"
            "# Test Guide\n"
            "Some content.\n"
        )
        with open(os.path.join(docs_dir, "test-guide.md"), "w") as fh:
            fh.write(md_content)

        count, md_files = _apply_update(
            self.tmpdir, categories=["docs"], base_dir=self.kb_dir,
        )
        self.assertEqual(count, 1)
        self.assertEqual(len(md_files), 1)

        path, category, title = md_files[0]
        self.assertTrue(path.endswith("test-guide.md"))
        self.assertEqual(category, "doc")
        self.assertEqual(title, "Test Guide")


# ---------------------------------------------------------------------------
# Seed preserves kb update files
# ---------------------------------------------------------------------------


class TestSeedPreservesUpdateFiles(unittest.TestCase):
    """Verify that kb seed preserves and re-embeds files from kb update."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Place a fake "kb update" file in registry/docs inside temp dir
        self.docs_dir = os.path.join(self.tmpdir, "registry", "docs")
        os.makedirs(self.docs_dir, exist_ok=True)
        self.test_file = os.path.join(self.docs_dir, "tailwindcss-v4-setup-guide.md")
        md_content = (
            "---\n"
            "title: Tailwind CSS v4 Setup Guide\n"
            "category: doc\n"
            "tags: tailwind,css,v4\n"
            "version: 1.0.0\n"
            "---\n"
            "# Tailwind CSS v4\n"
            "Use @import 'tailwindcss';\n"
        )
        with open(self.test_file, "w") as fh:
            fh.write(md_content)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_seed_preserves_update_file(self):
        """seed(embed=False) does not delete files from kb update."""
        from agentchanti.kb.global_kb.seeder import seed

        seed(embed=False, base_dir=self.tmpdir)
        self.assertTrue(
            os.path.isfile(self.test_file),
            "kb seed deleted a file from kb update",
        )

    def test_seed_includes_update_file_in_all_md_files(self):
        """collect_all_registry_md_files() discovers kb update files."""
        from agentchanti.kb.global_kb.seeder import (
            seed, collect_all_registry_md_files,
        )

        seed(embed=False, base_dir=self.tmpdir)
        registry_dir = os.path.join(self.tmpdir, "registry")

        # collect_all_registry_md_files with no exclusions returns everything
        all_files = collect_all_registry_md_files(registry_dir=registry_dir)
        filenames = [os.path.basename(p) for p, _, _ in all_files]
        self.assertIn(
            "tailwindcss-v4-setup-guide.md", filenames,
            "kb update file not discoverable for embedding",
        )

    def test_collect_excludes_given_paths(self):
        """collect_all_registry_md_files() respects exclude_paths."""
        from agentchanti.kb.global_kb.seeder import (
            seed, collect_all_registry_md_files,
        )

        seed(embed=False, base_dir=self.tmpdir)
        registry_dir = os.path.join(self.tmpdir, "registry")

        # Exclude the test file
        excluded = collect_all_registry_md_files(
            exclude_paths={self.test_file},
            registry_dir=registry_dir,
        )
        filenames = [os.path.basename(p) for p, _, _ in excluded]
        self.assertNotIn("tailwindcss-v4-setup-guide.md", filenames)


if __name__ == "__main__":
    unittest.main()


class TestLanguageScopedDocs(unittest.TestCase):
    """`language: "all"` on framework-specific docs disabled every filter.

    Live evidence, a Python/Pygame run: "React Component Export
    Instructions" was injected into EVERY step, alongside Django and
    Vitest material, at 2.8k-3.9k KB tokens per step. The filter reads
    `doc_lang != "all" and doc_lang != language`, and all six behavioral
    docs claimed "all" -- so it could never fire. React docs had to claim
    "all" because a single value cannot cover javascript AND typescript.
    """

    def test_all_still_matches_every_project(self):
        self.assertTrue(_language_matches("all", "python"))
        self.assertTrue(_language_matches("all", "javascript"))

    def test_missing_or_blank_language_is_permissive(self):
        self.assertTrue(_language_matches("", "python"))
        self.assertTrue(_language_matches(None, "python"))

    def test_single_language_scopes_correctly(self):
        self.assertTrue(_language_matches("python", "python"))
        self.assertFalse(_language_matches("python", "javascript"))

    def test_a_list_covers_a_family(self):
        for lang in ("javascript", "typescript"):
            self.assertTrue(
                _language_matches("javascript, typescript", lang))
        self.assertFalse(
            _language_matches("javascript, typescript", "python"))

    def test_case_and_spacing_are_ignored(self):
        self.assertTrue(_language_matches("  JavaScript ,TypeScript ",
                                          "typescript"))

    def test_shipped_react_docs_no_longer_reach_python_projects(self):
        """The registry metadata itself must be right, not just the helper.

        `agentchanti/kb/global_kb/registry/` is gitignored and pulled at
        runtime, so it is absent in CI. Skip there rather than scan an empty
        directory: an empty scan finds no offenders, which would report
        green while asserting nothing.
        """
        import glob
        import os
        registry = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(
                agentchanti.kb.global_kb.store.__file__))),
            "global_kb", "registry")
        paths = sorted(glob.glob(os.path.join(registry, "**", "*.md"),
                                 recursive=True))
        if not paths:
            self.skipTest("KB registry is gitignored/pulled at runtime — "
                          "not present here")
        offenders = []
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                head = fh.read(600)
            name = os.path.basename(path)
            if not any(t in name for t in ("react", "vitest", "npm",
                                           "threejs", "testing-library")):
                continue
            m = re.search(r'^language:\s*"([^"]*)"', head, re.M)
            if m and _language_matches(m.group(1), "python"):
                offenders.append((name, m.group(1)))
        self.assertEqual(offenders, [],
                         f"JS/TS-only docs still reachable from Python: "
                         f"{offenders}")

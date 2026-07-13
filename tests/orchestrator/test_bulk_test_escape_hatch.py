"""Regression tests for the BulkTest escape-hatch source-fix path.

When a test failure can't be resolved by rewriting the test file (the
root cause is in the source under test), the BulkTest fix loop's
"source-file protection" guard would block every fix and burn all
retries on test-only rewrites that can't help. The escape hatch lets
the loop apply ONE narrowly-scoped source-file fix when it detects the
loop is making no progress.

These tests pin the safety rails so the relaxation can't silently
regress into "the LLM rewrites whatever it wants in production source."

See: bugfix branch — pipeline.py:_attempt_targeted_source_fix.
"""
import unittest
from unittest.mock import MagicMock

from agentchanti.orchestrator.pipeline import (
    _attempt_targeted_source_fix,
    _diff_stats,
    _error_signature,
    _extract_stack_trace_files,
    _extract_top_level_exports,
    _is_additive_source_fix,
    _is_safe_source_fix,
    _should_trigger_escape_hatch,
)


class _FakeRoute:
    """Tiny stand-in for ErrorRoute — only needs source_type."""
    def __init__(self, source_type):
        self.source_type = source_type


# ── Helper unit tests ─────────────────────────────────────────────────────


class TestErrorSignature(unittest.TestCase):
    """Error signatures must collapse cosmetic churn so repeating
    failures can be detected across attempts."""

    def test_identical_errors_have_same_signature(self):
        err = "TypeError: Cannot read property 'foo' of null"
        self.assertEqual(_error_signature(err), _error_signature(err))

    def test_line_numbers_stripped(self):
        e1 = "at LinkWithRef src/Header.jsx:42:11"
        e2 = "at LinkWithRef src/Header.jsx:99:1"
        self.assertEqual(_error_signature(e1), _error_signature(e2))

    def test_test_durations_stripped(self):
        e1 = "× test failed 13ms"
        e2 = "× test failed 1234ms"
        self.assertEqual(_error_signature(e1), _error_signature(e2))

    def test_windows_paths_stripped(self):
        e1 = "Error in C:\\Users\\foo\\src\\App.jsx"
        e2 = "Error in C:\\Users\\bar\\src\\App.jsx"
        self.assertEqual(_error_signature(e1), _error_signature(e2))

    def test_distinct_errors_differ(self):
        a = "TypeError: x is null"
        b = "ReferenceError: y is not defined"
        self.assertNotEqual(_error_signature(a), _error_signature(b))

    def test_empty_error_returns_empty(self):
        self.assertEqual(_error_signature(""), "")
        self.assertEqual(_error_signature(None), "")


class TestExtractStackTraceFiles(unittest.TestCase):
    """Stack-trace parsing decides which files the escape hatch is
    allowed to touch — getting it wrong either lets the LLM modify
    arbitrary files or blocks every legitimate fix."""

    def test_js_parens_format(self):
        err = "at LinkWithRef (src/components/Header.jsx:42:11)"
        self.assertEqual(
            _extract_stack_trace_files(err),
            {"src/components/Header.jsx"},
        )

    def test_js_bare_format(self):
        err = "FAIL src/App.test.jsx:5:3"
        self.assertIn("src/App.test.jsx", _extract_stack_trace_files(err))

    def test_python_file_format(self):
        err = 'File "src/foo.py", line 42, in bar'
        self.assertEqual(_extract_stack_trace_files(err), {"src/foo.py"})

    def test_node_modules_filtered(self):
        err = (
            "at LinkWithRef "
            "node_modules/react-router/dist/development/chunk.mjs:10182:11\n"
            "at App src/components/Header.jsx:42:11"
        )
        files = _extract_stack_trace_files(err)
        self.assertNotIn(
            "node_modules/react-router/dist/development/chunk.mjs", files)
        self.assertIn("src/components/Header.jsx", files)

    def test_site_packages_filtered(self):
        err = 'File "/venv/lib/site-packages/django/foo.py", line 1'
        self.assertEqual(_extract_stack_trace_files(err), set())

    def test_windows_backslashes_normalised(self):
        err = "at Foo (src\\components\\Header.jsx:1:1)"
        self.assertEqual(
            _extract_stack_trace_files(err),
            {"src/components/Header.jsx"},
        )

    def test_real_log_scenario(self):
        """The exact stack from the user's failing session."""
        err = (
            "TypeError: Cannot destructure property 'basename' of "
            "'React10.useContext(...)' as it is null.\n"
            " ❯ LinkWithRef node_modules/react-router/dist/development/"
            "chunk-QFMPRPBF.mjs:10182:11\n"
            " ❯ Object.react_stack_bottom_frame node_modules/react-dom/"
            "cjs/react-dom-client.development.js:25904:20\n"
            " ❯ render src/components/Header.jsx:42:11\n"
        )
        files = _extract_stack_trace_files(err)
        self.assertIn("src/components/Header.jsx", files)
        self.assertFalse(
            any("node_modules" in f for f in files),
            f"node_modules paths leaked through: {files}",
        )


class TestExtractTopLevelExports(unittest.TestCase):
    """Export extraction must catch the file's public API so the
    escape hatch can verify nothing was dropped."""

    def test_js_default_function_export(self):
        src = "export default function Header() { return null }"
        self.assertEqual(_extract_top_level_exports(src), {"Header"})

    def test_js_default_identifier_export(self):
        src = "function Header() {}\nexport default Header;"
        self.assertEqual(_extract_top_level_exports(src), {"Header"})

    def test_js_named_const_export(self):
        src = "export const helper = () => 42"
        self.assertEqual(_extract_top_level_exports(src), {"helper"})

    def test_js_named_block_export(self):
        src = "export { Foo, Bar as Baz, Qux }"
        self.assertEqual(
            _extract_top_level_exports(src),
            {"Foo", "Bar", "Qux"},
        )

    def test_cjs_module_exports(self):
        src = "module.exports.Foo = function() {}"
        self.assertEqual(_extract_top_level_exports(src), {"Foo"})

    def test_python_def_and_class(self):
        src = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
        self.assertEqual(_extract_top_level_exports(src), {"foo", "Bar"})

    def test_empty_content(self):
        self.assertEqual(_extract_top_level_exports(""), set())


class TestDiffStats(unittest.TestCase):
    """The shared diff math powers both the additive guard (10%) and
    the escape-hatch cap (30%)."""

    def test_pure_addition(self):
        orig = "a\nb\nc\n"
        new = "a\nb\nc\nd\n"
        stats = _diff_stats(orig, new)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["removed"], 0)
        self.assertEqual(stats["changed"], 0)

    def test_pure_replace(self):
        orig = "a\nb\nc\n"
        new = "a\nB\nc\n"
        stats = _diff_stats(orig, new)
        self.assertEqual(stats["changed"], 1)
        self.assertEqual(stats["removed"], 0)

    def test_pure_deletion(self):
        orig = "a\nb\nc\n"
        new = "a\nc\n"
        stats = _diff_stats(orig, new)
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(stats["added"], 0)

    def test_ratio_against_orig_length(self):
        orig = "\n".join(str(i) for i in range(10)) + "\n"
        new = "\n".join(str(i) for i in range(10)) + "\nNEW\n"
        stats = _diff_stats(orig, new)
        self.assertAlmostEqual(stats["ratio"], 0.1, places=2)


class TestIsAdditiveSourceFix(unittest.TestCase):
    """Pin the existing additive guard — refactored to share _diff_stats."""

    def _make_memory(self, files):
        mem = MagicMock()
        mem.get.side_effect = lambda fp: files.get(fp)
        return mem

    def test_pure_addition_within_cap(self):
        # 10 lines, add 1 → 10% delta, should pass
        orig = "\n".join(f"line{i}" for i in range(10))
        new = orig + "\nline10"
        mem = self._make_memory({"f.js": orig})
        self.assertTrue(_is_additive_source_fix("f.js", new, mem))

    def test_replace_blocked(self):
        # Removing a line is not additive
        orig = "\n".join(f"line{i}" for i in range(10))
        new = "\n".join(f"line{i}" for i in range(9))  # dropped line9
        mem = self._make_memory({"f.js": orig})
        self.assertFalse(_is_additive_source_fix("f.js", new, mem))

    def test_too_large_blocked(self):
        # 30% of 10 lines = 3 → over the 10% cap
        orig = "\n".join(f"line{i}" for i in range(10))
        new = orig + "\nA\nB\nC\nD"
        mem = self._make_memory({"f.js": orig})
        self.assertFalse(_is_additive_source_fix("f.js", new, mem))


class TestIsSafeSourceFix(unittest.TestCase):
    """The relaxed BulkTest gate: judges the diff, not the response
    format, so full-file responses carrying a small real fix pass while
    wholesale rewrites and export-dropping changes stay blocked."""

    def _make_memory(self, files):
        mem = MagicMock()
        mem.get.side_effect = lambda fp: files.get(fp)
        return mem

    def test_one_line_insert_passes(self):
        # The observed case: {% load static %} added to a template —
        # blocked by the additive gate only because the model responded
        # in full-file format.
        orig = "\n".join(f"<div>line{i}</div>" for i in range(20))
        new = "{% load static %}\n" + orig
        mem = self._make_memory({"base.html": orig})
        self.assertTrue(_is_safe_source_fix("base.html", new, mem))

    def test_moderate_replace_passes_where_additive_blocks(self):
        # 4 of 20 lines rewritten (20%) — additive gate rejects
        # (delta > 10%), the relaxed gate accepts (≤30%).
        orig_lines = [f"line{i}" for i in range(20)]
        new_lines = list(orig_lines)
        for i in range(4):
            new_lines[i] = f"fixed{i}"
        orig, new = "\n".join(orig_lines), "\n".join(new_lines)
        mem = self._make_memory({"f.py": orig})
        self.assertFalse(_is_additive_source_fix("f.py", new, mem))
        self.assertTrue(_is_safe_source_fix("f.py", new, mem))

    def test_wholesale_rewrite_blocked(self):
        orig = "\n".join(f"line{i}" for i in range(20))
        new = "\n".join(f"other{i}" for i in range(20))
        mem = self._make_memory({"f.py": orig})
        self.assertFalse(_is_safe_source_fix("f.py", new, mem))

    def test_dropped_export_blocked(self):
        orig = ("def home(request):\n    return 1\n\n"
                + "\n".join(f"# pad{i}" for i in range(20)))
        new = ("def dashboard(request):\n    return 1\n\n"
               + "\n".join(f"# pad{i}" for i in range(20)))
        mem = self._make_memory({"views.py": orig})
        self.assertFalse(_is_safe_source_fix("views.py", new, mem))

    def test_new_file_blocked(self):
        mem = self._make_memory({})
        self.assertFalse(_is_safe_source_fix(
            "no/such/file_xyz.py", "def f():\n    pass", mem))

    def test_empty_new_content_blocked(self):
        orig = "\n".join(f"line{i}" for i in range(10))
        mem = self._make_memory({"f.py": orig})
        self.assertFalse(_is_safe_source_fix("f.py", "   \n", mem))

    def test_disk_fallback_when_memory_misses(self):
        import os
        import tempfile
        orig = "\n".join(f"line{i}" for i in range(20))
        fd, path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(orig)
            mem = self._make_memory({})  # not tracked by this run
            self.assertTrue(_is_safe_source_fix(path, orig + "\nextra", mem))
        finally:
            os.unlink(path)


# ── Escape-hatch helper ───────────────────────────────────────────────────


def _make_executor(parsed_files: dict | None = None):
    """Build a mock executor that returns *parsed_files* from
    parse_code_blocks (and the fuzzy fallback)."""
    ex = MagicMock()
    ex.parse_code_blocks.return_value = parsed_files or {}
    ex.parse_code_blocks_fuzzy.return_value = {}
    return ex


def _make_coder(response: str = "fix response"):
    coder = MagicMock()
    coder.llm_client.generate_response.return_value = response
    return coder


def _make_memory(files: dict | None = None):
    files = files or {}
    mem = MagicMock()
    mem.get.side_effect = lambda fp: files.get(fp)
    return mem


# Real-world Header.jsx — small, with one default export
HEADER_JSX_ORIG = """\
import { useState } from 'react'
import { Link } from 'react-router-dom'
import PropTypes from 'prop-types'

export default function Header({ title = 'BrandLogo' }) {
  const [open, setOpen] = useState(false)
  return (
    <header>
      <Link to="#home">{title}</Link>
      <button onClick={() => setOpen(true)}>Menu</button>
    </header>
  )
}

Header.propTypes = {
  title: PropTypes.string,
}
"""

# Minimal valid fix: drop react-router-dom import, swap Link → <a>
HEADER_JSX_FIXED = """\
import { useState } from 'react'
import PropTypes from 'prop-types'

export default function Header({ title = 'BrandLogo' }) {
  const [open, setOpen] = useState(false)
  return (
    <header>
      <a href="#home">{title}</a>
      <button onClick={() => setOpen(true)}>Menu</button>
    </header>
  )
}

Header.propTypes = {
  title: PropTypes.string,
}
"""

# Same file but the LLM dropped the default export
HEADER_JSX_BROKEN_EXPORTS = """\
import { useState } from 'react'

function Header({ title }) {
  return <header>{title}</header>
}
"""

# Stack trace pointing at the source file under test
SAMPLE_ERROR = (
    "TypeError: Cannot destructure property 'basename' of "
    "'React10.useContext(...)' as it is null.\n"
    " ❯ LinkWithRef node_modules/react-router/dist/development/"
    "chunk-QFMPRPBF.mjs:10182:11\n"
    " ❯ render src/components/Header.jsx:9:5\n"
)


class TestAttemptTargetedSourceFix(unittest.TestCase):
    """Pin the safety rails on the escape-hatch helper. Each test
    represents a class of LLM misbehaviour the rails must catch."""

    def _call(self, *, parsed_files, files=None, error=SAMPLE_ERROR):
        return _attempt_targeted_source_fix(
            test_path="src/components/Header.test.jsx",
            file_error=error,
            source_ctx="(source ctx)",
            coder=_make_coder("(llm response)"),
            executor=_make_executor(parsed_files),
            memory=_make_memory(files or {}),
            subproject_cwd=None,
            lang_tag="jsx",
            task="implement responsive header",
        )

    def test_happy_path_minimal_source_fix(self):
        result = self._call(
            parsed_files={"src/components/Header.jsx": HEADER_JSX_FIXED},
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
        )
        self.assertIsNotNone(result)
        self.assertEqual(set(result), {"src/components/Header.jsx"})
        self.assertIn("href=\"#home\"", result["src/components/Header.jsx"])

    def test_no_stack_trace_files_returns_none(self):
        """If the error mentions no source files, the hatch can't
        target anything and must abort."""
        result = self._call(
            parsed_files={"src/components/Header.jsx": HEADER_JSX_FIXED},
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
            error="generic failure with no stack trace",
        )
        self.assertIsNone(result)

    def test_multi_file_response_rejected(self):
        """LLM that returns >1 file is going nuclear — reject."""
        result = self._call(
            parsed_files={
                "src/components/Header.jsx": HEADER_JSX_FIXED,
                "src/components/HeroBanner.jsx": "// rewrite",
            },
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
        )
        self.assertIsNone(result)

    def test_test_file_response_rejected(self):
        """The hatch is source-only; rewriting the test belongs to
        the regular test-only retry path."""
        result = self._call(
            parsed_files={"src/components/Header.test.jsx": "// new test"},
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
        )
        self.assertIsNone(result)

    def test_unrelated_file_rejected(self):
        """LLM tries to fix a file the error doesn't reference —
        out of scope, reject."""
        result = self._call(
            parsed_files={"src/utils/helpers.js": "// unrelated rewrite"},
            files={
                "src/components/Header.jsx": HEADER_JSX_ORIG,
                "src/utils/helpers.js": "// orig",
            },
        )
        self.assertIsNone(result)

    def test_dropped_export_rejected(self):
        """LLM drops the default export — would break every importer."""
        result = self._call(
            parsed_files={
                "src/components/Header.jsx": HEADER_JSX_BROKEN_EXPORTS,
            },
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
        )
        self.assertIsNone(result)

    def test_oversized_diff_rejected(self):
        """LLM rewrote >30% of the file — reject as too risky."""
        # Original has ~17 lines; 30% = ~5 lines. Rewrite the body
        # entirely while keeping the export name.
        rewrite = (
            "import React from 'react'\n"
            "export default function Header() {\n"
            "  // entirely new implementation\n"
            "  const x = 1\n"
            "  const y = 2\n"
            "  const z = 3\n"
            "  return null\n"
            "}\n"
        )
        result = self._call(
            parsed_files={"src/components/Header.jsx": rewrite},
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
        )
        self.assertIsNone(result)

    def test_unparseable_response_returns_none(self):
        result = self._call(
            parsed_files={},  # parser found nothing
            files={"src/components/Header.jsx": HEADER_JSX_ORIG},
        )
        self.assertIsNone(result)

    def test_missing_original_content_returns_none(self):
        """Can't run safety checks against a file we can't read."""
        result = self._call(
            parsed_files={"src/components/Header.jsx": HEADER_JSX_FIXED},
            files={},  # not in memory and not on disk
        )
        self.assertIsNone(result)


# ── Trigger function ──────────────────────────────────────────────────────


class TestShouldTriggerEscapeHatch(unittest.TestCase):
    """Pin the trigger preconditions. The trigger is the single
    decision point that decides whether the escape hatch fires —
    every guard added here is the only thing standing between the
    LLM and the project's source files."""

    def _ok_kwargs(self, **overrides):
        """Baseline kwargs that satisfy every precondition."""
        kwargs = dict(
            used_escape_hatch=False,
            did_test_only_retry=True,
            error_sig_history=["sig_a", "sig_a"],
            route=_FakeRoute("code"),
        )
        kwargs.update(overrides)
        return kwargs

    def test_baseline_fires(self):
        """The happy path — every condition met."""
        self.assertTrue(_should_trigger_escape_hatch(**self._ok_kwargs()))

    def test_already_used_blocks(self):
        """One shot per test file — second invocation must skip."""
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(used_escape_hatch=True)))

    def test_test_only_retry_not_yet_done_blocks(self):
        """Test-only retry must be tried first (cheaper, safer)."""
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(did_test_only_retry=False)))

    def test_history_too_short_blocks(self):
        """First attempt has only one signature — can't compare yet."""
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(error_sig_history=["sig_a"])))
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(error_sig_history=[])))

    def test_signature_drift_blocks(self):
        """Different errors → loop is making progress, don't escalate."""
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(error_sig_history=["sig_a", "sig_b"])))

    def test_empty_signature_blocks(self):
        """Empty error string → can't trust the comparison."""
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(error_sig_history=["", ""])))

    def test_no_route_blocks(self):
        """ErrorRouter unavailable → don't take risky source action."""
        self.assertFalse(_should_trigger_escape_hatch(
            **self._ok_kwargs(route=None)))

    def test_non_code_route_blocks(self):
        """Hatch is for code bugs only — env / data / web get other
        remedies."""
        for source_type in ("web", "kb", "web+code", "data", "unknown"):
            with self.subTest(source_type=source_type):
                self.assertFalse(_should_trigger_escape_hatch(
                    **self._ok_kwargs(route=_FakeRoute(source_type))))

    def test_three_repeats_still_fires(self):
        """Stable error across 3+ attempts — definitely a stuck loop."""
        self.assertTrue(_should_trigger_escape_hatch(
            **self._ok_kwargs(
                error_sig_history=["sig_a", "sig_a", "sig_a"])))

    def test_only_last_two_matter(self):
        """If the last two match (loop just stalled), fire — even if
        earlier attempts had different errors."""
        self.assertTrue(_should_trigger_escape_hatch(
            **self._ok_kwargs(
                error_sig_history=["sig_x", "sig_y", "sig_a", "sig_a"])))

    def test_fires_on_final_attempt(self):
        """Regression: an earlier version of the trigger had a
        ``fix_attempt < MAX_BULK_TEST_FIX_ATTEMPTS`` guard which
        prevented the hatch from firing on the very attempt it was
        most needed — the last one, after test-only retries had
        demonstrably failed.  The signature of the function does NOT
        take fix_attempt anymore.  This test pins that absence: any
        future attempt to add a budget guard will need to update the
        signature, and these kwargs will then fail to bind."""
        # If a fix_attempt kwarg sneaks back in, this call still works
        # (because the function takes **kwargs in the future?) — guard
        # against that by also asserting the call signature explicitly.
        import inspect
        sig = inspect.signature(_should_trigger_escape_hatch)
        self.assertNotIn(
            "fix_attempt", sig.parameters,
            "fix_attempt budget guard must not be reintroduced — see "
            "the bugfix branch commit message for the failure mode it "
            "caused (hatch could never fire on the final attempt).",
        )
        # And the trigger must say "yes" on a stale-error case.
        self.assertTrue(_should_trigger_escape_hatch(**self._ok_kwargs()))


if __name__ == "__main__":
    unittest.main()

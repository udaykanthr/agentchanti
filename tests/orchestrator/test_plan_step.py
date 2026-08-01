"""Tests for PlanStep parsing, including inline code capture."""

from __future__ import annotations

import unittest

from agentchanti.orchestrator.plan_step import (
    parse_structured_plan, PlanStep, validate_plan,
    fix_import_dependencies,
    is_structured_plan, build_waves,
    plan_looks_truncated,
)


class TestPlanLooksTruncated(unittest.TestCase):

    def test_missing_end_marker_is_truncated(self):
        text = ("==PLAN==\n\n--STEP 1.1 [CMD] depends:none\nInstall\n"
                "> npm i\nproduces: package.json\n\n"
                "--STEP 2.1 [CODE] depends:1.1\nCreate the server and (")
        truncated, reason = plan_looks_truncated(text)
        self.assertTrue(truncated)
        self.assertIn("==END==", reason)

    def test_complete_plan_not_truncated(self):
        text = ("==PLAN==\n\n--STEP 1.1 [CMD] depends:none\nInstall\n"
                "> npm i\nproduces: package.json\n==END==\n")
        steps = parse_structured_plan(text)
        truncated, _ = plan_looks_truncated(text, steps)
        self.assertFalse(truncated)

    def test_last_step_without_body_is_truncated(self):
        # Parses to a CODE step carrying no target/verify/inline body.
        step = PlanStep(id="2.1", step_type="CODE", description="Create X")
        truncated, reason = plan_looks_truncated("some plan ==END==", [step])
        self.assertTrue(truncated)
        self.assertIn("2.1", reason)

    def test_empty_text_not_truncated(self):
        self.assertEqual(plan_looks_truncated(""), (False, ""))


class TestParseStructuredPlan(unittest.TestCase):

    def test_basic_parse(self):
        text = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Install express
> npm install express
produces: package.json

--STEP 2.1 [CODE] depends:1.1
Create server
target: src/server.js
exports: app
imports: none
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].step_type, "CMD")
        self.assertEqual(steps[0].command, "npm install express")
        self.assertEqual(steps[1].step_type, "CODE")
        self.assertEqual(steps[1].target_files, ["src/server.js"])
        self.assertEqual(steps[1].exports, ["app"])

    def test_is_structured_plan(self):
        self.assertTrue(is_structured_plan("--STEP 1.1 [CMD] depends:none\nfoo"))
        self.assertFalse(is_structured_plan("1. Install express\n2. Create server"))


class TestVerifyCmd(unittest.TestCase):
    """Per-step verify: line — the plan-declared acceptance command."""

    def test_bare_verify_line(self):
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create home view
target: main/views.py
exports: home
imports: none
verify: python manage.py test main --noinput
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(steps[0].verify_cmd,
                         "python manage.py test main --noinput")

    def test_gt_prefixed_verify_line(self):
        # Some models copy the CMD "> " prefix onto metadata lines
        text = """
==PLAN==
--STEP 1.1 [TEST] depends:none
Run tests
target: tests/test_app.py
> verify: npm test --silent
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(steps[0].verify_cmd, "npm test --silent")
        self.assertIsNone(steps[0].command)  # not treated as a CMD command

    def test_verify_none_and_absent(self):
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create thing
target: a.py
verify: none

--STEP 1.2 [CODE] depends:none
Create other
target: b.py
==END==
"""
        steps = parse_structured_plan(text)
        self.assertIsNone(steps[0].verify_cmd)
        self.assertIsNone(steps[1].verify_cmd)

    def test_backtick_wrapping_stripped(self):
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create thing
target: a.py
verify: `pytest -q tests/test_a.py`
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(steps[0].verify_cmd, "pytest -q tests/test_a.py")

    def test_serialization_roundtrip(self):
        step = PlanStep(id="1.1", step_type="CODE",
                        verify_cmd="python manage.py check")
        restored = PlanStep.from_dict(step.to_dict())
        self.assertEqual(restored.verify_cmd, "python manage.py check")
        # Absent stays absent (and doesn't bloat the dict)
        bare = PlanStep(id="1.2", step_type="CODE")
        self.assertNotIn("verify_cmd", bare.to_dict())
        self.assertIsNone(PlanStep.from_dict(bare.to_dict()).verify_cmd)


class TestInlineCode(unittest.TestCase):
    """Tests for ---file-content-start--- / ---file-content-end--- parsing."""

    def test_single_target_inline_code(self):
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create PostCSS config
target: postcss.config.mjs
exports: none
imports: none
---file-content-start---
export default {
  plugins: {
    "@tailwindcss/postcss": {},
  },
}
---file-content-end---
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 1)
        self.assertIn("postcss.config.mjs", steps[0].inline_code)
        self.assertIn("@tailwindcss/postcss", steps[0].inline_code["postcss.config.mjs"])

    def test_multi_target_with_file_headers(self):
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create components
target: src/Header.jsx, src/Footer.jsx
exports: Header, Footer
imports: none
---file-content-start---
// Header.jsx
import React from 'react';

export function Header() {
  return <header>Hello</header>;
}

// Footer.jsx
import React from 'react';

export function Footer() {
  return <footer>Bye</footer>;
}
---file-content-end---
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertIn("src/Header.jsx", step.inline_code)
        self.assertIn("src/Footer.jsx", step.inline_code)
        self.assertIn("export function Header", step.inline_code["src/Header.jsx"])
        self.assertIn("export function Footer", step.inline_code["src/Footer.jsx"])

    def test_multi_target_with_python_style_file_headers(self):
        """File-header matching must work for ``#``-style comments too,
        not just ``//``.

        Regression: ``_FILE_COMMENT_RE`` only recognised ``//`` headers,
        so for Python (and Ruby/shell/YAML) steps Strategy 1 silently
        found zero matches and fell through to Strategy 2's blind
        positional zip. When the planner emitted content blocks in a
        different order than the ``target:`` list (as happened for a
        real snake-game plan: target order snake.py/food.py/board.py but
        content order board.py/snake.py/food.py), the positional zip
        scrambled the Snake/Food/board logic across the wrong files —
        board.py's ``random_cell`` ended up under ``snake.py``, and so
        on — producing ImportErrors that looked like source bugs and
        burned a full diagnosis retry loop chasing the wrong root cause.
        """
        text = """
==PLAN==
--STEP 3.1 [CODE] depends:none
Create pure-logic modules
target: pkg/snake.py, pkg/food.py, pkg/board.py
exports: Snake, Food, random_cell
imports: none
---file-content-start---
# pkg/board.py
import random

def random_cell():
    return (0, 0)

# pkg/snake.py
class Snake:
    def __init__(self):
        self.segments = []

# pkg/food.py
class Food:
    def __init__(self):
        self.position = None
---file-content-end---
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertIn("def random_cell", step.inline_code["pkg/board.py"])
        self.assertIn("class Snake", step.inline_code["pkg/snake.py"])
        self.assertIn("class Food", step.inline_code["pkg/food.py"])

    def test_multi_target_with_separate_content_blocks(self):
        """Multi-target step where each target has its OWN ``content:``
        block (one per file).  Real-world planner output for a step that
        creates both ``vite.config.js`` and ``vitest.setup.js`` in one
        go.

        Regression: Strategy 3 fallback used to always assign to
        ``targets[0]``, so the second ``content:`` block silently
        overwrote the first.  vite.config.js ended up holding the
        setup-file content (a single import statement) and the actual
        Vite config was lost — breaking ``vite build`` / ``vite dev``
        even though the test runner's auto-create later masked the
        symptom from the pipeline's success report.
        """
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.2
Create the Vite + Vitest config and the testing-library setup file.
target: myapp/vite.config.js, myapp/vitest.setup.js
exports: default
imports: none
content:
```js
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './vitest.setup.js',
  },
})
```
---file-content-end---
content:
```js
import '@testing-library/jest-dom/vitest'
```
---file-content-end---
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 1)
        step = steps[0]
        # Both targets must be populated independently.
        self.assertIn("myapp/vite.config.js", step.inline_code)
        self.assertIn("myapp/vitest.setup.js", step.inline_code)
        # vite.config.js must hold the actual Vite config, not the
        # setup-file content.
        vite_cfg = step.inline_code["myapp/vite.config.js"]
        self.assertIn("defineConfig", vite_cfg)
        self.assertIn("plugins: [react()]", vite_cfg)
        # vitest.setup.js must hold the testing-library import, not
        # the Vite config.
        setup = step.inline_code["myapp/vitest.setup.js"]
        self.assertIn("@testing-library/jest-dom/vitest", setup)
        self.assertNotIn("defineConfig", setup)

    def test_content_marker_with_markdown_fence(self):
        """LLMs often use --- content --- with ```js fences instead of
        ---file-content-start--- / ---file-content-end---."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Configure vite.config.js
target: responsive-webapp/vite.config.js
exports: none
imports: none
--- content ---
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 1)
        self.assertIn("responsive-webapp/vite.config.js", steps[0].inline_code)
        code = steps[0].inline_code["responsive-webapp/vite.config.js"]
        self.assertIn("defineConfig", code)
        self.assertIn("plugins: [react()]", code)
        # Markdown fences should NOT appear in captured code
        self.assertNotIn("```", code)

    def test_content_marker_without_fence(self):
        """--- content --- marker without markdown fences."""
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create config
target: config.js
--- content ---
module.exports = { key: true };
==END==
"""
        steps = parse_structured_plan(text)
        self.assertIn("config.js", steps[0].inline_code)
        self.assertIn("module.exports", steps[0].inline_code["config.js"])

    def test_no_inline_code(self):
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create server
target: src/server.js
exports: app
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(steps[0].inline_code, {})

    def test_parenthesized_echo_cmd_reclassified_to_code(self):
        # Regression: the planner emitted a file as a Windows-broken
        # (echo ... > f) && (echo ... >> f) CMD chain. It must be rescued
        # into a CODE step that writes the file directly, not run as a
        # shell chain (which dies on the '[' / '{' in the content).
        text = """
==PLAN==
--STEP 1.3 [CMD] depends:1.2
Create the Vitest config and setup files.
> cd bootstrap-homepage && (echo import { defineConfig } from 'vitest/config' > vitest.config.js) && (echo.>> vitest.config.js) && (echo export default defineConfig({ >> vitest.config.js) && (echo   plugins: [react()], >> vitest.config.js) && (echo }) >> vitest.config.js) && (echo import '@testing-library/jest-dom/vitest' > vitest.setup.js)
produces: bootstrap-homepage\\vitest.config.js, bootstrap-homepage\\vitest.setup.js
==END==
"""
        step = parse_structured_plan(text)[0]
        self.assertEqual(step.step_type, "CODE")
        self.assertIsNone(step.command)  # the broken chain is gone
        # Paths carry the cd'd subdir prefix.
        self.assertIn("bootstrap-homepage/vitest.config.js", step.inline_code)
        self.assertIn("bootstrap-homepage/vitest.setup.js", step.inline_code)
        self.assertEqual(set(step.target_files), set(step.inline_code))
        cfg = step.inline_code["bootstrap-homepage/vitest.config.js"]
        self.assertIn("defineConfig", cfg)
        self.assertIn("plugins: [react()]", cfg)
        setup = step.inline_code["bootstrap-homepage/vitest.setup.js"]
        self.assertIn("@testing-library/jest-dom/vitest", setup)

    def test_unspaced_and_caret_echo_cmd_reclassified_to_code(self):
        # Second observed variant: compact redirects with NO spaces around
        # `>`/`>>` (echo x>f) and cmd.exe caret escapes for indentation
        # (echo ^  plugins:). Must still be rescued to CODE, and the caret
        # escapes should restore the leading indentation.
        text = """
==PLAN==
--STEP 1.2 [CMD] depends:1.1
Create the Vitest configuration and setup file.
> cd bootstrap_homepage && (echo import { defineConfig } from 'vitest/config'>vitest.config.js) && (echo.>>vitest.config.js) && (echo export default defineConfig({>>vitest.config.js) && (echo ^  plugins: [react()],>>vitest.config.js) && (echo })>>vitest.config.js) && (echo import '@testing-library/jest-dom/vitest'>vitest.setup.js)
produces: bootstrap_homepage\\vitest.config.js, bootstrap_homepage\\vitest.setup.js
==END==
"""
        step = parse_structured_plan(text)[0]
        self.assertEqual(step.step_type, "CODE")
        self.assertIsNone(step.command)
        cfg = step.inline_code["bootstrap_homepage/vitest.config.js"]
        self.assertIn("defineConfig", cfg)
        # Caret escape restored the indentation on the plugins line.
        self.assertIn("  plugins: [react()],", cfg)
        self.assertIn("@testing-library/jest-dom/vitest",
                      step.inline_code["bootstrap_homepage/vitest.setup.js"])

    def test_mixed_cmd_with_real_command_stays_cmd(self):
        # A chain that also runs a real command (npm install) must NOT be
        # converted — losing the install would break the build.
        text = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Install then write a config.
> cd app && npm install left-pad && (echo module.exports = {} > app.config.js)
produces: app/app.config.js
==END==
"""
        step = parse_structured_plan(text)[0]
        self.assertEqual(step.step_type, "CMD")
        self.assertIsNotNone(step.command)
        self.assertIn("npm install", step.command)

    def test_inline_code_preserves_indentation(self):
        text = """
==PLAN==
--STEP 1.1 [CODE] depends:none
Create config
target: config.js
---file-content-start---
module.exports = {
  nested: {
    deep: true,
  },
};
---file-content-end---
==END==
"""
        steps = parse_structured_plan(text)
        code = steps[0].inline_code["config.js"]
        self.assertIn("  nested:", code)
        self.assertIn("    deep:", code)

    def test_inline_code_serialization(self):
        """inline_code should survive to_dict / from_dict round-trip."""
        step = PlanStep(
            id="1.1", step_type="CODE",
            target_files=["a.js"],
            inline_code={"a.js": "const x = 1;"},
        )
        d = step.to_dict()
        self.assertEqual(d["inline_code"], {"a.js": "const x = 1;"})
        restored = PlanStep.from_dict(d)
        self.assertEqual(restored.inline_code, {"a.js": "const x = 1;"})

    def test_empty_inline_code_not_in_dict(self):
        """When inline_code is empty, it should not appear in to_dict()."""
        step = PlanStep(id="1.1", step_type="CODE")
        d = step.to_dict()
        self.assertNotIn("inline_code", d)

    def test_multi_step_with_mixed_inline(self):
        """Some steps have inline code, others don't."""
        text = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Install deps
> npm install express

--STEP 2.1 [CODE] depends:1.1
Create config
target: config.js
---file-content-start---
module.exports = {};
---file-content-end---

--STEP 3.1 [CODE] depends:2.1
Create server (complex, no inline)
target: src/server.js
exports: app
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].inline_code, {})  # CMD
        self.assertIn("config.js", steps[1].inline_code)  # has inline
        self.assertEqual(steps[2].inline_code, {})  # no inline


class TestValidation(unittest.TestCase):

    def test_valid_plan(self):
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),
            PlanStep(id="2.1", step_type="CODE", depends_on=["1.1"], index=1),
        ]
        self.assertEqual(validate_plan(steps), [])

    def test_unknown_dependency(self):
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),
            PlanStep(id="2.1", step_type="CODE", depends_on=["9.9"], index=1),
        ]
        errors = validate_plan(steps)
        self.assertTrue(any("9.9" in e for e in errors))


class TestFixImportDependencies(unittest.TestCase):

    def test_missing_dep_injected(self):
        """Step 11 creates ErrorBoundary, step 10 imports it but doesn't depend on 11."""
        steps = [
            PlanStep(id="10", step_type="CODE", index=0,
                     target_files=["src/App.jsx"],
                     imports_from={"src/components/ErrorBoundary.jsx": ["ErrorBoundary"]}),
            PlanStep(id="11", step_type="CODE", index=1,
                     target_files=["src/components/ErrorBoundary.jsx"],
                     exports=["ErrorBoundary"]),
        ]
        fixes = fix_import_dependencies(steps)
        self.assertEqual(len(fixes), 1)
        self.assertIn("11", steps[0].depends_on)

    def test_no_fix_when_dep_already_declared(self):
        steps = [
            PlanStep(id="10", step_type="CODE", index=0,
                     target_files=["src/components/ErrorBoundary.jsx"],
                     exports=["ErrorBoundary"]),
            PlanStep(id="11", step_type="CODE", index=1,
                     target_files=["src/App.jsx"],
                     depends_on=["10"],
                     imports_from={"src/components/ErrorBoundary.jsx": ["ErrorBoundary"]}),
        ]
        fixes = fix_import_dependencies(steps)
        self.assertEqual(fixes, [])

    def test_no_fix_for_external_files(self):
        """Imports from files not produced by any step (existing project files)."""
        steps = [
            PlanStep(id="1", step_type="CODE", index=0,
                     target_files=["src/App.jsx"],
                     imports_from={"src/utils/existing.js": ["helper"]}),
        ]
        fixes = fix_import_dependencies(steps)
        self.assertEqual(fixes, [])

    def test_self_reference_ignored(self):
        """A step that imports from its own target file."""
        steps = [
            PlanStep(id="1", step_type="CODE", index=0,
                     target_files=["src/App.jsx"],
                     imports_from={"src/App.jsx": ["App"]}),
        ]
        fixes = fix_import_dependencies(steps)
        self.assertEqual(fixes, [])

    def test_multiple_missing_deps(self):
        """Multiple import deps missing across steps."""
        steps = [
            PlanStep(id="3", step_type="CODE", index=0,
                     target_files=["src/App.jsx"],
                     imports_from={
                         "src/Header.jsx": ["Header"],
                         "src/Footer.jsx": ["Footer"],
                     }),
            PlanStep(id="1", step_type="CODE", index=1,
                     target_files=["src/Header.jsx"],
                     exports=["Header"]),
            PlanStep(id="2", step_type="CODE", index=2,
                     target_files=["src/Footer.jsx"],
                     exports=["Footer"]),
        ]
        fixes = fix_import_dependencies(steps)
        self.assertEqual(len(fixes), 2)
        self.assertIn("1", steps[0].depends_on)
        self.assertIn("2", steps[0].depends_on)

    def test_waves_reordered_after_fix(self):
        """After fix, wave builder should schedule producer before consumer."""
        steps = [
            PlanStep(id="1.1", step_type="CODE", index=0,
                     target_files=["src/App.jsx"],
                     imports_from={"src/components/ErrorBoundary.jsx": ["ErrorBoundary"]}),
            PlanStep(id="1.2", step_type="CODE", index=1,
                     target_files=["src/components/ErrorBoundary.jsx"],
                     exports=["ErrorBoundary"]),
        ]
        fix_import_dependencies(steps)
        waves = build_waves(steps)
        # ErrorBoundary (step 1.2) must be in an earlier wave than App (step 1.1)
        wave_of = {}
        for wi, wave in enumerate(waves):
            for s in wave:
                wave_of[s.id] = wi
        self.assertLess(wave_of["1.2"], wave_of["1.1"])

    def test_backslash_import_still_matches_producer(self):
        """Separator style must not defeat producer lookup.

        The planner writes `target: src\\map.py` and `imports: src\\map.py:Map`.
        Target paths are normalised at parse time, import paths historically
        were not, so the exact-string lookup missed and NO dependency was
        injected — producer and consumer then ran concurrently in the same
        wave and the consumer rewrote the producer's file.
        """
        steps = [
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["src/map.py"], exports=["Map"]),
            PlanStep(id="2.2", step_type="CODE", index=1,
                     target_files=["src/player.py"],
                     imports_from={"src\\map.py": ["Map"]}),
        ]
        fixes = fix_import_dependencies(steps)
        self.assertEqual(len(fixes), 1)
        self.assertIn("2.1", steps[1].depends_on)
        # ...and the two must no longer share a wave.
        waves = build_waves(steps)
        wave_of = {s.id: wi for wi, w in enumerate(waves) for s in w}
        self.assertLess(wave_of["2.1"], wave_of["2.2"])

    def test_dotted_module_import_matches_producer(self):
        """Python module notation ('src.map') resolves to target 'src/map.py'."""
        steps = [
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["src/map.py"], exports=["Map"]),
            PlanStep(id="2.2", step_type="CODE", index=1,
                     target_files=["src/player.py"],
                     imports_from={"src.map": ["Map"]}),
        ]
        fix_import_dependencies(steps)
        self.assertIn("2.1", steps[1].depends_on)

    def test_same_basename_different_dirs_not_linked(self):
        """No basename fuzz: a wrong edge here would reorder waves wrongly."""
        steps = [
            PlanStep(id="2.1", step_type="CODE", index=0,
                     target_files=["src/admin/index.js"]),
            PlanStep(id="2.2", step_type="CODE", index=1,
                     target_files=["src/public/page.js"],
                     imports_from={"src/public/index.js": ["thing"]}),
        ]
        self.assertEqual(fix_import_dependencies(steps), [])
        self.assertEqual(steps[1].depends_on, [])


class TestBuildWaves(unittest.TestCase):

    def test_linear_dependencies(self):
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),
            PlanStep(id="2.1", step_type="CODE", depends_on=["1.1"], index=1),
            PlanStep(id="3.1", step_type="TEST", depends_on=["2.1"], index=2),
        ]
        waves = build_waves(steps)
        self.assertEqual(len(waves), 3)

    def test_parallel_steps(self):
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),
            PlanStep(id="2.1", step_type="CODE", depends_on=["1.1"], index=1),
            PlanStep(id="2.2", step_type="CODE", depends_on=["1.1"], index=2),
        ]
        waves = build_waves(steps)
        self.assertEqual(len(waves), 2)
        self.assertEqual(len(waves[1]), 2)  # 2.1 and 2.2 in parallel

    def test_phase_ordering(self):
        """Steps 2.x should always execute after all 1.x waves, even with depends:none."""
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),
            PlanStep(id="1.2", step_type="CMD", depends_on=["1.1"], index=1),
            PlanStep(id="2.1", step_type="CODE", index=2),  # depends:none
            PlanStep(id="2.2", step_type="CODE", index=3),  # depends:none
            PlanStep(id="2.3", step_type="CODE", depends_on=["2.2"], index=4),
        ]
        waves = build_waves(steps)
        # Phase 1: wave[0]=[1.1], wave[1]=[1.2]
        # Phase 2: wave[2]=[2.1, 2.2], wave[3]=[2.3]
        self.assertEqual(len(waves), 4)
        self.assertEqual([s.id for s in waves[0]], ["1.1"])
        self.assertEqual([s.id for s in waves[1]], ["1.2"])
        phase2_ids = {s.id for w in waves[2:] for s in w}
        self.assertEqual(phase2_ids, {"2.1", "2.2", "2.3"})
        # 2.1 and 2.2 should be parallel in the same wave
        self.assertEqual(len(waves[2]), 2)

    def test_cmd_implicit_dep_within_same_phase(self):
        """CMD scaffold + CODE writes all in phase 1 — CODE must wait for CMD.

        Reproduces the bug seen when the LLM puts npm-create (1.1 [CMD]) and
        all file-write steps (1.4–1.9 [CODE] depends:none) in the same phase,
        causing them to execute concurrently.
        """
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),   # npm create vite
            PlanStep(id="1.2", step_type="CODE", depends_on=["1.1"], index=1),
            PlanStep(id="1.3", step_type="CODE", depends_on=["1.2"], index=2),
            PlanStep(id="1.4", step_type="CODE", index=3),   # depends:none
            PlanStep(id="1.5", step_type="CODE", index=4),   # depends:none
            PlanStep(id="1.6", step_type="CODE", index=5),   # depends:none
        ]
        waves = build_waves(steps)

        # 1.1 must be alone in the first wave
        self.assertEqual([s.id for s in waves[0]], ["1.1"])

        # 1.4, 1.5, 1.6 must NOT appear in the first wave (they should
        # implicitly depend on 1.1 now)
        first_wave_ids = {s.id for s in waves[0]}
        self.assertNotIn("1.4", first_wave_ids)
        self.assertNotIn("1.5", first_wave_ids)
        self.assertNotIn("1.6", first_wave_ids)

        # All step IDs must appear in exactly one wave
        all_ids_in_waves = [s.id for w in waves for s in w]
        self.assertEqual(sorted(all_ids_in_waves),
                         ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"])

    def test_phase_ordering_real_world(self):
        """Real-world plan: CMD setup then CODE generation."""
        steps = [
            PlanStep(id="1.1", step_type="CMD", index=0),  # npm create
            PlanStep(id="1.2", step_type="CMD", depends_on=["1.1"], index=1),  # npm install
            PlanStep(id="1.3", step_type="CMD", depends_on=["1.2"], index=2),  # npm install tailwind
            PlanStep(id="2.1", step_type="CODE", index=3),  # index.css - depends:none
            PlanStep(id="2.2", step_type="CODE", index=4),  # App.jsx - depends:none
            PlanStep(id="2.3", step_type="CODE", depends_on=["2.2"], index=5),  # App.test.js
            PlanStep(id="2.4", step_type="CODE", index=6),  # vitest.config.js
            PlanStep(id="2.5", step_type="CODE", index=7),  # vitest.setup.js
        ]
        waves = build_waves(steps)
        # Phase 1 should be first 3 waves
        phase1_waves = [w for w in waves if any(s.id.startswith("1.") for s in w)]
        phase2_waves = [w for w in waves if any(s.id.startswith("2.") for s in w)]
        # All phase 1 waves come before all phase 2 waves
        last_phase1_idx = max(waves.index(w) for w in phase1_waves)
        first_phase2_idx = min(waves.index(w) for w in phase2_waves)
        self.assertLess(last_phase1_idx, first_phase2_idx)


class TestEchoInlineCode(unittest.TestCase):
    """Tests for echo command parsing into inline_code during plan parsing."""

    def test_echo_single_file(self):
        """Echo command in CODE step populates inline_code."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create PostCSS config
target: postcss.config.mjs
> echo "export default { plugins: {} }" > postcss.config.mjs
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 1)
        self.assertIn("postcss.config.mjs", steps[0].inline_code)
        self.assertIn("plugins", steps[0].inline_code["postcss.config.mjs"])

    def test_echo_append_multiple_lines(self):
        """Multiple echo >> commands build up file content."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create CSS file
target: src/index.css
> echo "@import 'tailwindcss';" >> src/index.css
> echo ".container { max-width: 1200px; }" >> src/index.css
==END==
"""
        steps = parse_structured_plan(text)
        code = steps[0].inline_code.get("src/index.css", "")
        self.assertIn("@import 'tailwindcss';", code)
        self.assertIn(".container", code)

    def test_echo_with_cd_prefix(self):
        """cd dir && echo ... chains are handled."""
        text = """
==PLAN==
--STEP 3.1 [CODE] depends:1.1
Create config
target: config.js
> cd my-app && echo "module.exports = {};" > config.js
==END==
"""
        steps = parse_structured_plan(text)
        self.assertIn("config.js", steps[0].inline_code)
        self.assertIn("module.exports", steps[0].inline_code["config.js"])

    def test_dot_path_fixing(self):
        """src.App.jsx is fixed to src/App.jsx."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create App component
target: src/App.jsx
> echo "export default function App() { return null; }" > src.App.jsx
==END==
"""
        steps = parse_structured_plan(text)
        self.assertIn("src/App.jsx", steps[0].inline_code)

    def test_file_content_start_takes_priority(self):
        """---file-content-start--- blocks take priority over echo commands."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create config
target: config.js
---file-content-start---
module.exports = { priority: true };
---file-content-end---
> echo "module.exports = { echo_version: true };" > config.js
==END==
"""
        steps = parse_structured_plan(text)
        code = steps[0].inline_code.get("config.js", "")
        # file-content-start takes priority, echo should be ignored
        self.assertIn("priority", code)
        self.assertNotIn("echo_version", code)

    def test_touch_creates_empty_file(self):
        """touch file creates an empty entry in inline_code."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create files
target: src/index.css
> touch src/index.css
==END==
"""
        steps = parse_structured_plan(text)
        self.assertIn("src/index.css", steps[0].inline_code)
        self.assertEqual(steps[0].inline_code["src/index.css"], "")

    def test_cmd_step_echo_populates_inline_code(self):
        """CMD steps with echo commands also get inline_code."""
        text = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Scaffold project and create configs
> npm create vite@latest my-app -- --template react
> echo "module.exports = { plugins: {} };" > postcss.config.mjs
produces: package.json, postcss.config.mjs
==END==
"""
        steps = parse_structured_plan(text)
        self.assertIn("postcss.config.mjs", steps[0].inline_code)

    def test_mixed_steps_with_and_without_echo(self):
        """Some steps have echo, some don't."""
        text = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Install deps
> npm install express

--STEP 2.1 [CODE] depends:1.1
Create config
target: config.js
> echo "module.exports = {};" > config.js

--STEP 3.1 [CODE] depends:2.1
Create server (complex)
target: src/server.js
exports: app
==END==
"""
        steps = parse_structured_plan(text)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].inline_code, {})  # just npm install, no echo
        self.assertIn("config.js", steps[1].inline_code)  # has echo
        self.assertEqual(steps[2].inline_code, {})  # no echo

    def test_newline_escape_in_echo(self):
        """\\n in echo content becomes actual newlines."""
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:1.1
Create config
target: config.js
> echo "line1\\nline2\\nline3" > config.js
==END==
"""
        steps = parse_structured_plan(text)
        code = steps[0].inline_code.get("config.js", "")
        self.assertIn("line1\nline2\nline3", code)


if __name__ == "__main__":
    unittest.main()


class TestDeriveImportedByModuleNotation(unittest.TestCase):
    """imports: written in Python module notation (src.snake) must link to
    the producer step targeting src/snake.py."""

    def test_module_dot_notation_links_producer(self):
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:none
Create Snake class
target: src/snake.py
exports: Snake
imports: none

--STEP 3.1 [TEST] depends:2.1
Add unit tests
target: tests/test_game_logic.py
imports: src.snake:Snake
==END==
"""
        steps = parse_structured_plan(text)
        producer = next(s for s in steps if s.id == "2.1")
        self.assertIn("tests/test_game_logic.py", producer.imported_by)

    def test_file_path_notation_still_links(self):
        text = """
==PLAN==
--STEP 2.1 [CODE] depends:none
Create Snake class
target: src/snake.py
exports: Snake
imports: none

--STEP 3.1 [TEST] depends:2.1
Add unit tests
target: tests/test_game_logic.py
imports: src/snake.py:Snake
==END==
"""
        steps = parse_structured_plan(text)
        producer = next(s for s in steps if s.id == "2.1")
        self.assertIn("tests/test_game_logic.py", producer.imported_by)

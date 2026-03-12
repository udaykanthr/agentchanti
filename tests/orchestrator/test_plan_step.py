"""Tests for PlanStep parsing, including inline code capture."""

from __future__ import annotations

import unittest

from multi_agent_coder.orchestrator.plan_step import (
    parse_structured_plan, PlanStep, validate_plan,
    fix_import_dependencies,
    is_structured_plan, build_waves,
)


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
            PlanStep(id="10", step_type="CODE", index=0,
                     target_files=["src/App.jsx"],
                     imports_from={"src/components/ErrorBoundary.jsx": ["ErrorBoundary"]}),
            PlanStep(id="11", step_type="CODE", index=1,
                     target_files=["src/components/ErrorBoundary.jsx"],
                     exports=["ErrorBoundary"]),
        ]
        fix_import_dependencies(steps)
        waves = build_waves(steps)
        # ErrorBoundary (step 11) must be in an earlier wave than App (step 10)
        wave_of = {}
        for wi, wave in enumerate(waves):
            for s in wave:
                wave_of[s.id] = wi
        self.assertLess(wave_of["11"], wave_of["10"])


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


if __name__ == "__main__":
    unittest.main()

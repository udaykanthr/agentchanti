"""Tests for PlanStep parsing, including inline code capture."""

from __future__ import annotations

import unittest

from multi_agent_coder.orchestrator.plan_step import (
    parse_structured_plan, PlanStep, validate_plan,
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

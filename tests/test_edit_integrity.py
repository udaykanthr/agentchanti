"""Tests for multi-target edit attribution, blind-edit routing, and the
syntax gates that keep corrupt content (merged files, invalid JSON) off disk.

Regression source: a two-target step's edit blocks were both applied to
target[0] and merge-promoted into one file (duplicate `App` declaration
broke the build); a minimal-diff patch wrote a trailing comma into
package.json (broke every npm invocation)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import AgentTools
from agentchanti.orchestrator.pipeline import _syntax_gate
from agentchanti.orchestrator.plan_step import (
    dedupe_redundant_cd,
    parse_structured_plan,
    route_blind_edits,
    validate_plan,
)


_TWO_TARGET_PLAN = """
==PLAN==
--STEP 2.2 [CODE] depends:1.1
Wire the homepage into the app root and import Bootstrap CSS.
target: app/src/App.jsx, app/src/main.jsx
exports: App
imports: none
edit:
<<<FIND>>>
old app content
<<<REPLACE>>>
import { HomePage } from './components/HomePage'
function App() { return <HomePage /> }
export default App
<<<END>>>
edit:
<<<FIND>>>
old main content
<<<REPLACE>>>
import 'bootstrap/dist/css/bootstrap.min.css'
import App from './App.jsx'
<<<END>>>
==END==
"""


class TestEditBlockAttribution(unittest.TestCase):

    def test_bare_edit_blocks_map_to_targets_in_order(self):
        step = parse_structured_plan(_TWO_TARGET_PLAN)[0]
        self.assertIn("app/src/App.jsx", step.inline_edits)
        self.assertIn("app/src/main.jsx", step.inline_edits)
        # First block → first target, second block → second target.
        self.assertIn("HomePage", step.inline_edits["app/src/App.jsx"][0][1])
        self.assertIn("bootstrap",
                      step.inline_edits["app/src/main.jsx"][0][1])

    def test_single_target_keeps_all_blocks(self):
        plan = _TWO_TARGET_PLAN.replace(
            "target: app/src/App.jsx, app/src/main.jsx",
            "target: app/src/App.jsx")
        step = parse_structured_plan(plan)[0]
        # Both blocks clamp to the only target — pre-existing behavior.
        self.assertEqual(list(step.inline_edits), ["app/src/App.jsx"])
        self.assertEqual(len(step.inline_edits["app/src/App.jsx"]), 2)

    def test_explicit_edit_path_still_wins(self):
        plan = _TWO_TARGET_PLAN.replace(
            "edit:\n<<<FIND>>>\nold main content",
            "edit: app/src/other.jsx\n<<<FIND>>>\nold main content")
        step = parse_structured_plan(plan)[0]
        self.assertIn("app/src/other.jsx", step.inline_edits)


class TestRouteBlindEdits(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="blindedit_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_complete_replace_on_missing_file_becomes_overwrite(self):
        steps = parse_structured_plan(_TWO_TARGET_PLAN)
        notes = route_blind_edits(steps, project_root=self.root)
        step = steps[0]
        # App.jsx replace has an export → full-file write; main.jsx replace
        # is import-only (no export) → dropped, step goes grounded for it.
        self.assertIn("app/src/App.jsx", step.inline_code)
        self.assertNotIn("app/src/App.jsx", step.inline_edits)
        self.assertNotIn("app/src/main.jsx", step.inline_edits)
        self.assertNotIn("app/src/main.jsx", step.inline_code)
        self.assertEqual(len(notes), 2)

    def test_existing_file_edits_untouched(self):
        os.makedirs(os.path.join(self.root, "app", "src"))
        for name in ("App.jsx", "main.jsx"):
            with open(os.path.join(self.root, "app", "src", name), "w") as f:
                f.write("// existing\n")
        steps = parse_structured_plan(_TWO_TARGET_PLAN)
        notes = route_blind_edits(steps, project_root=self.root)
        self.assertEqual(notes, [])
        self.assertEqual(len(steps[0].inline_edits), 2)


class TestSyntaxGate(unittest.TestCase):

    def test_json_trailing_comma_rejected(self):
        self.assertIn("invalid JSON",
                      _syntax_gate("app/package.json",
                                   '{"scripts": {"test": "vitest",}}'))

    def test_valid_json_passes(self):
        self.assertIsNone(_syntax_gate("app/package.json",
                                       '{"scripts": {"test": "vitest"}}'))

    def test_tsconfig_jsonc_exempt(self):
        self.assertIsNone(_syntax_gate(
            "app/tsconfig.json", '{\n  // comment\n  "strict": true,\n}'))

    def test_python_syntax_error_rejected(self):
        self.assertIn("syntax error",
                      _syntax_gate("x.py", "def broken(:\n"))
        self.assertIsNone(_syntax_gate("x.py", "def ok():\n    return 1\n"))


class TestAgentToolsJsonGate(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="jsongate_")
        self.tools = AgentTools(project_root=self.root, executor=MagicMock())
        with open(os.path.join(self.root, "package.json"), "w") as f:
            f.write('{\n  "scripts": {\n    "build": "vite build"\n  }\n}\n')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_edit_producing_invalid_json_rejected(self):
        result = self.tools._tool_edit_file(
            "package.json",
            '"build": "vite build"',
            '"build": "vite build",')  # introduces a trailing comma
        self.assertTrue(result.startswith("ERROR"))
        self.assertIn("invalid", result)
        # File untouched on disk.
        with open(os.path.join(self.root, "package.json")) as f:
            self.assertNotIn('build",\n', f.read())

    def test_valid_json_edit_accepted(self):
        result = self.tools._tool_edit_file(
            "package.json",
            '"build": "vite build"',
            '"build": "vite build",\n    "test": "vitest"')
        self.assertTrue(result.startswith("OK"), result)


class TestDedupeRedundantCd(unittest.TestCase):

    def test_repeated_cd_dropped(self):
        # Exact shape from the failing run: four plan lines each assuming
        # the project root, joined into one && chain.
        cmd = ("npm create vite@latest app -- --template react --yes"
               " && cd app && npm install"
               " && cd app && npm install bootstrap"
               " && cd app && npm install --save-dev vitest")
        out = dedupe_redundant_cd(cmd)
        self.assertEqual(out.count("cd app"), 1)
        self.assertIn("npm install bootstrap", out)
        self.assertIn("npm install --save-dev vitest", out)

    def test_distinct_cds_kept(self):
        cmd = "cd app && npm install && cd docs && npm run build"
        self.assertEqual(dedupe_redundant_cd(cmd), cmd)

    def test_single_and_none_unchanged(self):
        self.assertEqual(dedupe_redundant_cd("npm install"), "npm install")
        self.assertIsNone(dedupe_redundant_cd(None))

    def test_parsed_multiline_cmd_step_deduped(self):
        plan = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Scaffold and install.
> npm create vite@latest app -- --template react --yes
> cd app && npm install
> cd app && npm install bootstrap
produces: app/package.json
==END==
"""
        step = parse_structured_plan(plan)[0]
        self.assertEqual(step.command.count("cd app"), 1)


class TestGlobProducesValidation(unittest.TestCase):

    def test_scaffold_glob_covers_imported_files(self):
        # A CMD step produces `app/src/*` (glob). Inline code importing
        # scaffold-created files (./index.css) must NOT be treated as
        # dangling — that false positive cleared a correct main.jsx and
        # cost an 8-turn loop.
        plan = """
==PLAN==
--STEP 1.1 [CMD] depends:none
Scaffold the app.
> npm create vite@latest app -- --template react --yes
produces: app/package.json, app/src/*

--STEP 1.2 [CODE] depends:1.1
Wire the entry point.
target: app/src/main.jsx
exports: none
imports: none
content:
```jsx
import 'bootstrap/dist/css/bootstrap.min.css'
import './index.css'
import App from './App.jsx'
```
---file-content-end---
==END==
"""
        steps = parse_structured_plan(plan)
        errors = validate_plan(steps)
        dangling = [e for e in errors if "no step produces" in e]
        self.assertEqual(dangling, [])
        # Inline code survived validation.
        self.assertIn("app/src/main.jsx", steps[1].inline_code)


if __name__ == "__main__":
    unittest.main()

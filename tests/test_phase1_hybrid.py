"""Phase 1 hybrid-planner tests: adaptive inline cap + no-blind-edit prompt
rules, truncation salvage, deterministic vitest bootstrap, and the
test-infra guard exemption."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agents.planner import PlannerAgent
from agentchanti.orchestrator.pipeline import _is_test_infra_file
from agentchanti.orchestrator.plan_step import (
    PlanStep,
    plan_salvageable,
)
from agentchanti.orchestrator.step_handlers import ensure_vitest_env


def _captured_prompt(plan_mode: str) -> str:
    planner = PlannerAgent("P", "Architect", "Plan tasks", MagicMock())
    planner.llm_client.generate_response = lambda prompt: prompt
    return planner.process("build an api", context="ctx",
                           plan_mode=plan_mode)


class TestAdaptiveCapPromptRules(unittest.TestCase):

    def test_content_mode_has_inline_budget(self):
        prompt = _captured_prompt("content")
        self.assertIn("Inline code budget", prompt)
        self.assertIn("under ~150 lines", prompt)

    def test_content_mode_bans_blind_edits(self):
        # edit: blocks for files an earlier step of the plan creates
        # (scaffold output) hallucinate FIND text — the prompt must forbid
        # them and the checklist must enforce it.
        prompt = _captured_prompt("content")
        self.assertIn("NEVER write an edit: block for a file that does\n"
                      "      not exist yet", prompt)
        self.assertIn("NO edit: block targets a file created by an earlier "
                      "step of this plan", prompt)

    def test_intent_mode_unaffected(self):
        # Intent mode has no inline code at all — the cap is content-mode
        # machinery and must not leak into intent prompts.
        prompt = _captured_prompt("intent")
        self.assertNotIn("Inline code budget", prompt)
        self.assertIn("NO inline code (intent mode)", prompt)


class TestPlanSalvageable(unittest.TestCase):

    def _step(self, id_, **kw):
        return PlanStep(id=id_, step_type=kw.pop("step_type", "CODE"), **kw)

    def test_complete_last_step_is_salvageable(self):
        steps = [
            self._step("1.1", step_type="CMD", command="npm create vite"),
            self._step("2.1", target_files=["src/A.jsx"],
                       inline_code={"src/A.jsx": "x"}),
            self._step("3.1", step_type="CMD", command="npm run build"),
        ]
        self.assertTrue(plan_salvageable(steps))

    def test_bodyless_last_step_not_salvageable(self):
        steps = [
            self._step("1.1", step_type="CMD", command="npm i"),
            self._step("2.1", target_files=["src/A.jsx"]),
            self._step("3.1"),  # cut mid-step: no body at all
        ]
        self.assertFalse(plan_salvageable(steps))

    def test_too_few_steps_not_salvageable(self):
        steps = [self._step("1.1", step_type="CMD", command="npm i")]
        self.assertFalse(plan_salvageable(steps))
        self.assertFalse(plan_salvageable([]))
        self.assertFalse(plan_salvageable(None))


class TestEnsureVitestEnv(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vitestenv_")
        self.executor = MagicMock()
        self.executor.run_command.return_value = (True, "ok")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_pkg(self, dev_deps=None):
        pkg = {"name": "x", "devDependencies": dev_deps or {}}
        with open(os.path.join(self.root, "package.json"), "w",
                  encoding="utf-8") as f:
            json.dump(pkg, f)

    _DOM_TESTS = {"src/App.test.jsx":
                  "import { render } from '@testing-library/react';"}

    def test_bootstraps_missing_env(self):
        self._write_pkg(dev_deps={"@vitejs/plugin-react": "^5"})
        memory = MagicMock()
        ensure_vitest_env(self.executor, self.root, self._DOM_TESTS,
                          memory=memory)
        # Missing deps installed
        install_cmd = self.executor.run_command.call_args[0][0]
        self.assertIn("npm install -D", install_cmd)
        self.assertIn("@testing-library/react", install_cmd)
        self.assertIn("jsdom", install_cmd)
        # Config + setup written, jsdom env, react plugin included
        cfg = open(os.path.join(self.root, "vitest.config.js"),
                   encoding="utf-8").read()
        self.assertIn("environment: 'jsdom'", cfg)
        self.assertIn("plugins: [react()]", cfg)
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, "vitest.setup.js")))
        memory.update.assert_called_once()

    def test_no_react_plugin_when_not_installed(self):
        self._write_pkg()
        ensure_vitest_env(self.executor, self.root, self._DOM_TESTS)
        cfg = open(os.path.join(self.root, "vitest.config.js"),
                   encoding="utf-8").read()
        self.assertNotIn("react()", cfg)
        self.assertIn("environment: 'jsdom'", cfg)

    def test_existing_config_not_overwritten(self):
        self._write_pkg(dev_deps={p: "1" for p in (
            "vitest", "jsdom", "@testing-library/react",
            "@testing-library/dom", "@testing-library/jest-dom")})
        existing = "export default { custom: true }\n"
        with open(os.path.join(self.root, "vitest.config.js"), "w",
                  encoding="utf-8") as f:
            f.write(existing)
        ensure_vitest_env(self.executor, self.root, self._DOM_TESTS)
        # All deps present → no install; config untouched
        self.executor.run_command.assert_not_called()
        self.assertEqual(
            open(os.path.join(self.root, "vitest.config.js"),
                 encoding="utf-8").read(), existing)

    def test_non_dom_tests_are_noop(self):
        self._write_pkg()
        ensure_vitest_env(self.executor, self.root,
                          {"src/math.test.js":
                           "import { expect, it } from 'vitest'"})
        self.executor.run_command.assert_not_called()
        self.assertFalse(os.path.isfile(
            os.path.join(self.root, "vitest.config.js")))

    def test_no_package_json_is_noop(self):
        ensure_vitest_env(self.executor, self.root, self._DOM_TESTS)
        self.executor.run_command.assert_not_called()


class TestTestInfraGuardExemption(unittest.TestCase):

    def test_infra_files_recognized(self):
        for p in ("herbanner_site/vitest.config.js", "vitest.config.ts",
                  "app/vitest.setup.js", "jest.config.cjs",
                  "jest.setup.ts", "src/setupTests.js", "conftest.py"):
            self.assertTrue(_is_test_infra_file(p), p)

    def test_source_files_still_protected(self):
        for p in ("src/App.jsx", "src/main.jsx", "vite.config.js",
                  "src/Homepage.jsx", "package.json"):
            self.assertFalse(_is_test_infra_file(p), p)


if __name__ == "__main__":
    unittest.main()

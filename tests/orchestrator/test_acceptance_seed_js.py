"""A JavaScript project can earn a verified result too.

Measured 2026-09-17. A user asked a Next.js + TypeScript project for a
home page. Every step ran, the build reported `✓ Compiled successfully in
1337ms`, and the run exited 1:

    [AcceptanceSeed] skipped: language is typescript
    [Evidence] require_independent_evidence is set, but nothing can
               satisfy it in this run
    Pipeline failed: nothing outside this run's own output verified it

`require_independent_evidence` is satisfiable three ways — user
`acceptance_cmds`, a pre-existing suite, or a seeded contract — and the
seeder was Python-only, so for a TypeScript project none of the three
could exist. The run was doomed before it started, however well it went.

The contract is a PLAIN NODE SCRIPT rather than a vitest/jest suite: it is
written before any code exists, so it cannot know which runner the project
will end up with, and one needing an install nobody has run yet could
never judge anything. `node file.mjs` works wherever a JS project works.
"""
import os

import pytest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME_JS,
    _looks_like_a_suite_js,
    js_platform_defect_reason,
    js_server_orchestration_reason,
    seed_acceptance_tests,
    seed_state,
    weak_js_contract_reason,
)
from agentchanti.orchestrator.evidence import _runner_command

GOOD = """\
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

const pkg = JSON.parse(readFileSync("package.json", "utf8"));
assert.ok(pkg.scripts && pkg.scripts.build, "the project must build");

const out = execSync("npm run build", { timeout: 120000, encoding: "utf8" });
assert.match(out, /compiled|built|ready/i, "the build must succeed");

console.log(`ACCEPTANCE: 2 checks passed`);
"""


class _Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_response(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return self.responses.pop(0) if self.responses else "nope"


class TestTheSanityGate:

    def test_a_real_contract_is_accepted(self):
        assert _looks_like_a_suite_js(GOOD)

    @pytest.mark.parametrize("src,why", [
        ("Here is the contract you asked for.", "prose"),
        ('import assert from "node:assert/strict";\n'
         'assert.ok(1);\nassert.ok(2);', "no reported count"),
        ('import assert from "node:assert/strict";\n'
         'console.log("ACCEPTANCE: 0 checks passed");',
         "imports assert and never asserts — a stub"),
        ('import { test } from "vitest";\ntest("x", () => {});\n'
         'console.log("ACCEPTANCE: 1 checks passed");',
         "a runner that needs an install nobody has run yet"),
    ])
    def test_what_is_refused(self, src, why):
        assert not _looks_like_a_suite_js(src), why

    def test_one_assertion_is_enough(self):
        """Counting literal assert calls was wrong about real JavaScript.

        Measured across three live samples: every one wrapped the
        assertion in a `check()` helper and called it fourteen times, so
        the file holds exactly ONE literal `assert.ok(`. Requiring two
        refused three good contracts.
        """
        src = ('import assert from "node:assert/strict";\n'
               'let n = 0;\n'
               'function check(c, m) { n += 1; assert.ok(c, m); }\n'
               'check(true, "a");\ncheck(true, "b");\ncheck(true, "c");\n'
               'console.log(`ACCEPTANCE: ${n} checks passed`);\n')
        assert _looks_like_a_suite_js(src)

    def test_a_regex_inside_a_template_literal_is_not_a_syntax_error(self):
        """The line that defeated the brace-balance scan, verbatim."""
        src = ('import assert from "node:assert/strict";\n'
               'assert.ok(1);\n'
               "const cssPath = href.startsWith('/') ? href : "
               "`/${href.replace(/^\\.\\//, '')}`;\n"
               'console.log("ACCEPTANCE: 1 checks passed");\n')
        assert _looks_like_a_suite_js(src)

    @pytest.mark.parametrize("src,style", [
        ('import assert from "assert";\n'
         'assert.ok(1);\nassert.ok(2);\n'
         'console.log("ACCEPTANCE: 2 checks passed");',
         "the bare `assert` specifier, no node: prefix"),
        ('import assert from "node:assert/strict";\nlet n = 0;\n'
         'assert.ok(1); n++;\nassert.ok(2); n++;\n'
         'console.log("ACCEPTANCE: " + n + " checks passed");',
         "a counted total, concatenated rather than literal"),
        ('import { strict as assert } from "node:assert";\n'
         'assert.ok(1);\nassert.equal(1, 1);\n'
         'console.log(`ACCEPTANCE: 2 checks passed`);',
         "a named import of assert"),
        ('const assert = require("assert");\n'
         'assert.ok(1);\nassert.ok(2);\n'
         'console.log("ACCEPTANCE: 2 checks passed");',
         "require(), unprefixed"),
    ])
    def test_ordinary_styles_a_model_actually_writes(self, src, style):
        """Measured 2026-09-17: a live run produced two usable contracts
        and the gate refused both — one for `from "assert"` and one for
        printing its count by concatenation. A sanity gate that cannot be
        satisfied is the same defect as an unsatisfiable verify: line, and
        it cost that run its evidence."""
        assert _looks_like_a_suite_js(src), style

    def test_a_brace_inside_a_string_does_not_unbalance_it(self):
        src = GOOD + 'const s = "a { b";\n'
        assert _looks_like_a_suite_js(src)


class TestSeeding:

    def test_typescript_is_seeded_not_skipped(self, tmp_path):
        client = _Client(f"```js\n{GOOD}```")

        path = seed_acceptance_tests(
            "build a home page", str(tmp_path), client, language="typescript")

        assert path is not None, "this is the run that used to be doomed"
        assert os.path.basename(path) == SEED_BASENAME_JS

    def test_javascript_is_seeded(self, tmp_path):
        client = _Client(f"```javascript\n{GOOD}```")
        assert seed_acceptance_tests("t", str(tmp_path), client,
                                     language="javascript")

    def test_the_file_carries_a_js_comment_header(self, tmp_path):
        client = _Client(f"```js\n{GOOD}```")
        path = seed_acceptance_tests("t", str(tmp_path), client,
                                     language="typescript")

        first = open(path, encoding="utf-8").readline()
        assert first.startswith("// agentchanti:acceptance-seed")
        assert seed_state(path) is not None, "the run must recognise it as ours"

    def test_an_unusable_response_is_retried(self, tmp_path):
        """The model-written path, which is now opt-in."""
        from agentchanti.orchestrator.acceptance_seed import _seed_js
        client = _Client("I cannot help with that.", f"```js\n{GOOD}```")

        path = _seed_js("t", str(tmp_path), client, model_written=True)

        assert path is not None
        assert client.calls == 2

    def test_python_is_unaffected(self, tmp_path):
        """The Python path must not be routed into the JS one."""
        client = _Client("```python\nimport unittest\n\n\n"
                         "class C(unittest.TestCase):\n"
                         "    def test_a(self):\n        self.assertEqual(1, 1)\n"
                         "    def test_b(self):\n        self.assertEqual(2, 2)\n```")

        path = seed_acceptance_tests("t", str(tmp_path), client,
                                     language="python")

        assert path and path.endswith(".py")

    def test_an_unknown_language_is_still_skipped(self, tmp_path):
        client = _Client(f"```js\n{GOOD}```")
        assert seed_acceptance_tests("t", str(tmp_path), client,
                                     language="go") is None


class TestItIsRunnable:

    def test_the_evidence_layer_runs_it_with_node(self, tmp_path):
        assert _runner_command(str(tmp_path), SEED_BASENAME_JS) == \
            "node " + SEED_BASENAME_JS

    def test_the_contract_is_a_recognised_test_file(self):
        """So the survivor snapshot picks it up like any other suite."""
        from agentchanti.orchestrator.pipeline import _is_test_file
        assert _is_test_file(SEED_BASENAME_JS)


class TestReSeeding:

    def _seed(self, tmp_path, task="build a home page"):
        return seed_acceptance_tests(task, str(tmp_path),
                                     _Client(f"```js\n{GOOD}```"),
                                     language="typescript")

    def test_the_same_task_is_not_re_seeded(self, tmp_path):
        self._seed(tmp_path)
        client = _Client(f"```js\n{GOOD}```")

        again = seed_acceptance_tests("build a home page", str(tmp_path),
                                      client, language="typescript")

        assert again is None and client.calls == 0

    def test_a_different_task_re_seeds(self, tmp_path):
        self._seed(tmp_path)
        client = _Client(f"```js\n{GOOD}```")

        again = seed_acceptance_tests("build a checkout page", str(tmp_path),
                                      client, language="typescript")

        assert again is not None
        assert client.calls == 0, "the built-in contract costs no tokens"

    def test_an_edited_contract_is_left_alone(self, tmp_path):
        path = self._seed(tmp_path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('\nassert.ok(true, "mine now");\n')
        client = _Client(f"```js\n{GOOD}```")

        again = seed_acceptance_tests("build a home page", str(tmp_path),
                                      client, language="typescript")

        assert again is None and client.calls == 0


class TestItCanRunItsOwnCommands:
    """Measured 2026-09-17, the first Node contract a live run seeded.

    Fourteen good checks, and it reported `The production build must
    succeed` against a project whose `npm run build` prints
    `Compiled successfully` by hand:

        spawnSync("npm", ["run","build"], {cwd})  -> status null, ENOENT
        spawnSync("npm", [...], {shell: true})    -> status 0

    On Windows npm/npx/yarn/pnpm are .cmd batch scripts. This is the
    JavaScript counterpart of `platform_signal_reason`.
    """

    BROKEN = 'spawnSync("npm", ["run", "build"], { cwd: appDirectory })'
    FIXED = ('spawnSync("npm", ["run", "build"], '
             '{ cwd: appDirectory, shell: true })')

    def test_the_measured_call_is_flagged_on_windows(self):
        assert js_platform_defect_reason(self.BROKEN, platform="win32")

    def test_shell_true_is_accepted(self):
        assert js_platform_defect_reason(self.FIXED, platform="win32") is None

    def test_execsync_already_has_a_shell(self):
        assert js_platform_defect_reason(
            'execSync("npm run build", { cwd: d })', platform="win32") is None

    def test_posix_is_never_accused(self):
        """The same call is correct there — flagging it would invent a bug."""
        assert js_platform_defect_reason(self.BROKEN, platform="linux") is None

    @pytest.mark.parametrize("fn", ["spawn", "spawnSync",
                                    "execFile", "execFileSync"])
    def test_every_spawning_form(self, fn):
        src = fn + '("npx", ["next", "build"], { cwd: d })'
        assert js_platform_defect_reason(src, platform="win32")

    def test_an_unrelated_binary_is_not_flagged(self):
        """`node` itself is a real executable, not a .cmd shim."""
        assert js_platform_defect_reason(
            'spawnSync("node", ["server.js"], { cwd: d })',
            platform="win32") is None

    def test_the_real_fixture_carries_the_defect(self):
        import pathlib
        src = (pathlib.Path(__file__).parent / "fixtures"
               / "js_contract_helper_style.mjs").read_text(encoding="utf-8")
        assert js_platform_defect_reason(src, platform="win32")


class TestItDoesNotOrchestrateAServer:
    """Measured 2026-09-17 across three consecutive live runs, each of
    which failed a project that was correct:

        spawnSync("npm", ...)                     ENOENT (npm is a .cmd)
        "project root must contain package.json"  the app was in my-app/
        "the development server did not serve the home page"

    Different causes, one shape: the contract was orchestrating an
    environment rather than judging an artifact. Every live sample
    collected started a dev server.
    """

    @pytest.mark.parametrize("src,why", [
        ('spawn("npm", ["run", "dev"], { shell: true })', "npm run dev"),
        ('execSync("npx next start -p 3000")', "next start"),
        ("server.listen(3000)", "listening on a socket"),
        ('await fetch("http://localhost:3000/")', "fetching from localhost"),
        ('const r = await fetch(`http://127.0.0.1:${port}/`)', "127.0.0.1"),
    ])
    def test_server_orchestration_is_flagged(self, src, why):
        assert js_server_orchestration_reason(src), why

    def test_a_build_only_contract_is_accepted(self):
        src = ('const out = execSync("npm run build", '
               '{ cwd: dir, shell: true, timeout: 300000 });\n'
               'assert.ok(existsSync(".next/server/app/page.js"));\n')
        assert js_server_orchestration_reason(src) is None

    def test_the_real_dev_server_fixture(self):
        import pathlib
        src = (pathlib.Path(__file__).parent / "fixtures"
               / "js_contract_dev_server.mjs").read_text(encoding="utf-8")
        assert js_server_orchestration_reason(src)


class TestItCanFailOnWrongBehaviour:
    """`weak_contract_reason`'s counterpart — the gap 0.8.6 named."""

    def test_existence_checks_alone_are_weak(self):
        src = ('import assert from "node:assert";\n'
               'assert.ok(existsSync("a"));\nassert.ok(existsSync("b"));\n')
        assert weak_js_contract_reason(src)

    def test_a_tautology_is_weak(self):
        src = 'import assert from "node:assert";\nassert.ok(true);\n'
        assert weak_js_contract_reason(src)

    def test_one_real_check_is_not_enough(self):
        """One is within reach of code written to satisfy exactly that one."""
        src = ('assert.ok(existsSync("a"));\n'
               'assert.ok(out.includes("compiled"));\n')
        assert weak_js_contract_reason(src)

    def test_two_discriminating_checks_pass(self):
        src = ('assert.ok(html.includes("<header"));\n'
               "assert.match(html, /<title>[^<]+<" + chr(92) + "/title>/);" + chr(10))
        assert weak_js_contract_reason(src) is None

    def test_the_helper_style_is_counted(self):
        """Assertions are usually made through a check() helper."""
        src = ('function check(c, m) { assert.ok(c, m); }\n'
               'check(html.includes("<header"), "a");\n'
               'check(/<title>/.test(html), "b");\n')
        assert weak_js_contract_reason(src) is None

    @pytest.mark.parametrize("name", ["js_contract_helper_style.mjs",
                                      "js_contract_dev_server.mjs"])
    def test_the_real_contracts_are_strong(self, name):
        import pathlib
        src = (pathlib.Path(__file__).parent / "fixtures"
               / name).read_text(encoding="utf-8")
        assert weak_js_contract_reason(src) is None

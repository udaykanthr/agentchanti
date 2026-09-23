"""`tsc <file>` discards tsconfig.json, so the file can never pass.

Measured 2026-09-23, kimi-k2.7-code on a Next.js page. The plan's gate was

    cd my-app && npx tsc --noEmit --jsx react-jsx app/page.tsx

which returned the identical 218-character error over two versions of the
file. `observe_gate_verdict` called it STALLED (correctly) and the run was
reported failed — over a page that builds and passes the acceptance
contract. Naming a file makes TypeScript ignore the project config, so the
jsx setting, `types` and the `@/*` aliases are all dropped.
"""
import os

import pytest

from agentchanti.orchestrator.gate_integrity import (
    platform_equivalent_variants, tsc_explicit_file_reason,
    tsc_project_variant)
from agentchanti.orchestrator.plan_step import unrunnable_gate_reason

MEASURED = "cd my-app && npx tsc --noEmit --jsx react-jsx app/page.tsx"


@pytest.fixture
def next_project(tmp_path, monkeypatch):
    (tmp_path / "my-app").mkdir()
    (tmp_path / "my-app" / "tsconfig.json").write_text('{"compilerOptions":{}}')
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestDetection:

    def test_the_measured_gate_is_named(self, next_project):
        why = tsc_explicit_file_reason(MEASURED)
        assert "app/page.tsx" in why and "tsconfig.json" in why

    def test_it_is_a_variant_not_a_refusal(self, next_project):
        """Measured against a real scaffold the file form can exit 0, so
        refusing it up front would reject a gate that works."""
        assert unrunnable_gate_reason(MEASURED) is None

    def test_the_project_form_is_fine(self, next_project):
        assert tsc_explicit_file_reason("cd my-app && npx tsc --noEmit") is None

    def test_an_explicit_project_flag_is_deliberate(self, next_project):
        assert tsc_explicit_file_reason(
            "cd my-app && npx tsc -p tsconfig.json app/page.tsx") is None

    def test_without_a_tsconfig_explicit_files_are_the_only_way(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert tsc_explicit_file_reason("npx tsc --noEmit src/a.ts") is None

    def test_other_commands_are_untouched(self, next_project):
        for cmd in ("cd my-app && npm run build",
                    "cd my-app && npx next build",
                    "node -e \"require('./my-app/app/page.tsx')\""):
            assert tsc_explicit_file_reason(cmd) is None


class TestTheVariant:

    def test_the_file_argument_is_dropped(self, next_project):
        variant = dict(platform_equivalent_variants(MEASURED))
        assert variant["tsc-project-config"] == "cd my-app && npx tsc --noEmit"

    def test_the_variant_is_runnable(self, next_project):
        assert unrunnable_gate_reason(
            tsc_project_variant(MEASURED)) is None

    def test_a_bare_tsc_keeps_its_runner(self, next_project):
        assert tsc_project_variant("cd my-app && tsc --noEmit app/page.tsx") \
            == "cd my-app && tsc --noEmit"


@pytest.mark.skipif(not os.environ.get("AGENTCHANTI_LIVE_TSC"),
                    reason="needs a real Next.js project (set "
                           "AGENTCHANTI_LIVE_TSC=<project dir>)")
def test_live_project_accepts_the_project_form():
    """The variant must pass against a real scaffold — that is the whole
    claim. The file form is NOT asserted to fail: measured 2026-09-23 it
    exits 0 for a page needing nothing from tsconfig.json, which is why
    this is a variant rather than a refusal."""
    import subprocess
    root = os.environ["AGENTCHANTI_LIVE_TSC"]
    good = subprocess.run(tsc_project_variant(MEASURED), shell=True, cwd=root,
                          capture_output=True)
    assert good.returncode == 0, good.stdout[-400:]

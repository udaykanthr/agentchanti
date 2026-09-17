"""A wrong premise must not be able to delete a user's work.

`gate_safety` has refused destructive `verify:` lines since a
`taskkill /im python.exe` killed the pipeline mid-run, and its own header
records the hole that left: nothing asked the same question of a plan's
CMD step, a recovery command, or an agent `run_command`.

Measured 2026-09-17. The project scan reported a hand-made, uncommitted
Next.js application as empty, so the planner wrote:

    --STEP 1.1 [CMD] ... in the blank `my-app` shell, replacing only the
    empty directory structure
    > rmdir /s /q my-app && npm create next-app@latest my-app -- --js

Nineteen files of the user's TypeScript app were deleted and replaced with
a JavaScript scaffold. The same run also executed `taskkill /f /im
node.exe`, killing every node process on the machine.

The scan defect is fixed where it lives. This is the second line of
defence: whichever subsystem forms the wrong premise, the command that
would destroy work is refused.
"""
import os

import pytest

from agentchanti.executor import Executor
from agentchanti.orchestrator.gate_safety import (
    command_destructive_reason, removal_targets,
)

MEASURED = ('rmdir /s /q my-app && npm create next-app@latest my-app '
            '-- --js --tailwind --eslint --app --no-src-dir --use-npm --yes')


def _app(tmp_path):
    """The user's project: real files, plus ordinary build output."""
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "my-app", "app"))
    for name in ("page.tsx", "layout.tsx"):
        with open(os.path.join(root, "my-app", "app", name), "w") as fh:
            fh.write("export default function X() { return null }\n")
    os.makedirs(os.path.join(root, "my-app", "node_modules", "react"))
    with open(os.path.join(root, "my-app", "node_modules", "react",
                           "index.js"), "w") as fh:
        fh.write("module.exports = {}\n")
    return root


class TestTheMeasuredIncident:

    def test_the_command_is_refused(self, tmp_path):
        root = _app(tmp_path)
        assert command_destructive_reason(MEASURED, root)

    def test_the_files_survive_a_real_executor_call(self, tmp_path):
        root = _app(tmp_path)

        ok, out = Executor().run_command(MEASURED, cwd=root)

        assert not ok
        assert "refused to run" in out
        assert os.path.exists(os.path.join(root, "my-app", "app", "page.tsx"))

    def test_the_refusal_explains_the_premise(self, tmp_path):
        root = _app(tmp_path)
        _ok, out = Executor().run_command(MEASURED, cwd=root)
        assert "premise is probably wrong" in out
        assert "node_modules" in out, "it must say what IS still allowed"

    def test_the_machine_wide_process_kill_is_refused(self, tmp_path):
        root = _app(tmp_path)
        ok, out = Executor().run_command("taskkill /f /im node.exe", cwd=root)
        assert not ok and "refused to run" in out


class TestOrdinaryWorkIsUntouched:

    def test_clearing_node_modules_is_allowed(self, tmp_path):
        root = _app(tmp_path)

        reason = command_destructive_reason(
            r"rmdir /s /q my-app\node_modules", root)

        assert reason is None, "rm -rf node_modules && npm install is normal"

    def test_clearing_a_build_dir_is_allowed(self, tmp_path):
        root = _app(tmp_path)
        assert command_destructive_reason("rm -rf dist", root) is None
        assert command_destructive_reason("rm -rf build", root) is None

    def test_removing_something_that_does_not_exist_is_allowed(self, tmp_path):
        root = _app(tmp_path)
        assert command_destructive_reason("rm -rf nowhere", root) is None

    def test_removing_an_empty_directory_is_allowed(self, tmp_path):
        root = _app(tmp_path)
        os.makedirs(os.path.join(root, "scratch"))
        assert command_destructive_reason("rm -rf scratch", root) is None

    def test_an_ordinary_command_runs(self, tmp_path):
        ok, _out = Executor().run_command("echo hello", cwd=str(tmp_path))
        assert ok

    def test_an_install_is_not_confused_for_a_delete(self, tmp_path):
        root = _app(tmp_path)
        assert command_destructive_reason("npm install", root) is None
        assert command_destructive_reason("python -m pip install -r r.txt",
                                          root) is None


class TestTargets:

    @pytest.mark.parametrize("cmd,want", [
        ("rm -rf my-app", ["my-app"]),
        ("rm -rf ./my-app/", ["./my-app"]),
        (r"rmdir /s /q my-app", ["my-app"]),
        (r"rmdir /s /q my-app\node_modules", ["my-app/node_modules"]),
        ('Remove-Item -Recurse -Force "my-app"', ["my-app"]),
    ])
    def test_the_path_operand_is_found(self, cmd, want):
        assert removal_targets(cmd) == want

    def test_deleting_the_project_root_is_never_allowed(self, tmp_path):
        root = _app(tmp_path)
        assert command_destructive_reason("rm -rf .", root)

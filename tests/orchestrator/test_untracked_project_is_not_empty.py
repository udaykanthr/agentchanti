"""A project the user has not committed yet is still a project.

Measured 2026-09-17. A user created a Next.js + TypeScript application in
`my-app/` by hand, never committed it, and asked for a home page:

    09:39:03 Language: javascript (JavaScript)
    09:39:03 Project scan: 0 files detected, 0 source files collected
    09:39:07 Creating git checkpoint branch...
    ...
    --STEP 1.1 [CMD] Initialize a JavaScript Next.js App Router project
    ... in the blank `my-app` shell, replacing only the empty directory
    structure
    > rmdir /s /q my-app && npm create next-app@latest my-app -- --js

Nineteen files were deleted and replaced with a JavaScript scaffold. The
scan ran four seconds before the run's own git checkpoint — the commit
that made those files tracked, and the only reason they could be restored
afterwards — so at scan time every one of them was untracked and therefore
invisible. `_is_gitignored` was named for one question and implemented
another: absent from `git ls-files` is *untracked*, not *ignored*.

The intent agent had read `layout.tsx` moments earlier and knew the
application was there. One subsystem knew while another asserted the
opposite, and the empty scan is what reached the planner.
"""
import os
import subprocess

import pytest

from agentchanti.project_scanner import scan_project


def _repo(tmp_path):
    root = str(tmp_path)
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"]):
        subprocess.run(cmd, cwd=root, capture_output=True)
    return root


def _write(root, rel, text="x\n"):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _commit(root):
    subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "c"], cwd=root, capture_output=True)


class TestTheMeasuredIncident:

    def test_an_uncommitted_app_is_visible(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, "README.md", "# repo\n")
        _commit(root)                      # the repo has one commit
        # what the user made by hand, never committed
        _write(root, "my-app/package.json", '{"name":"my-app"}\n')
        _write(root, "my-app/app/page.tsx")
        _write(root, "my-app/app/layout.tsx")

        result = scan_project(root)

        assert result["file_count"] >= 4, "the app must not read as empty"
        for name in ("package.json", "page.tsx", "layout.tsx"):
            assert name in result["tree"], f"{name} was invisible to the plan"

    def test_a_repo_with_no_commits_at_all_is_visible(self, tmp_path):
        """`git init` then write code — nothing is tracked yet."""
        root = _repo(tmp_path)
        _write(root, "my-app/app/page.tsx")

        assert scan_project(root)["file_count"] >= 1


class TestWhatMustStayHidden:

    def test_a_gitignored_file_is_still_skipped(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, ".gitignore", "secret.env\nnode_modules/\n")
        _write(root, "secret.env", "KEY=1\n")
        _write(root, "app.js")
        _commit(root)

        tree = scan_project(root)["tree"]

        assert "app.js" in tree
        assert "secret.env" not in tree, "ignored means ignored"

    def test_an_ignored_directory_is_still_skipped(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, ".gitignore", "node_modules/\n")
        _write(root, "node_modules/react/index.js")
        _write(root, "app.js")
        _commit(root)

        assert "react" not in scan_project(root)["tree"]

    def test_a_tracked_file_is_still_visible(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, "app.js")
        _commit(root)

        assert "app.js" in scan_project(root)["tree"]


class TestWithoutGit:

    def test_a_plain_directory_still_scans(self, tmp_path):
        """No git at all — the hardcoded skip lists carry it, as before."""
        root = str(tmp_path)
        _write(root, "app.js")
        _write(root, "src/main.js")

        result = scan_project(root)

        assert result["file_count"] >= 2
        assert "app.js" in result["tree"]

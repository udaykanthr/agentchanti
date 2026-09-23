"""A nested git repository is still part of the project.

Measured 2026-09-21: `create-next-app` ran `git init` inside `my-app/`, the
snapshot repo at the parent committed it as a gitlink, and the next scan
reported `0 files detected` over a 22-file app — telling the planner the
directory was BLANK.
"""
import os
import shutil
import subprocess

import pytest

from agentchanti.orchestrator.wave_snapshots import ProjectSnapshots
from agentchanti.project_scanner import _get_git_tracked_files

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="needs git")

_ID = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _git(cwd, *args):
    subprocess.run(["git", *_ID, *args], cwd=cwd, check=True,
                   capture_output=True)


def _app(root):
    app = os.path.join(root, "my-app")
    os.makedirs(os.path.join(app, "app"))
    os.makedirs(os.path.join(app, "node_modules", "react"))
    for rel, body in {"app/page.tsx": "export default function P(){}\n",
                      "package.json": "{}\n",
                      ".gitignore": "node_modules/\n",   # as create-next-app writes
                      "node_modules/react/index.js": "x\n"}.items():
        with open(os.path.join(app, *rel.split("/")), "w") as fh:
            fh.write(body)
    _git(app, "init", "-q")
    _git(app, "add", "-A")
    _git(app, "commit", "-qm", "scaffold")
    return app


def test_a_committed_gitlink_is_walked(tmp_path):
    root = str(tmp_path)
    _app(root)
    _git(root, "init", "-q")
    _git(root, "add", "-A")                    # records my-app as a gitlink
    _git(root, "commit", "-qm", "snapshot")

    files = _get_git_tracked_files(root)

    assert "my-app/app/page.tsx" in files
    assert "my-app" not in files


def test_an_untracked_nested_repo_is_walked(tmp_path):
    root = str(tmp_path)
    _app(root)
    _git(root, "init", "-q")

    files = _get_git_tracked_files(root)

    assert "my-app/app/page.tsx" in files
    assert not any("node_modules" in f for f in files)


def test_snapshots_report_what_they_cannot_cover(tmp_path, caplog):
    root = str(tmp_path)
    _app(root)

    snaps = ProjectSnapshots(root)
    assert snaps.nested_repos() == ["my-app"]
    with caplog.at_level("WARNING"):
        snaps.start()
    assert "do NOT cover" in caplog.text and "--restore" in caplog.text

"""Whatever existed before the run can be restored afterwards.

Measured 2026-09-17. A user created a Next.js + TypeScript application in
`my-app/` by hand, never committed it, and asked for a home page. The
project scan counted only git-tracked files, reported `0 files detected`,
and the planner — handed what looked like an empty directory — wrote
`rmdir /s /q my-app && npm create next-app@latest my-app`. Nineteen files
of real work were deleted.

Both causes are fixed elsewhere: the scan sees untracked files now, and
`command_destructive_reason` refuses the command. This module exists
because neither is a guarantee. The recovery that day worked by luck —
`create_checkpoint_branch` is gated on the directory already being a git
repository, and in `cli.py` it ran 324 lines AFTER the scan that formed
the wrong premise. In a plain folder there was no snapshot at all.

Guards decide what is allowed to happen. This decides what can be undone.
"""
import json
import os
import shutil

from agentchanti.snapshot import (
    MANIFEST_NAME,
    SNAPSHOT_ROOT,
    latest_snapshot,
    restore_snapshot,
    take_snapshot,
)


def _project(root, with_deps=True):
    """A stock create-next-app layout, as the incident had it."""
    os.makedirs(os.path.join(root, "my-app", "app"))
    files = {
        "my-app/app/page.tsx": "export default function Home() { return null }\n",
        "my-app/app/layout.tsx": "export default function L() { return null }\n",
        "my-app/package.json": '{"name": "my-app"}\n',
        "README.md": "# project\n",
    }
    for rel, body in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
    if with_deps:
        dep = os.path.join(root, "my-app", "node_modules", "react")
        os.makedirs(dep)
        with open(os.path.join(dep, "index.js"), "w") as fh:
            fh.write("module.exports = {}\n")
    return files


class TestTheMeasuredIncident:

    def test_a_deleted_project_comes_back(self, tmp_path):
        root = str(tmp_path)
        files = _project(root)

        take_snapshot(root)
        shutil.rmtree(os.path.join(root, "my-app"))     # the rmdir /s /q
        ok, detail = restore_snapshot(root)

        assert ok, detail
        for rel, body in files.items():
            path = os.path.join(root, *rel.split("/"))
            assert os.path.isfile(path), rel
            assert open(path, encoding="utf-8").read() == body

    def test_it_works_without_git(self, tmp_path):
        """The gap that made the incident unrecoverable in a plain folder."""
        root = str(tmp_path)
        assert not os.path.exists(os.path.join(root, ".git"))
        _project(root)

        assert take_snapshot(root) is not None
        shutil.rmtree(os.path.join(root, "my-app"))
        assert restore_snapshot(root)[0]


class TestWhatItCopies:

    def test_dependency_trees_are_not_copied(self, tmp_path):
        """Large, reproducible, and not what anyone mourns."""
        root = str(tmp_path)
        _project(root)

        snap = take_snapshot(root)

        manifest = json.load(open(os.path.join(snap, MANIFEST_NAME),
                                  encoding="utf-8"))
        assert not any("node_modules" in f for f in manifest["files"])
        assert "my-app/app/page.tsx" in manifest["files"]

    def test_its_own_directory_is_not_copied(self, tmp_path):
        """A snapshot of the snapshots grows without bound."""
        root = str(tmp_path)
        _project(root)
        take_snapshot(root)

        snap2 = take_snapshot(root)

        manifest = json.load(open(os.path.join(snap2, MANIFEST_NAME),
                                  encoding="utf-8"))
        assert not any(f.startswith(".agentchanti") for f in manifest["files"])

    def test_an_empty_project_snapshots_nothing(self, tmp_path):
        assert take_snapshot(str(tmp_path)) is None


class TestBounds:

    def test_a_huge_tree_is_refused_not_half_copied(self, tmp_path, monkeypatch):
        """A partial snapshot that looks complete is worse than none."""
        import agentchanti.snapshot as snap
        monkeypatch.setattr(snap, "MAX_FILES", 2)
        root = str(tmp_path)
        _project(root)

        assert snap.take_snapshot(root) is None
        assert snap.latest_snapshot(root) is None

    def test_the_size_bound_is_honoured(self, tmp_path, monkeypatch):
        import agentchanti.snapshot as snap
        monkeypatch.setattr(snap, "MAX_BYTES", 10)
        root = str(tmp_path)
        _project(root)

        assert snap.take_snapshot(root) is None


class TestRestoreIsNotItselfDestructive:

    def test_files_the_run_added_are_kept(self, tmp_path):
        """Undo must not destroy work either — the new file is the user's."""
        root = str(tmp_path)
        _project(root)
        take_snapshot(root)
        added = os.path.join(root, "my-app", "app", "about.tsx")
        with open(added, "w", encoding="utf-8") as fh:
            fh.write("export default function A() { return null }\n")

        restore_snapshot(root)

        assert os.path.isfile(added), "restore must not delete new work"

    def test_a_modified_file_is_put_back(self, tmp_path):
        root = str(tmp_path)
        _project(root)
        take_snapshot(root)
        page = os.path.join(root, "my-app", "app", "page.tsx")
        with open(page, "w", encoding="utf-8") as fh:
            fh.write("BROKEN\n")

        restore_snapshot(root)

        assert "export default" in open(page, encoding="utf-8").read()

    def test_restoring_with_no_snapshot_says_so(self, tmp_path):
        ok, detail = restore_snapshot(str(tmp_path))
        assert not ok and "no snapshot" in detail


class TestLatest:

    def test_the_most_recent_snapshot_is_chosen(self, tmp_path):
        root = str(tmp_path)
        _project(root)
        first = take_snapshot(root)
        # a second snapshot taken later in the same second still sorts last
        second = os.path.join(root, SNAPSHOT_ROOT, "29990101_000000")
        os.makedirs(second)
        with open(os.path.join(second, MANIFEST_NAME), "w") as fh:
            json.dump({"files": []}, fh)

        assert latest_snapshot(root) == second
        assert first != second

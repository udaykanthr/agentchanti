"""Protected-manifest guard: blind rewrites blocked, grounded edits allowed.

A run asked to add an npm script to package.json applied the edit
in memory (exact FIND match against the real file), then the Executor's
protected-file guard silently skipped the write — and the pipeline
reported 'wrote 1 file(s)' anyway. Editing a manifest is sometimes the
entire task; content derived from the file's current on-disk state is
grounded and must be writable.
"""

import json
import os
import shutil
import tempfile
import unittest

from agentchanti.executor import Executor
from agentchanti.orchestrator.memory import FileMemory


class ProtectedWriteBase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="protw_")
        self.prev = os.getcwd()
        os.chdir(self.root)
        self.pkg = {"name": "x", "scripts": {"test": "jest"}}
        with open("package.json", "w") as f:
            json.dump(self.pkg, f)

    def tearDown(self):
        os.chdir(self.prev)
        shutil.rmtree(self.root, ignore_errors=True)


class TestExecutorProtectedWrites(ProtectedWriteBase):

    def test_blind_rewrite_still_blocked(self):
        written = Executor.write_files({"package.json": "{}"})
        self.assertEqual(written, [])
        with open("package.json") as f:
            self.assertEqual(json.load(f), self.pkg)

    def test_grounded_edit_is_written(self):
        new = json.dumps({"name": "x",
                          "scripts": {"test": "jest", "build:css": "tw"}})
        written = Executor.write_files(
            {"package.json": new}, allow_protected={"package.json"})
        self.assertEqual(len(written), 1)
        with open("package.json") as f:
            self.assertIn("build:css", f.read())

    def test_return_reflects_actual_writes(self):
        written = Executor.write_files({
            "package.json": "{}",              # blocked
            "app.py": "x = 1\n",               # written
        })
        self.assertEqual([os.path.basename(w) for w in written], ["app.py"])


class TestMemoryProtectedUpdates(ProtectedWriteBase):

    def test_blocked_without_allow(self):
        memory = FileMemory()
        memory.update({"package.json": "{}"})
        self.assertIsNone(memory.get("package.json"))

    def test_allowed_with_grounded_flag(self):
        memory = FileMemory()
        memory.update({"package.json": '{"scripts": {}}'},
                      allow_protected={"package.json"})
        self.assertEqual(memory.get("package.json"), '{"scripts": {}}')


if __name__ == "__main__":
    unittest.main()

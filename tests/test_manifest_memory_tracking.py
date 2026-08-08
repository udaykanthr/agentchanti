"""A manifest a step creates must reach FileMemory.

FileMemory's protected-basename guard tests `os.path.isfile()`, which is
true the moment the file is written — so a manifest the run had just
created looked pre-existing and was dropped, while the log claimed a skip
that protected nothing.

This has been fixed three times, once per writer, each found by a benchmark
run rather than by reading the code:

  * agent_tools.AgentTools._record          (agent loop)
  * pipeline, the plan-step inline writer   (classic, plan-declared files)
  * step_handlers, the CODE-step writer     (classic, plain CODE step)

A source survey found roughly fifteen further `write_files` → `memory.update`
pairs with the same omission (diagnosis fixes, generated tests, integration
fixes). They are latent rather than observed: they only bite when the files
being written include a protected basename. Patching each call site is the
wrong shape of fix — see the note at the bottom of this file.
"""

import os
import tempfile
import unittest

from agentchanti.executor import Executor
from agentchanti.orchestrator.memory import FileMemory


REAL_MANIFEST = "pygame==2.6.1\nnumpy==2.0\n"


class ManifestReachesMemoryTest(unittest.TestCase):
    """The write-then-record sequence the classic writers perform."""

    def _run(self, pre_existing: bool):
        d = tempfile.mkdtemp()
        prev = os.getcwd()
        os.chdir(d)
        try:
            if pre_existing:
                with open("requirements.txt", "w") as f:
                    f.write(REAL_MANIFEST)
            files = {"requirements.txt": "pygame"}
            existing_before = {p for p in files if os.path.exists(p)}
            Executor.write_files(files)
            memory = FileMemory()
            memory.update(
                files,
                allow_protected={p for p in files if p not in existing_before})
            with open("requirements.txt") as f:
                return memory.get("requirements.txt"), f.read()
        finally:
            os.chdir(prev)

    def test_created_manifest_is_tracked(self):
        tracked, disk = self._run(pre_existing=False)
        self.assertEqual(tracked, "pygame")
        self.assertEqual(disk, "pygame")

    def test_pre_existing_manifest_is_still_protected(self):
        tracked, disk = self._run(pre_existing=True)
        self.assertIsNone(tracked, "LLM content overwrote a real manifest")
        self.assertEqual(disk, REAL_MANIFEST, "real manifest was clobbered")


class GuardPlacementTest(unittest.TestCase):
    """Documents why the remaining call sites are not patched one by one.

    `Executor.write_files` already refuses to overwrite a pre-existing
    protected file — that is the guard that actually protects anything,
    because it runs before the bytes hit disk. FileMemory's copy of the
    same check runs *after*, when `os.path.isfile()` can no longer
    distinguish "existed before this run" from "we just wrote it", so it
    cannot protect and can only desynchronise memory from disk.

    The systemic fix is to let the disk layer own the guard and have
    FileMemory record what it is given. This test pins the asymmetry that
    makes the current arrangement wrong, so the reasoning survives.
    """

    def test_disk_layer_refuses_before_writing(self):
        d = tempfile.mkdtemp()
        prev = os.getcwd()
        os.chdir(d)
        try:
            with open("requirements.txt", "w") as f:
                f.write(REAL_MANIFEST)
            written = Executor.write_files({"requirements.txt": "pygame"})
            self.assertEqual(written, [],
                             "write_files must refuse a pre-existing manifest")
            with open("requirements.txt") as f:
                self.assertEqual(f.read(), REAL_MANIFEST)
        finally:
            os.chdir(prev)

    def test_memory_guard_cannot_tell_created_from_pre_existing(self):
        """Both cases look identical to FileMemory by the time it runs."""
        d = tempfile.mkdtemp()
        prev = os.getcwd()
        os.chdir(d)
        try:
            Executor.write_files({"requirements.txt": "pygame"})
            memory = FileMemory()
            memory.update({"requirements.txt": "pygame"})   # no allowlist
            self.assertIsNone(
                memory.get("requirements.txt"),
                "if this now passes, the guard was moved and this file's "
                "premise should be revisited")
        finally:
            os.chdir(prev)


if __name__ == "__main__":
    unittest.main()

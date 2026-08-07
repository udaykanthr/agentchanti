"""Tests that the agent loop cannot silently replace a real manifest.

The classic writer has always refused to overwrite a dependency manifest it
did not create (`Executor.write_files`). The loop's `write_file` went
straight to disk with no such check, so a model could replace a project's
real requirements.txt or package.json with a shorter regenerated one and
drop dependencies — every later step then building against a different
dependency set than the project actually has.

The test is create-versus-overwrite, not existence: creating a manifest is
legitimate and common (5 of 8 benchmark runs did it). Across steps the
answer comes from FileMemory, because build_step_tools() makes a fresh
AgentTools per step.
"""

import os
import tempfile
import unittest

from agentchanti.agent_tools import AgentTools
from agentchanti.llm.chat_types import ToolCall
from agentchanti.orchestrator.memory import FileMemory


REAL_MANIFEST = "pygame==2.6.1\nnumpy==2.0\n"


class LoopProtectedManifestTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.memory = FileMemory()
        self.tools = AgentTools(project_root=self.root, memory=self.memory)

    def _write(self, tools, path, content):
        return tools.execute(ToolCall(name="write_file",
                                      arguments={"path": path,
                                                 "content": content}))

    def _disk(self, name):
        with open(os.path.join(self.root, name), encoding="utf-8") as f:
            return f.read()

    # ── the regression ────────────────────────────────────────────────
    def test_pre_existing_manifest_is_not_overwritten(self):
        with open(os.path.join(self.root, "requirements.txt"), "w") as f:
            f.write(REAL_MANIFEST)
        result = self._write(self.tools, "requirements.txt", "pygame")
        self.assertTrue(result.startswith("ERROR:"), result)
        self.assertEqual(self._disk("requirements.txt"), REAL_MANIFEST,
                         "the real manifest was clobbered")

    def test_refusal_names_the_grounded_alternative(self):
        with open(os.path.join(self.root, "package.json"), "w") as f:
            f.write('{"dependencies": {"react": "^19.0.0"}}')
        result = self._write(self.tools, "package.json", "{}")
        self.assertIn("edit_file", result,
                      "refusal must point at the tool that can do this safely")

    def test_edit_file_still_works_on_a_protected_manifest(self):
        path = os.path.join(self.root, "requirements.txt")
        with open(path, "w") as f:
            f.write(REAL_MANIFEST)
        result = self.tools.execute(ToolCall(
            name="edit_file",
            arguments={"path": "requirements.txt",
                       "old_text": "numpy==2.0",
                       "new_text": "numpy==2.0\nrequests==2.32"}))
        self.assertTrue(result.startswith("OK:"), result)
        self.assertIn("pygame==2.6.1", self._disk("requirements.txt"))
        self.assertIn("requests==2.32", self._disk("requirements.txt"))

    # ── creating and updating within the run stay allowed ─────────────
    def test_creating_a_manifest_is_allowed(self):
        result = self._write(self.tools, "requirements.txt", "pygame")
        self.assertTrue(result.startswith("OK:"), result)
        self.assertEqual(self._disk("requirements.txt"), "pygame")

    def test_rewriting_within_the_same_step_is_allowed(self):
        self._write(self.tools, "requirements.txt", "pygame")
        result = self._write(self.tools, "requirements.txt", "pygame\nnumpy")
        self.assertTrue(result.startswith("OK:"), result)

    def test_rewriting_in_a_later_step_of_the_same_run_is_allowed(self):
        """build_step_tools() makes a new AgentTools; FileMemory carries."""
        self._write(self.tools, "requirements.txt", "pygame")
        next_step = AgentTools(project_root=self.root, memory=self.memory)
        result = self._write(next_step, "requirements.txt", "pygame\nnumpy")
        self.assertTrue(result.startswith("OK:"), result)
        self.assertEqual(self._disk("requirements.txt"), "pygame\nnumpy")

    def test_ordinary_source_files_are_unaffected(self):
        self._write(self.tools, "game.py", "x = 1")
        result = self._write(self.tools, "game.py", "x = 2")
        self.assertTrue(result.startswith("OK:"), result)
        self.assertEqual(self._disk("game.py"), "x = 2")

    def test_guard_does_not_block_when_no_memory_is_attached(self):
        """A created manifest is still rewritable without FileMemory."""
        tools = AgentTools(project_root=self.root, memory=None)
        self._write(tools, "requirements.txt", "pygame")
        result = self._write(tools, "requirements.txt", "pygame\nnumpy")
        self.assertTrue(result.startswith("OK:"), result)


if __name__ == "__main__":
    unittest.main()

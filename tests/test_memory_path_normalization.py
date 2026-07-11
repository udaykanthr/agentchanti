"""FileMemory must treat foo/bar.py and foo\\bar.py as one file.

Pipeline writers use forward slashes; agent-loop tool writes use OS
separators. Un-normalised keys tracked the same file twice, which
double-reported every DepCheck finding.
"""

import unittest

from agentchanti.orchestrator.memory import FileMemory


class TestPathNormalization(unittest.TestCase):

    def test_mixed_separators_collapse_to_one_entry(self):
        memory = FileMemory()
        memory.update({"app/config/settings.py": "v1"})
        memory.update({"app\\config\\settings.py": "v2"})
        self.assertEqual(len(memory.all_files()), 1)
        self.assertEqual(memory.get("app/config/settings.py"), "v2")

    def test_get_accepts_either_separator(self):
        memory = FileMemory()
        memory.update({"pkg/mod.py": "x = 1"})
        self.assertEqual(memory.get("pkg\\mod.py"), "x = 1")
        self.assertEqual(memory.get("pkg/mod.py"), "x = 1")

    def test_delete_accepts_either_separator(self):
        memory = FileMemory()
        memory.update({"pkg/mod.py": "x = 1"})
        memory.delete("pkg\\mod.py")
        self.assertEqual(memory.all_files(), {})


if __name__ == "__main__":
    unittest.main()

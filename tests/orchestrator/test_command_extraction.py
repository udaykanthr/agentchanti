import unittest
from agentchanti.orchestrator.classification import _extract_commands_from_text

class TestCommandExtraction(unittest.TestCase):
    def test_multiline_merge_and_cleanup(self):
        # Case 1: Standard bash-style line continuation with trailing slash
        text = "```bash\nnpm install \\\n  package1 \\\n  package2\n```"
        cmds = _extract_commands_from_text(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0], "npm install package1 package2")

    def test_dangling_operators(self):
        # Case 2: Dangling && \ (from user's request)
        text = "```bash\nnpm install && \\\n```"
        cmds = _extract_commands_from_text(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0], "npm install")

    def test_dangling_backslash(self):
        # Case 3: Dangling backslash without &&
        text = "```bash\ndir \\\n```"
        cmds = _extract_commands_from_text(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0], "dir")

    def test_complex_multiline(self):
        # Case 4: Multiple commands, one multiline
        text = """
Step 1:
```bash
cd myapp && \\
npm install
ls -la
```
"""
        cmds = _extract_commands_from_text(text)
        # "cd myapp && \" -> merges with "npm install" -> "cd myapp && npm install"
        # "ls -la"
        self.assertEqual(len(cmds), 2)
        self.assertEqual(cmds[0], "cd myapp && npm install")
        self.assertEqual(cmds[1], "ls -la")

    def test_inline_cleanup(self):
        # Case 5: Inline backticks with dangling chars
        text = "Run `npm install && \\` to continue."
        cmds = _extract_commands_from_text(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0], "npm install")

    def test_heredoc_preservation(self):
        # Case 6: Heredoc should NOT be merged/split incorrectly
        text = "```bash\ncat << 'EOF' > file.txt\nline 1\nline 2\nEOF\n```"
        cmds = _extract_commands_from_text(text)
        self.assertEqual(len(cmds), 1)
        self.assertIn("cat << 'EOF'", cmds[0])
        self.assertIn("EOF", cmds[0])

if __name__ == '__main__':
    unittest.main()

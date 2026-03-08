"""
Tests for the PlanOptimizer — KB command extraction and override logic.
"""

from __future__ import annotations

import unittest


class TestExtractCommandsFromKBDoc(unittest.TestCase):
    """Tests for _extract_commands_from_kb_doc negation awareness."""

    def test_extracts_normal_commands(self):
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "## Setup\n"
            "```bash\n"
            "npm install tailwindcss @tailwindcss/vite\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        self.assertEqual(cmds, ["npm install tailwindcss @tailwindcss/vite"])

    def test_skips_deprecated_block(self):
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "## v4 Setup\n\n"
            "Install the package:\n"
            "```bash\n"
            "npm install tailwindcss @tailwindcss/vite\n"
            "```\n\n"
            "The following command is **deprecated** in v4:\n"
            "```bash\n"
            "npx tailwindcss init\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        self.assertEqual(cmds, ["npm install tailwindcss @tailwindcss/vite"])
        self.assertNotIn("npx tailwindcss init", cmds)

    def test_skips_do_not_block(self):
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "## Correct way\n"
            "```bash\n"
            "npm install tailwindcss\n"
            "```\n\n"
            "Do NOT use the old v3 init command:\n"
            "```bash\n"
            "npx tailwindcss init -p\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        self.assertEqual(cmds, ["npm install tailwindcss"])

    def test_skips_no_longer_needed(self):
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "The init command is no longer needed in v4:\n"
            "```bash\n"
            "npx tailwindcss init\n"
            "```\n\n"
            # Pad to push the next block beyond the 200-char negation window
            "## Correct v4 Setup\n\n"
            "Add the Tailwind CSS import to your main CSS file. "
            "This replaces all the old directives and config files. "
            "Simply add the following line to get started with v4. "
            "No configuration file is needed.\n\n"
            "```bash\n"
            "echo '@import \"tailwindcss\";' > src/index.css\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        # "no longer" should skip the first block
        self.assertNotIn("npx tailwindcss init", cmds)
        # The echo command is far enough from negation text to be included
        self.assertEqual(len(cmds), 1)

    def test_skips_legacy_block(self):
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "## Modern setup\n"
            "```bash\n"
            "npm install tailwindcss\n"
            "```\n\n"
            "Legacy / old way (v3):\n"
            "```bash\n"
            "npx tailwindcss init -p\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        self.assertEqual(cmds, ["npm install tailwindcss"])


    def test_negation_separated_by_heading_keeps_block(self):
        """Negation in a DIFFERENT section (separated by ##) should not
        suppress the code block in the NEXT section.

        Reproduces the real-world bug where:
          *** Do not create TS files ***
          ## 1. Required Packages
          ```bash
          npm install -D vitest jsdom
          ```
        incorrectly skipped the install command.
        """
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "*** Do not create TS or TSX files if JS or JSX already exists ***\n\n"
            "## 1. Required Packages\n\n"
            "Install the following development dependencies:\n\n"
            "```bash\n"
            "npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        self.assertEqual(
            cmds,
            ["npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom"],
        )

    def test_negation_same_section_still_skips(self):
        """Negation in the SAME section (no heading between) should still skip."""
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _extract_commands_from_kb_doc,
        )

        content = (
            "## Deprecated Commands\n\n"
            "Do NOT use the old init command:\n"
            "```bash\n"
            "npx tailwindcss init -p\n"
            "```\n"
        )
        cmds = _extract_commands_from_kb_doc(content)
        self.assertEqual(cmds, [])


class TestIsDeprecatedCommand(unittest.TestCase):
    """Tests for _is_deprecated_command error dict check."""

    def test_deprecated_init_detected(self):
        """tailwindcss init matches TailwindCSSDeprecatedInit in error dict."""
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _is_deprecated_command,
        )
        from multi_agent_coder.kb.context_builder import ContextBuilder
        from multi_agent_coder.kb.global_kb.seeder import seed

        # Ensure error dict is seeded
        seed(embed=False)

        cb = ContextBuilder()
        cb._ensure_global()

        result = _is_deprecated_command("npx tailwindcss init", cb)
        self.assertTrue(result)

    def test_normal_command_not_deprecated(self):
        from multi_agent_coder.orchestrator.plan_optimizer import (
            _is_deprecated_command,
        )
        from multi_agent_coder.kb.context_builder import ContextBuilder

        cb = ContextBuilder()
        cb._ensure_global()

        result = _is_deprecated_command("npm install tailwindcss", cb)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

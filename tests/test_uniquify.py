"""Tests for the uniquify_context utility in agents.base."""

from multi_agent_coder.agents.base import uniquify_context


class TestUniquifyContext:
    """Test duplicate block removal from LLM context."""

    def test_empty_string(self):
        assert uniquify_context("") == ""

    def test_no_duplicates(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert uniquify_context(text) == text

    def test_exact_duplicate_removed(self):
        block = "A" * 120  # above min_size
        text = f"{block}\n\n{block}"
        assert uniquify_context(text) == block

    def test_whitespace_normalized_duplicate(self):
        """Blocks differing only in whitespace should be treated as duplicates."""
        block1 = "This is a long enough paragraph that exceeds the minimum size threshold for deduplication to kick in properly."
        block2 = "This is a long enough paragraph  that  exceeds  the  minimum  size  threshold  for  deduplication  to  kick  in  properly."
        text = f"{block1}\n\n{block2}"
        result = uniquify_context(text)
        assert result == block1

    def test_short_blocks_always_kept(self):
        """Blocks shorter than min_size should never be removed, even if duplicated."""
        short = "short block"
        text = f"{short}\n\n{short}\n\n{short}"
        assert uniquify_context(text) == text

    def test_mixed_short_and_long_duplicates(self):
        """Short duplicates kept, long duplicates removed."""
        short = "---"
        long_block = "X" * 150
        text = f"{short}\n\n{long_block}\n\n{short}\n\n{long_block}"
        result = uniquify_context(text)
        assert result == f"{short}\n\n{long_block}\n\n{short}"

    def test_preserves_first_occurrence(self):
        """The first occurrence of a duplicated block should be kept."""
        block_a = "A" * 120
        block_b = "B" * 120
        text = f"{block_a}\n\n{block_b}\n\n{block_a}"
        result = uniquify_context(text)
        assert result == f"{block_a}\n\n{block_b}"

    def test_case_insensitive_dedup(self):
        """Dedup should be case-insensitive."""
        block_lower = "a" * 120
        block_upper = "A" * 120
        text = f"{block_lower}\n\n{block_upper}"
        result = uniquify_context(text)
        assert result == block_lower

    def test_custom_min_size(self):
        """Custom min_size changes the threshold."""
        block = "Medium block here"  # 17 chars
        text = f"{block}\n\n{block}"
        # With default min_size=100, both kept
        assert uniquify_context(text) == text
        # With min_size=10, second is removed
        assert uniquify_context(text, min_size=10) == block

    def test_realistic_kb_duplication(self):
        """Simulate the real-world KB duplication issue: same guide appearing twice.

        Each section must be >100 chars (min_size) so dedup kicks in.
        In practice, KB guide sections are each 200+ chars.
        """
        section1 = (
            "## Overview\n"
            "Tailwind CSS v4 uses a completely new installation and configuration approach. "
            "The old npx tailwindcss init command and tailwind.config.js configuration file "
            "are no longer supported."
        )
        section2 = (
            "## Installation\n"
            "Run npm create vite@latest my-app -- --template react to create a new project. "
            "Then install tailwindcss and the Vite plugin with npm install tailwindcss "
            "@tailwindcss/vite --save-dev."
        )
        other_context = "Project uses React with Vite build system and has 5 components."
        # Both sections appear twice, separated by other content
        text = f"{section1}\n\n{section2}\n\n{other_context}\n\n{section1}\n\n{section2}"
        result = uniquify_context(text)
        assert result == f"{section1}\n\n{section2}\n\n{other_context}"

    def test_none_returns_none(self):
        assert uniquify_context(None) is None

    def test_triple_newline_split(self):
        """Blocks separated by 3+ newlines should still be split and deduped."""
        block = "Z" * 120
        text = f"{block}\n\n\n{block}"
        result = uniquify_context(text)
        assert result == block

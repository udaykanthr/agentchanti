"""Tests for Executor.parse_code_blocks with nested Markdown fences.

The extractor used a non-greedy body (``(.*?)\\n```py``), which ends at the
FIRST fence line inside the block. Right for source files, catastrophic for
Markdown, whose content is full of fences.

Observed in the test1 Pac-Man benchmark (classic mode): an 808-token README
completion was written as 15 lines / 417 bytes, cut off mid-sentence at
"install the required dependency:". Every command the step's verify gate
looked for — ``python -m pip install -r requirements.txt``, ``python main.py``,
``python -m unittest -v`` — lived inside a fence and was dropped. The gate
failed correctly, but the truncation is deterministic: all three diagnosis
attempts regenerated the same document, got the same 15 lines, logged
"previous fix changed nothing", and the pipeline halted at step 11 of 12
having never written the tests.
"""

import unittest

from agentchanti.executor import Executor


README_RESPONSE = """\
Here is the documentation file.

#### [FILE]: README.md
```markdown
# Pac-Man Clone

## Setup

Install the required dependency:

```
python -m pip install -r requirements.txt
```

## Running

```
python main.py
```

## Tests

```
python -m unittest -v
```
```
"""

GATE_STRINGS = (
    "python -m pip install -r requirements.txt",
    "python main.py",
    "python -m unittest -v",
)


class NestedFenceTest(unittest.TestCase):
    # ── the regression ────────────────────────────────────────────────
    def test_readme_keeps_content_after_the_first_inner_fence(self):
        files = Executor.parse_code_blocks(README_RESPONSE)
        self.assertIn("README.md", files)
        readme = files["README.md"]
        for needle in GATE_STRINGS:
            self.assertIn(needle, readme,
                          f"{needle!r} was dropped — fence truncation is back")

    def test_readme_is_not_cut_off_mid_document(self):
        readme = Executor.parse_code_blocks(README_RESPONSE)["README.md"]
        self.assertIn("## Tests", readme, "document ends before its last section")
        self.assertNotIn("#### [FILE]", readme, "marker leaked into content")

    # ── the old behaviour must not come back by another route ─────────
    def test_source_file_stops_at_its_own_closing_fence(self):
        """A follow-up example block must not be swallowed into the file."""
        text = ("#### [FILE]: mod.py\n```python\n"
                "def f():\n    return 1\n```\n"
                "Example usage:\n```python\nprint(f())\n```\n")
        self.assertEqual(Executor.parse_code_blocks(text)["mod.py"],
                         "def f():\n    return 1")

    def test_longer_outer_fence_is_parsed_exactly(self):
        """A ```` fence around ``` content needs no guessing at all."""
        text = ("#### [FILE]: doc.md\n````markdown\n"
                "# Title\n\n```\ninner code\n```\n````\n")
        self.assertEqual(Executor.parse_code_blocks(text)["doc.md"],
                         "# Title\n\n```\ninner code\n```")

    # ── unchanged behaviour ───────────────────────────────────────────
    def test_multiple_files_still_parse(self):
        text = ("#### [FILE]: a.py\n```python\nA = 1\nB = 2\nC = 3\n```\n"
                "#### [FILE]: b.py\n```python\nD = 4\nE = 5\nF = 6\n```\n")
        files = Executor.parse_code_blocks(text)
        self.assertEqual(sorted(files), ["a.py", "b.py"])
        self.assertEqual(files["a.py"], "A = 1\nB = 2\nC = 3")

    def test_a_markdown_file_between_two_source_files(self):
        """The nested fences must not swallow the file that follows."""
        text = ("#### [FILE]: a.py\n```python\nA = 1\nB = 2\nC = 3\n```\n"
                "#### [FILE]: README.md\n```markdown\n# Doc\n\n```\nrun me\n```\n```\n"
                "#### [FILE]: b.py\n```python\nD = 4\nE = 5\nF = 6\n```\n")
        files = Executor.parse_code_blocks(text)
        self.assertEqual(sorted(files), ["README.md", "a.py", "b.py"])
        self.assertIn("run me", files["README.md"])
        self.assertEqual(files["b.py"], "D = 4\nE = 5\nF = 6")

    def test_unclosed_fence_yields_nothing(self):
        text = "#### [FILE]: a.py\n```python\nA = 1\nB = 2\n"
        self.assertEqual(Executor.parse_code_blocks(text), {})

    def test_no_marker_yields_nothing(self):
        self.assertEqual(Executor.parse_code_blocks("```python\nA = 1\n```"), {})


if __name__ == "__main__":
    unittest.main()

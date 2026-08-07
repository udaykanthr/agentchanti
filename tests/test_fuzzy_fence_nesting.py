"""Tests for parse_code_blocks_fuzzy with nested and line-internal fences.

Every pattern in the fuzzy parser searched for its block body with an
unanchored, non-greedy ``(.*?)``` ``. Two consequences, both observed:

1. A block ended at the first ``` ANYWHERE, including one inside a Markdown
   document's own body. Run against the README that halted a benchmark
   pipeline, the fuzzy parser returned the same truncated few lines the
   strict parser did.
2. Worse, the lines inside that document then kept matching Pattern 3, whose
   filename comes from the line ABOVE a fence. The README's own usage
   examples became files to write: phantom ``requirements.txt`` and
   ``main.py`` entries generated out of its install and run instructions.

An unanchored search also ends a diff block on its own ``+``` `` line, so a
diff touching any Markdown file was truncated at the first fence it added.
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
```
"""


class FuzzyNestedFenceTest(unittest.TestCase):
    # ── the regressions ───────────────────────────────────────────────
    def test_readme_is_not_truncated_at_the_first_inner_fence(self):
        files = Executor.parse_code_blocks_fuzzy(README_RESPONSE)
        self.assertIn("README.md", files)
        readme = files["README.md"]
        self.assertIn("python -m pip install -r requirements.txt", readme)
        self.assertIn("python main.py", readme)
        self.assertIn("## Running", readme)

    def test_no_phantom_files_from_the_documents_own_examples(self):
        """The README's examples are its content, not files to write."""
        files = Executor.parse_code_blocks_fuzzy(README_RESPONSE)
        self.assertEqual(sorted(files), ["README.md"])

    def test_diff_block_survives_its_own_added_fences(self):
        text = ("```diff\n"
                "+#### [FILE]: docs.md\n"
                "+# Title\n"
                "+\n"
                "+```\n"
                "+run me\n"
                "+```\n"
                "+done\n"
                "```\n")
        files = Executor.parse_code_blocks_fuzzy(text)
        self.assertIn("docs.md", files)
        self.assertIn("run me", files["docs.md"])
        self.assertIn("done", files["docs.md"],
                      "block ended early on the diff's own fence")

    # ── unchanged behaviour ───────────────────────────────────────────
    def test_marker_inside_a_python_block_still_parses(self):
        text = ("```python\n#### [FILE]: mod.py\n"
                "A = 1\nB = 2\nC = 3\n```\n")
        self.assertEqual(Executor.parse_code_blocks_fuzzy(text)["mod.py"],
                         "A = 1\nB = 2\nC = 3")

    def test_filepath_comment_first_line_still_parses(self):
        text = "```python\n# pkg/mod.py\nA = 1\nB = 2\nC = 3\n```\n"
        self.assertEqual(Executor.parse_code_blocks_fuzzy(text)["pkg/mod.py"],
                         "A = 1\nB = 2\nC = 3")

    def test_named_source_file_before_a_plain_block_still_parses(self):
        text = "Update `mod.py`:\n```python\nA = 1\nB = 2\nC = 3\n```\n"
        self.assertEqual(Executor.parse_code_blocks_fuzzy(text)["mod.py"],
                         "A = 1\nB = 2\nC = 3")

    def test_two_named_source_files_both_parse(self):
        """Consumption tracking must not swallow the second file."""
        text = ("`a.py`:\n```python\nA = 1\nB = 2\nC = 3\n```\n"
                "`b.py`:\n```python\nD = 4\nE = 5\nF = 6\n```\n")
        files = Executor.parse_code_blocks_fuzzy(text)
        self.assertEqual(sorted(files), ["a.py", "b.py"])
        self.assertEqual(files["b.py"], "D = 4\nE = 5\nF = 6")


class TopLevelBlockIterTest(unittest.TestCase):
    def test_inner_fences_do_not_yield_their_own_blocks(self):
        text = "```markdown\n# T\n\n```\ninner\n```\n```\n"
        blocks = list(Executor._iter_fenced_blocks(text))
        self.assertEqual(len(blocks), 1)
        self.assertIn("inner", blocks[0][1])

    def test_sequential_blocks_are_all_yielded(self):
        text = "```python\nA = 1\n```\n```python\nB = 2\n```\n"
        blocks = list(Executor._iter_fenced_blocks(text))
        self.assertEqual([b[1] for b in blocks], ["A = 1", "B = 2"])


if __name__ == "__main__":
    unittest.main()

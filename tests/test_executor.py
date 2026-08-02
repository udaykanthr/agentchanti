"""Code-block extraction from a coder response."""

from agentchanti.executor import Executor


class TestSingleTargetBlockAttribution:
    """A model that answers in prose with a bare fence produced NOTHING.

    Every other extractor needs the model to name the file — a
    ``#### [FILE]:`` marker, a path after the fence language, a ``# path``
    first line — and the KB symbol fallback needs an index a blank project
    does not have at step 2.

    Observed on Gemini: correct, complete code with no filename anywhere,
    "No files parsed from coder response" twice, two diagnosis rounds,
    then the pipeline halted after 12 minutes and 129k tokens having
    written nothing. When the step declares exactly ONE target there is
    nothing to guess.
    """

    RESPONSE = (
        "Here is the implementation.\n\n"
        "```python\n"
        "class Map:\n"
        "    def __init__(self):\n"
        "        self.grid = []\n"
        "```\n\n"
        "That validates invariant 3."
    )

    def test_attributes_the_block_to_the_only_target(self):
        files = Executor.parse_blocks_for_single_target(
            self.RESPONSE, "pacman.py")
        assert list(files) == ["pacman.py"]
        assert "class Map" in files["pacman.py"]

    def test_the_largest_block_wins(self):
        """Explanatory snippets are short; the implementation is not."""
        text = ("```python\na = 1\nb = 2\nc = 3\n```\n"
                "and the real file:\n"
                "```python\nclass Big:\n    x = 1\n    y = 2\n"
                "    z = 3\n    w = 4\n```")
        got = Executor.parse_blocks_for_single_target(text, "t.py")["t.py"]
        assert "class Big" in got

    def test_a_short_fragment_is_not_a_file(self):
        """Two lines is something being discussed, not a deliverable."""
        assert Executor.parse_blocks_for_single_target(
            "see ```\nx = 1\n```", "a.py") == {}

    def test_no_target_means_no_guess(self):
        assert Executor.parse_blocks_for_single_target(self.RESPONSE, "") == {}

    def test_no_code_blocks_yields_nothing(self):
        assert Executor.parse_blocks_for_single_target(
            "just prose, no code", "a.py") == {}

    def test_empty_input_is_safe(self):
        assert Executor.parse_blocks_for_single_target("", "a.py") == {}
        assert Executor.parse_blocks_for_single_target(None, "a.py") == {}

    def test_a_fence_with_no_language_still_works(self):
        text = "```\nclass A:\n    x = 1\n    y = 2\n```"
        assert "class A" in Executor.parse_blocks_for_single_target(
            text, "a.py")["a.py"]

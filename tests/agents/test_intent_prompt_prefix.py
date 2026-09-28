"""The intent loop's prompts must share a cacheable prefix.

Profiled 2026-09-26: intent was the largest single phase of a run —
27–33% of sent tokens, 7 calls per run on the snake task — and cached
**0%** in every profiled run.

The cause was ordering, not content. `_build_prompt` put the evidence
(which grows every iteration) ahead of ~2,582 tokens of invariant
instructions, so two successive prompts shared a common prefix of 155
characters, about 38 tokens. OpenAI's automatic prompt cache needs 1,024
tokens of shared prefix, so nothing could ever be cached.

Moving the evidence to the end makes the prefix ~2,621 tokens. This test
pins that, because the defect is invisible in behaviour — the prompts are
correct either way, the run simply costs more.
"""
import pytest

from agentchanti.agents.intent import IntentAgent, _evidence_block

# OpenAI caches only a prefix of at least this many tokens.
CACHE_MINIMUM_TOKENS = 1024
TASK = "Build a production-ready command-line todo manager in Python."


def _agent():
    a = object.__new__(IntentAgent)
    a.role = "Requirements analyst"
    a.goal = "Turn a user request into a REQUIREMENTS_SPEC"
    return a


def _common_prefix_chars(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


@pytest.fixture
def prompts():
    a = _agent()
    ctx2 = "KB_SEARCH result: " + ("x" * 4000)
    ctx3 = ctx2 + "\nREAD_FILE main.py: " + ("y" * 3000)
    return (a._build_prompt(TASK, ""),
            a._build_prompt(TASK, ctx2),
            a._build_prompt(TASK, ctx3))


class TestThePrefixIsCacheable:

    def test_successive_iterations_share_more_than_the_cache_minimum(self, prompts):
        p1, p2, p3 = prompts
        for label, (x, y) in (("1<->2", (p1, p2)), ("2<->3", (p2, p3)),
                              ("1<->3", (p1, p3))):
            shared_tokens = _common_prefix_chars(x, y) // 4
            assert shared_tokens >= CACHE_MINIMUM_TOKENS, (
                f"{label} shares only ~{shared_tokens} tokens; the cache "
                f"needs {CACHE_MINIMUM_TOKENS}")

    def test_the_evidence_comes_after_the_instructions(self, prompts):
        _p1, p2, _p3 = prompts
        assert p2.index("STEP 1") < p2.index("Evidence gathered so far")

    def test_the_conclude_branch_is_ordered_the_same_way(self):
        p = _agent()._build_prompt(TASK, "some evidence", conclude=True)
        assert p.index("REQUIREMENTS_SPEC") < p.index("Evidence gathered so far")


class TestNothingWasLost:
    """A cheaper prompt that says less would not be an optimisation."""

    def test_the_evidence_is_still_present(self, prompts):
        _p1, p2, p3 = prompts
        assert "KB_SEARCH result" in p2
        assert "READ_FILE main.py" in p3

    def test_the_task_is_still_present(self, prompts):
        for p in prompts:
            assert TASK in p

    def test_the_first_iteration_has_no_evidence_header(self, prompts):
        assert "Evidence gathered so far" not in prompts[0]

    def test_longer_evidence_makes_a_longer_prompt(self, prompts):
        p1, p2, p3 = prompts
        assert len(p1) < len(p2) < len(p3)


class TestTheHelperIsShared:
    """Both branches must append a BYTE-IDENTICAL block: a second copy of
    this string is exactly how the prefix stops matching."""

    def test_empty_context_adds_nothing(self):
        assert _evidence_block("") == ""
        assert _evidence_block(None or "") == ""

    def test_both_branches_use_the_helper(self):
        import inspect

        from agentchanti.agents import intent
        src = inspect.getsource(intent)
        assert src.count("_evidence_block(accumulated_context)") == 2

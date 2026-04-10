"""Tests for IntentAgent directive sanitization and LLM reasoning-tag stripping.

These cover the helpers added to defend against reasoning models (glm-5,
deepseek-r1, qwq, o1, Claude extended thinking) that leak <think> blocks
or hallucinate fake system banners into directive arguments.
"""
from agentchanti.agents.intent import (
    _normalize_dedup_key,
    _sanitize_directive_arg,
)
from agentchanti.llm.base import _strip_reasoning


# ── _strip_reasoning ────────────────────────────────────────────────────────


def test_strip_reasoning_removes_paired_block():
    text = "<think>internal reasoning here</think>actual answer"
    assert _strip_reasoning(text) == "actual answer"


def test_strip_reasoning_removes_multiline_block():
    text = "<think>line one\nline two\nline three</think>\nfinal answer"
    assert _strip_reasoning(text) == "final answer"


def test_strip_reasoning_handles_dangling_close():
    # Opener was lost (e.g. truncated by streaming), only </think> survives.
    text = "leaked thoughts</think>real answer"
    assert _strip_reasoning(text) == "real answer"


def test_strip_reasoning_handles_dangling_open():
    # Unterminated opener at the start — drop only the opener, keep content.
    text = "<think>\nthe rest of the response"
    assert _strip_reasoning(text) == "the rest of the response"


def test_strip_reasoning_passthrough_for_normal_text():
    text = "REQUIREMENTS_SPEC:\nGoal: fix the bug"
    assert _strip_reasoning(text) == text


def test_strip_reasoning_handles_empty():
    assert _strip_reasoning("") == ""
    assert _strip_reasoning(None) is None


def test_strip_reasoning_case_insensitive():
    text = "<THINK>foo</THINK>bar"
    assert _strip_reasoning(text) == "bar"


# ── _sanitize_directive_arg ─────────────────────────────────────────────────


def test_sanitize_truncates_at_dangling_think_close():
    # Exact failure mode from the production log.
    raw = "config.py</think>│─ SAVED CONTEXT RESTORED ─│"
    assert _sanitize_directive_arg(raw) == "config.py"


def test_sanitize_truncates_at_fake_banner_pipe():
    raw = "snake.py │Error: KB_SEARCH failed│"
    assert _sanitize_directive_arg(raw) == "snake.py"


def test_sanitize_truncates_at_chatml_sentinel():
    raw = "food.py <|tool_call|>"
    assert _sanitize_directive_arg(raw) == "food.py"


def test_sanitize_truncates_at_code_fence():
    raw = "config.py ```python"
    assert _sanitize_directive_arg(raw) == "config.py"


def test_sanitize_passes_through_clean_query():
    assert _sanitize_directive_arg("HEADER_ROWS constant") == "HEADER_ROWS constant"


def test_sanitize_strips_whitespace():
    assert _sanitize_directive_arg("  food.py  ") == "food.py"


def test_sanitize_returns_empty_for_pure_garbage():
    assert _sanitize_directive_arg("</think>garbage") == ""
    assert _sanitize_directive_arg("") == ""


def test_sanitize_caps_length():
    raw = "x" * 1000
    result = _sanitize_directive_arg(raw)
    assert len(result) <= 500
    assert result == "x" * 500


def test_sanitize_preserves_punctuation_in_query():
    # Commas/semicolons must survive — KB_SEARCH splits on them downstream.
    assert _sanitize_directive_arg("Food.jsx; SnakeSegment.jsx") == "Food.jsx; SnakeSegment.jsx"


# ── _normalize_dedup_key ────────────────────────────────────────────────────


def test_normalize_collapses_whitespace():
    assert _normalize_dedup_key("config.py") == _normalize_dedup_key("config.py ")
    assert _normalize_dedup_key("a  b") == _normalize_dedup_key("a b")


def test_normalize_lowercases():
    assert _normalize_dedup_key("Config.PY") == _normalize_dedup_key("config.py")


def test_normalize_strips_trailing_punctuation():
    assert _normalize_dedup_key("config.py.") == _normalize_dedup_key("config.py")
    assert _normalize_dedup_key("config.py?") == _normalize_dedup_key("config.py")


def test_normalize_loop_collision_from_production_log():
    # The 6 variants seen in the stuck loop should all collapse to one key.
    variants = [
        "config.py",
        "config.py ",
        "config.py HEADER_ROWS",
        "config.py HEADER_ROWS",  # exact dup
        "Config.py HEADER_ROWS ",
    ]
    keys = {_normalize_dedup_key(v) for v in variants}
    # Two distinct queries: "config.py" and "config.py header_rows"
    assert len(keys) == 2


def test_normalize_handles_empty():
    assert _normalize_dedup_key("") == ""
    assert _normalize_dedup_key("   ") == ""

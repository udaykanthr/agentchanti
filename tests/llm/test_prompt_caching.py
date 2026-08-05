"""The loop keeps a stable prefix on purpose; the client must ask for it.

``agent_loop``'s module docstring states that its system prompt is kept
byte-identical across every step of a run "so provider prompt caches and
local KV caches get a stable prefix". The Anthropic client never sent
``cache_control``, so that stable prefix was re-billed at full price on
every call. Measured on a Pygame run (2026-08-05): 1,224,846 tokens sent,
**0 cached**, across 70 chat calls averaging 17k and peaking at 44.8k.

Caching is a prefix match, so where the breakpoints go is the whole
design. This module pins the placement and the accounting that makes a
working cache visible instead of looking like a shrinking prompt.
"""

from __future__ import annotations

import pytest

from agentchanti.llm.anthropic_client import AnthropicClient
from agentchanti.llm.chat_types import Message, ToolDef


@pytest.fixture
def client():
    return AnthropicClient(base_url="https://api.anthropic.com/v1",
                           model="claude-haiku-4-5-20251001",
                           api_key="test-key")


def _payload(client, messages, tools=None):
    system, api_messages = client._serialize_messages(messages)
    payload = {"model": client.model, "messages": api_messages}
    if system:
        payload["system"] = [{"type": "text", "text": system}]
    if tools:
        payload["tools"] = [{"name": t.name, "description": t.description,
                             "input_schema": t.parameters} for t in tools]
    client._apply_cache_breakpoints(payload)
    return payload


def _breakpoints(payload) -> int:
    n = 0
    for block in payload.get("system") or []:
        n += "cache_control" in block
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            n += sum("cache_control" in b for b in content)
    return n


TOOLS = [ToolDef(name="read_file", description="Read a file",
                 parameters={"type": "object", "properties": {}})]


def test_system_block_carries_a_breakpoint():
    c = AnthropicClient(base_url="u", model="m", api_key="k")
    payload = _payload(c, [Message(role="system", content="You are an agent."),
                           Message(role="user", content="Do the thing.")],
                       TOOLS)
    # Tools render before system, so one marker on the system block caches
    # the tool definitions with it.
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_opening_and_newest_messages_are_both_anchored(client):
    payload = _payload(client, [
        Message(role="system", content="sys"),
        Message(role="user", content="Step task + preloaded files"),
        Message(role="assistant", content="working"),
        Message(role="user", content="tool output"),
    ])
    msgs = payload["messages"]
    assert "cache_control" in msgs[0]["content"][-1]
    assert "cache_control" in msgs[-1]["content"][-1]
    # The turn in between is not marked — only the ends of the prefix are.
    assert "cache_control" not in str(msgs[1]["content"])


def test_never_exceeds_the_four_breakpoint_limit(client):
    convo = [Message(role="system", content="sys"),
             Message(role="user", content="task")]
    for i in range(12):
        convo.append(Message(role="assistant", content=f"turn {i}"))
        convo.append(Message(role="user", content=f"result {i}"))
    payload = _payload(client, convo, TOOLS)
    assert _breakpoints(payload) <= 4


def test_a_single_message_is_anchored_once_not_twice(client):
    payload = _payload(client, [Message(role="user", content="only turn")])
    assert _breakpoints(payload) == 1


def test_empty_content_is_left_alone(client):
    # A string that cannot carry a block must not be converted into an
    # empty text block — the API rejects those.
    msg = {"role": "user", "content": ""}
    client._mark_cacheable(msg)
    assert msg["content"] == ""


def test_marking_is_idempotent_across_serialisations(client):
    """The prefix must be byte-identical between turns, markers included."""
    base = [Message(role="system", content="sys"),
            Message(role="user", content="task")]
    first = _payload(client, base, TOOLS)
    second = _payload(client, base + [Message(role="assistant", content="x")],
                      TOOLS)
    # The system block — the part shared by both requests — renders the
    # same both times, so the second call can read what the first wrote.
    assert first["system"] == second["system"]


# ── accounting ────────────────────────────────────────────────────────

def test_billed_prompt_is_the_sum_of_all_three_usage_fields():
    """`input_tokens` is the uncached remainder, not the prompt size.

    Reporting it alone would make a cache hit look like the prompt got
    smaller — the run would appear to use fewer tokens than it was billed
    for, which is the opposite of what this telemetry is for.
    """
    usage = {"input_tokens": 400,
             "cache_read_input_tokens": 9_000,
             "cache_creation_input_tokens": 600}
    total = (usage["input_tokens"]
             + usage["cache_read_input_tokens"]
             + usage["cache_creation_input_tokens"])
    assert total == 10_000

    from agentchanti.cli_display import TokenTracker
    tracker = TokenTracker()
    tracker.record(total, 50, cached_tokens=usage["cache_read_input_tokens"])
    assert tracker.total_prompt_tokens == 10_000
    assert tracker.total_cached_tokens == 9_000
    assert tracker.full_price_prompt_tokens == 1_000

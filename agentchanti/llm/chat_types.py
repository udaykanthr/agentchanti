"""
Chat-native message types for the LLM layer.

These back :meth:`LLMClient.chat` — the structured, multi-turn,
tool-calling entry point. Providers with a native chat endpoint
(Ollama ``/api/chat``, Anthropic Messages API) translate these types
to their wire format; providers without one fall back to flattening
the conversation into a single text prompt via :func:`flatten_messages`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDef:
    """A tool the model may call.

    ``parameters`` is a JSON Schema object describing the arguments,
    e.g. ``{"type": "object", "properties": {"path": {"type": "string"}},
    "required": ["path"]}``.
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Provider-assigned call id (Anthropic ``tool_use.id``). Empty for
    #: providers that don't assign ids (Ollama, Gemini).
    id: str = ""
    #: Opaque provider state that must be echoed back verbatim when this
    #: call is replayed in conversation history. Gemini 3.x rejects a
    #: replayed ``functionCall`` without its ``thoughtSignature``:
    #: "Function call is missing a thought_signature ... required for
    #: tools to work correctly". Ignored by providers that do not use it.
    provider_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in a chat conversation.

    ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.

    - Assistant messages may carry ``tool_calls`` (what the model invoked).
    - Tool messages carry the result of one call; ``tool_call_id`` links the
      result back to the assistant's :class:`ToolCall` and ``tool_name``
      names the tool (used by providers without call ids).
    """
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""


@dataclass
class ChatResponse:
    """Result of one :meth:`LLMClient.chat` call."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Provider stop reason, normalised only loosely: Ollama ``done_reason``
    #: ("stop", "length", …), Anthropic ``stop_reason`` ("end_turn",
    #: "tool_use", "max_tokens", …), or "stop" for the text fallback.
    stop_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_empty(self) -> bool:
        """True when the model produced neither text nor tool calls."""
        return not self.tool_calls and not self.text.strip()

    def to_message(self) -> Message:
        """Convert to an assistant :class:`Message` for conversation history."""
        return Message(role="assistant", content=self.text,
                       tool_calls=list(self.tool_calls))


_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool result",
}


def flatten_messages(messages: list[Message]) -> str:
    """Flatten a chat conversation into a single text prompt.

    Fallback path for providers/models without a native chat endpoint.
    Tool calls and results are rendered as labelled text so the model
    retains the full interaction history, and the prompt ends with an
    ``Assistant`` cue for the next completion.
    """
    parts: list[str] = []
    for msg in messages:
        label = _ROLE_LABELS.get(msg.role, msg.role.capitalize())
        content = msg.content or ""
        if msg.role == "assistant" and msg.tool_calls:
            calls = "\n".join(
                f"[tool call] {tc.name}({json.dumps(tc.arguments)})"
                for tc in msg.tool_calls)
            content = f"{content}\n{calls}".strip()
        elif msg.role == "tool":
            source = msg.tool_name or msg.tool_call_id
            if source:
                label = f"Tool result ({source})"
        parts.append(f"### {label}\n{content}")
    parts.append("### Assistant\n")
    return "\n\n".join(parts)

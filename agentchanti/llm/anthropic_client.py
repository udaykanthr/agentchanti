"""
Anthropic Claude LLM client — calls the Anthropic Messages API directly.
"""

import json
import requests
from typing import List, Optional

from .base import LLMClient
from .chat_types import ChatResponse, Message, ToolCall, ToolDef
from .cancellation import streaming_response, check_cancelled
from ..cli_display import token_tracker, log


class AnthropicClient(LLMClient):

    ANTHROPIC_VERSION = "2023-06-01"
    NATIVE_CHAT = True

    def __init__(self, base_url: str, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
        }

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Anthropic] Sending ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)
        log.debug(f"[Anthropic] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        url = f"{self.base_url}/messages"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        response.raise_for_status()
        data = response.json()

        # Extract token counts
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", est_tokens)
        completion_tokens = usage.get("output_tokens", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
        )
        log.debug(f"[Anthropic] Usage: prompt={prompt_tokens} completion={completion_tokens}")

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        response_text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        log.debug(f"[Anthropic] Response:\n{response_text}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Anthropic] Streaming ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)

        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        url = f"{self.base_url}/messages"

        content_parts: list[str] = []
        tokens_generated = 0
        prompt_tokens = est_tokens
        completion_tokens = 0

        response = requests.post(url, headers=self._headers(), json=payload,
                                 stream=True, timeout=(10, 120))
        response.raise_for_status()

        with streaming_response(response):
            for line in response.iter_lines(decode_unicode=True):
                check_cancelled()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        event_type = event.get("type", "")

                        if event_type == "message_start":
                            # First event — contains real input_tokens count
                            usage = event.get("message", {}).get("usage", {})
                            if usage.get("input_tokens"):
                                prompt_tokens = usage["input_tokens"]

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                token = delta.get("text", "")
                                if token:
                                    content_parts.append(token)
                                    tokens_generated += 1
                                    if self._stream_callback and tokens_generated % 10 == 0:
                                        self._stream_callback(tokens_generated)

                        elif event_type == "message_delta":
                            # Final event — contains real output_tokens count
                            usage = event.get("usage", {})
                            if usage.get("output_tokens"):
                                completion_tokens = usage["output_tokens"]

                        elif event_type == "message_stop":
                            break

                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        result = "".join(content_parts)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else tokens_generated,
            model_name=self.model,
        )
        log.debug(f"[Anthropic] Streamed usage: prompt={prompt_tokens} completion={completion_tokens}")
        log.debug(f"[Anthropic] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Native chat (Messages API with tools) ──

    @staticmethod
    def _serialize_messages(messages: List[Message]) -> tuple[str, list[dict]]:
        """Convert chat messages to (system_prompt, api_messages).

        Anthropic takes system prompts as a top-level parameter, renders
        assistant tool calls as ``tool_use`` content blocks, and expects
        tool results as ``tool_result`` blocks inside a *user* message —
        consecutive tool results (parallel calls) merge into one message.
        """
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content)
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for i, tc in enumerate(m.tool_calls):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id or f"call_{len(out)}_{i}",
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content or "",
                }
                prev = out[-1] if out else None
                if (prev and prev["role"] == "user"
                        and isinstance(prev["content"], list)
                        and any(b.get("type") == "tool_result"
                                for b in prev["content"])):
                    prev["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                prev = out[-1] if out else None
                if m.role == "user" and prev and prev["role"] == "user":
                    # The API requires alternating roles — merge consecutive
                    # user messages (e.g. tool results followed by loop
                    # feedback) into one message's content blocks.
                    if isinstance(prev["content"], str):
                        prev["content"] = [{"type": "text",
                                            "text": prev["content"]}]
                    prev["content"].append({"type": "text",
                                            "text": m.content or ""})
                else:
                    out.append({"role": m.role, "content": m.content or ""})
        return "\n\n".join(system_parts), out

    # Prompt caching is a PREFIX match: everything up to a breakpoint is
    # cached together, and any byte change before it invalidates the rest.
    # Reads bill at ~0.1x, writes at ~1.25x, so this only pays where the
    # prefix recurs — which is exactly the agent loop's shape. Both halves
    # of that shape are already stable by design: the loop's system prompt
    # is deliberately byte-identical across steps (see agent_loop's module
    # docstring, which says so and names prompt caches as the reason), and
    # within a step every turn re-sends the whole conversation so far.
    #
    # Nothing was ever marked, so none of it cached. Measured on a Pygame
    # run: 1,224,846 tokens sent, 0 cached — 70 chat calls averaging 17k
    # and peaking at 44.8k, every one of them billed at full price.
    _CACHE_CONTROL = {"type": "ephemeral"}

    @staticmethod
    def _mark_cacheable(message: dict) -> None:
        """Put a cache breakpoint on *message*'s last content block."""
        content = message.get("content")
        if isinstance(content, str):
            if not content:
                return
            content = [{"type": "text", "text": content}]
            message["content"] = content
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = dict(
                AnthropicClient._CACHE_CONTROL)

    def _apply_cache_breakpoints(self, payload: dict) -> None:
        """Mark the stable prefix of *payload* (max 4 breakpoints allowed).

        Three are used, in render order (tools -> system -> messages):

        * the system block, which carries the tools before it;
        * the opening user message, which holds the step's task and its
          pre-loaded files and does not change for the life of the step;
        * the newest message, so the next turn reads everything up to here.

        The middle one is not redundant. A breakpoint searches back at most
        20 content blocks for a prior entry, and a tool-calling turn adds
        two blocks a time, so a long step can outrun the window; the anchor
        on the opening message keeps the expensive part of the prefix
        reachable even then.
        """
        if isinstance(payload.get("system"), list) and payload["system"]:
            payload["system"][-1]["cache_control"] = dict(self._CACHE_CONTROL)

        api_messages = payload.get("messages") or []
        if not api_messages:
            return
        self._mark_cacheable(api_messages[0])
        if len(api_messages) > 1:
            self._mark_cacheable(api_messages[-1])

    def _chat(self, messages: List[Message],
              tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        est_tokens = int(sum(len((m.content or "").split()) for m in messages) * 1.3)
        log.debug(f"[Anthropic] Chat: ~{est_tokens} est. tokens, "
                  f"{len(messages)} messages, {len(tools or [])} tools")
        token_tracker.set_context(est_tokens)

        system, api_messages = self._serialize_messages(messages)
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": api_messages,
        }
        if system:
            # Block form so a cache breakpoint can be attached; a plain
            # string cannot carry one.
            payload["system"] = [{"type": "text", "text": system}]
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.parameters}
                for t in tools
            ]
        self._apply_cache_breakpoints(payload)

        url = f"{self.base_url}/messages"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        if response.status_code >= 400:
            # `raise_for_status` reports the status and drops the body, and
            # the body is where the API says what is actually wrong — an
            # exhausted credit balance and a malformed request are both a
            # bare 400 without it, which is a materially different thing to
            # go and fix.
            log.error("[Anthropic] HTTP %s: %s",
                      response.status_code, response.text[:600])
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", est_tokens)
        completion_tokens = usage.get("output_tokens", 0)
        # `input_tokens` counts only the UNCACHED remainder, so the billed
        # prompt is the sum of the three. Reporting input_tokens alone would
        # make a working cache look like a shrinking prompt.
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_write = usage.get("cache_creation_input_tokens") or 0
        if not isinstance(prompt_tokens, int):
            prompt_tokens = est_tokens
        total_prompt = prompt_tokens + cache_read + cache_write
        token_tracker.record(
            total_prompt,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
            cached_tokens=cache_read,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.get("name", ""),
                    arguments=block.get("input") or {},
                    id=block.get("id", ""),
                ))

        log.debug(f"[Anthropic] Chat usage: prompt={total_prompt} "
                  f"completion={completion_tokens} tool_calls={len(tool_calls)} "
                  f"cache_read={cache_read} cache_write={cache_write}")
        return ChatResponse(text="".join(text_parts), tool_calls=tool_calls,
                            stop_reason=data.get("stop_reason", "") or "")

    # ── Embeddings ──

    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        log.warning(
            "[Anthropic] Anthropic does not provide an embedding API. "
            "Consider using a different provider (e.g. OpenAI or Gemini) for embeddings."
        )
        return []

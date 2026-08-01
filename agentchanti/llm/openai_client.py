"""
OpenAI-compatible LLM client — works with OpenAI, Groq, Together.ai,
and any other provider that implements the OpenAI chat/completions API.
"""

import json
import requests
from typing import List, Optional

from .base import LLMClient, NonRetryableLLMError, ToolsNotSupportedError
from .chat_types import ChatResponse, Message, ToolCall, ToolDef
from .cancellation import streaming_response, check_cancelled
from ..cli_display import token_tracker, log

# A 400 whose message points at the tools/functions part of the request.
# Kept narrow on purpose: misreading an unrelated 400 as "no tool support"
# would silently drop the whole agent-loop path for the session.
_TOOLS_REJECTED_MARKERS = (
    "does not support tools",
    "does not support function",
    "tools are not supported",
    "function calling is not supported",
    "unsupported parameter: 'tools'",
    "unsupported value: 'tools'",
    "invalid parameter: 'tools'",
    "unknown parameter: 'tools'",
)


def _looks_like_tools_rejection(response) -> bool:
    body = (response.text or "").lower()
    return any(marker in body for marker in _TOOLS_REJECTED_MARKERS)


def _reasoning_blocks_tools(response) -> bool:
    """True when the model refuses tools *while reasoning is enabled*.

    Some reasoning models default ``reasoning_effort`` to a non-``none``
    value server-side and then reject function tools on
    ``/v1/chat/completions``::

        Function tools with reasoning_effort are not supported for
        gpt-5.6-terra in /v1/chat/completions. To use function tools, use
        /v1/responses or set reasoning_effort to 'none'.

    This is recoverable and worth distinguishing from a model that has no
    tool support at all: sending ``reasoning_effort: "none"`` keeps native
    tool calling — and therefore the whole agent-loop path — instead of
    falling back to the text protocol for the rest of the session.
    """
    body = (response.text or "").lower()
    return "reasoning_effort" in body and (
        "tool" in body or "function" in body)


_PARAM_REJECTED_WORDS = ("unsupported", "unrecognized", "unrecognised",
                         "unknown", "invalid", "not supported")


def _param_rejected(response, param: str) -> bool:
    """True when a 400 specifically complains about *param*.

    The ``max_completion_tokens`` → ``max_tokens`` fallback used to fire on
    *any* 400, which meant an unrelated rejection silently retried with the
    legacy parameter and then reported that second failure — "Unsupported
    parameter: 'max_tokens' ... Use 'max_completion_tokens' instead" — the
    exact opposite of the real problem, with the first error discarded.
    """
    body = (response.text or "").lower()
    if param.lower() not in body:
        return False
    return any(word in body for word in _PARAM_REJECTED_WORDS)


def _raise_for_status_with_body(response, model: str = "",
                                tool_count: int = 0,
                                payload: Optional[dict] = None) -> None:
    """``raise_for_status`` that keeps the provider's explanation.

    ``requests`` raises ``400 Client Error: Bad Request for url: ...`` and
    throws the response body away — but the body is the only thing that
    says *what* was wrong. Observed: a model rejected every tool-calling
    request and three runs' worth of logs showed nothing but the bare
    status line, leaving no way to tell a bad tool schema from an
    unsupported parameter.

    A 4xx other than 408/429 is also raised as
    :class:`NonRetryableLLMError`: the request is malformed, so sending it
    twice more only spends time and money to fail identically.
    """
    if response.status_code < 400:
        return

    detail = ""
    try:
        body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            detail = err.get("message") or json.dumps(err)[:500]
        elif body:
            detail = json.dumps(body)[:500]
    except Exception:
        detail = (response.text or "")[:500]

    context = f"model={model}" if model else ""
    if tool_count:
        context += f", {tool_count} tool(s)"
    if payload:
        # Naming the request parameters turns "this model rejected the
        # request" into something actionable without another round trip.
        context += ", params=" + ",".join(sorted(payload))
    message = (f"{response.status_code} from {response.url.split('?')[0]}"
               + (f" [{context}]" if context else "")
               + (f": {detail}" if detail else ""))

    if 400 <= response.status_code < 500 and response.status_code not in (
            408, 409, 429):
        raise NonRetryableLLMError(message)
    response.raise_for_status()


class OpenAIClient(LLMClient):

    NATIVE_CHAT = True

    # Models whose completion budget is shared with hidden reasoning
    # tokens and that accept the ``reasoning_effort`` parameter.
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(self, base_url: str, model: str, api_key: str, **kwargs):
        # Popped before super(): LLMClient takes named parameters, not
        # **kwargs, so an unknown key would raise TypeError.
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # One-shot reasoning-effort downgrade: armed by
        # _prepare_token_limit_retry, consumed by the next _chat call.
        self._retry_reasoning_effort: Optional[str] = None
        # Session-sticky downgrade, latched after the FIRST reasoning burn.
        # The one-shot alone only rescued the call that burned: every later
        # request went back to the configured effort and burned again.
        # Measured on one Pac-Man run: 4 burns x 16384 = 65,536 completion
        # tokens for zero visible output — 43% of everything received.
        # Mirrors _native_tools_ok: once the model has demonstrated it
        # cannot work at this setting, stop paying to rediscover that.
        self._session_effort_floor: Optional[str] = None
        # Latched once the model reports that function tools and reasoning
        # cannot be combined on /chat/completions. Tool-calling turns then
        # go to /v1/responses, which supports both, instead of paying the
        # 400 again or throwing the model's reasoning away.
        self._tools_need_responses_api = False
        # Reasoning effort for the Responses path. None leaves it to the
        # provider default, which is what keeps reasoning ON without
        # pinning an effort level the caller did not ask for.
        self.reasoning_effort: Optional[str] = reasoning_effort

    @staticmethod
    def _cached_tokens(usage: dict) -> int:
        """Extract cache-hit prompt tokens from an OpenAI usage block.

        OpenAI reports automatic prompt-cache hits at
        ``usage.prompt_tokens_details.cached_tokens`` (a subset of
        ``prompt_tokens``, billed at a discount). Returns 0 when the field
        is absent, so non-OpenAI compatible backends stay unaffected.
        """
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        return cached if isinstance(cached, int) else 0

    def _is_reasoning_model(self) -> bool:
        return self.model.lower().startswith(self._REASONING_MODEL_PREFIXES)

    def _effort(self) -> Optional[str]:
        """Reasoning effort to send, or None to leave it to the provider.

        A one-shot downgrade armed after a reasoning burn outranks the
        configured value — that retry exists precisely because the model
        spent the whole completion budget thinking. Gated on the model
        looking like a reasoning model, since sending the parameter to one
        that does not accept it is a 400.
        """
        if self._retry_reasoning_effort:
            effort = self._retry_reasoning_effort
            self._retry_reasoning_effort = None
            return effort
        if self._session_effort_floor:
            return self._session_effort_floor
        if self.reasoning_effort and self._is_reasoning_model():
            return self.reasoning_effort
        return None

    def _prepare_token_limit_retry(self) -> None:
        """Reasoning models can burn the whole completion budget on hidden
        reasoning tokens, returning an empty response with
        ``finish_reason: length``. Retrying verbatim is a coin flip —
        request low reasoning effort for the retry instead.

        The downgrade also latches for the rest of the session. A model
        that burned its entire budget once will do it again on the next
        comparable request, and each repeat costs a full completion cap
        plus a wasted round trip before the retry rescues it.
        """
        if not self._is_reasoning_model():
            return
        self._retry_reasoning_effort = "low"
        if self._session_effort_floor is None:
            self._session_effort_floor = "low"
            log.warning(
                "[OpenAI] %s spent its entire output budget on reasoning at "
                "effort=%s — dropping to 'low' for the rest of this session "
                "rather than paying the same burn on every later call.",
                self.model, self.reasoning_effort or "provider default")

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[OpenAI] Sending ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)
        # log.debug(f"[OpenAI] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        # Newer OpenAI models (GPT-4o+, o1+, GPT-5+) require
        # max_completion_tokens; older models and compatible APIs use
        # max_tokens.  Try the new name first, fall back on 400 error.
        _tok_key = "max_completion_tokens"
        payload[_tok_key] = self.max_output_tokens
        _effort = self._effort()
        if _effort:
            payload["reasoning_effort"] = _effort
        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        if response.status_code == 400 and _param_rejected(response, _tok_key):
            # Legacy parameter name for older models/APIs (see _chat).
            first_failure = response
            del payload[_tok_key]
            payload["max_tokens"] = self.max_output_tokens
            response = requests.post(url, headers=self._headers(), json=payload,
                                     timeout=(10, 300))
            if response.status_code >= 400:
                response = first_failure
        _raise_for_status_with_body(response, model=self.model,
                                    payload=payload)
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", est_tokens)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = self._cached_tokens(usage)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
            cached_tokens=cached_tokens,
        )
        log.debug(f"[OpenAI] Usage: prompt={prompt_tokens} "
                  f"(cached={cached_tokens}) completion={completion_tokens}")

        response_text = data["choices"][0]["message"]["content"]
        self._last_stop_reason = data["choices"][0].get("finish_reason", "") or ""
        log.debug(f"[OpenAI] Response:\n{completion_tokens}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[OpenAI] Streaming ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)

        # log.debug(f"[OpenAI] Prompt:\n{prompt}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self.max_output_tokens,
        }
        _effort = self._effort()
        if _effort:
            payload["reasoning_effort"] = _effort
        url = f"{self.base_url}/chat/completions"

        content_parts: list[str] = []
        tokens_generated = 0
        prompt_tokens = est_tokens
        completion_tokens = 0
        cached_tokens = 0
        self._last_stop_reason = ""

        response = requests.post(url, headers=self._headers(), json=payload,
                                 stream=True, timeout=(10, 120))
        if response.status_code == 400 and \
                _param_rejected(response, "max_completion_tokens"):
            # Legacy parameter name for older models/APIs (see _chat).
            first_failure = response
            del payload["max_completion_tokens"]
            payload["max_tokens"] = self.max_output_tokens
            response = requests.post(url, headers=self._headers(), json=payload,
                                     stream=True, timeout=(10, 120))
            if response.status_code >= 400:
                response = first_failure
        _raise_for_status_with_body(response, model=self.model,
                                    payload=payload)

        with streaming_response(response):
            for line in response.iter_lines(decode_unicode=True):
                check_cancelled()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Final chunk from stream_options.include_usage carries real usage
                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                            cached_tokens = self._cached_tokens(usage)
                        _choice0 = chunk.get("choices", [{}])[0]
                        _fr = _choice0.get("finish_reason")
                        if _fr:
                            self._last_stop_reason = _fr
                        delta = _choice0.get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            content_parts.append(token)
                            tokens_generated += 1
                            if self._stream_callback and tokens_generated % 10 == 0:
                                self._stream_callback(tokens_generated)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        result = "".join(content_parts)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else tokens_generated,
            model_name=self.model,
            cached_tokens=cached_tokens,
        )
        log.debug(f"[OpenAI] Streamed usage: prompt={prompt_tokens} "
                  f"(cached={cached_tokens}) completion={completion_tokens}")
        log.debug(f"[OpenAI] Response:\n{completion_tokens}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Native chat (/v1/responses with tools + reasoning) ──

    @staticmethod
    def _responses_input(messages: List[Message]) -> list[dict]:
        """Translate chat messages to Responses API ``input`` items.

        The shapes differ from /chat/completions in three ways that all
        matter: a tool call is a top-level ``function_call`` item rather
        than a field on the assistant message, a tool result is a
        ``function_call_output`` item rather than a ``tool`` role, and both
        are linked by ``call_id`` rather than ``tool_call_id``.
        """
        out: list[dict] = []
        for i, m in enumerate(messages):
            if m.role == "assistant" and m.tool_calls:
                if m.content:
                    out.append({"role": "assistant", "content": m.content})
                for j, tc in enumerate(m.tool_calls):
                    out.append({
                        "type": "function_call",
                        "call_id": tc.id or f"call_{i}_{j}",
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    })
            elif m.role == "tool":
                out.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content or "",
                })
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    @staticmethod
    def _parse_responses_output(data: dict) -> tuple[str, list[ToolCall]]:
        """Pull assistant text and tool calls out of a Responses payload.

        ``output`` is a heterogeneous list — reasoning items, message
        items whose ``content`` holds ``output_text`` parts, and
        ``function_call`` items. Reasoning items are skipped: they carry
        no user-visible text and the loop resends full history each turn.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "function_call":
                raw = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {"_raw": raw}
                tool_calls.append(ToolCall(
                    name=item.get("name", ""), arguments=args,
                    id=item.get("call_id") or item.get("id") or ""))
            elif kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        text_parts.append(part.get("text") or "")
        # Convenience field some responses include alongside `output`.
        if not text_parts and isinstance(data.get("output_text"), str):
            text_parts.append(data["output_text"])
        return "".join(text_parts), tool_calls

    def _chat_via_responses(self, messages: List[Message],
                            tools: Optional[List[ToolDef]],
                            est_tokens: int) -> ChatResponse:
        """Tool-calling turn over ``/v1/responses``.

        Used for models that refuse function tools on
        ``/chat/completions`` while reasoning is enabled. Going through
        this endpoint keeps BOTH — the alternative (reasoning_effort
        "none") keeps tools but throws the model's reasoning away, which
        is the capability the agent loop leans on hardest.
        """
        payload: dict = {
            "model": self.model,
            "input": self._responses_input(messages),
            "max_output_tokens": self.max_output_tokens,
            # The loop resends the full conversation every turn, so there
            # is nothing to gain from server-side retention.
            "store": False,
        }
        effort = self._effort()
        if effort:
            payload["reasoning"] = {"effort": effort}
        if tools:
            # Flat here — no nested "function" wrapper as in completions.
            payload["tools"] = [
                {"type": "function", "name": t.name,
                 "description": t.description, "parameters": t.parameters}
                for t in tools
            ]

        url = f"{self.base_url}/responses"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        _raise_for_status_with_body(response, model=self.model,
                                    tool_count=len(tools or []),
                                    payload=payload)
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", est_tokens)
        completion_tokens = usage.get("output_tokens", 0)
        cached = (usage.get("input_tokens_details") or {}).get(
            "cached_tokens", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
            cached_tokens=cached if isinstance(cached, int) else 0,
        )

        text, tool_calls = self._parse_responses_output(data)
        stop_reason = data.get("status", "") or ""
        incomplete = data.get("incomplete_details") or {}
        if incomplete.get("reason") == "max_output_tokens":
            stop_reason = "length"
        log.debug(f"[OpenAI] Responses usage: prompt={prompt_tokens} "
                  f"(cached={cached}) completion={completion_tokens} "
                  f"tool_calls={len(tool_calls)}")
        return ChatResponse(text=text, tool_calls=tool_calls,
                            stop_reason=stop_reason)

    # ── Native chat (/chat/completions with tools) ──

    @staticmethod
    def _serialize_messages(messages: List[Message]) -> list[dict]:
        out: list[dict] = []
        for i, m in enumerate(messages):
            if m.role == "assistant" and m.tool_calls:
                entry: dict = {"role": "assistant",
                               "content": m.content or None}
                entry["tool_calls"] = [
                    {"id": tc.id or f"call_{i}_{j}",
                     "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments)}}
                    for j, tc in enumerate(m.tool_calls)
                ]
                out.append(entry)
            elif m.role == "tool":
                out.append({"role": "tool",
                            "tool_call_id": m.tool_call_id,
                            "content": m.content or ""})
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _chat(self, messages: List[Message],
              tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        est_tokens = int(sum(len((m.content or "").split()) for m in messages) * 1.3)
        log.debug(f"[OpenAI] Chat: ~{est_tokens} est. tokens, "
                  f"{len(messages)} messages, {len(tools or [])} tools")
        token_tracker.set_context(est_tokens)

        if tools and self._tools_need_responses_api:
            return self._chat_via_responses(messages, tools, est_tokens)

        payload: dict = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "stream": False,
            "max_completion_tokens": self.max_output_tokens,
        }
        _effort = self._effort()
        if _effort:
            payload["reasoning_effort"] = _effort
        if tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in tools
            ]

        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        if response.status_code == 400 and tools and \
                _reasoning_blocks_tools(response):
            # Deliberately NOT gated on the latch being unset. Wave steps
            # share one client across threads: two concurrent tool calls
            # both got this 400, the first latched and recovered, and the
            # second saw the latch already true, skipped the retry and
            # raised — costing that step a failure and a recovery loop.
            # Retrying is idempotent, so just always do it.
            # This model defaults reasoning on server-side and refuses
            # function tools here. /v1/responses supports both, so switch
            # endpoints rather than dropping reasoning (which is what the
            # agent loop leans on hardest) or dropping tools (which loses
            # the agent loop entirely). Latched: one 400 per session.
            log.info(
                "[OpenAI] %s refuses function tools on /chat/completions "
                "while reasoning is on — switching to /v1/responses for "
                "tool calls this session", self.model)
            self._tools_need_responses_api = True
            return self._chat_via_responses(messages, tools, est_tokens)
        if response.status_code == 400 and \
                _param_rejected(response, "max_completion_tokens"):
            # Legacy parameter name for older models/APIs. Only on a 400
            # that actually names the parameter: firing on any 400 masked
            # the real error behind a bogus "use max_completion_tokens".
            first_failure = response
            del payload["max_completion_tokens"]
            payload["max_tokens"] = self.max_output_tokens
            response = requests.post(url, headers=self._headers(), json=payload,
                                     timeout=(10, 300))
            if response.status_code >= 400:
                response = first_failure   # the original is the real one
        if response.status_code == 400 and tools and \
                _looks_like_tools_rejection(response):
            # Degrade to the text path for the rest of the session instead
            # of dying. Only Ollama ever raised this, so a model that
            # rejects tools over the OpenAI API used to take the whole run
            # down: observed with a model whose every tool-calling request
            # 400'd, killing the pipeline on the first CODE step.
            raise ToolsNotSupportedError(
                f"model '{self.model}' rejected tools: "
                f"{(response.text or '')[:300]}")
        _raise_for_status_with_body(response, model=self.model,
                                    tool_count=len(tools or []),
                                    payload=payload)
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", est_tokens)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = self._cached_tokens(usage)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
            cached_tokens=cached_tokens,
        )

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(ToolCall(
                name=fn.get("name", ""), arguments=args,
                id=tc.get("id", "")))

        log.debug(f"[OpenAI] Chat usage: prompt={prompt_tokens} "
                  f"(cached={cached_tokens}) completion={completion_tokens} "
                  f"tool_calls={len(tool_calls)}")
        return ChatResponse(text=text, tool_calls=tool_calls,
                            stop_reason=choice.get("finish_reason", "") or "")

    # ── Embeddings ──

    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        embed_model = model or self.model
        url = f"{self.base_url}/embeddings"
        payload = {"model": embed_model, "input": text}

        dimensions = kwargs.get("dimensions")
        if dimensions:
            payload["dimensions"] = dimensions

        try:
            response = requests.post(url, headers=self._headers(), json=payload,
                                     timeout=(10, 120))
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if items:
                return items[0].get("embedding", [])
            return []
        except requests.exceptions.RequestException as e:
            log.error(f"[OpenAI] Embedding error: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error(f"[OpenAI] Embedding parse error: {e}")
            return []

    def generate_embeddings_batch(self, texts: list[str], model: Optional[str] = None, **kwargs) -> list[list[float]]:
        """Embed multiple texts in a single OpenAI /embeddings call."""
        if not texts:
            return []
        embed_model = model or self.model
        url = f"{self.base_url}/embeddings"
        payload = {"model": embed_model, "input": texts}
        dimensions = kwargs.get("dimensions")
        if dimensions:
            payload["dimensions"] = dimensions
        try:
            response = requests.post(url, headers=self._headers(), json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            items.sort(key=lambda x: x.get("index", 0))
            vectors = [item.get("embedding", []) for item in items]
            if len(vectors) == len(texts):
                return vectors
            log.warning("[OpenAI] Batch embed returned %d vectors for %d texts",
                        len(vectors), len(texts))
            while len(vectors) < len(texts):
                vectors.append([])
            return vectors
        except Exception as e:
            log.warning(f"[OpenAI] Batch embedding failed, falling back to sequential: {e}")
            return [self.generate_embedding(t, model=model, **kwargs) for t in texts]

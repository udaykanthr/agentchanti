"""
Google Gemini LLM client — calls the Gemini REST API directly.
"""

import json
import requests
from typing import List, Optional

from .base import (LLMClient, LLMError, NonRetryableLLMError,
                   ToolsNotSupportedError)
from .cancellation import streaming_response, check_cancelled
from .chat_types import ChatResponse, Message, ToolCall, ToolDef
from ..cli_display import token_tracker, log


# Phrases a provider uses when it will not accept function declarations.
# Kept narrow: misreading an unrelated 400 as "no tool support" would
# silently drop the whole agent-loop path for the rest of the session.
_TOOLS_REJECTED_MARKERS = (
    "function calling is not supported",
    "does not support function",
    "tools are not supported",
    "unsupported field: tools",
    "unknown name \"tools\"",
    "functiondeclarations",
)


def _looks_like_tools_rejection(detail: str) -> bool:
    low = (detail or "").lower()
    return any(m in low for m in _TOOLS_REJECTED_MARKERS)


class GeminiClient(LLMClient):

    NATIVE_CHAT = True

    # Gemini rejects JSON Schema keywords it does not implement, and a 400
    # here would silently drop the whole agent-loop path for the session.
    _SCHEMA_DROP_KEYS = frozenset({
        "additionalProperties", "$schema", "$id", "definitions", "$defs",
        "examples", "default", "exclusiveMinimum", "exclusiveMaximum",
    })

    def __init__(self, base_url: str, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # Session-sticky thinking cap, latched after the model spends its
        # whole output budget on hidden thoughts. Mirrors the OpenAI
        # client's reasoning-effort latch.
        self._thinking_budget: Optional[int] = None

    # ── Native chat (structured tool calling) ──

    @classmethod
    def _clean_schema(cls, node):
        """Strip JSON Schema keywords Gemini's function declarations reject.

        The API accepts an OpenAPI-flavoured subset; unknown keywords are a
        400, and ``AgentTools`` schemas carry several of them.
        """
        if isinstance(node, dict):
            return {k: cls._clean_schema(v) for k, v in node.items()
                    if k not in cls._SCHEMA_DROP_KEYS}
        if isinstance(node, list):
            return [cls._clean_schema(v) for v in node]
        return node

    def _headers(self) -> dict:
        """Auth by header, never in the URL.

        The key used to travel as ``?key=...``, which put it into every
        request exception, proxy log and debug trace — it leaked into a
        traceback during development. ``x-goog-api-key`` keeps it out of
        the URL entirely.
        """
        return {"Content-Type": "application/json",
                "x-goog-api-key": self.api_key}

    def _generation_config(self) -> dict:
        cfg: dict = {"maxOutputTokens": self.max_output_tokens}
        if self._thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self._thinking_budget}
        return cfg

    def _prepare_token_limit_retry(self) -> None:
        """Cap hidden thinking after it consumed the whole output budget.

        Gemini 3.x spends output tokens on thoughts before any visible
        text. Measured: a 200-token cap produced
        ``thoughtsTokenCount: 190`` and six visible tokens, finishReason
        MAX_TOKENS. At the real 16k cap that is an empty response and a
        wasted call. Retrying verbatim repeats it, so the retry pins a
        small thinking budget instead — and it latches, because a model
        that burned once will burn again on the next comparable request.
        """
        if self._thinking_budget is None:
            self._thinking_budget = 512
            log.warning(
                "[Gemini] %s spent its entire output budget on thinking "
                "(finishReason MAX_TOKENS, no visible text) — capping "
                "thinkingBudget at 512 for the rest of this session.",
                self.model)

    @staticmethod
    def _cached_tokens(usage: dict) -> int:
        """Prompt tokens served from Gemini's implicit cache.

        Gemini caches a repeated prefix automatically and reports the hit
        at ``usageMetadata.cachedContentTokenCount`` — measured at 16,362
        of 24,011 prompt tokens (68%) on a repeated request. Not reading
        it made every Gemini token look full-price and overstated a run's
        cost roughly threefold against the OpenAI client, which does
        report its cache hits.
        """
        cached = (usage or {}).get("cachedContentTokenCount", 0)
        return cached if isinstance(cached, int) else 0

    @staticmethod
    def _system_and_contents(messages):
        """Split messages into Gemini's systemInstruction + contents.

        Gemini has no system role and only knows ``user`` and ``model``.
        Tool results go back as a ``functionResponse`` part, which the API
        expects on a ``user`` turn.
        """
        system_bits: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    system_bits.append(m.content)
                continue
            if m.role == "tool":
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {
                        "name": m.tool_name or "tool",
                        # The API requires an object here, not a bare string.
                        "response": {"result": m.content or ""},
                    }}]})
                continue
            if m.role == "assistant":
                parts: list[dict] = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.tool_calls:
                    part = {"functionCall": {"name": tc.name,
                                             "args": tc.arguments or {}}}
                    # Gemini 3.x rejects a replayed functionCall whose
                    # thoughtSignature is missing, so it must be returned
                    # verbatim on the same part it arrived on.
                    sig = (tc.provider_state or {}).get("thoughtSignature")
                    if sig:
                        part["thoughtSignature"] = sig
                    parts.append(part)
                if not parts:
                    parts = [{"text": ""}]
                contents.append({"role": "model", "parts": parts})
                continue
            contents.append({"role": "user",
                             "parts": [{"text": m.content or ""}]})
        return "\n\n".join(system_bits), contents

    def _chat(self, messages: List[Message],
              tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        est_tokens = int(sum(len((m.content or "").split())
                             for m in messages) * 1.3)
        log.debug(f"[Gemini] Chat: ~{est_tokens} est. tokens, "
                  f"{len(messages)} messages, {len(tools or [])} tools")
        token_tracker.set_context(est_tokens)

        system, contents = self._system_and_contents(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": self._generation_config(),
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": [
                {"name": t.name,
                 "description": t.description,
                 "parameters": self._clean_schema(t.parameters)}
                for t in tools]}]

        url = f"{self.base_url}/models/{self.model}:generateContent"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        if response.status_code >= 400:
            detail = ""
            try:
                err = (response.json() or {}).get("error") or {}
                detail = err.get("message", "")
            except Exception:
                detail = (response.text or "")[:400]
            if tools and _looks_like_tools_rejection(detail):
                # Same contract as the OpenAI client: let the caller fall
                # back to the text protocol instead of failing the step.
                raise ToolsNotSupportedError(
                    f"{self.model} rejected function declarations: {detail}")
            message = (f"{response.status_code} from Gemini "
                       f"[model={self.model}, {len(tools or [])} tool(s)]"
                       f"{': ' + detail if detail else ''}")
            # A malformed request or a bad key does not become valid by
            # being resent; retrying spends latency to fail identically.
            if 400 <= response.status_code < 500 and                     response.status_code not in (408, 409, 429):
                raise NonRetryableLLMError(message)
            raise LLMError(message)
        data = response.json()

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", est_tokens)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        cached_tokens = self._cached_tokens(usage)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
            cached_tokens=cached_tokens,
        )

        candidates = data.get("candidates") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = ""
        if candidates:
            stop_reason = candidates[0].get("finishReason", "") or ""
            for part in candidates[0].get("content", {}).get("parts", []):
                if "text" in part and part["text"]:
                    text_parts.append(part["text"])
                fc = part.get("functionCall")
                if fc:
                    sig = part.get("thoughtSignature")
                    tool_calls.append(ToolCall(
                        name=fc.get("name", ""),
                        arguments=fc.get("args") or {},
                        # Gemini assigns no call ids; the tool NAME is the
                        # link back, which Message.tool_name carries.
                        id="",
                        provider_state={"thoughtSignature": sig} if sig else {}))

        log.debug(f"[Gemini] Chat usage: prompt={prompt_tokens} "
                  f"completion={completion_tokens} "
                  f"tool_calls={len(tool_calls)}")
        return ChatResponse(text="".join(text_parts), tool_calls=tool_calls,
                            stop_reason=stop_reason)

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Gemini] Sending ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)
        # log.debug(f"[Gemini] Prompt:\n{prompt}")

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": self._generation_config(),
        }
        url = f"{self.base_url}/models/{self.model}:generateContent"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        response.raise_for_status()
        data = response.json()

        # Extract token counts from usageMetadata
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", est_tokens)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
            cached_tokens=self._cached_tokens(usage),
        )
        log.debug(f"[Gemini] Usage: prompt={prompt_tokens} "
                  f"completion={completion_tokens} "
                  f"cached={self._cached_tokens(usage)}")

        # Extract text from candidates
        candidates = data.get("candidates", [])
        if not candidates:
            self._last_stop_reason = ""
            return ""
        # MAX_TOKENS here means thinking (or output) hit the cap. The base
        # class maps it to a token-limit retry via _TOKEN_LIMIT_REASONS;
        # leaving it unset made every burn invisible.
        self._last_stop_reason = candidates[0].get("finishReason", "") or ""
        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = "".join(p.get("text", "") for p in parts)
        log.debug(f"[Gemini] Response:\n{response_text}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Gemini] Streaming ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": self._generation_config(),
        }
        url = (f"{self.base_url}/models/{self.model}"
               f":streamGenerateContent?alt=sse")

        self._last_stop_reason = ""
        content_parts: list[str] = []
        tokens_generated = 0
        prompt_tokens = est_tokens
        completion_tokens = 0
        cached_tokens = 0

        response = requests.post(url, headers=self._headers(), json=payload,
                                 stream=True, timeout=(10, 120))
        response.raise_for_status()
        # Logging the headers as the body stream can't be read before iter_lines
        log.debug(f"[Gemini] Response Status: {response.status_code}, Headers: {dict(response.headers)}")
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
                        # log.debug(f"[Gemini] Chunk: {chunk}")
                        # usageMetadata is present in the final chunk with real counts
                        usage = chunk.get("usageMetadata", {})
                        if usage:
                            prompt_tokens = usage.get("promptTokenCount", prompt_tokens)
                            completion_tokens = usage.get("candidatesTokenCount", completion_tokens)
                            cached_tokens = self._cached_tokens(usage) or cached_tokens
                        candidates = chunk.get("candidates", [])
                        if not candidates:
                            continue
                        # The final chunk carries finishReason; MAX_TOKENS
                        # there is a thinking/output burn the base class
                        # retries at a reduced thinking budget.
                        _fr = candidates[0].get("finishReason")
                        if _fr:
                            self._last_stop_reason = _fr
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            token = part.get("text", "")
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
        log.debug(f"[Gemini] Streamed usage: prompt={prompt_tokens} "
                  f"completion={completion_tokens} cached={cached_tokens}")
        log.debug(f"[Gemini] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Embeddings ──

    # Models that are only available locally (not on the Gemini REST API).
    # If one of these is passed as *model*, fall back to the Gemini default.
    _LOCAL_ONLY_EMBED_MODELS = {
        "nomic-embed-text", "all-minilm", "mxbai-embed-large",
        "snowflake-arctic-embed", "bge-large", "bge-small",
    }

    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        # Ignore local-only model names that aren't valid on the Gemini API
        if model and (model in self._LOCAL_ONLY_EMBED_MODELS
                      or not model.startswith(("text-embedding", "embedding-", "models/", "gemini-embedding"))):
            log.warning(f"[Gemini] Embedding model '{model}' is not a valid "
                        f"Gemini API model, using 'text-embedding-004' instead")
            model = None
        embed_model = model or "text-embedding-004"
        url = (
            f"{self.base_url}/models/{embed_model}"
            f":embedContent"
        )
        payload = {
            "model": f"models/{embed_model}",
            "content": {
                "parts": [{"text": text}]
            },
        }
        
        # Support specifying the embedding output dimension directly
        dimensions = kwargs.get("dimensions")
        if dimensions:
            payload["outputDimensionality"] = dimensions

        try:
            response = requests.post(url, headers=self._headers(),
                                     json=payload, timeout=(10, 60))
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", {}).get("values", [])
        except requests.exceptions.RequestException as e:
            log.error(f"[Gemini] Embedding error: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error(f"[Gemini] Embedding parse error: {e}")
            return []

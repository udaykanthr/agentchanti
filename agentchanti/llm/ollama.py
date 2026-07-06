import json
import requests
from typing import List, Optional

from .base import LLMClient, ToolsNotSupportedError
from .chat_types import ChatResponse, Message, ToolCall, ToolDef
from .cancellation import streaming_response, check_cancelled
from ..cli_display import token_tracker, log


class OllamaClient(LLMClient):

    NATIVE_CHAT = True

    def __init__(self, base_url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.model = model
        # Derive the API root for endpoints like /api/embed
        if "/api/" in base_url:
            self._api_root = base_url.rsplit("/api/", 1)[0]
        else:
            self._api_root = base_url.rstrip("/")

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Ollama] Sending ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)
        # log.debug(f"[Ollama] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": self.max_output_tokens},
        }
        response = requests.post(self.base_url, json=payload, timeout=(10, 300))
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "")

        prompt_tokens = data.get("prompt_eval_count", est_tokens)
        completion_tokens = data.get("eval_count", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
        )
        log.debug(f"[Ollama] Usage: prompt={prompt_tokens} completion={completion_tokens}")
        log.debug(f"[Ollama] Response:\n{result}")
        return result

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Ollama] Streaming ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": self.max_output_tokens},
        }
        content_parts: list[str] = []
        tokens_generated = 0
        prompt_tokens = est_tokens

        response = requests.post(self.base_url, json=payload,
                                 stream=True, timeout=(10, 120))
        response.raise_for_status()

        with streaming_response(response):
            for line in response.iter_lines(decode_unicode=True):
                check_cancelled()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        content_parts.append(token)
                        tokens_generated += 1
                        if self._stream_callback and tokens_generated % 10 == 0:
                            self._stream_callback(tokens_generated)

                    # Final chunk contains token counts
                    if chunk.get("done", False):
                        prompt_tokens = chunk.get("prompt_eval_count", est_tokens)
                        eval_count = chunk.get("eval_count", tokens_generated)
                        tokens_generated = eval_count if isinstance(eval_count, int) else tokens_generated
                except (json.JSONDecodeError, KeyError):
                    continue

        result = "".join(content_parts)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            tokens_generated,
            model_name=self.model,
        )
        log.debug(f"[Ollama] Streamed {tokens_generated} tokens")
        log.debug(f"[Ollama] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Native chat (/api/chat) ──

    @staticmethod
    def _serialize_messages(messages: List[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content or ""}
            if m.role == "assistant" and m.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in m.tool_calls
                ]
            elif m.role == "tool" and m.tool_name:
                entry["tool_name"] = m.tool_name
            out.append(entry)
        return out

    def _chat(self, messages: List[Message],
              tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        est_tokens = int(sum(len((m.content or "").split()) for m in messages) * 1.3)
        log.debug(f"[Ollama] Chat: ~{est_tokens} est. tokens, "
                  f"{len(messages)} messages, {len(tools or [])} tools")
        token_tracker.set_context(est_tokens)

        payload: dict = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "stream": False,
            "options": {"num_predict": self.max_output_tokens},
        }
        if tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in tools
            ]

        url = f"{self._api_root}/api/chat"
        response = requests.post(url, json=payload, timeout=(10, 300))
        if response.status_code == 400 and tools and \
                "does not support tools" in response.text.lower():
            raise ToolsNotSupportedError(
                f"model '{self.model}' rejected tools: {response.text}")
        response.raise_for_status()
        data = response.json()

        message = data.get("message") or {}
        text = message.get("content", "") or ""
        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            # Some models return arguments as a JSON string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(
                name=fn.get("name", ""), arguments=args, id=str(i)))

        prompt_tokens = data.get("prompt_eval_count", est_tokens)
        completion_tokens = data.get("eval_count", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
        )
        log.debug(f"[Ollama] Chat usage: prompt={prompt_tokens} "
                  f"completion={completion_tokens} tool_calls={len(tool_calls)}")
        return ChatResponse(text=text, tool_calls=tool_calls,
                            stop_reason=data.get("done_reason", "") or "")

    # ── Embeddings (unchanged) ──

    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        embed_model = model or self.model
        url = f"{self._api_root}/api/embed"
        payload = {"model": embed_model, "input": text}
        try:
            response = requests.post(url, json=payload, timeout=(10, 120))
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
        except requests.exceptions.RequestException as e:
            log.error(f"[Ollama] Embedding error: {e}")
            return []

    def generate_embeddings_batch(self, texts: list[str], model: Optional[str] = None, **kwargs) -> list[list[float]]:
        """Embed multiple texts in a single Ollama /api/embed call.

        Ollama natively supports ``"input": ["text1", "text2", ...]``
        and returns all embeddings in one response, which is much faster
        than N sequential calls (avoids per-request model load overhead).
        """
        if not texts:
            return []
        embed_model = model or self.model
        url = f"{self._api_root}/api/embed"
        payload = {"model": embed_model, "input": texts}
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) == len(texts):
                return embeddings
            # Fallback: pad missing with empty vectors
            log.warning("[Ollama] Batch embed returned %d vectors for %d texts",
                        len(embeddings), len(texts))
            while len(embeddings) < len(texts):
                embeddings.append([])
            return embeddings
        except requests.exceptions.RequestException as e:
            log.warning(f"[Ollama] Batch embedding failed, falling back to sequential: {e}")
            return [self.generate_embedding(t, model=model, **kwargs) for t in texts]

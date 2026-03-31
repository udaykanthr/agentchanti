import json
import requests
from typing import List, Optional

from .base import LLMClient
from ..cli_display import token_tracker, log


class LMStudioClient(LLMClient):

    def __init__(self, base_url: str, model: str,
                 reasoning_effort: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[LM Studio] Sending ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)
        # log.debug(f"[LM Studio] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=headers, json=payload,
                                 timeout=(10, 300))
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", est_tokens)
        completion_tokens = usage.get("completion_tokens", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
        )
        log.debug(f"[LM Studio] Usage: prompt={prompt_tokens} completion={completion_tokens}")

        response_text = data["choices"][0]["message"]["content"]
        log.debug(f"[LM Studio] Response:\n{response_text}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[LM Studio] Streaming ~{est_tokens} est. tokens")
        token_tracker.set_context(est_tokens)
        # log.debug(f"[LM Studio] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
            log.debug(f"[LM Studio] reasoning_effort={self.reasoning_effort}")
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        content_parts: list[str] = []
        tokens_generated = 0
        prompt_tokens = est_tokens

        # timeout: (connect, read-per-chunk); generous read timeout for slow models
        response = requests.post(url, headers=headers, json=payload,
                                 stream=True, timeout=(10, 120))
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        content_parts.append(token)
                        tokens_generated += 1
                        if self._stream_callback and tokens_generated % 10 == 0:
                            self._stream_callback(tokens_generated)
                    # Extract actual usage from the final chunk if available
                    usage = chunk.get("usage")
                    if usage:
                        pt = usage.get("prompt_tokens")
                        ct = usage.get("completion_tokens")
                        if isinstance(pt, int):
                            prompt_tokens = pt
                        if isinstance(ct, int):
                            tokens_generated = ct
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        result = "".join(content_parts)
        token_tracker.record(prompt_tokens, tokens_generated,
                             model_name=self.model)
        log.debug(f"[LM Studio] Streamed {tokens_generated} tokens")
        log.debug(f"[LM Studio] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Embeddings (unchanged) ──

    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        embed_model = model or self.model
        url = f"{self.base_url}/embeddings"
        payload = {"model": embed_model, "input": text}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload,
                                     timeout=(10, 120))
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if items:
                return items[0].get("embedding", [])
            return []
        except requests.exceptions.RequestException as e:
            log.error(f"[LM Studio] Embedding error: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error(f"[LM Studio] Embedding parse error: {e}")
            return []

    def generate_embeddings_batch(self, texts: list[str], model: Optional[str] = None, **kwargs) -> list[list[float]]:
        """Embed multiple texts in a single /embeddings call (OpenAI-compatible batch)."""
        if not texts:
            return []
        embed_model = model or self.model
        url = f"{self.base_url}/embeddings"
        payload = {"model": embed_model, "input": texts}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            # Sort by index to guarantee order
            items.sort(key=lambda x: x.get("index", 0))
            vectors = [item.get("embedding", []) for item in items]
            if len(vectors) == len(texts):
                return vectors
            log.warning("[LM Studio] Batch embed returned %d vectors for %d texts",
                        len(vectors), len(texts))
            while len(vectors) < len(texts):
                vectors.append([])
            return vectors
        except Exception as e:
            log.warning(f"[LM Studio] Batch embedding failed, falling back to sequential: {e}")
            results = [self.generate_embedding(t, model=model, **kwargs) for t in texts]
            if all(not v for v in results):
                raise RuntimeError(
                    f"LM Studio embedding unavailable: batch and all {len(texts)} sequential "
                    f"attempts returned empty vectors. The embedding model may have crashed "
                    f"(check LM Studio logs for GPU/Vulkan errors like 'ErrorDeviceLost')."
                )
            return results

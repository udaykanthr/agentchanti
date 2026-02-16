import requests
from typing import List, Optional
from .base import LLMClient
from ..cli_display import token_tracker, log


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        # Derive the API root for endpoints like /api/embed
        # base_url is typically http://localhost:11434/api/generate
        if "/api/" in base_url:
            self._api_root = base_url.rsplit("/api/", 1)[0]
        else:
            self._api_root = base_url.rstrip("/")

    def generate_response(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Ollama] Sending ~{est_tokens} est. tokens")
        log.debug(f"[Ollama] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "")

            prompt_tokens = data.get("prompt_eval_count", est_tokens)
            completion_tokens = data.get("eval_count", 0)
            token_tracker.record(
                prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
                completion_tokens if isinstance(completion_tokens, int) else 0
            )
            log.debug(f"[Ollama] Usage: prompt={prompt_tokens} completion={completion_tokens}")
            log.debug(f"[Ollama] Response:\n{result}")

            return result
        except requests.exceptions.RequestException as e:
            log.error(f"[Ollama] Connection error: {e}")
            return ""

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        embed_model = model or self.model
        url = f"{self._api_root}/api/embed"
        payload = {"model": embed_model, "input": text}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
        except requests.exceptions.RequestException as e:
            log.error(f"[Ollama] Embedding error: {e}")
            return []

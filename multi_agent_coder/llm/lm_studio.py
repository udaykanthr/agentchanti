import requests
import json
from .base import LLMClient

class LMStudioClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt: str) -> str:
        # Estimate tokens sent (rough: ~1.3 tokens per word)
        est_tokens = int(len(prompt.split()) * 1.3)
        print(f"  [LLM] Sending ~{est_tokens} tokens...")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        headers = {
            "Content-Type": "application/json"
        }
        try:
            # LM Studio uses /v1/chat/completions endpoint structure
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Read actual token usage from API response
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")
            print(f"  [LLM] Received: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
            response_text = data['choices'][0]['message']['content']
            print(f"  [LLM] Response: {response_text}")

            return response_text
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with LM Studio: {e}")
            return ""
        except (KeyError, IndexError, json.JSONDecodeError) as e:
             print(f"Error parsing response from LM Studio: {e}")
             return ""

import requests
import json
from .base import LLMClient

class LMStudioClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt: str) -> str:
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
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with LM Studio: {e}")
            return ""
        except (KeyError, IndexError, json.JSONDecodeError) as e:
             print(f"Error parsing response from LM Studio: {e}")
             return ""

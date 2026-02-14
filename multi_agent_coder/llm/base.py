from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

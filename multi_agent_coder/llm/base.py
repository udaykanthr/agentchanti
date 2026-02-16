from abc import ABC, abstractmethod
from typing import List, Optional

class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate an embedding vector for the given text."""
        pass

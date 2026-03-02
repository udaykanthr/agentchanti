from abc import ABC, abstractmethod
from ..llm.base import LLMClient

class Agent(ABC):
    def __init__(self, name: str, role: str, goal: str, llm_client: LLMClient,
                 prompt_suffix: str = ""):
        self.name = name
        self.role = role
        self.goal = goal
        self.llm_client = llm_client
        self.prompt_suffix = prompt_suffix

    @abstractmethod
    def process(self, task: str, context: str = "") -> str:
        """
        Process the given task and return the result.
        """
        pass

    def _build_prompt(self, task: str, context: str, language: str | None = None) -> str:
        import os as _os
        prompt = f"Role: {self.role}\nGoal: {self.goal}\n\n"
        if language:
            from ..language import get_language_name
            prompt += f"Language: {get_language_name(language)}\n\n"
        if _os.name == 'nt':
            prompt += "Platform: Windows (use cmd.exe-compatible commands)\n\n"
        else:
            import platform as _platform
            _sysname = _platform.system()
            _os_label = "macOS" if _sysname == "Darwin" else _sysname
            prompt += f"Platform: {_os_label}\n\n"
        if self.prompt_suffix:
            prompt += f"Instructions: {self.prompt_suffix}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Task: {task}\n\nResponse:"
        return prompt

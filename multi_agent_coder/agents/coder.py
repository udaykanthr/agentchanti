from .base import Agent

class CoderAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += "\n\nPlease write the Python code to implement the plan. Provide only the code, ensuring it is complete and runnable. Wrap the code in ```python ... ``` blocks."
        return self.llm_client.generate_response(prompt)

from .base import Agent

class ReviewerAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += "\n\nPlease review the provided code for any errors, bugs, or potential improvements. If the code looks good, simply state 'Code looks good.' otherwise provide specific feedback."
        return self.llm_client.generate_response(prompt)

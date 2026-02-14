from .base import Agent

class PlannerAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += "\n\nPlease provide a detailed step-by-step plan to solve the task. Format the plan as a numbered list."
        return self.llm_client.generate_response(prompt)

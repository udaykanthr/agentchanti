from .base import Agent


class PlannerAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """

Provide a step-by-step plan as a numbered list.
Keep each step short and actionable. Do NOT include code in this plan.
"""
        return self.llm_client.generate_response(prompt)

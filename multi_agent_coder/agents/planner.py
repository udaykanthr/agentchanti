from .base import Agent


class PlannerAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """

Provide a step-by-step plan as a numbered list.
Keep each step short and actionable. Do NOT include code in this plan.
For steps that involve running shell commands (scanning files, listing directories,
installing packages, etc.), include the exact command in backticks, e.g.:
  1. List all project files with `tree /F` or `Get-ChildItem -Recurse`
  2. Install dependencies with `pip install -r requirements.txt`
"""
        return self.llm_client.generate_response(prompt)

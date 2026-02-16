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

For each step, if it depends on a previous step being completed first, add (depends: N) or (depends: N, M) at the end of the step.
Steps with no dependencies can run in parallel.
Example:
  1. Create the data model
  2. Create the API endpoints (depends: 1)
  3. Write unit tests (depends: 1, 2)
"""
        return self.llm_client.generate_response(prompt)

import unittest
from unittest.mock import MagicMock
from agentchanti.llm.base import LLMClient
from agentchanti.agents.planner import PlannerAgent
from agentchanti.agents.coder import CoderAgent
from agentchanti.agents.reviewer import ReviewerAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.generate_response.side_effect = [
            "1. Step one\n2. Step two", # Plan
            "```python\nprint('Hello World')\n```", # Code
            "Code looks good." # Review
        ]

    def test_agent_flow(self):
        planner = PlannerAgent("Planner", "Role", "Goal", self.mock_llm)
        coder = CoderAgent("Coder", "Role", "Goal", self.mock_llm)
        reviewer = ReviewerAgent("Reviewer", "Role", "Goal", self.mock_llm)

        plan = planner.process("Test Task")
        self.assertIn("Step one", plan)

        code = coder.process(f"Implement: {plan}")
        self.assertIn("print('Hello World')", code)

        review = reviewer.process(f"Review: {code}")
        self.assertIn("Code looks good", review)

if __name__ == '__main__':
    unittest.main()

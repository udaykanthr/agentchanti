import unittest
from unittest.mock import MagicMock, patch
import re

# We need to mock some imports before importing pipeline to avoid side effects
with patch('multi_agent_coder.orchestrator.pipeline.log'), \
     patch('multi_agent_coder.orchestrator.pipeline._logger'):
    from multi_agent_coder.orchestrator.pipeline import _execute_step

class TestReclassification(unittest.TestCase):
    def setUp(self):
        self.task = "Test Task"
        self.steps = ["Step 1: Install deps\n> npm install"]
        self.llm_client = MagicMock()
        self.executor = MagicMock()
        self.memory = MagicMock()
        self.display = MagicMock()
        self.coder = MagicMock()
        self.tester = MagicMock()
        self.reviewer = MagicMock()
        
        # Setup display.steps to avoid KeyError
        self.display.steps = {0: {"type": "CODE"}}
        
        # Mock memory.all_files to return empty dict
        self.memory.all_files.return_value = {}

    @patch('multi_agent_coder.orchestrator.pipeline._handle_cmd_step')
    @patch('multi_agent_coder.orchestrator.pipeline._handle_code_step')
    def test_reclassify_code_to_cmd(self, mock_handle_code, mock_handle_cmd):
        from multi_agent_coder.orchestrator.plan_step import PlanStep
        
        # 1. Setup: A CODE step that looks like a CMD step (has markers, no target:, no inline)
        plan_step = PlanStep(id="1.1", step_type="CODE", description="Step 1: Install deps\n> npm install")
        all_plan_steps = [plan_step]
        
        # 2. Mock LLM to return CMD upon re-classification
        self.llm_client.generate_response.return_value = "CMD"
        
        # 3. Mock success for handlers
        mock_handle_cmd.return_value = (True, "")
        
        # 4. Call _execute_step
        _execute_step(
            0, self.steps[0], 
            steps=self.steps,
            llm_client=self.llm_client, 
            executor=self.executor, 
            coder=self.coder, 
            reviewer=self.reviewer, 
            tester=self.tester,
            task=self.task, 
            memory=self.memory, 
            display=self.display,
            language="javascript",
            plan_step=plan_step,
            all_plan_steps=all_plan_steps
        )
        
        # 5. Verify:
        # - LLM was called for re-classification because it matched our heuristic
        #   (CODE step, no target:, no inline, has markers)
        self.llm_client.generate_response.assert_called()
        
        # - step_type was updated to CMD
        self.assertEqual(plan_step.step_type, "CMD")
        
        # - _handle_cmd_step was called, NOT _handle_code_step
        mock_handle_cmd.assert_called()
        mock_handle_code.assert_not_called()

if __name__ == '__main__':
    unittest.main()

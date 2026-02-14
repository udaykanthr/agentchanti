from .base import Agent

class TesterAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """
Please generate unit tests for the provided code. 
The tests should use the `pytest` framework.
Format your response by specifying the filename for each test file using the following format:
#### [FILE]: path/to/test_file.py
```python
# test code here
```
Make sure the tests are comprehensive and cover the logic described in the task.
"""
        return self.llm_client.generate_response(prompt)

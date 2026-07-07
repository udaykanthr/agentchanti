"""
agentchanti — Multi-Agent Local Coder (AgentChanti).

Public API for library usage::

    from agentchanti import run_task, TaskResult

    result = run_task(task="Add logging to all endpoints", auto=True)
"""

from .api import run_task, TaskResult
from .agent_tools import AgentTools

__all__ = ["run_task", "TaskResult", "AgentTools"]

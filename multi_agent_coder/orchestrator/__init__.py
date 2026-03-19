"""
Orchestrator package — multi-agent pipeline for code generation.

Backward-compatible re-exports so that existing code like
``from multi_agent_coder.orchestrator import FileMemory`` continues to work.
"""

from .memory import FileMemory
from .pipeline import build_step_waves
from .plan_step import PlanStep, parse_structured_plan, parse_heuristic_plan, build_waves, fix_import_dependencies
from .cli import main
from .step_handlers import (
    _shell_instructions,
    _shell_examples,
    MAX_STEP_RETRIES,
)

__all__ = [
    "FileMemory",
    "build_step_waves",
    "PlanStep",
    "parse_structured_plan",
    "parse_heuristic_plan",
    "fix_import_dependencies",
    "build_waves",
    "main",
]

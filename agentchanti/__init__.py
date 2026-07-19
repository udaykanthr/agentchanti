"""
agentchanti — Multi-Agent Local Coder (AgentChanti).

Public API for library usage::

    from agentchanti import run_task, TaskResult

    result = run_task(task="Add logging to all endpoints", auto=True)
"""

from importlib import metadata as _metadata

try:
    # Reflect the version of the *installed* distribution, so
    # `agentchanti --version` (and `agentchanti.__version__`) always
    # match what `pip install` actually put on the system rather than a
    # value hard-coded here that can drift from pyproject.toml.
    __version__ = _metadata.version("agentchanti")
except _metadata.PackageNotFoundError:
    # Running from a source tree that was never installed (e.g. `python -c`
    # from the repo root without `pip install -e .`).
    __version__ = "0.0.0+unknown"

from .api import run_task, TaskResult
from .agent_tools import AgentTools

__all__ = ["run_task", "TaskResult", "AgentTools", "__version__"]

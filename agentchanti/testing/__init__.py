"""
AgentChanti Testing — agent-driven browser & API end-to-end testing.

Records a user's browser session through an MCP server, normalizes raw
events into a semantic, agent-understandable test spec, and replays it
with self-healing element location plus request/response schema validation.

Public surface (lazy-imported — importing this package does NOT pull in
Playwright or MCP client libraries; those load only when you instantiate
the relevant class):

    from agentchanti.testing import Recorder, Normalizer, Replayer, Validator, Reporter

Install the optional dependencies with::

    pip install agentchanti[testing]
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = [
    "Recorder", "Normalizer", "Replayer", "Validator", "Reporter",
    # Recording schema — pure-Python, no heavy deps, safe to import eagerly
    # via `from agentchanti.testing import Spec, Step, Locator, ...`.
    "Spec", "Step", "Locator", "NetworkExpectation", "Assertion",
]


def __getattr__(name: str):
    # PEP 562 — lazy attribute access at module level.
    # Heavy deps (playwright, mcp) only import when the class is first referenced.
    if name == "Recorder":
        from .recorder import Recorder
        return Recorder
    if name == "Normalizer":
        from .normalizer import Normalizer
        return Normalizer
    if name == "Replayer":
        from .replayer import Replayer
        return Replayer
    if name == "Validator":
        from .validator import Validator
        return Validator
    if name == "Reporter":
        from .reporter import Reporter
        return Reporter
    if name in {"Spec", "Step", "Locator", "NetworkExpectation", "Assertion"}:
        from . import spec as _spec
        return getattr(_spec, name)
    raise AttributeError(f"module 'agentchanti.testing' has no attribute {name!r}")

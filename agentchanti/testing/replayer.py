"""
Replayer — executes a semantic test spec against a live browser.

For each step in the spec, the replayer uses the semantic label + locator
fallbacks to locate elements against the *current* DOM. When cached
locators still work, no LLM call is made; when they fail, the LLM is
consulted to re-derive a working locator (self-healing). Network traffic
observed during replay is handed to ``Validator`` for schema checks.
"""

from __future__ import annotations

from pathlib import Path


class Replayer:
    """Drive a browser through a semantic test spec via an MCP browser server."""

    def __init__(self, mcp_server_url: str, llm_client):
        self.mcp_server_url = mcp_server_url
        self.llm_client = llm_client

    def replay(self, spec_path: str | Path) -> dict:
        """Execute every step in the spec. Return a run result dict for Validator."""
        raise NotImplementedError("Replayer.replay is not implemented yet")

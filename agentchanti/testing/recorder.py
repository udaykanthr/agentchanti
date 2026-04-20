"""
Recorder — captures a user's browser session through an MCP browser server.

Connects to a browser MCP server (e.g., Playwright MCP), attaches listeners
for DOM interaction events and network traffic, and writes a raw session
trace to disk. The raw trace is later consumed by ``Normalizer`` which
converts it into a semantic, agent-understandable test spec.
"""

from __future__ import annotations

from pathlib import Path


class Recorder:
    """Capture a browser session via an MCP browser server.

    Parameters
    ----------
    mcp_server_url:
        Endpoint of the browser MCP server (e.g., Playwright MCP).
    output_path:
        Where to write the raw session trace (JSON lines).
    """

    def __init__(self, mcp_server_url: str, output_path: str | Path):
        self.mcp_server_url = mcp_server_url
        self.output_path = Path(output_path)

    def start(self, start_url: str) -> None:
        """Open the browser at ``start_url`` and begin capturing events."""
        raise NotImplementedError("Recorder.start is not implemented yet")

    def stop(self) -> Path:
        """Stop capturing and flush the raw trace to ``output_path``."""
        raise NotImplementedError("Recorder.stop is not implemented yet")

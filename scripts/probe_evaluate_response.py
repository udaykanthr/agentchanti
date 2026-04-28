"""Probe what browser_evaluate's response text looks like for the live recorder.

We need to know exactly how Playwright MCP renders return values so the
Python polling loop can parse them deterministically. Returns are tested
in three shapes: primitive string, plain object, JSON-stringified object.
"""

from __future__ import annotations

from agentchanti.testing.mcp_client import DEFAULT_MCP_SERVER_URL, BrowserMCPClient


def main() -> None:
    with BrowserMCPClient(DEFAULT_MCP_SERVER_URL) as mcp:
        mcp.navigate("data:text/html,<h1>x</h1>")

        for label, fn in [
            ("string", '() => "hello"'),
            ("object", '() => ({events: [{type:"click",x:42}], url: location.href})'),
            ("stringified-object", '() => JSON.stringify({events:[{type:"click"}], url: location.href})'),
        ]:
            text = mcp._call_tool("browser_evaluate", {"function": fn})
            print(f"--- {label} ---")
            print(repr(text))
            print()


if __name__ == "__main__":
    main()

"""One-off: print the raw snapshot text Playwright MCP returns so we can
write a parser against the actual format.

Requires a running Playwright MCP server on the default port.
"""

from __future__ import annotations

from agentchanti.testing.mcp_client import DEFAULT_MCP_SERVER_URL, BrowserMCPClient

HTML = """
<h1>Test Page</h1>
<form>
  <label for="email">Email</label>
  <input id="email" name="email" type="email" data-testid="email-input">
  <button type="submit" data-testid="submit-btn">Sign in</button>
</form>
<a href="/help" aria-label="Open help">?</a>
"""


def main() -> None:
    url = f"data:text/html,{HTML}"
    with BrowserMCPClient(DEFAULT_MCP_SERVER_URL) as mcp:
        mcp.navigate(url)
        snap = mcp.snapshot()
    raw = snap.get("raw", "")
    print("=" * 60)
    print(f"snapshot length: {len(raw)} chars")
    print("=" * 60)
    print(raw)
    print("=" * 60)


if __name__ == "__main__":
    main()

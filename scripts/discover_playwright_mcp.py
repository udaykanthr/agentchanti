"""
One-off: connect to a running Playwright MCP server and dump its tool surface.

Run once while `npx @playwright/mcp@latest --port 8931` is listening to see
exactly which tools / input schemas we need to map onto BrowserMCPClient.
Not part of the library — lives under scripts/ so it's obvious it's temporary
exploration, not shipped code.
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8931/mcp"


async def main() -> None:
    async with streamablehttp_client(URL) as (read, write, _get_sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

            print(f"{len(result.tools)} tool(s) exposed by {URL}\n" + "=" * 60)
            for t in result.tools:
                print(f"\n* {t.name}")
                if t.description:
                    first = t.description.strip().splitlines()[0][:120]
                    print(f"    {first}")
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                if props:
                    print("    args:")
                    for name, spec in props.items():
                        mark = "*" if name in required else " "
                        typ = spec.get("type", "?")
                        desc = (spec.get("description") or "").strip().splitlines()
                        desc_short = desc[0][:80] if desc else ""
                        print(f"      {mark} {name}: {typ}  {desc_short}")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from mcp_client import MCPClient

async def main():
    client = MCPClient()
    await client.connect()
    try:
        res = await client.call_tool("refresh_unity", {})
        print("Refresh Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())

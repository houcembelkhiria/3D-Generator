import asyncio
from fastmcp.client import Client

async def test():
    try:
        c = Client('http://localhost:6400/mcp')
        await c.initialize()
        print("Tools:", await c.list_tools())
        await c.close()
    except Exception as e:
        print("Failed:", type(e), e)

asyncio.run(test())

import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("ws://localhost:6400/mcp") as ws:
            print("Connected to ws://localhost:6400/mcp!")
            await ws.close()
    except Exception as e:
        print("Failed to connect:", type(e), e)

asyncio.run(test())

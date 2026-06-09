"""MCP client using the FastMCP SDK for StreamableHTTP support.

Connects to http://localhost:6400/mcp (Unity Editor with MCP for Unity package).
"""
from __future__ import annotations

import logging
from typing import Any

from fastmcp.client import Client

from config import UNITY_MCP_HOST, UNITY_MCP_PORT

logger = logging.getLogger(__name__)

# FastMCP uses StreamableHTTP or SSE on this URL
_MCP_URI = f"http://{UNITY_MCP_HOST}:{UNITY_MCP_PORT}/mcp"


from contextlib import AsyncExitStack

class MCPClient:
    """Persistent FastMCP client."""

    def __init__(self) -> None:
        self.client: Client | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        """Initialize connection to Unity FastMCP server."""
        try:
            logger.info("Connecting to Unity FastMCP server at %s", _MCP_URI)
            self._exit_stack = AsyncExitStack()
            c = Client(_MCP_URI, auto_initialize=True)
            self.client = await self._exit_stack.enter_async_context(c)
            logger.info("FastMCP connected and initialized.")
        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection."""
        await self._exit_stack.aclose()
        self.client = None

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool via the MCP session."""
        if not self.is_connected:
            raise RuntimeError("MCP client not connected")
        
        # In fastmcp.client.Client, call_tool returns the raw response string or object
        return await self.client.call_tool(name, arguments)

    async def list_tools(self) -> list[dict]:
        """Return the list of tools exposed by Unity MCP."""
        if not self.is_connected:
            raise RuntimeError("MCP client not connected")
        
        tools = await self.client.list_tools()
        # FastMCP client list_tools returns a list of Tool objects or dicts
        # We handle both in case of API differences
        result = []
        for t in tools:
            if hasattr(t, "name"):
                result.append({"name": t.name, "description": getattr(t, "description", "")})
            else:
                result.append(t)
        return result

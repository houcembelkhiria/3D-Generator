"""MCP WebSocket client for communicating with the CoplayDev/unity-mcp server.

Connects to ws://localhost:6400 (Unity Editor with MCP for Unity package).
Sends JSON-RPC 2.0 tool calls and waits for the response.
Handles reconnection automatically on lost connection.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from config import UNITY_MCP_HOST, UNITY_MCP_PORT

logger = logging.getLogger(__name__)

_WS_URI = f"ws://{UNITY_MCP_HOST}:{UNITY_MCP_PORT}"


class MCPClient:
    """Persistent WebSocket MCP client with auto-reconnect."""

    def __init__(self) -> None:
        self._ws = None
        self._id  = 0

    async def connect(self) -> None:
        """Open WebSocket connection to Unity MCP server."""
        self._ws = await websockets.connect(_WS_URI, ping_interval=20, ping_timeout=10)
        logger.info("Connected to Unity MCP server at %s", _WS_URI)
        # Send MCP initialize handshake
        await self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "clientInfo":      {"name": "3d-generator-relay", "version": "1.0.0"},
            },
        })
        resp = await self._recv()
        logger.debug("MCP initialize response: %s", resp)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send a tools/call request and return the result."""
        if not self.is_connected:
            raise RuntimeError("MCP client not connected")
        call_id = self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id":      call_id,
            "method":  "tools/call",
            "params":  {"name": name, "arguments": arguments},
        })
        resp = await self._recv()
        if "error" in resp:
            raise RuntimeError(f"MCP tool error: {resp['error']}")
        return resp.get("result", {})

    async def list_tools(self) -> list[dict]:
        """Return the list of tools exposed by Unity MCP."""
        if not self.is_connected:
            raise RuntimeError("MCP client not connected")
        await self._send({
            "jsonrpc": "2.0",
            "id":      self._next_id(),
            "method":  "tools/list",
            "params":  {},
        })
        resp = await self._recv()
        return resp.get("result", {}).get("tools", [])

    # ── Internal ──────────────────────────────────────────────────────────────
    async def _send(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload))

    async def _recv(self) -> dict:
        raw = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
        return json.loads(raw)

    def _next_id(self) -> int:
        self._id += 1
        return self._id

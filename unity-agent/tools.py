"""MCP tool calls for Unity spawn operations.

Single entry point: spawn_glb(mcp_client, event)
  → calls spawn_glb_from_url on the Unity MCP server
  → Unity downloads the GLB and spawns it in the scene
"""
from __future__ import annotations

import logging
from typing import Any

from mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Name of the custom MCP tool registered in MCPSpawnTool.cs
SPAWN_TOOL = "spawn_glb_from_url"


async def spawn_glb(client: MCPClient, event: dict[str, Any]) -> str:
    """Call the Unity MCP spawn tool with event data.

    Args:
        client: Connected MCPClient instance.
        event:  Spawn event dict with keys: url, name, scene, id.

    Returns:
        Result message from Unity.
    """
    url   = event.get("url", "")
    name  = event.get("name", "")
    scene = event.get("scene", "existing")

    if not url:
        raise ValueError(f"Event {event.get('id')} has no url")

    logger.info("Calling MCP tool '%s': url=%s name=%s scene=%s",
                SPAWN_TOOL, url, name, scene)

    result = await client.call_tool(SPAWN_TOOL, {
        "url":   url,
        "name":  name,
        "scene": scene,
    })

    logger.info("MCP tool result: %s", result)
    return str(result)

"""Unity MCP Relay Agent — main entry point.

Bridges the FastAPI backend event queue with the Unity MCP server:

  [Backend /unity/pending-events]
          ↓  HTTP poll (every POLL_INTERVAL_SEC)
  [This agent]
          ↓  MCP WebSocket call: spawn_glb_from_url(url, name, scene)
  [Unity Editor + CoplayDev/unity-mcp package]
          ↓
  Model spawned in scene

Usage:
    cd unity-agent
    cp .env.example .env      # edit BACKEND_URL, UNITY_AGENT_TOKEN
    pip install -r requirements.txt
    python agent.py
"""
from __future__ import annotations

import asyncio
import logging
import sys

from backend_client import ack_event, fetch_pending_events
from config import POLL_INTERVAL_SEC, UNITY_MCP_HOST, UNITY_MCP_PORT
from mcp_client import MCPClient
from tools import spawn_glb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("unity-agent")


async def _connect_with_retry(client: MCPClient, max_wait: float = 30.0) -> bool:
    """Try to connect to Unity MCP, with exponential backoff."""
    delay = 1.0
    while True:
        try:
            await client.connect()
            return True
        except Exception as exc:
            logger.warning("Unity MCP not reachable (%s). Retrying in %.0fs…", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_wait)


async def main() -> None:
    client = MCPClient()
    logger.info("Unity MCP Relay Agent starting…")
    logger.info("Unity MCP target : ws://%s:%s", UNITY_MCP_HOST, UNITY_MCP_PORT)

    await _connect_with_retry(client)

    # Print available tools on startup for verification
    try:
        tools = await client.list_tools()
        names = [t.get("name") for t in tools]
        logger.info("Unity MCP tools available: %s", names)
        if "spawn_glb_from_url" not in names:
            logger.warning(
                "Tool 'spawn_glb_from_url' not found in Unity. "
                "Make sure MCPSpawnTool.cs is compiled and Unity is open."
            )
    except Exception as exc:
        logger.warning("Could not list MCP tools: %s", exc)

    logger.info("Polling backend every %.1fs…", POLL_INTERVAL_SEC)

    while True:
        # ── reconnect if Unity MCP dropped ────────────────────────────────
        if not client.is_connected:
            logger.info("MCP connection lost — reconnecting…")
            await _connect_with_retry(client)

        # ── fetch pending events from backend ──────────────────────────────
        events = fetch_pending_events()

        for event in events:
            event_id = event.get("id", "")
            logger.info("Processing event %s (url=%s)", event_id, event.get("url", ""))
            try:
                await spawn_glb(client, event)
                ack_event(event_id)
                logger.info("✅ Event %s acked", event_id)
            except Exception as exc:
                logger.error("❌ Failed to spawn event %s: %s", event_id, exc)
                # Do NOT ack on failure — the agent will retry on next poll

        await asyncio.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
        sys.exit(0)

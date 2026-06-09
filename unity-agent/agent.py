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


async def _connect_with_retry(client: MCPClient, max_wait: float = 30.0, max_retries: int = -1) -> bool:
    """Try to connect to Unity MCP, with exponential backoff.
    If max_retries >= 0, returns False after max_retries failures.
    """
    delay = 1.0
    retries = 0
    while True:
        try:
            await client.connect()
            return True
        except Exception as exc:
            if max_retries >= 0 and retries >= max_retries:
                logger.warning("Max retries reached. Unity MCP still not reachable.")
                return False
            
            logger.warning("Unity MCP not reachable (%s). Retrying in %.0fs…", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_wait)
            retries += 1


async def main() -> None:
    client = MCPClient()
    logger.info("Unity MCP Relay Agent starting…")
    logger.info("Unity MCP target : ws://%s:%s", UNITY_MCP_HOST, UNITY_MCP_PORT)

    try:
        await client.connect()
    except Exception:
        logger.warning("Unity MCP not reachable at startup. Will launch automatically on first event.")

    # Print available tools on startup for verification if connected
    if client.is_connected:
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
        # ── fetch pending events from backend ──────────────────────────────
        events = fetch_pending_events()

        if events:
            # We have events! If Unity is not connected, launch it.
            if not client.is_connected:
                logger.info("Events found, but Unity is not connected. Launching Unity...")
                try:
                    import subprocess
                    from config import UNITY_EDITOR_APP_PATH, UNITY_PROJECT_PATH
                    subprocess.Popen([
                        "open", "-a", UNITY_EDITOR_APP_PATH, "--args", "-projectPath", UNITY_PROJECT_PATH
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    logger.error("Failed to launch Unity: %s", e)
                
                logger.info("Waiting for Unity MCP server to come online...")
                connected = await _connect_with_retry(client, max_wait=5.0, max_retries=10)
                if not connected:
                    logger.error("Could not connect to Unity MCP. Aborting spawn attempt.")
                    if "--one-shot" in sys.argv:
                        sys.exit(1)
                    continue

            for event in events:
                event_id = event.get("id", "")
                logger.info("Processing event %s (url=%s)", event_id, event.get("url", ""))
                
                if not client.is_connected:
                    logger.error("❌ Failed to spawn event %s: Client is not connected.", event_id)
                    continue

                try:
                    await spawn_glb(client, event)
                    ack_event(event_id)
                    logger.info("✅ Event %s acked", event_id)
                except Exception as exc:
                    logger.error("❌ Failed to spawn event %s: %s", event_id, exc)
                    # Do NOT ack on failure — the agent will retry on next poll
        else:
            # If no events, and we lost connection, just silently reconnect in background if it comes back
            if not client.is_connected:
                try:
                    await client.connect()
                except Exception:
                    pass

        if "--one-shot" in sys.argv:
            logger.info("One-shot mode: Exiting after one poll.")
            break

        await asyncio.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
        sys.exit(0)

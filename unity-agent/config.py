"""Configuration for the Unity MCP Relay Agent.

Copy .env.example to .env and edit the values for your environment.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Backend connection ────────────────────────────────────────────────────────
# URL of the FastAPI backend.
# Local:      http://localhost:8000
# Production: https://your-server.com
BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Shared secret set via UNITY_AGENT_TOKEN env var on the backend.
UNITY_AGENT_TOKEN: str = os.environ.get("UNITY_AGENT_TOKEN", "dev-token")

# ── Unity MCP Server ──────────────────────────────────────────────────────────
# Host and port where the CoplayDev/unity-mcp package listens.
# Unity must be open with the MCP for Unity package installed.
UNITY_MCP_HOST: str = os.environ.get("UNITY_MCP_HOST", "localhost")
UNITY_MCP_PORT: int = int(os.environ.get("UNITY_MCP_PORT", "6400"))

# ── Unity Auto-Launch ────────────────────────────────────────────────────────
UNITY_EDITOR_APP_PATH: str = os.environ.get("UNITY_EDITOR_APP_PATH", "/Applications/Unity/Hub/Editor/6000.3.7f1/Unity.app")
UNITY_PROJECT_PATH: str = os.environ.get("UNITY_PROJECT_PATH", "/Users/houcem/My project")

# ── Polling ───────────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC: float = float(os.environ.get("POLL_INTERVAL_SEC", "1.0"))

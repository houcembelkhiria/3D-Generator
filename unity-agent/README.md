# Unity MCP Relay Agent

Bridges the 3D Generator backend with Unity Editor via the
[CoplayDev/unity-mcp](https://github.com/CoplayDev/unity-mcp) package.

```
[Backend] → EventBus → [This Agent] → MCP WebSocket → [Unity Editor]
```

## Prerequisites

| Requirement | Details |
|---|---|
| Python 3.10+ | `python --version` |
| Unity open | `UnityProject/` must be open with the MCP for Unity package installed |
| Backend running | `make dev` from repo root |

## Install Unity MCP package

In Unity Editor: **Window → Package Manager → + → Add package from git URL**

```
https://github.com/CoplayDev/unity-mcp.git?path=/MCPForUnity#main
```

After install: **Window → MCP for Unity → Settings** — note the port (default **6400**).

## Setup

```bash
cd unity-agent
cp .env.example .env
# Edit .env — set BACKEND_URL and UNITY_AGENT_TOKEN to match your backend
pip install -r requirements.txt
```

## Run

```bash
python agent.py
```

You should see:
```
Connected to Unity MCP server at ws://localhost:6400
Unity MCP tools available: ['spawn_glb_from_url', ...]
Polling backend every 1.0s…
```

## Local vs Production

Only the `.env` file changes — the agent code is identical:

```bash
# Local (.env)
BACKEND_URL=http://localhost:8000
UNITY_AGENT_TOKEN=dev-token

# Production (.env)
BACKEND_URL=https://your-server.com
UNITY_AGENT_TOKEN=your-production-secret
```

## How it works

1. Agent polls `GET /api/v1/unity/pending-events` every second
2. For each event, calls MCP tool `spawn_glb_from_url(url, name, scene)`
3. Unity downloads the GLB, imports it via glTFast, and spawns it in the scene
4. Agent calls `POST /api/v1/unity/ack` to confirm processing

## Troubleshooting

**`Unity MCP not reachable`** — Unity is not open, or the MCP package is not installed/configured.

**`Tool 'spawn_glb_from_url' not found`** — `MCPSpawnTool.cs` is not compiled. 
Open Unity, check the Console for errors, and ensure the assembly definition includes the file.

**`401 Invalid or missing UNITY_AGENT_TOKEN`** — Token mismatch between agent `.env` and backend `UNITY_AGENT_TOKEN` env var.

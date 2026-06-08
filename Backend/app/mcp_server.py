"""
MCP server for 3D Generator — exposes Unity 3D generation tools via the MCP protocol.

Usage:
    # stdio transport (Claude Code, Claude Desktop)
    python -m app.mcp_server

    # SSE transport (persistent HTTP service on port 8002)
    python -m app.mcp_server --sse
    python -m app.mcp_server --sse --port 8002

Set BACKEND_URL env var to point at the FastAPI backend (default: http://localhost:8000).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Activity log tailed by Unity MCPMonitor editor window
_LOG_FILE = Path(__file__).resolve().parents[2] / "Backend" / "generated" / "mcp_activity.log"


def _log(tool: str, args: dict, result: object) -> None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": tool,
        "args": args,
        "result": result if isinstance(result, (dict, list)) else {"value": str(result)},
    }
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


mcp = FastMCP(
    "3D Generator",
    instructions=(
        "You control a local Hunyuan3D generation pipeline connected to Unity Editor. "
        "To create a 3D asset: call generate_3d_from_text (or generate_3d_from_image_url), "
        "then poll get_generation_status until status='completed', "
        "then call spawn_in_unity to import and place it in the scene. "
        "Or use generate_and_spawn to do all three steps automatically."
    ),
)


# ---------------------------------------------------------------------------
# Tool: list_models
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_models() -> list[dict]:
    """List all 3D models that have been generated and are stored on disk.

    Returns a list of model objects with fields: id, prompt, source,
    previewUrl, downloadUrl, createdAt.
    """
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BACKEND_URL}/api/v1/models")
        r.raise_for_status()
        data = r.json()
    _log("list_models", {}, {"count": len(data)})
    return data


# ---------------------------------------------------------------------------
# Tool: generate_3d_from_text
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_3d_from_text(
    prompt: str,
    steps: int = 5,
    texture: bool = False,
    octree_resolution: int = 128,
) -> dict:
    """Generate a 3D mesh from a text description (non-blocking).

    Returns immediately with {"uid": "...", "status": "processing"}.
    Call get_generation_status(uid) to poll for completion, then
    spawn_in_unity(uid) to load it into the scene.

    Args:
        prompt: Description of the object, e.g. 'a red sports car'.
        steps: Diffusion steps — 5 is fast, 20 is higher quality (max 100).
        texture: Generate PBR texture map (slower). Default False.
        octree_resolution: Mesh resolution: 64 / 128 / 192. Default 128.
    """
    payload = {
        "text": prompt,
        "num_inference_steps": max(1, min(steps, 100)),
        "texture": texture,
        "octree_resolution": octree_resolution,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{BACKEND_URL}/api/v1/text-to-3d/async", json=payload)
        r.raise_for_status()
        data = r.json()
    _log("generate_3d_from_text", {"prompt": prompt, "steps": steps}, data)
    return data


# ---------------------------------------------------------------------------
# Tool: generate_3d_from_image_url
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_3d_from_image_url(
    image_url: str,
    steps: int = 5,
    texture: bool = False,
    octree_resolution: int = 128,
) -> dict:
    """Generate a 3D mesh from a publicly accessible image URL (non-blocking).

    Downloads the image, base64-encodes it, and submits to the pipeline.
    Best results with a white or transparent background.

    Returns {"uid": "...", "status": "processing"}.

    Args:
        image_url: HTTP/HTTPS URL of a PNG or JPG image.
        steps: Diffusion steps (1-100). Default 5.
        texture: Generate PBR texture. Default False.
        octree_resolution: 64 / 128 / 192. Default 128.
    """
    async with httpx.AsyncClient(timeout=60) as c:
        img_r = await c.get(image_url)
        img_r.raise_for_status()
        b64 = base64.b64encode(img_r.content).decode()

    payload = {
        "image": b64,
        "num_inference_steps": max(1, min(steps, 100)),
        "texture": texture,
        "octree_resolution": octree_resolution,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{BACKEND_URL}/api/v1/image-to-3d/async", json=payload)
        r.raise_for_status()
        data = r.json()
    _log("generate_3d_from_image_url", {"image_url": image_url, "steps": steps}, data)
    return data


# ---------------------------------------------------------------------------
# Tool: get_generation_status
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_generation_status(uid: str) -> dict:
    """Check the status of a 3D generation job.

    Returns:
        status: 'processing' | 'completed' | 'failed'
        progress: 0-100 (when available)
        preview_url: relative URL to the .glb file (when completed)
        error: error message (when failed)

    Args:
        uid: The uid returned by generate_3d_from_text or generate_3d_from_image_url.
    """
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BACKEND_URL}/api/v1/generation-status/{uid}")
        r.raise_for_status()
        data = r.json()
    _log("get_generation_status", {"uid": uid}, data)
    return data


# ---------------------------------------------------------------------------
# Tool: spawn_in_unity
# ---------------------------------------------------------------------------

@mcp.tool()
async def spawn_in_unity(
    model_id: str,
    scene: str = "existing",
    name: str = "",
) -> dict:
    """Spawn a generated 3D model into the open Unity Editor scene.

    Requires Unity Editor to be open with the UnityProject/ project.
    The model is downloaded and imported automatically by SpawnBridge.cs.

    Args:
        model_id: uid of a completed generation (from list_models or generate_*).
        scene: 'existing' to add to current scene, 'new' to open a fresh scene.
        name: Display name for the spawned GameObject (optional).
    """
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BACKEND_URL}/api/v1/models")
        r.raise_for_status()
        models = r.json()

    model = next((m for m in models if m.get("id") == model_id), None)
    if model is None:
        raise ValueError(
            f"Model '{model_id}' not found in gallery. "
            "Call list_models() to see what's available."
        )

    url = model.get("previewUrl") or model.get("downloadUrl") or ""
    if url and not url.startswith("http"):
        url = f"{BACKEND_URL}{url}"

    payload = {
        "url": url,
        "id": model_id,
        "scene": scene if scene in ("existing", "new") else "existing",
        "name": (name or model.get("prompt") or model_id)[:60],
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{BACKEND_URL}/api/v1/unity/spawn", json=payload)
        r.raise_for_status()

    result = {"spawned": True, "model_id": model_id, "scene": scene}
    _log("spawn_in_unity", {"model_id": model_id, "scene": scene}, result)
    return result


# ---------------------------------------------------------------------------
# Tool: generate_and_spawn  (all-in-one agentic tool)
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_and_spawn(
    prompt: str,
    scene: str = "existing",
    steps: int = 5,
    texture: bool = False,
    poll_interval: float = 5.0,
    timeout: float = 300.0,
) -> dict:
    """Generate a 3D model from text AND automatically spawn it in Unity.

    This all-in-one tool runs the full pipeline: text → mesh → Unity.
    It blocks (polls internally) until generation completes or times out.

    Args:
        prompt: Text description of the 3D object.
        scene: 'existing' or 'new'.
        steps: Diffusion steps (5 fast, 20 quality).
        texture: Generate PBR texture.
        poll_interval: Seconds between status checks. Default 5.
        timeout: Max total wait seconds. Default 300.
    """
    # 1. Kick off generation
    gen = await generate_3d_from_text(prompt=prompt, steps=steps, texture=texture)
    uid = gen.get("uid")
    if not uid:
        raise RuntimeError(f"Backend did not return a uid: {gen}")

    # 2. Poll until done
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await get_generation_status(uid)
        s = status.get("status", "")
        if s == "completed":
            break
        if s == "failed":
            raise RuntimeError(f"Generation failed: {status.get('error', 'unknown error')}")
        await asyncio.sleep(poll_interval)
    else:
        raise TimeoutError(f"Generation timed out after {timeout}s (uid={uid})")

    # 3. Spawn
    spawn_result = await spawn_in_unity(model_id=uid, scene=scene, name=prompt)
    result = {"uid": uid, "prompt": prompt, **spawn_result}
    _log("generate_and_spawn", {"prompt": prompt, "scene": scene}, result)
    return result


# ---------------------------------------------------------------------------
# Resource: mcp_activity_log
# ---------------------------------------------------------------------------

@mcp.resource("file://mcp-activity-log")
def mcp_activity_log() -> str:
    """The last 50 lines of the MCP activity log (also shown in Unity MCPMonitor)."""
    if not _LOG_FILE.exists():
        return "No activity yet."
    lines = _LOG_FILE.read_text().splitlines()
    return "\n".join(lines[-50:])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Generator MCP Server")
    parser.add_argument("--sse", action="store_true", help="Use SSE transport (HTTP)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.sse:
        print(f"Starting MCP server (SSE) on http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("Starting MCP server (stdio) — connect via Claude Code or Claude Desktop")
        mcp.run(transport="stdio")

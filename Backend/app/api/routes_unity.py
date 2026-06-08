"""Unity launcher registration endpoints + MCP event queue.

Installs a tiny macOS .app bundle under ~/Applications that registers the
`unity3dgen://` URL scheme with Launch Services. Once installed, any click on
a `unity3dgen://spawn?...` URL routes to UnityLauncher.app.

The spawn workflow now goes through the MCP EventBus:
  POST /unity/spawn  →  EventBus.publish()  →  GET /unity/pending-events
  →  Relay Agent  →  MCP tool call  →  Unity MCPSpawnTool.cs

The register-launcher / launcher-status endpoints remain for the URI scheme
installer (macOS only, host-native only).
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/unity", tags=["Unity Integration"])

# Resolve repo root from this file's location.
# Backend/app/api/routes_unity.py -> Backend/app/api -> Backend/app -> Backend -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_DIR = REPO_ROOT / "scripts" / "UnityLauncher"

APP_NAME = "UnityLauncher.app"
INSTALL_DIR = Path.home() / "Applications"
APP_PATH = INSTALL_DIR / APP_NAME
LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework"
    "/Frameworks/LaunchServices.framework/Support/lsregister"
)

_URL_TYPES_XML = """<array>
  <dict>
    <key>CFBundleURLName</key>
    <string>com.3dgen.unitylauncher.spawn</string>
    <key>CFBundleURLSchemes</key>
    <array>
      <string>unity3dgen</string>
    </array>
  </dict>
</array>"""


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_docker() -> bool:
    return Path("/.dockerenv").exists()


def _guard_host() -> None:
    if not _is_macos():
        raise HTTPException(
            400,
            f"Unity launcher registration is macOS-only (running on {platform.system()}).",
        )
    if _is_docker():
        raise HTTPException(
            400,
            "Backend is running inside Docker and cannot modify the host. "
            "Run the backend natively (uvicorn) to use the in-app installer, "
            "or install the launcher manually — see docs/UNITY_INTEGRATION.md.",
        )


def _lsregister_has_scheme() -> bool:
    try:
        out = subprocess.run(
            [LSREGISTER, "-dump"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return "unity3dgen" in out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


@router.get("/launcher-status")
def launcher_status():
    """Report whether the `unity3dgen://` URL handler is installed on this host."""
    if not _is_macos():
        return {
            "supported": False,
            "reason": f"Unity launcher is macOS-only (running on {platform.system()}).",
            "installed": False,
            "registered": False,
        }
    if _is_docker():
        return {
            "supported": False,
            "reason": (
                "Backend is running in Docker; the launcher must be installed on the host. "
                "Run the backend natively or install the launcher manually."
            ),
            "installed": False,
            "registered": False,
        }

    installed = APP_PATH.is_dir()
    handler_sh = APP_PATH / "Contents" / "Resources" / "handler.sh"
    current_repo_match = False
    if handler_sh.is_file():
        try:
            current_repo_match = f'REPO="{REPO_ROOT}"' in handler_sh.read_text()
        except OSError:
            current_repo_match = False

    return {
        "supported": True,
        "installed": installed,
        "registered": installed and _lsregister_has_scheme(),
        "repo_match": current_repo_match,
        "app_path": str(APP_PATH),
        "repo_root": str(REPO_ROOT),
    }


@router.post("/register-launcher")
def register_launcher():
    """Install UnityLauncher.app under ~/Applications and register the URL scheme."""
    _guard_host()

    applescript = LAUNCHER_DIR / "handler.applescript"
    handler_template = LAUNCHER_DIR / "handler.sh.template"
    for f in (applescript, handler_template):
        if not f.is_file():
            raise HTTPException(500, f"Missing launcher source: {f}")

    for tool in ("osacompile", "plutil"):
        if shutil.which(tool) is None:
            raise HTTPException(500, f"{tool} not found on PATH.")

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    if APP_PATH.exists():
        shutil.rmtree(APP_PATH)

    try:
        subprocess.run(
            ["osacompile", "-o", str(APP_PATH), str(applescript)],
            check=True,
            capture_output=True,
            text=True,
        )

        info_plist = APP_PATH / "Contents" / "Info.plist"

        subprocess.run(
            ["plutil", "-remove", "CFBundleURLTypes", str(info_plist)],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["plutil", "-insert", "CFBundleURLTypes", "-xml", _URL_TYPES_XML, str(info_plist)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["plutil", "-replace", "CFBundleIdentifier", "-string",
             "com.3dgen.unitylauncher", str(info_plist)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["plutil", "-replace", "CFBundleName", "-string", "UnityLauncher", str(info_plist)],
            check=True, capture_output=True, text=True,
        )
        if subprocess.run(
            ["plutil", "-replace", "LSUIElement", "-bool", "true", str(info_plist)],
            capture_output=True,
        ).returncode != 0:
            subprocess.run(
                ["plutil", "-insert", "LSUIElement", "-bool", "true", str(info_plist)],
                check=True, capture_output=True, text=True,
            )

        resources = APP_PATH / "Contents" / "Resources"
        resources.mkdir(parents=True, exist_ok=True)
        handler_sh = resources / "handler.sh"
        handler_sh.write_text(
            handler_template.read_text().replace("__REPO_ROOT__", str(REPO_ROOT))
        )
        handler_sh.chmod(0o755)

        # Strip quarantine so macOS doesn't block the URL handler.
        subprocess.run(
            ["xattr", "-cr", str(APP_PATH)],
            capture_output=True, text=True,
        )
        # Unregister any stale/duplicate bindings first.
        subprocess.run(
            [LSREGISTER, "-u", str(APP_PATH)],
            capture_output=True, text=True,
        )
        subprocess.run(
            [LSREGISTER, "-R", "-f", str(APP_PATH)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.exception("Launcher install failed")
        detail = exc.stderr or exc.stdout or str(exc)
        raise HTTPException(500, f"Launcher install failed: {detail.strip()}")

    return {
        "registered": True,
        "installed": True,
        "supported": True,
        "repo_match": True,
        "app_path": str(APP_PATH),
        "repo_root": str(REPO_ROOT),
    }


class _SpawnBody(BaseModel):
    url: str
    scene: str = "existing"
    id: str = ""
    name: str = ""
    has_texture: bool = False


@router.post("/spawn")
def spawn_model(body: _SpawnBody):
    """Publish a spawn event to the EventBus.

    The Relay Agent (unity-agent/agent.py) polls GET /unity/pending-events,
    picks up the event, and calls the Unity MCP tool spawn_glb_from_url.
    No file-system access required — works locally and on remote servers.
    """
    from app.events.factory import get_event_bus

    bus = get_event_bus()
    event = {
        "url":         body.url,
        "scene":       body.scene if body.scene in ("new", "existing") else "existing",
        "name":        body.name[:60],
        "has_texture": body.has_texture,
    }
    event_id = bus.publish(event)
    return {"spawned": True, "event_id": event_id}


# ── MCP Event Queue endpoints ────────────────────────────────────────────────

class _AckBody(BaseModel):
    event_id: str


def _check_agent_token(authorization: str = "") -> None:
    """Validate Bearer token from Relay Agent."""
    import os
    from fastapi import HTTPException
    expected = os.environ.get("UNITY_AGENT_TOKEN", "dev-token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(401, "Invalid or missing UNITY_AGENT_TOKEN.")


@router.get("/pending-events")
def pending_events(authorization: str = ""):
    """Return unacknowledged spawn events for the Relay Agent.

    The Relay Agent polls this endpoint, picks up events, calls the Unity
    MCP tool spawn_glb_from_url, then POSTs /ack for each processed event.

    Header: Authorization: Bearer <UNITY_AGENT_TOKEN>
    """
    _check_agent_token(authorization)
    from app.events.factory import get_event_bus
    events = get_event_bus().consume(limit=20)
    return {"events": events}


@router.post("/ack")
def ack_event(body: _AckBody, authorization: str = ""):
    """Acknowledge a processed spawn event.

    Header: Authorization: Bearer <UNITY_AGENT_TOKEN>
    """
    _check_agent_token(authorization)
    from app.events.factory import get_event_bus
    get_event_bus().ack(body.event_id)
    return {"acked": True, "event_id": body.event_id}


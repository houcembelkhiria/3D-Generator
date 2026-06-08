"""HTTP client for polling the FastAPI backend event queue."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import BACKEND_URL, UNITY_AGENT_TOKEN

logger = logging.getLogger(__name__)

_HEADERS = {"Authorization": f"Bearer {UNITY_AGENT_TOKEN}"}


def fetch_pending_events() -> list[dict[str, Any]]:
    """GET /api/v1/unity/pending-events → list of unacked spawn events."""
    try:
        r = httpx.get(
            f"{BACKEND_URL}/api/v1/unity/pending-events",
            headers=_HEADERS,
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json().get("events", [])
    except httpx.HTTPStatusError as exc:
        logger.error("Backend returned %s: %s", exc.response.status_code, exc.response.text)
        return []
    except Exception as exc:
        logger.warning("Could not reach backend: %s", exc)
        return []


def ack_event(event_id: str) -> bool:
    """POST /api/v1/unity/ack — mark event as processed."""
    try:
        r = httpx.post(
            f"{BACKEND_URL}/api/v1/unity/ack",
            json={"event_id": event_id},
            headers=_HEADERS,
            timeout=5.0,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Failed to ack event %s: %s", event_id, exc)
        return False

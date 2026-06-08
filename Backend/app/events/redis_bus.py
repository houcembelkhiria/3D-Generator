"""Redis-backed EventBus for production.

Uses a Redis List as a reliable queue:
  - publish → HSET + LPUSH  (store payload + add id to pending list)
  - consume → LRANGE        (peek without removing)
  - ack     → LREM + HDEL   (remove from pending list + delete payload)

Requires REDIS_URL to be set (already in docker-compose.yml).
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import redis

from app.events.bus import EventBus

_QUEUE_KEY   = "unity:spawn:queue"    # Redis List of event IDs (pending)
_PAYLOAD_KEY = "unity:spawn:payloads" # Redis Hash  id → json payload


class RedisBus(EventBus):
    """Production EventBus backed by Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._r = redis.from_url(redis_url, decode_responses=True)

    def publish(self, event: dict[str, Any]) -> str:
        event_id = str(uuid.uuid4())
        event = dict(event, id=event_id)
        pipe = self._r.pipeline()
        pipe.hset(_PAYLOAD_KEY, event_id, json.dumps(event))
        pipe.rpush(_QUEUE_KEY, event_id)
        pipe.execute()
        return event_id

    def consume(self, limit: int = 20) -> list[dict[str, Any]]:
        ids = self._r.lrange(_QUEUE_KEY, 0, limit - 1)
        if not ids:
            return []
        payloads = self._r.hmget(_PAYLOAD_KEY, *ids)
        result = []
        for raw in payloads:
            if raw:
                result.append(json.loads(raw))
        return result

    def ack(self, event_id: str) -> None:
        pipe = self._r.pipeline()
        pipe.lrem(_QUEUE_KEY, 1, event_id)
        pipe.hdel(_PAYLOAD_KEY, event_id)
        pipe.execute()

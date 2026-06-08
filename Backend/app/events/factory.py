"""EventBus factory — reads EVENT_BUS_TYPE from environment.

Usage:
    from app.events.factory import get_event_bus
    bus = get_event_bus()   # singleton, safe to call multiple times

Environment variables:
    EVENT_BUS_TYPE   = "sqlite" (default) | "redis"
    REDIS_URL        = redis://localhost:6379/0 (used when type=redis)
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.events.bus import EventBus


@lru_cache(maxsize=1)
def get_event_bus() -> EventBus:
    bus_type = os.environ.get("EVENT_BUS_TYPE", "sqlite").lower()

    if bus_type == "redis":
        from app.events.redis_bus import RedisBus
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return RedisBus(redis_url=redis_url)

    # Default: SQLite (no extra infra needed)
    from app.events.sqlite_bus import SQLiteBus
    return SQLiteBus()

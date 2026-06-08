"""Abstract EventBus interface for Unity spawn events.

Two implementations:
- SQLiteBus  → local dev (no Redis required)
- RedisBus   → production

Chosen by factory.py based on EVENT_BUS_TYPE env var.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EventBus(ABC):
    """Minimal publish/consume/ack interface for spawn events."""

    @abstractmethod
    def publish(self, event: dict[str, Any]) -> str:
        """Store a spawn event. Returns the generated event_id."""

    @abstractmethod
    def consume(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return up to *limit* unacknowledged events (oldest first)."""

    @abstractmethod
    def ack(self, event_id: str) -> None:
        """Mark an event as processed so it won't be returned again."""

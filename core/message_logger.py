# -*- coding: utf-8 -*-
"""MessageLogger — zero-intrusion persistence for agent messages."""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum metadata size to persist (bytes). Larger payloads are summarised.
_MAX_METADATA_BYTES = 2048


def _summarise_metadata(meta: dict) -> dict:
    """Reduce large metadata to summary form for storage."""
    summary = {}
    for key, value in meta.items():
        if isinstance(value, list) and len(value) > 3:
            # Store count + first few IDs for large lists (e.g. posts)
            ids = [
                item.get("id", str(item)[:40]) if isinstance(item, dict) else str(item)[:40]
                for item in value[:3]
            ]
            summary[key] = {"_count": len(value), "_sample_ids": ids}
        elif isinstance(value, dict) and len(json.dumps(value, default=str)) > 512:
            summary[key] = {"_truncated": True, "_keys": list(value.keys())[:10]}
        else:
            summary[key] = value
    return summary


class MessageLogger:
    """Zero-intrusion message persistence recorder.

    Designed to be injected by the runner into MsgHub and BaseAgent
    without those classes depending on the storage layer.
    """

    def __init__(self, structured_store: Optional[Any] = None) -> None:
        self._store = structured_store
        self._cycle_count: int = 0

    def set_cycle(self, n: int) -> None:
        """Update the current cycle counter."""
        self._cycle_count = n

    async def log_send(self, sender: str, receiver: str, msg: Any) -> None:
        """Record a point-to-point message (fan-out from __call__)."""
        if not self._store:
            return
        try:
            meta = getattr(msg, "metadata", {}) or {}
            stored_meta = _summarise_metadata(meta)
            await self._store.record_agent_message({
                "message_id": getattr(msg, "id", ""),
                "cycle_count": self._cycle_count,
                "sender": sender,
                "receiver": receiver,
                "direction": "send",
                "message_type": meta.get("type", ""),
                "content": (getattr(msg, "content", "") or "")[:500],
                "metadata": stored_meta,
            })
        except Exception as e:
            logger.debug("MessageLogger.log_send failed: %s", e)

    async def log_broadcast(self, sender: str, msg: Any) -> None:
        """Record a broadcast message (one row, receiver=NULL)."""
        if not self._store:
            return
        try:
            meta = getattr(msg, "metadata", {}) or {}
            stored_meta = _summarise_metadata(meta)
            await self._store.record_agent_message({
                "message_id": getattr(msg, "id", ""),
                "cycle_count": self._cycle_count,
                "sender": sender,
                "receiver": None,
                "direction": "broadcast",
                "message_type": meta.get("type", ""),
                "content": (getattr(msg, "content", "") or "")[:500],
                "metadata": stored_meta,
            })
        except Exception as e:
            logger.debug("MessageLogger.log_broadcast failed: %s", e)

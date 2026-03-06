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


def _extract_envelope(msg: Any) -> dict:
    """Extract envelope fields from Message-like objects when available."""
    return {
        "topic": getattr(msg, "topic", "") or "",
        "target": getattr(msg, "target", None),
        "trace_id": getattr(msg, "trace_id", "") or "",
        "causation_id": getattr(msg, "causation_id", None),
        "ttl_seconds": getattr(msg, "ttl_seconds", None),
        "hops": getattr(msg, "hops", None),
        "max_hops": getattr(msg, "max_hops", None),
    }


def _merge_metadata_with_envelope(meta: dict, msg: Any) -> dict:
    """Merge user metadata with envelope fields for observability."""
    merged = dict(meta or {})
    envelope = _extract_envelope(msg)
    for key, value in envelope.items():
        if value is None or value == "":
            continue
        if key not in merged:
            merged[key] = value
    return merged


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
            stored_meta = _summarise_metadata(_merge_metadata_with_envelope(meta, msg))
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
            stored_meta = _summarise_metadata(_merge_metadata_with_envelope(meta, msg))
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

    async def log_decision(self, agent_name: str, input_msg: Any, decision: dict) -> None:
        """Record an agent's handling decision for observability."""
        if not self._store:
            return
        try:
            input_meta = getattr(input_msg, "metadata", {}) or {}
            merged_meta = _merge_metadata_with_envelope(input_meta, input_msg)
            await self._store.record_agent_message({
                "message_id": getattr(input_msg, "id", ""),
                "cycle_count": self._cycle_count,
                "sender": agent_name,
                "receiver": None,
                "direction": "decision",
                "message_type": "agent_decision",
                "content": decision.get("reason", "")[:500],
                "metadata": _summarise_metadata({
                    "decision": decision,
                    "input_type": input_meta.get("type", ""),
                    "topic": merged_meta.get("topic", ""),
                    "trace_id": merged_meta.get("trace_id", ""),
                    "hops": merged_meta.get("hops"),
                    "max_hops": merged_meta.get("max_hops"),
                }),
            })
        except Exception as e:
            logger.debug("MessageLogger.log_decision failed: %s", e)

    async def log_hub_drop(self, msg: Any, reason: str, route: str = "") -> None:
        """Record a MsgHub guard/drop event for replay diagnostics."""
        if not self._store:
            return
        try:
            base_meta = getattr(msg, "metadata", {}) or {}
            merged_meta = _merge_metadata_with_envelope(base_meta, msg)
            merged_meta["drop_reason"] = reason
            if route:
                merged_meta["route"] = route
            await self._store.record_agent_message({
                "message_id": getattr(msg, "id", ""),
                "cycle_count": self._cycle_count,
                "sender": "msghub",
                "receiver": None,
                "direction": "hub_drop",
                "message_type": f"hub_drop.{reason}",
                "content": route[:500],
                "metadata": _summarise_metadata(merged_meta),
            })
        except Exception as e:
            logger.debug("MessageLogger.log_hub_drop failed: %s", e)

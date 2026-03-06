# -*- coding: utf-8 -*-
"""Message — core data class for agent communication."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Message:
    """Immutable message exchanged between agents.

    Attributes:
        name: Sender agent name.
        role: One of "user", "assistant", "system".
        content: Free-form text payload.
        metadata: Structured data (analysis results, proposals, etc.).
        id: Auto-generated UUID4 string.
        timestamp: Auto-generated creation time.
    """

    name: str
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.now)
    # Envelope fields (v2, all optional/backward-compatible)
    topic: str = ""
    target: Optional[str] = None
    trace_id: str = ""
    causation_id: Optional[str] = None
    ttl_seconds: Optional[int] = None
    hops: int = 0
    max_hops: int = 8

    def __post_init__(self) -> None:
        # Legacy compatibility: if no trace id is provided, bind it to message id.
        if not self.trace_id:
            self.trace_id = self.id
        if self.hops < 0:
            self.hops = 0
        if self.max_hops < 0:
            self.max_hops = 0
        if self.ttl_seconds is not None and self.ttl_seconds < 0:
            self.ttl_seconds = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe except datetime)."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "topic": self.topic,
            "target": self.target,
            "trace_id": self.trace_id,
            "causation_id": self.causation_id,
            "ttl_seconds": self.ttl_seconds,
            "hops": self.hops,
            "max_hops": self.max_hops,
        }

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Return True when TTL is set and the message has expired."""
        if self.ttl_seconds is None:
            return False
        if self.timestamp.tzinfo is not None:
            current = now or datetime.now(tz=self.timestamp.tzinfo)
        else:
            current = now or datetime.now()
        age_s = (current - self.timestamp).total_seconds()
        return age_s > self.ttl_seconds

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize from dict."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now()
        return cls(
            name=data.get("name", ""),
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            id=data.get("id", uuid.uuid4().hex),
            timestamp=ts,
            topic=data.get("topic", ""),
            target=data.get("target"),
            trace_id=data.get("trace_id", data.get("id", "")),
            causation_id=data.get("causation_id"),
            ttl_seconds=data.get("ttl_seconds"),
            hops=data.get("hops", 0),
            max_hops=data.get("max_hops", 8),
        )

# -*- coding: utf-8 -*-
"""Message — core data class for agent communication."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe except datetime)."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

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
        )

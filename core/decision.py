# -*- coding: utf-8 -*-
"""Agent decision protocol models."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentDecision:
    """Unified decision result for agent message handling."""

    memory_action: str = "cache"  # drop | cache
    output_action: str = "reply"  # none | reply | forward
    reason: str = ""
    confidence: float = 1.0
    forward_target: Optional[str] = None
    forward_topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "AgentDecision":
        memory_action = self.memory_action if self.memory_action in ("drop", "cache") else "cache"
        output_action = self.output_action if self.output_action in ("none", "reply", "forward") else "reply"
        confidence = min(max(float(self.confidence), 0.0), 1.0)
        return AgentDecision(
            memory_action=memory_action,
            output_action=output_action,
            reason=self.reason,
            confidence=confidence,
            forward_target=self.forward_target,
            forward_topic=self.forward_topic,
            metadata=dict(self.metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_action": self.memory_action,
            "output_action": self.output_action,
            "reason": self.reason,
            "confidence": self.confidence,
            "forward_target": self.forward_target,
            "forward_topic": self.forward_topic,
            "metadata": self.metadata,
        }

    @classmethod
    def from_any(cls, raw: Any) -> "AgentDecision":
        if isinstance(raw, AgentDecision):
            return raw.normalized()
        if isinstance(raw, dict):
            return cls(
                memory_action=raw.get("memory_action", "cache"),
                output_action=raw.get("output_action", "reply"),
                reason=raw.get("reason", ""),
                confidence=raw.get("confidence", 1.0),
                forward_target=raw.get("forward_target"),
                forward_topic=raw.get("forward_topic"),
                metadata=raw.get("metadata", {}),
            ).normalized()
        return cls().normalized()

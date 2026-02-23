# -*- coding: utf-8 -*-
"""Proposal — data class for agent bidding / action proposals."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Proposal:
    """An action proposal submitted by an Action Agent to the Coordinator.

    Attributes:
        agent_name: Proposing agent.
        action: One of "comment", "post", "upvote", "follow".
        target_id: Target post/user ID (empty for "post" actions).
        priority: 0-1 priority score.
        reasoning: Why this action is proposed.
        strategy: "A" or "B" (which A/B variant produced this).
        metadata: Extra data (post content, generated text, etc.).
    """

    agent_name: str
    action: str
    target_id: str = ""
    priority: float = 0.5
    reasoning: str = ""
    strategy: str = "A"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "action": self.action,
            "target_id": self.target_id,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proposal":
        return cls(
            agent_name=data.get("agent_name", ""),
            action=data.get("action", ""),
            target_id=data.get("target_id", ""),
            priority=data.get("priority", 0.5),
            reasoning=data.get("reasoning", ""),
            strategy=data.get("strategy", "A"),
            metadata=data.get("metadata", {}),
        )

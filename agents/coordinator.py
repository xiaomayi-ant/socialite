# -*- coding: utf-8 -*-
"""CoordinatorAgent — proposal arbitration (pure rules v1).

Collects Proposals from all action agents → sorts by priority →
checks API rate limits + budget → approves/rejects.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.message import Message
from core.proposal import Proposal

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """Arbitrates proposals from action agents.

    Pure rule-based (no LLM). Enforces:
    - Budget limits
    - API rate limits (max comments/upvotes per cycle)
    - Priority ordering
    - Deduplication
    """

    def __init__(
        self,
        name: str = "coordinator",
        structured_store=None,
        daily_budget_usd: float = 5.0,
        max_comments_per_cycle: int = 3,
        max_upvotes_per_cycle: int = 5,
        max_posts_per_cycle: int = 1,
        max_follows_per_cycle: int = 2,
    ) -> None:
        super().__init__(name)
        self.structured_store = structured_store
        self.daily_budget_usd = daily_budget_usd
        self.max_comments = max_comments_per_cycle
        self.max_upvotes = max_upvotes_per_cycle
        self.max_posts = max_posts_per_cycle
        self.max_follows = max_follows_per_cycle
        self._cycle_count: int = 0

    async def observe(self, msg: Message) -> None:
        if msg.metadata.get("type") == "proposals":
            proposals_data = msg.metadata.get("proposals", [])
            proposals = [
                Proposal.from_dict(p) for p in proposals_data
            ]
            self._cycle_count += 1
            result = await self.arbitrate(
                proposals, self._cycle_count
            )
            # Broadcast approved proposals
            if self._hub:
                await self._hub.broadcast(Message(
                    name=self.name,
                    role="assistant",
                    content="arbitration_complete",
                    metadata={
                        "type": "proposal_approved",
                        "approved": result.get("approved", []),
                        "rejected": result.get(
                            "rejected", []
                        ),
                        "budget_remaining": result.get(
                            "budget_remaining", 0
                        ),
                    },
                    causation_id=msg.id,
                ))
            return
        await super().observe(msg)

    async def arbitrate(
        self, proposals: List[Proposal], cycle_count: int = 0
    ) -> Dict[str, Any]:
        """Sort proposals by priority, apply limits, return approved/rejected."""
        # Check budget
        remaining_budget = await self._get_remaining_budget()
        if remaining_budget <= 0.10:
            logger.warning("Budget exhausted ($%.2f remaining), rejecting all", remaining_budget)
            return {
                "approved": [],
                "rejected": [p.to_dict() for p in proposals],
                "budget_remaining": remaining_budget,
            }

        # Sort by priority (descending)
        sorted_proposals = sorted(proposals, key=lambda p: p.priority, reverse=True)

        # Apply per-action limits
        action_counts = {"comment": 0, "upvote": 0, "post": 0, "follow": 0}
        action_limits = {
            "comment": self.max_comments,
            "upvote": self.max_upvotes,
            "post": self.max_posts,
            "follow": self.max_follows,
        }

        approved: List[Proposal] = []
        rejected: List[Proposal] = []
        seen_targets: set = set()

        for proposal in sorted_proposals:
            action = proposal.action
            target = proposal.target_id

            # Dedup: same target+action
            dedup_key = f"{action}:{target}"
            if dedup_key in seen_targets:
                rejected.append(proposal)
                continue
            seen_targets.add(dedup_key)

            # Check limit
            limit = action_limits.get(action, 1)
            if action_counts.get(action, 0) >= limit:
                rejected.append(proposal)
                continue

            action_counts[action] = action_counts.get(action, 0) + 1
            approved.append(proposal)

        # Record proposals if structured_store available
        if self.structured_store:
            for p in approved:
                await self._record_proposal(p, "approved", cycle_count)
            for p in rejected:
                await self._record_proposal(p, "rejected", cycle_count)

        logger.info(
            "Arbitration: %d approved, %d rejected (budget: $%.2f)",
            len(approved), len(rejected), remaining_budget,
        )
        for a in approved:
            logger.info(
                "  ✓ %s %s (p=%.2f, strategy=%s) by %s",
                a.action, a.target_id[:12] if a.target_id else "new",
                a.priority, a.strategy, a.agent_name,
            )

        return {
            "approved": [p.to_dict() for p in approved],
            "rejected": [p.to_dict() for p in rejected],
            "budget_remaining": remaining_budget,
            "action_counts": action_counts,
        }

    async def _get_remaining_budget(self) -> float:
        if not self.structured_store:
            return self.daily_budget_usd
        try:
            spent = await self.structured_store.get_daily_cost()
            return max(self.daily_budget_usd - spent, 0.0)
        except Exception:
            return self.daily_budget_usd

    async def _record_proposal(
        self, proposal: Proposal, status: str, cycle_count: int
    ) -> None:
        try:
            await self.structured_store.record_agent_proposal({
                "cycle_count": cycle_count,
                "agent_name": proposal.agent_name,
                "action": proposal.action,
                "target_id": proposal.target_id,
                "priority": proposal.priority,
                "reasoning": proposal.reasoning,
                "strategy": proposal.strategy,
                "status": status,
                "metadata": proposal.metadata,
            })
        except Exception as e:
            logger.debug("Record proposal failed: %s", e)

    # ── BaseAgent interface ──────────────────────────────────

    async def reply(self, msg: Message) -> Message:
        """Handle arbitration request via message."""
        proposals_data = msg.metadata.get("proposals", [])
        proposals = [Proposal.from_dict(p) for p in proposals_data]
        cycle = msg.metadata.get("cycle_count", 0)
        result = await self.arbitrate(proposals, cycle_count=cycle)
        return Message(
            name=self.name,
            role="assistant",
            content="arbitration_complete",
            metadata=result,
        )

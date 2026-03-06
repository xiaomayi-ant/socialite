# -*- coding: utf-8 -*-
"""FollowAgent — follow/unfollow action agent.

Analyses social graph to decide which agents to follow.
Uses knowledge graph (Neo4j) for PageRank-based influencer detection.
"""

import logging
from typing import Any, Dict, List, Optional

from core.ab_strategy import ABSelector
from core.base_agent import BaseAgent
from core.message import Message
from core.proposal import Proposal

logger = logging.getLogger(__name__)


class FollowAgent(BaseAgent):
    """Generates follow/unfollow Proposals based on social graph analysis."""

    MAX_FOLLOWS_PER_DAY = 10

    def __init__(
        self,
        name: str = "follow",
        social_memory=None,
        ab_selector: Optional[ABSelector] = None,
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.ab_selector = ab_selector or ABSelector(mode="alternate")
        self._analysis_data: Dict[str, Any] = {}
        self._cycle_count: int = 0
        self._followed_ids: set = set()

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        msg_type = msg.metadata.get("type")

        if msg_type == "action_completed":
            if (
                msg.metadata.get("action") == "follow"
                and msg.metadata.get("success")
            ):
                target = msg.metadata.get("target_id", "")
                if target:
                    self._followed_ids.add(target)
            return

        if msg_type == "analysis_result":
            self._analysis_data = msg.metadata
            self._cycle_count += 1
            if len(self._followed_ids) >= self.MAX_FOLLOWS_PER_DAY:
                logger.info(
                    "FollowAgent: daily limit reached (%d/%d), skipping",
                    len(self._followed_ids), self.MAX_FOLLOWS_PER_DAY,
                )
                return
            posts = msg.metadata.get("posts", [])
            if posts:
                proposals = await self.propose(
                    posts, self._cycle_count,
                    already_following=self._followed_ids,
                )
                if proposals and self._hub:
                    await self._hub.send_to(
                        "coordinator",
                        Message(
                            name=self.name,
                            role="assistant",
                            content="follow_proposals",
                            metadata={
                                "type": "proposals",
                                "proposals": [
                                    p.to_dict() for p in proposals
                                ],
                            },
                            causation_id=msg.id,
                        ),
                    )
                    logger.info(
                        "FollowAgent sent %d proposals "
                        "to coordinator",
                        len(proposals),
                    )

        elif msg_type == "proposal_approved":
            approved = msg.metadata.get("approved", [])
            for p in approved:
                if (
                    p.get("agent_name") == self.name
                    and p.get("action") == "follow"
                ):
                    if self._hub:
                        await self._hub.send_to(
                            "sensor",
                            Message(
                                name=self.name,
                                role="assistant",
                                content="exec_command",
                                metadata={
                                    "type": "exec_command",
                                    "action": "follow",
                                    "target_id": p.get(
                                        "target_id", ""
                                    ),
                                    "strategy": p.get(
                                        "strategy", ""
                                    ),
                                },
                                causation_id=msg.id,
                            ),
                        )
                        logger.info(
                            "FollowAgent exec follow %s",
                            p.get("target_id", ""),
                        )

    async def propose(
        self,
        posts: List[Dict[str, Any]],
        cycle_count: int = 0,
        already_following: Optional[set] = None,
    ) -> List[Proposal]:
        """Generate follow proposals based on interaction patterns."""
        strategy = self.ab_selector.select(cycle_count)
        following = already_following or set()
        proposals = []

        if strategy == "A":
            proposals = self._propose_engagement_based(posts, following, strategy)
        else:
            proposals = await self._propose_graph_based(posts, following, strategy)

        return proposals

    def _propose_engagement_based(
        self, posts: List[Dict[str, Any]], following: set, strategy: str
    ) -> List[Proposal]:
        """Strategy A: Follow authors of high-engagement posts."""
        author_engagement: Dict[str, float] = {}
        for p in posts:
            author = p.get("author_name", "unknown")
            if author in following or author == "SocialLearnerBot":
                continue
            score = p.get("upvotes", 0) * 0.6 + p.get("comment_count", 0) * 0.4
            author_engagement[author] = max(author_engagement.get(author, 0), score)

        proposals = []
        sorted_authors = sorted(author_engagement.items(), key=lambda x: -x[1])
        for author, score in sorted_authors[:3]:
            if score < 5:
                continue
            priority = min(score / 50, 0.8)
            proposals.append(Proposal(
                agent_name=self.name,
                action="follow",
                target_id=author,
                priority=round(priority, 3),
                reasoning=f"engagement-based: score={score:.1f}",
                strategy=strategy,
            ))
        return proposals

    async def _propose_graph_based(
        self, posts: List[Dict[str, Any]], following: set, strategy: str
    ) -> List[Proposal]:
        """Strategy B: Use knowledge graph PageRank to find influencers."""
        if not self.social_memory or not hasattr(self.social_memory, 'graph_store'):
            return self._propose_engagement_based(posts, following, strategy)

        try:
            graph_store = self.social_memory.graph_store
            if not graph_store:
                return self._propose_engagement_based(posts, following, strategy)

            pagerank = await graph_store.compute_pagerank()
            # compute_pagerank returns List[Dict] with keys:
            #   user_id, username, influence_score
            if not pagerank:
                return self._propose_engagement_based(posts, following, strategy)

            sorted_pr = sorted(
                pagerank,
                key=lambda r: r.get("influence_score", 0),
                reverse=True,
            )

            proposals = []
            for record in sorted_pr[:5]:
                name = record.get("username") or record.get("user_id", "")
                score = record.get("influence_score", 0)
                if not name or name in following or name == "SocialLearnerBot":
                    continue
                if score <= 0:
                    continue
                proposals.append(Proposal(
                    agent_name=self.name,
                    action="follow",
                    target_id=name,
                    priority=round(min(score * 10, 0.9), 3),
                    reasoning=f"graph-pagerank: score={score:.4f}",
                    strategy=strategy,
                ))
                if len(proposals) >= 3:
                    break
            # If graph has no meaningful data, fall back to engagement
            if not proposals:
                return self._propose_engagement_based(posts, following, strategy)
            return proposals
        except Exception as e:
            logger.warning("Graph-based follow failed: %s", e)
            return self._propose_engagement_based(posts, following, strategy)

    async def reply(self, msg: Message) -> Message:
        return Message(
            name=self.name, role="assistant",
            content="FollowAgent uses propose() interface",
        )

# -*- coding: utf-8 -*-
"""SensorAgent — intelligent feed collection + Moltbook API I/O.

Multi-strategy collection: hot feed, submolt feed, semantic search.
Accepts LearnerAgent collection suggestions via observe().
"""

import json
import logging
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.message import Message

logger = logging.getLogger(__name__)


class SensorAgent(BaseAgent):
    """Pure I/O gateway — no LLM attached.

    Wraps MoltbookClient for feed fetching, upvoting, commenting, posting.
    Listens for collection_suggestion messages from LearnerAgent.
    """

    def __init__(self, name: str = "sensor", client=None) -> None:
        super().__init__(name)
        self.client = client
        self._collection_hints: List[str] = []  # submolt/topic suggestions from Learner

    async def observe(self, msg: Message) -> None:
        """Accept collection hints from LearnerAgent."""
        await super().observe(msg)
        if msg.metadata.get("type") == "collection_suggestion":
            hints = msg.metadata.get("suggested_submolts", [])
            self._collection_hints = hints[:5]
            logger.debug("Received collection hints: %s", self._collection_hints)

    # ── Feed collection ──────────────────────────────────────

    def fetch_feed(self, sort: str = "hot", limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch feed posts from Moltbook."""
        from moltbook.models import MoltbookPost
        feed: List[MoltbookPost] = []
        for s in (sort, "new"):
            try:
                feed = self.client.get_feed(sort=s, limit=limit)
                if feed:
                    break
            except Exception as e:
                logger.warning("Feed (%s) failed: %s", s, e)

        return [
            {
                "id": p.id,
                "title": p.title,
                "content": p.content or "",
                "submolt": p.submolt_name or p.submolt or "general",
                "author_name": p.author.name if p.author else "unknown",
                "upvotes": p.upvotes,
                "downvotes": p.downvotes,
                "comment_count": p.comment_count,
                "user_vote": p.user_vote,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in feed
        ]

    # ── API actions ──────────────────────────────────────────

    def upvote(self, post_id: str) -> bool:
        try:
            return self.client.upvote_post(post_id)
        except Exception as e:
            logger.warning("Upvote failed %s: %s", post_id, e)
            return False

    def create_comment(self, post_id: str, content: str) -> Optional[Dict[str, Any]]:
        try:
            comment = self.client.create_comment(post_id, content)
            return {"id": comment.id, "post_id": post_id, "content": content}
        except Exception as e:
            logger.warning("Comment failed %s: %s", post_id, e)
            return None

    def create_post(self, submolt: str, title: str, content: str) -> Optional[Dict[str, Any]]:
        try:
            post = self.client.create_post(submolt=submolt, title=title, content=content)
            return {
                "id": post.id, "title": post.title,
                "content": post.content, "submolt": submolt,
            }
        except Exception as e:
            logger.warning("Post creation failed: %s", e)
            return None

    def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        try:
            post = self.client.get_post(post_id)
            if post:
                return {
                    "id": post.id, "title": post.title,
                    "upvotes": post.upvotes, "downvotes": post.downvotes,
                    "comment_count": post.comment_count,
                }
        except Exception as e:
            logger.warning("Get post failed %s: %s", post_id, e)
        return None

    def get_agent_profile(self, agent_name: str):
        try:
            return self.client.get_agent_profile(agent_name)
        except Exception as e:
            logger.warning("Get profile failed %s: %s", agent_name, e)
            return None

    def get_post_comments(self, post_id: str, sort: str = "top") -> List[Dict[str, Any]]:
        try:
            comments = self.client.get_comments(post_id, sort=sort)
            return [
                {
                    "id": c.id,
                    "post_id": c.post_id or post_id,
                    "content": c.content,
                    "author_name": c.author.name if c.author else "unknown",
                    "parent_id": c.parent_id,
                    "upvotes": c.upvotes,
                    "downvotes": c.downvotes,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in comments
            ]
        except Exception as e:
            logger.warning("Get comments failed %s: %s", post_id, e)
            return []

    def check_comment_feedback(
        self, post_id: str, our_comment_ids: List[str]
    ) -> List[Dict[str, Any]]:
        try:
            all_comments = self.client.get_comments(post_id)
            our_ids_set = set(our_comment_ids)
            return [
                {"comment_id": c.id, "upvotes": c.upvotes, "downvotes": c.downvotes}
                for c in all_comments if c.id in our_ids_set
            ]
        except Exception as e:
            logger.debug("Check comment feedback failed %s: %s", post_id, e)
            return []

    # ── BaseAgent interface ──────────────────────────────────

    async def reply(self, msg: Message) -> Message:
        """Handle collect requests → return raw_feed message."""
        action = msg.metadata.get("action", "fetch_feed")
        if action == "fetch_feed":
            sort = msg.metadata.get("sort", "hot")
            limit = msg.metadata.get("limit", 25)
            posts = self.fetch_feed(sort=sort, limit=limit)
            return Message(
                name=self.name,
                role="assistant",
                content=json.dumps({"type": "raw_feed", "count": len(posts)}),
                metadata={"type": "raw_feed", "posts": posts, "count": len(posts)},
            )
        elif action == "check_performance":
            post_ids = msg.metadata.get("post_ids", [])
            results = [d for pid in post_ids if (d := self.get_post(pid))]
            return Message(
                name=self.name,
                role="assistant",
                content=json.dumps({"type": "performance_feedback", "posts": results}),
                metadata={"type": "performance_feedback", "posts": results},
            )
        return Message(
            name=self.name, role="assistant",
            content=json.dumps({"type": "error", "message": f"Unknown: {action}"}),
        )

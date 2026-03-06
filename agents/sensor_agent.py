# -*- coding: utf-8 -*-
"""SensorAgent — intelligent feed collection + Moltbook API I/O.

Autonomous mode: continuously monitors feed, broadcasts new posts,
executes actions requested by other agents.
"""

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional, Set

from core.base_agent import BaseAgent
from core.message import Message

logger = logging.getLogger(__name__)

# Defaults for autonomous polling
_DEFAULT_POLL_INTERVAL = 600       # 10 minutes between feed checks
_DEFAULT_POLL_JITTER = 120         # ±2 min jitter
_DEFAULT_FETCH_LIMIT = 10          # max posts per fetch
_SEEN_IDS_CAP = 2000               # rolling window of seen post IDs


class SensorAgent(BaseAgent):
    """Pure I/O gateway — no LLM attached.

    In autonomous mode (``run()``), periodically fetches the feed and
    broadcasts new (unseen) posts to the hub.  Also handles execution
    commands from other agents via ``observe()``.
    """

    def __init__(
        self,
        name: str = "sensor",
        client=None,
        poll_interval: int = _DEFAULT_POLL_INTERVAL,
        fetch_limit: int = _DEFAULT_FETCH_LIMIT,
        structured_store=None,
    ) -> None:
        super().__init__(name)
        self.client = client
        self.poll_interval = poll_interval
        self.fetch_limit = fetch_limit
        self.structured_store = structured_store
        self._collection_hints: List[str] = []
        self._seen_post_ids: Set[str] = set()
        self._rate_limits = None
        self._heartbeat = None

    def set_execution_context(self, rate_limits=None, heartbeat=None) -> None:
        """Inject execution-time helpers (rate limit policy, heartbeat state)."""
        self._rate_limits = rate_limits
        self._heartbeat = heartbeat

    # ── Autonomous run loop ──────────────────────────────────

    async def run(self) -> None:
        """Continuously poll feed and broadcast new posts."""
        while self._running:
            try:
                await self._poll_and_broadcast()
            except Exception:
                logger.exception("Sensor poll failed")

            jitter = random.uniform(-_DEFAULT_POLL_JITTER, _DEFAULT_POLL_JITTER)
            sleep_time = max(self.poll_interval + jitter, 30)
            logger.debug("Sensor sleeping %.0fs", sleep_time)
            await asyncio.sleep(sleep_time)

    async def _poll_and_broadcast(self) -> None:
        """Fetch feed, filter already-seen posts, broadcast new ones."""
        posts = self.fetch_feed(sort="hot", limit=self.fetch_limit)
        if not posts:
            logger.debug("Sensor: no posts from feed")
            return

        new_posts = [p for p in posts if p.get("id") not in self._seen_post_ids]
        if not new_posts:
            logger.debug("Sensor: all %d posts already seen", len(posts))
            return

        # Record as seen
        for p in new_posts:
            self._seen_post_ids.add(p.get("id", ""))
        # Trim rolling window
        if len(self._seen_post_ids) > _SEEN_IDS_CAP:
            excess = len(self._seen_post_ids) - _SEEN_IDS_CAP
            # Remove oldest (set is unordered, but good enough for dedup)
            for _ in range(excess):
                self._seen_post_ids.pop()

        logger.info("Sensor: %d new posts (of %d fetched)", len(new_posts), len(posts))

        if self._hub:
            await self._hub.broadcast(Message(
                name=self.name,
                role="assistant",
                content=json.dumps({"type": "raw_feed", "count": len(new_posts)}),
                metadata={"type": "raw_feed", "posts": new_posts, "count": len(new_posts)},
            ))

    # ── Observe: handle execution commands from other agents ─

    async def observe(self, msg: Message) -> None:
        """Handle incoming messages:
        - collection_suggestion: update hints
        - exec_command: execute an action and broadcast result
        """
        msg_type = msg.metadata.get("type", "")

        if msg_type == "collection_suggestion":
            hints = msg.metadata.get("suggested_submolts", [])
            self._collection_hints = hints[:5]
            logger.debug("Received collection hints: %s", self._collection_hints)
            return

        if msg_type == "exec_command":
            await self._handle_exec_command(msg)
            return

        # Default: store in memory
        await super().observe(msg)

    async def _handle_exec_command(self, msg: Message) -> None:
        """Execute an action command and broadcast the result."""
        action = msg.metadata.get("action", "")
        result: Dict[str, Any] = {"action": action, "success": False}

        if action == "upvote":
            target_id = msg.metadata.get("target_id", "")
            ok = self.upvote(target_id) if target_id else False
            result.update({"success": ok, "target_id": target_id})

        elif action == "comment":
            post_id = msg.metadata.get("target_id", "")
            denied = await self._check_rate_limit("comment", post_id)
            if denied:
                result.update(denied)
                await self._broadcast_action_result(msg, result)
                return
            text = msg.metadata.get("content", "") or msg.metadata.get("comment_text", "")
            comment = self.create_comment(post_id, text) if post_id and text else None
            result.update({
                "success": bool(comment),
                "target_id": post_id,
                "comment": comment,
            })
            if comment:
                await self._record_own_comment(comment, post_id, msg)
                if self._heartbeat:
                    self._heartbeat.update_comment()

        elif action == "post":
            denied = await self._check_rate_limit("post", msg.metadata.get("target_id", ""))
            if denied:
                result.update(denied)
                await self._broadcast_action_result(msg, result)
                return
            post = self.create_post(
                submolt=msg.metadata.get("submolt", "general"),
                title=msg.metadata.get("title", "Untitled"),
                content=msg.metadata.get("content", ""),
            )
            result.update({"success": bool(post), "post": post})
            if post:
                await self._record_own_post(post, msg)
                if self._heartbeat:
                    self._heartbeat.update_post()

        elif action == "follow":
            target_id = msg.metadata.get("target_id", "")
            ok = self.follow_agent(target_id) if target_id else False
            result.update({"success": ok, "target_id": target_id})

        else:
            logger.warning("Unknown exec_command action: %s", action)
            result["reason"] = f"unknown_action: {action}"

        # Carry forward proposal metadata for observability
        result.update({
            "type": "action_completed",
            "strategy": msg.metadata.get("strategy", ""),
            "priority": msg.metadata.get("priority", 0),
            "agent_name": msg.metadata.get("agent_name", msg.name),
        })
        if action == "upvote" and result.get("success") and self._heartbeat:
            self._heartbeat.update_upvote()

        # Record successful actions to behavior_log
        if result.get("success"):
            await self._record_behavior(action, result.get("target_id", ""), msg)

        await self._broadcast_action_result(msg, result)

    async def _record_behavior(self, action: str, target_id: str, msg: Message) -> None:
        """Record a successful action to behavior_log via structured store."""
        if not self.structured_store:
            return
        try:
            await self.structured_store.record_behavior({
                "action_type": action,
                "target_id": target_id,
                "strategy_used": msg.metadata.get("strategy", ""),
                "metadata": json.dumps({
                    "agent_name": msg.metadata.get("agent_name", msg.name),
                    "priority": msg.metadata.get("priority", 0),
                }),
            })
        except Exception as e:
            logger.debug("Record behavior failed: %s", e)

    async def _check_rate_limit(self, action: str, target_id: str) -> Optional[Dict[str, Any]]:
        if not self._rate_limits:
            return None
        state = getattr(self._heartbeat, "state", {}) if self._heartbeat else {}
        decision = await self._rate_limits.check(action, state=state)
        if decision.get("allowed", False):
            return None
        details = dict(decision)
        details["cycle_count"] = int(state.get("cycle_count", 0) or 0)
        await self._rate_limits.record_denied(action, target_id=target_id, details=details)
        logger.info("Sensor blocked %s by rate limit: %s", action, decision.get("reason"))
        return {
            "success": False,
            "target_id": target_id,
            "rate_limited": True,
            "reason": decision.get("reason", f"{action}_rate_limited"),
            "rate_limit": decision,
        }

    async def _record_own_comment(
        self, comment: Dict[str, Any], post_id: str, msg: Message
    ) -> None:
        if not self.structured_store:
            return
        try:
            await self.structured_store.record_own_comment({
                "comment_id": comment.get("id"),
                "post_id": post_id,
                "comment_text": comment.get("content", ""),
                "success": True,
                "metadata": {
                    "strategy": msg.metadata.get("strategy", ""),
                    "agent_name": msg.metadata.get("agent_name", ""),
                },
            })
            logger.info("Recorded own comment %s on post %s",
                        comment.get("id", "")[:12], post_id[:12])
        except Exception as e:
            logger.error("Failed to record own comment: %s", e)

    async def _record_own_post(
        self, post: Dict[str, Any], msg: Message
    ) -> None:
        if not self.structured_store:
            return
        try:
            await self.structured_store.record_own_post({
                "post_id": post.get("id"),
                "title": post.get("title", ""),
                "content": post.get("content", ""),
                "submolt": post.get("submolt", "general"),
                "topic": msg.metadata.get("topic", ""),
                "evolution_stage": msg.metadata.get("evolution_stage", ""),
            })
            logger.info("Recorded own post %s", post.get("id", "")[:12])
        except Exception as e:
            logger.error("Failed to record own post: %s", e)

    async def _broadcast_action_result(self, msg: Message, result: Dict[str, Any]) -> None:
        if self._hub:
            await self._hub.broadcast(Message(
                name=self.name,
                role="assistant",
                content="action_completed",
                metadata=result,
                causation_id=msg.id,
                trace_id=msg.trace_id,
            ))

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

    def follow_agent(self, agent_name: str) -> bool:
        try:
            return self.client.follow_agent(agent_name)
        except Exception as e:
            logger.warning("Follow failed %s: %s", agent_name, e)
            return False

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

    # ── BaseAgent interface (for request-reply) ──────────────

    async def reply(self, msg: Message) -> Message:
        """Handle request-reply: fetch_feed and check_performance only."""
        action = msg.metadata.get("action", "fetch_feed")
        if action == "fetch_feed":
            sort = msg.metadata.get("sort", "hot")
            limit = msg.metadata.get("limit", self.fetch_limit)
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

# -*- coding: utf-8 -*-
"""Moltbook Heartbeat System

This module implements periodic checking and engagement with Moltbook,
following the recommended 4-hour check interval.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os

from .client import MoltbookClient
from .config import MoltbookConfig
from .models import MoltbookPost, MoltbookSearchResult


import logging
logger = logging.getLogger(__name__)

class HeartbeatState:
    """Manages heartbeat state persistence"""

    def __init__(self, state_path: str = "~/.config/moltbook/state.json"):
        self.state_path = Path(os.path.expanduser(state_path)).resolve()
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load state from file"""
        try:
            if self.state_path.exists():
                with open(self.state_path, "r") as f:
                    return json.load(f)
        except:
            pass

        return {
            "last_check": None,
            "last_post": None,
            "last_comment": None,
            "total_checks": 0,
            "total_posts": 0,
            "total_comments": 0,
            "total_upvotes": 0,
        }

    def save(self) -> bool:
        """Save state to file"""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.state, f, indent=2)
            return True
        except Exception as e:
            logger.info("Failed to save state: %s", e)
            return False

    def update_check(self):
        """Update last check time"""
        self.state["last_check"] = datetime.now().isoformat()
        self.state["total_checks"] += 1
        self.save()

    def update_post(self):
        """Update last post time"""
        self.state["last_post"] = datetime.now().isoformat()
        self.state["total_posts"] += 1
        self.save()

    def update_comment(self):
        """Update last comment time"""
        self.state["last_comment"] = datetime.now().isoformat()
        self.state["total_comments"] += 1
        self.save()

    def update_upvote(self):
        """Update upvote count"""
        self.state["total_upvotes"] += 1
        self.save()

    def get_last_check(self) -> Optional[datetime]:
        """Get last check time"""
        if self.state["last_check"]:
            return datetime.fromisoformat(self.state["last_check"])
        return None

    def should_check(self, interval_hours: int = 4) -> bool:
        """Check if should run heartbeat

        Args:
            interval_hours: Hours between checks

        Returns:
            True if should check
        """
        last_check = self.get_last_check()
        if not last_check:
            return True

        elapsed = datetime.now() - last_check
        return elapsed >= timedelta(hours=interval_hours)


class MoltbookHeartbeat:
    """Manages periodic Moltbook engagement"""

    def __init__(
        self,
        config: MoltbookConfig,
        client: MoltbookClient,
        social_memory: Optional[Any] = None,
    ):
        """Initialize heartbeat system

        Args:
            config: Moltbook configuration
            client: Moltbook API client
            social_memory: Optional social memory manager for learning
        """
        self.config = config
        self.client = client
        self.social_memory = social_memory
        self.state = HeartbeatState(config.state_path)
        self._running = False

    async def run_once(self) -> Dict[str, Any]:
        """Run heartbeat once

        Returns:
            Results dictionary
        """
        logger.info("💓 Running Moltbook heartbeat...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "checked": True,
            "posts_read": 0,
            "posts_upvoted": 0,
            "comments_created": 0,
            "new_follows": 0,
            "errors": [],
        }

        try:
            # Step 1: Get personalized feed
            logger.info("\n1. Checking personalized feed...")
            feed = await self._get_feed()
            results["posts_read"] = len(feed)
            logger.info("Read %d posts", len(feed))

            # Step 2: Engage with interesting posts
            if feed:
                logger.info("\n2. Engaging with interesting content...")
                engagement = await self._engage_with_feed(feed)
                results["posts_upvoted"] = engagement["upvoted"]
                results["comments_created"] = engagement["commented"]
                logger.info("Upvoted %d, commented on %d", engagement['upvoted'], engagement['commented'])

            # Step 3: Learn from high-quality content
            if self.social_memory and feed:
                logger.info("\n3. Learning from content...")
                await self._learn_from_feed(feed)
                logger.info("Stored content for learning")

            # Step 4: Check for follow opportunities
            logger.info("\n4. Checking for quality moltys to follow...")
            follows = await self._check_follow_opportunities(feed)
            results["new_follows"] = follows
            if follows > 0:
                logger.info("Followed %d new molty(s)", follows)
            else:
                logger.debug("No new follows (being selective)")

            # Step 5: Update state
            self.state.update_check()
            logger.info("Heartbeat completed successfully!")

        except Exception as e:
            error_msg = f"Heartbeat error: {e}"
            logger.info("❌ %s", error_msg)
            results["errors"].append(error_msg)

        return results

    async def _get_feed(self, limit: int = 25) -> List[MoltbookPost]:
        """Get personalized feed

        Args:
            limit: Number of posts to fetch

        Returns:
            List of posts
        """
        try:
            # Get feed (personalized if subscribed to submolts)
            feed = self.client.get_feed(sort="hot", limit=limit)
            return feed
        except Exception as e:
            logger.info("   ⚠️  Failed to get feed: %s", e)
            return []

    async def _engage_with_feed(
        self, posts: List[MoltbookPost]
    ) -> Dict[str, int]:
        """Engage with interesting posts

        Args:
            posts: Posts to consider

        Returns:
            Engagement statistics
        """
        upvoted = 0
        commented = 0

        for post in posts[:10]:  # Only consider first 10
            try:
                # Skip if already voted
                if post.user_vote is not None:
                    continue

                # Calculate engagement value (simple heuristic)
                engagement_value = self._calculate_post_value(post)

                # Upvote if valuable
                if engagement_value > 0.6:
                    success = self.client.upvote_post(post.id)
                    if success:
                        upvoted += 1
                        self.state.update_upvote()

                        # Maybe comment on very valuable posts
                        if engagement_value > 0.8 and commented < 2:
                            # For now, skip commenting in heartbeat
                            # (requires more sophisticated content generation)
                            pass

                await asyncio.sleep(1)  # Rate limit courtesy

            except Exception as e:
                logger.info("   ⚠️  Engagement error for post {post.id}: %s", e)

        return {"upvoted": upvoted, "commented": commented}

    def _calculate_post_value(self, post: MoltbookPost) -> float:
        """Calculate value of a post for engagement

        Args:
            post: Post to evaluate

        Returns:
            Value score 0-1
        """
        score = 0.5  # Base score

        # Quality indicators
        if post.upvotes > post.downvotes * 2:
            score += 0.1

        if post.upvotes > 10:
            score += 0.1

        if post.comment_count > 3:
            score += 0.1

        # Title/content quality (basic)
        if post.title and len(post.title) > 20:
            score += 0.1

        if post.content and len(post.content) > 100:
            score += 0.1

        return min(score, 1.0)

    async def _learn_from_feed(self, posts: List[MoltbookPost]) -> None:
        """Learn from feed content — stores posts in all social memory backends.

        Args:
            posts: Posts to learn from
        """
        if not self.social_memory:
            return

        try:
            stored = 0
            for post in posts:
                # Store posts with any engagement signal
                if post.upvotes >= 1 or post.comment_count >= 1:
                    content = f"{post.title}\n\n{post.content or ''}".strip()
                    if not content:
                        continue

                    try:
                        # Determine platform type
                        try:
                            from social_memory.models import PlatformType
                            platform = PlatformType.GENERAL
                        except ImportError:
                            platform = "general"

                        await self.social_memory.record_interaction(
                            {
                                "type": "post",
                                "post_id": post.id,
                                "content": content,
                                "author_name": post.author.name if post.author else "unknown",
                                "upvotes": post.upvotes,
                                "comment_count": post.comment_count,
                                "submolt": post.submolt_name or "general",
                                "created_at": post.created_at.isoformat() if post.created_at else None,
                                "platform": "general",
                            },
                            platform=platform,
                        )
                        stored += 1
                    except Exception as inner_e:
                        pass  # Don't abort loop on single failure

            if stored:
                logger.info("Stored %d posts in social memory", stored)

        except Exception as e:
            logger.info("   ⚠️  Learning error: %s", e)

    async def _check_follow_opportunities(
        self, posts: List[MoltbookPost]
    ) -> int:
        """Check for agents worth following

        Args:
            posts: Recent posts to analyze

        Returns:
            Number of new follows
        """
        # Following should be VERY selective
        # For now, return 0 (we'll implement proper follow logic later)
        return 0

    async def start(self):
        """Start periodic heartbeat"""
        self._running = True
        logger.info("Starting Moltbook heartbeat (every %sh)", self.config.heartbeat_interval_hours)

        while self._running:
            # Check if should run
            if self.state.should_check(self.config.heartbeat_interval_hours):
                await self.run_once()

            # Wait before checking again (check every 15 minutes if it's time)
            await asyncio.sleep(900)  # 15 minutes

    def stop(self):
        """Stop periodic heartbeat"""
        self._running = False
        logger.info("💓 Stopping Moltbook heartbeat")

    def get_stats(self) -> Dict[str, Any]:
        """Get heartbeat statistics

        Returns:
            Statistics dictionary
        """
        last_check = self.state.get_last_check()

        return {
            "last_check": last_check.isoformat() if last_check else None,
            "total_checks": self.state.state["total_checks"],
            "total_posts": self.state.state["total_posts"],
            "total_comments": self.state.state["total_comments"],
            "total_upvotes": self.state.state["total_upvotes"],
            "should_check_now": self.state.should_check(
                self.config.heartbeat_interval_hours
            ),
        }

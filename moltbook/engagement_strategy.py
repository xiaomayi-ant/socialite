# -*- coding: utf-8 -*-
"""Moltbook Engagement Strategy Engine

This module implements intelligent decision-making for when and how to engage
with Moltbook content, following community guidelines and best practices.
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

from .models import MoltbookPost, MoltbookComment, MoltbookAgent, MoltbookSearchResult
from .config import MoltbookConfig


class EngagementStrategy:
    """Intelligent engagement decision engine"""

    def __init__(self, config: MoltbookConfig, social_memory: Optional[Any] = None):
        """Initialize engagement strategy

        Args:
            config: Moltbook configuration
            social_memory: Optional social memory for learning-based decisions
        """
        self.config = config
        self.social_memory = social_memory

        # Track engagement history for decision-making
        self.engagement_history: List[Dict[str, Any]] = []
        self.follow_candidates: Dict[str, Dict[str, Any]] = {}

    # ==================== Post Decision ====================

    def should_create_post(
        self, content_quality_score: float, topic: str, last_post_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Decide if should create a post

        Args:
            content_quality_score: Quality score of generated content (0-1)
            topic: Post topic
            last_post_time: Time of last post

        Returns:
            (should_post, reason)
        """
        # Check quality threshold
        if content_quality_score < self.config.min_post_quality_score:
            return (
                False,
                f"Quality too low: {content_quality_score:.2f} < {self.config.min_post_quality_score}",
            )

        # Check cooldown
        if last_post_time:
            elapsed_minutes = (datetime.now() - last_post_time).total_seconds() / 60
            if elapsed_minutes < self.config.post_cooldown_minutes:
                remaining = int(self.config.post_cooldown_minutes - elapsed_minutes)
                return False, f"Cooldown: {remaining} minutes remaining"

        # Check if topic is avoided
        if topic.lower() in [t.lower() for t in self.config.avoid_topics]:
            return False, f"Topic '{topic}' is in avoid list"

        # Conservative posting strategy: don't post too frequently even if allowed
        if last_post_time:
            elapsed_hours = (datetime.now() - last_post_time).total_seconds() / 3600
            if elapsed_hours < 2:  # Internal conservative limit
                return False, "Being conservative: wait at least 2 hours between posts"

        return True, "Quality and timing are good"

    def select_submolt(self, topic: str, available_submolts: List[str]) -> str:
        """Select best submolt for a topic

        Args:
            topic: Post topic
            available_submolts: Available submolt names

        Returns:
            Selected submolt name
        """
        # Check preferred submolts first
        for preferred in self.config.preferred_submolts:
            if preferred in available_submolts:
                return preferred

        # Topic-based selection (simple mapping)
        topic_lower = topic.lower()
        if "ai" in topic_lower or "agent" in topic_lower:
            if "aithoughts" in available_submolts:
                return "aithoughts"

        # Default to general
        return "general" if "general" in available_submolts else available_submolts[0]

    # ==================== Comment Decision ====================

    def should_comment_on_post(
        self,
        post: MoltbookPost,
        relevance_score: float,
        comment_quality_score: float,
        daily_comment_count: int,
    ) -> Tuple[bool, str]:
        """Decide if should comment on a post

        Args:
            post: Post to potentially comment on
            relevance_score: How relevant the post is (0-1)
            comment_quality_score: Quality of generated comment (0-1)
            daily_comment_count: Number of comments today

        Returns:
            (should_comment, reason)
        """
        # Check daily limit
        if daily_comment_count >= self.config.max_comments_per_day:
            return False, "Daily comment limit reached"

        # Check relevance
        if relevance_score < self.config.min_comment_relevance_score:
            return (
                False,
                f"Not relevant enough: {relevance_score:.2f} < {self.config.min_comment_relevance_score}",
            )

        # Check comment quality
        if comment_quality_score < 0.7:  # High bar for comments
            return False, f"Comment quality too low: {comment_quality_score:.2f}"

        # Don't comment on very new posts (let discussion develop)
        if post.created_at:
            now = datetime.now(timezone.utc) if post.created_at.tzinfo else datetime.now()
            age_minutes = (now - post.created_at).total_seconds() / 60
            if age_minutes < 10:
                return False, "Post too new, let it develop"

        # Don't comment on posts with no engagement yet (might be spam)
        if post.upvotes == 0 and post.comment_count == 0:
            return False, "No engagement yet, waiting for validation"

        # Don't comment on controversial posts (more downvotes than upvotes)
        if post.downvotes > post.upvotes:
            return False, "Post is controversial, avoiding"

        return True, "Good opportunity to add value"

    def rank_posts_for_commenting(
        self, posts: List[MoltbookPost], limit: int = 5
    ) -> List[MoltbookPost]:
        """Rank posts by comment priority

        Args:
            posts: Posts to rank
            limit: Number to return

        Returns:
            Top posts to consider commenting on
        """
        scored_posts = []

        for post in posts:
            score = self._calculate_comment_priority_score(post)
            scored_posts.append((score, post))

        # Sort by score descending
        scored_posts.sort(key=lambda x: x[0], reverse=True)

        return [post for _, post in scored_posts[:limit]]

    def _calculate_comment_priority_score(self, post: MoltbookPost) -> float:
        """Calculate priority score for commenting on a post

        Args:
            post: Post to evaluate

        Returns:
            Priority score (higher = better)
        """
        score = 0.0

        # Engagement indicates quality
        if post.upvotes > post.downvotes * 2:
            score += 2.0

        # Active discussions are good
        if 1 <= post.comment_count <= 10:
            score += 1.5  # Sweet spot for adding value

        # Not too popular (hard to add value)
        if post.comment_count > 20:
            score -= 1.0

        # Recent but not too new
        if post.created_at:
            age_hours = (datetime.now() - post.created_at).total_seconds() / 3600
            if 1 <= age_hours <= 24:
                score += 1.0

        # Quality content (long post)
        if post.content and len(post.content) > 200:
            score += 0.5

        return score

    # ==================== Upvote Decision ====================

    def should_upvote_post(self, post: MoltbookPost) -> Tuple[bool, str]:
        """Decide if should upvote a post

        Args:
            post: Post to evaluate

        Returns:
            (should_upvote, reason)
        """
        # Already voted
        if post.user_vote is not None:
            return False, "Already voted"

        # Calculate value
        value = self._calculate_post_engagement_value(post)

        if value > 0.6:
            return True, f"Valuable content (score: {value:.2f})"

        return False, f"Not valuable enough (score: {value:.2f})"

    def should_upvote_comment(self, comment: MoltbookComment) -> Tuple[bool, str]:
        """Decide if should upvote a comment

        Args:
            comment: Comment to evaluate

        Returns:
            (should_upvote, reason)
        """
        # Already voted
        if comment.user_vote is not None:
            return False, "Already voted"

        # Simple quality check
        if comment.upvotes > comment.downvotes * 2 and len(comment.content) > 50:
            return True, "Quality comment"

        return False, "Not engaging enough"

    def _calculate_post_engagement_value(self, post: MoltbookPost) -> float:
        """Calculate engagement value of a post

        Args:
            post: Post to evaluate

        Returns:
            Value score (0-1)
        """
        score = 0.5  # Base

        # Quality signals
        if post.upvotes > post.downvotes * 3:
            score += 0.2

        if post.upvotes > 10:
            score += 0.1

        if 3 <= post.comment_count <= 20:
            score += 0.1  # Active but not overwhelming

        # Content quality
        if post.content:
            if len(post.content) > 200:
                score += 0.1
            if len(post.content) > 500:
                score += 0.1

        return min(score, 1.0)

    # ==================== Follow Decision ====================

    def should_follow_agent(
        self, agent: MoltbookAgent, interaction_count: int, avg_content_quality: float
    ) -> Tuple[bool, str]:
        """Decide if should follow an agent (VERY selective)

        Args:
            agent: Agent to evaluate
            interaction_count: Number of interactions with this agent
            avg_content_quality: Average quality of their content (0-1)

        Returns:
            (should_follow, reason)
        """
        # VERY selective following criteria
        min_interactions = self.config.follow_threshold_interactions
        min_quality = self.config.follow_threshold_quality

        # Not enough interactions
        if interaction_count < min_interactions:
            return (
                False,
                f"Need more interactions: {interaction_count} < {min_interactions}",
            )

        # Quality not high enough
        if avg_content_quality < min_quality:
            return (
                False,
                f"Quality not high enough: {avg_content_quality:.2f} < {min_quality}",
            )

        # Agent should be active
        if agent.last_active:
            days_since_active = (datetime.now() - agent.last_active).days
            if days_since_active > 7:
                return False, f"Not active recently ({days_since_active} days)"

        # Agent should have reasonable karma (indicator of quality)
        if agent.karma < 10:
            return False, "Low karma, not established yet"

        return True, "Consistently high-quality contributor"

    def record_interaction(
        self, agent_name: str, interaction_type: str, quality_score: float
    ) -> None:
        """Record interaction with an agent for follow consideration

        Args:
            agent_name: Name of agent
            interaction_type: Type of interaction ("upvote", "comment", etc.)
            quality_score: Quality of their content (0-1)
        """
        if agent_name not in self.follow_candidates:
            self.follow_candidates[agent_name] = {
                "first_interaction": datetime.now(),
                "interactions": [],
                "quality_scores": [],
            }

        candidate = self.follow_candidates[agent_name]
        candidate["interactions"].append(
            {"type": interaction_type, "timestamp": datetime.now()}
        )
        candidate["quality_scores"].append(quality_score)

    def get_follow_candidates(self, min_interactions: int = 3) -> List[Dict[str, Any]]:
        """Get agents we've interacted with enough to consider following

        Args:
            min_interactions: Minimum interactions required

        Returns:
            List of candidate info
        """
        candidates = []

        for agent_name, data in self.follow_candidates.items():
            interaction_count = len(data["interactions"])

            if interaction_count >= min_interactions:
                avg_quality = (
                    sum(data["quality_scores"]) / len(data["quality_scores"])
                    if data["quality_scores"]
                    else 0
                )

                candidates.append(
                    {
                        "agent_name": agent_name,
                        "interaction_count": interaction_count,
                        "avg_quality": avg_quality,
                        "first_interaction": data["first_interaction"],
                    }
                )

        # Sort by quality
        candidates.sort(key=lambda x: x["avg_quality"], reverse=True)
        return candidates

    # ==================== Content Analysis ====================

    def analyze_post_for_learning(self, post: MoltbookPost) -> Dict[str, Any]:
        """Analyze if a post is worth learning from

        Args:
            post: Post to analyze

        Returns:
            Analysis results
        """
        learning_value = 0.0

        # High engagement indicates valuable content
        if post.upvotes > 20:
            learning_value += 0.3

        # Active discussion indicates interesting topic
        if post.comment_count > 10:
            learning_value += 0.2

        # Quality ratio
        if post.upvotes > 0:
            quality_ratio = post.upvotes / max(post.upvotes + post.downvotes, 1)
            learning_value += quality_ratio * 0.3

        # Content depth
        if post.content and len(post.content) > 300:
            learning_value += 0.2

        return {
            "learning_value": min(learning_value, 1.0),
            "should_learn": learning_value > 0.7,
            "reason": (
                "High-quality popular content"
                if learning_value > 0.7
                else "Not significant enough"
            ),
        }

# -*- coding: utf-8 -*-
"""Social Interaction Recorder for Social Memory System"""

import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import (
    SocialPost,
    SocialComment,
    SocialInteraction,
    SocialUser,
    EngagementMetrics,
    InteractionType,
    AuthorType,
    PlatformType,
)


class SocialInteractionRecorder:
    """Recorder for social interactions"""

    def __init__(self, agent_user_id: str, agent_username: str):
        """Initialize social interaction recorder

        Args:
            agent_user_id: Agent's user ID
            agent_username: Agent's username
        """
        self.agent_user_id = agent_user_id
        self.agent_username = agent_username
        self.interaction_history: List[SocialInteraction] = []

    def create_agent_user(
        self,
        platform: PlatformType,
        followers_count: Optional[int] = 0,
        following_count: Optional[int] = 0,
        verified: bool = False,
    ) -> SocialUser:
        """Create the agent user object

        Args:
            platform: Platform type
            followers_count: Number of followers
            following_count: Number of following
            verified: Verification status

        Returns:
            Social user object
        """
        return SocialUser(
            user_id=self.agent_user_id,
            username=self.agent_username,
            platform=platform,
            followers_count=followers_count,
            following_count=following_count,
            verified=verified,
        )

    def record_post(
        self,
        content: str,
        platform: PlatformType,
        post_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
        sentiment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a post created by the agent

        Args:
            content: Post content
            platform: Platform type
            post_id: Optional post ID (generated if not provided)
            topics: Optional list of topics
            sentiment: Optional sentiment analysis result
            metadata: Optional additional metadata

        Returns:
            Dictionary containing post and interaction information
        """
        if post_id is None:
            post_id = str(uuid.uuid4())

        # Create agent user
        agent_user = self.create_agent_user(platform)

        # Create post
        post = SocialPost(
            post_id=post_id,
            content=content,
            platform=platform,
            author=agent_user,
            timestamp=datetime.now(),
            topics=topics,
            sentiment=None,  # Will be analyzed later
            engagement=EngagementMetrics(
                upvotes=0,
                downvotes=0,
                comment_count=0,
                timestamp=datetime.now(),
            ),
            metadata=metadata,
        )

        # Create interaction record
        interaction = SocialInteraction(
            interaction_id=str(uuid.uuid4()),
            interaction_type=InteractionType.POST,
            content=content,
            platform=platform,
            target_post_id=None,
            actor=agent_user,
            timestamp=datetime.now(),
            learning_value=0.0,  # Will be calculated based on engagement
            metadata=metadata,
        )

        self.interaction_history.append(interaction)

        return {
            "post": post,
            "interaction": interaction,
            "success": True,
            "message": "Post recorded successfully",
        }

    def record_comment(
        self,
        content: str,
        platform: PlatformType,
        target_post_id: str,
        target_post_content: Optional[str] = None,
        comment_id: Optional[str] = None,
        reply_to_comment_id: Optional[str] = None,
        sentiment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a comment made by the agent

        Args:
            content: Comment content
            platform: Platform type
            target_post_id: Target post ID
            target_post_content: Optional target post content for context
            comment_id: Optional comment ID (generated if not provided)
            reply_to_comment_id: Optional parent comment ID if this is a reply
            sentiment: Optional sentiment analysis result
            metadata: Optional additional metadata

        Returns:
            Dictionary containing comment and interaction information
        """
        if comment_id is None:
            comment_id = str(uuid.uuid4())

        # Create agent user
        agent_user = self.create_agent_user(platform)

        # Create comment
        comment = SocialComment(
            comment_id=comment_id,
            content=content,
            platform=platform,
            target_post_id=target_post_id,
            target_post_content=target_post_content,
            author=agent_user,
            author_type=AuthorType.ME,
            timestamp=datetime.now(),
            sentiment=None,  # Will be analyzed later
            reply_to_comment_id=reply_to_comment_id,
            engagement=EngagementMetrics(
                upvotes=0,
                downvotes=0,
                comment_count=0,
                timestamp=datetime.now(),
            ),
            learning_value=0.0,  # Will be calculated based on engagement
            metadata=metadata,
        )

        # Create interaction record
        interaction = SocialInteraction(
            interaction_id=str(uuid.uuid4()),
            interaction_type=InteractionType.COMMENT,
            content=content,
            platform=platform,
            target_post_id=target_post_id,
            target_comment_id=reply_to_comment_id,
            actor=agent_user,
            timestamp=datetime.now(),
            learning_value=0.0,
            metadata=metadata,
        )

        self.interaction_history.append(interaction)

        return {
            "comment": comment,
            "interaction": interaction,
            "success": True,
            "message": "Comment recorded successfully",
        }

    def record_like(
        self,
        platform: PlatformType,
        target_post_id: str,
        target_post_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a like action by the agent

        Args:
            platform: Platform type
            target_post_id: Target post ID
            target_post_content: Optional target post content
            metadata: Optional additional metadata

        Returns:
            Dictionary containing interaction information
        """
        # Create agent user
        agent_user = self.create_agent_user(platform)

        # Create interaction record
        interaction = SocialInteraction(
            interaction_id=str(uuid.uuid4()),
            interaction_type=InteractionType.LIKE,
            content="",  # Like actions don't have content
            platform=platform,
            target_post_id=target_post_id,
            target_comment_id=None,
            actor=agent_user,
            timestamp=datetime.now(),
            learning_value=0.0,
            metadata=metadata,
        )

        self.interaction_history.append(interaction)

        return {
            "interaction": interaction,
            "success": True,
            "message": "Like recorded successfully",
        }

    def record_other_user_post(
        self,
        post_id: str,
        content: str,
        platform: PlatformType,
        author_id: str,
        author_username: str,
        engagement_data: Optional[Dict[str, int]] = None,
        sentiment: Optional[str] = None,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a post from another user (for learning purposes)

        Args:
            post_id: Post ID
            content: Post content
            platform: Platform type
            author_id: Author user ID
            author_username: Author username
            engagement_data: Optional engagement data
            sentiment: Optional sentiment
            topics: Optional topics
            metadata: Optional additional metadata

        Returns:
            Dictionary containing post information
        """
        # Create other user object
        other_user = SocialUser(
            user_id=author_id,
            username=author_username,
            platform=platform,
        )

        # Create post
        post = SocialPost(
            post_id=post_id,
            content=content,
            platform=platform,
            author=other_user,
            timestamp=datetime.now(),
            topics=topics,
            sentiment=None,  # Will be analyzed later
            engagement=EngagementMetrics(
                upvotes=engagement_data.get("upvotes", 0) if engagement_data else 0,
                downvotes=engagement_data.get("downvotes", 0) if engagement_data else 0,
                comment_count=engagement_data.get("comment_count", 0) if engagement_data else 0,
                timestamp=datetime.now(),
            ),
            metadata=metadata,
        )

        return {
            "post": post,
            "success": True,
            "message": "Other user's post recorded successfully",
        }

    def record_other_user_comment(
        self,
        comment_id: str,
        content: str,
        platform: PlatformType,
        target_post_id: str,
        author_id: str,
        author_username: str,
        target_post_content: Optional[str] = None,
        reply_to_comment_id: Optional[str] = None,
        engagement_data: Optional[Dict[str, int]] = None,
        sentiment: Optional[str] = None,
        learning_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a comment from another user (for learning purposes)

        Args:
            comment_id: Comment ID
            content: Comment content
            platform: Platform type
            target_post_id: Target post ID
            target_post_content: Optional target post content
            author_id: Author user ID
            author_username: Author username
            reply_to_comment_id: Optional parent comment ID
            engagement_data: Optional engagement data
            sentiment: Optional sentiment
            learning_value: Optional learning value (how useful this comment is)
            metadata: Optional additional metadata

        Returns:
            Dictionary containing comment information
        """
        # Create other user object
        other_user = SocialUser(
            user_id=author_id,
            username=author_username,
            platform=platform,
        )

        # Create comment
        comment = SocialComment(
            comment_id=comment_id,
            content=content,
            platform=platform,
            target_post_id=target_post_id,
            target_post_content=target_post_content,
            author=other_user,
            author_type=AuthorType.OTHER,
            timestamp=datetime.now(),
            sentiment=None,  # Will be analyzed later
            reply_to_comment_id=reply_to_comment_id,
            engagement=EngagementMetrics(
                upvotes=engagement_data.get("upvotes", 0) if engagement_data else 0,
                downvotes=engagement_data.get("downvotes", 0) if engagement_data else 0,
                comment_count=engagement_data.get("comment_count", 0) if engagement_data else 0,
                timestamp=datetime.now(),
            ),
            learning_value=learning_value or 0.0,
            metadata=metadata,
        )

        return {
            "comment": comment,
            "success": True,
            "message": "Other user's comment recorded successfully",
        }

    def record_reply_to_agent(
        self,
        comment_id: str,
        content: str,
        platform: PlatformType,
        target_post_id: str,
        author_id: str,
        author_username: str,
        target_post_content: Optional[str] = None,
        reply_to_comment_id: Optional[str] = None,
        engagement_data: Optional[Dict[str, int]] = None,
        sentiment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a reply to the agent's comment (for learning)

        Args:
            comment_id: Comment ID
            content: Comment content
            platform: Platform type
            target_post_id: Target post ID
            target_post_content: Optional target post content
            author_id: Author user ID
            author_username: Author username
            reply_to_comment_id: The agent's comment ID being replied to
            engagement_data: Optional engagement data
            sentiment: Optional sentiment
            metadata: Optional additional metadata

        Returns:
            Dictionary containing comment information
        """
        return self.record_other_user_comment(
            comment_id=comment_id,
            content=content,
            platform=platform,
            target_post_id=target_post_id,
            target_post_content=target_post_content,
            author_id=author_id,
            author_username=author_username,
            reply_to_comment_id=reply_to_comment_id,
            engagement_data=engagement_data,
            sentiment=sentiment,
            learning_value=1.0,  # High learning value for direct replies
            metadata=metadata,
        )

    def get_interaction_summary(
        self,
        limit: int = 100,
        platform: Optional[PlatformType] = None,
    ) -> Dict[str, Any]:
        """Get summary of recorded interactions

        Args:
            limit: Maximum number of interactions to include
            platform: Optional platform filter

        Returns:
            Dictionary containing interaction summary
        """
        interactions = self.interaction_history[-limit:]

        if platform:
            interactions = [i for i in interactions if i.platform == platform]

        summary = {
            "total_interactions": len(interactions),
            "posts": len(
                [i for i in interactions if i.interaction_type == InteractionType.POST]
            ),
            "comments": len(
                [
                    i
                    for i in interactions
                    if i.interaction_type == InteractionType.COMMENT
                ]
            ),
            "upvotes": len(
                [i for i in interactions if i.interaction_type == InteractionType.LIKE]
            ),
            "platform_breakdown": {},
            "interactions": [
                {
                    "id": i.interaction_id,
                    "type": i.interaction_type.value,
                    "content": i.content[:100] + "..."
                    if len(i.content) > 100
                    else i.content,
                    "platform": i.platform.value,
                    "timestamp": i.timestamp.isoformat(),
                }
                for i in interactions
            ],
        }

        # Platform breakdown
        for interaction in interactions:
            platform_name = interaction.platform.value
            if platform_name not in summary["platform_breakdown"]:
                summary["platform_breakdown"][platform_name] = 0
            summary["platform_breakdown"][platform_name] += 1

        return summary

    def clear_history(self) -> None:
        """Clear interaction history"""
        self.interaction_history.clear()

    def get_interaction_count(self, platform: Optional[PlatformType] = None) -> int:
        """Get count of recorded interactions

        Args:
            platform: Optional platform filter

        Returns:
            Number of interactions
        """
        if platform:
            return len([i for i in self.interaction_history if i.platform == platform])
        return len(self.interaction_history)

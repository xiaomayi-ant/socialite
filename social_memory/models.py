# -*- coding: utf-8 -*-
"""Data Models for Social Memory System"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class InteractionType(str, Enum):
    """Types of social interactions"""

    POST = "post"
    COMMENT = "comment"
    REPLY = "reply"
    LIKE = "like"


class PlatformType(str, Enum):
    """Supported social media platforms"""

    WEIBO = "weibo"
    ZHIHU = "zhihu"
    XIAOHONGSHU = "xiaohongshu"
    WECHAT = "wechat"
    GENERAL = "general"


class SentimentType(str, Enum):
    """Sentiment categories for content analysis"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AuthorType(str, Enum):
    """Types of content authors"""

    ME = "me"
    OTHER = "other"


class SocialUser(BaseModel):
    """Social user information"""

    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    platform: PlatformType = Field(..., description="Platform type")
    profile_url: Optional[str] = Field(default=None, description="Profile URL")
    followers_count: Optional[int] = Field(
        default=None, description="Number of followers"
    )
    following_count: Optional[int] = Field(
        default=None, description="Number of following"
    )
    verified: Optional[bool] = Field(default=None, description="Verification status")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class EngagementMetrics(BaseModel):
    """Engagement metrics for social content (aligned with Moltbook API)"""

    upvotes: int = Field(default=0, description="Number of upvotes")
    downvotes: int = Field(default=0, description="Number of downvotes")
    comment_count: int = Field(default=0, description="Number of comments/replies")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Metrics timestamp"
    )


class SocialPost(BaseModel):
    """Social media post model"""

    post_id: str = Field(..., description="Unique post identifier")
    content: str = Field(..., description="Post content")
    platform: PlatformType = Field(..., description="Platform type")
    author: SocialUser = Field(..., description="Post author")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Post timestamp"
    )

    # Content analysis
    sentiment: Optional[SentimentType] = Field(
        default=None, description="Content sentiment"
    )
    topics: Optional[List[str]] = Field(default=None, description="Extracted topics")

    # Engagement metrics
    engagement: EngagementMetrics = Field(
        default_factory=EngagementMetrics, description="Engagement metrics"
    )

    # Vector representation (for semantic search)
    embedding: Optional[List[float]] = Field(
        default=None, description="Content embedding vector"
    )

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class SocialComment(BaseModel):
    """Social media comment model"""

    comment_id: str = Field(..., description="Unique comment identifier")
    content: str = Field(..., description="Comment content")
    platform: PlatformType = Field(..., description="Platform type")
    target_post_id: str = Field(..., description="Target post ID")
    target_post_content: Optional[str] = Field(
        default=None, description="Target post content for context"
    )

    # Author information
    author: SocialUser = Field(..., description="Comment author")
    author_type: AuthorType = Field(..., description="Author type (me or other)")

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Comment timestamp"
    )

    # Content analysis
    sentiment: Optional[SentimentType] = Field(
        default=None, description="Comment sentiment"
    )
    reply_to_comment_id: Optional[str] = Field(
        default=None, description="If this is a reply to another comment"
    )

    # Engagement
    engagement: EngagementMetrics = Field(
        default_factory=EngagementMetrics, description="Engagement metrics"
    )

    # Vector representation
    embedding: Optional[List[float]] = Field(
        default=None, description="Comment embedding vector"
    )

    # Learning value (how useful this comment is for learning)
    learning_value: Optional[float] = Field(
        default=None, description="Learning value score (0-1)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class SocialInteraction(BaseModel):
    """Record of a social interaction for learning"""

    interaction_id: str = Field(..., description="Unique interaction identifier")
    interaction_type: InteractionType = Field(..., description="Type of interaction")

    # Content information
    content: str = Field(..., description="Interaction content")
    platform: PlatformType = Field(..., description="Platform type")

    # Context information
    target_post_id: Optional[str] = Field(
        default=None, description="Target post ID if relevant"
    )
    target_comment_id: Optional[str] = Field(
        default=None, description="Target comment ID if relevant"
    )

    # Actor information
    actor: SocialUser = Field(..., description="User who performed the interaction")

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Interaction timestamp"
    )

    # Results and feedback
    engagement_result: Optional[EngagementMetrics] = Field(
        default=None, description="Engagement result"
    )

    # Learning value
    learning_value: float = Field(default=0.0, description="Learning value score (0-1)")
    feedback_score: Optional[float] = Field(
        default=None, description="User feedback score (0-1)"
    )

    # Vector representation
    embedding: Optional[List[float]] = Field(
        default=None, description="Interaction embedding vector"
    )

    # Pattern information
    pattern_id: Optional[str] = Field(
        default=None, description="Associated learning pattern ID"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class LearningPattern(BaseModel):
    """Learned social interaction pattern"""

    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: InteractionType = Field(..., description="Type of pattern")

    # Pattern characteristics
    trigger_context: str = Field(..., description="Context that triggers this pattern")
    response_content: str = Field(..., description="Typical response for this pattern")

    # Platform and topic information
    platform: PlatformType = Field(..., description="Platform where pattern applies")
    topics: Optional[List[str]] = Field(
        default=None, description="Topics where pattern applies"
    )

    # Performance metrics
    success_rate: float = Field(default=0.0, description="Pattern success rate (0-1)")
    usage_count: int = Field(default=0, description="Number of times pattern was used")

    # Learning metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Pattern creation timestamp"
    )
    last_used: Optional[datetime] = Field(
        default=None, description="Last usage timestamp"
    )
    learning_source: str = Field(default="", description="Source of this pattern")

    # Vector representation
    trigger_embedding: Optional[List[float]] = Field(
        default=None, description="Trigger context embedding"
    )
    response_embedding: Optional[List[float]] = Field(
        default=None, description="Response content embedding"
    )

    # Adaptability
    adaptability_score: float = Field(
        default=0.5, description="How adaptable the pattern is (0-1)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class SocialInteractionSummary(BaseModel):
    """Summary of social interactions for analysis"""

    period_start: datetime = Field(..., description="Summary period start")
    period_end: datetime = Field(..., description="Summary period end")

    # Interaction counts
    total_posts: int = Field(default=0, description="Total number of posts")
    total_comments: int = Field(default=0, description="Total number of comments")
    total_upvotes: int = Field(default=0, description="Total number of upvotes")

    # Platform breakdown
    platform_breakdown: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Interaction breakdown by platform"
    )

    # Engagement statistics
    average_upvotes: Optional[float] = Field(
        default=None, description="Average upvotes per post"
    )
    average_comments: Optional[float] = Field(
        default=None, description="Average comments per post"
    )
    top_performing_posts: List[str] = Field(
        default_factory=list, description="IDs of top performing posts"
    )

    # Sentiment analysis
    sentiment_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Distribution of sentiment types"
    )

    # Learning statistics
    new_patterns_learned: int = Field(
        default=0, description="Number of new patterns learned"
    )
    pattern_success_rate: Optional[float] = Field(
        default=None, description="Overall pattern success rate"
    )

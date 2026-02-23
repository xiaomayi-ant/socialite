# -*- coding: utf-8 -*-
"""Moltbook Data Models"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class MoltbookAgent(BaseModel):
    """Moltbook Agent/Molty model - flexible to handle both full and partial responses"""

    model_config = {"extra": "ignore"}

    name: str
    description: Optional[str] = None
    karma: int = 0
    follower_count: int = 0
    following_count: int = 0
    is_claimed: bool = False
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    avatar_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MoltbookSubmolt(BaseModel):
    """Moltbook Submolt (Community) model"""

    model_config = {"extra": "ignore"}

    name: str
    display_name: str = ""
    description: Optional[str] = None
    subscriber_count: int = 0
    post_count: int = 0
    created_at: Optional[datetime] = None
    avatar_url: Optional[str] = None
    banner_url: Optional[str] = None
    banner_color: Optional[str] = None
    theme_color: Optional[str] = None
    your_role: Optional[str] = None  # "owner", "moderator", or None


class MoltbookPost(BaseModel):
    """Moltbook Post model - handles API response field variations"""

    model_config = {"extra": "ignore"}

    id: Optional[str] = None
    submolt: Optional[str] = None       # used when creating a post
    submolt_name: Optional[str] = None  # returned by feed/get APIs
    title: str
    content: Optional[str] = None
    url: Optional[str] = None
    author: Optional[MoltbookAgent] = None
    upvotes: int = 0
    downvotes: int = 0
    comment_count: int = 0
    created_at: Optional[datetime] = None
    is_pinned: bool = False
    user_vote: Optional[int] = None  # 1 for upvote, -1 for downvote, None for no vote
    you_follow_author: bool = False

    @property
    def submolt_display(self) -> str:
        """Get submolt name regardless of which field is used"""
        return self.submolt or self.submolt_name or "unknown"


class MoltbookComment(BaseModel):
    """Moltbook Comment model"""

    model_config = {"extra": "ignore"}

    id: Optional[str] = None
    post_id: Optional[str] = None
    content: str
    author: Optional[MoltbookAgent] = None
    parent_id: Optional[str] = None  # For nested comments
    upvotes: int = 0
    downvotes: int = 0
    created_at: Optional[datetime] = None
    user_vote: Optional[int] = None


class MoltbookFeedItem(BaseModel):
    """Unified feed item that can be a post or comment"""

    type: str  # "post" or "comment"
    id: str
    post: Optional[MoltbookPost] = None
    comment: Optional[MoltbookComment] = None


class MoltbookSearchResult(BaseModel):
    """Semantic search result"""

    model_config = {"extra": "ignore"}

    id: str
    type: str  # "post" or "comment"
    title: Optional[str] = None
    content: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    relevance: Optional[float] = None   # API field name
    similarity: Optional[float] = None  # alias for relevance
    author: Optional[MoltbookAgent] = None
    post_id: Optional[str] = None
    created_at: Optional[datetime] = None

    @model_validator(mode="after")
    def unify_score(self) -> "MoltbookSearchResult":
        """Make similarity and relevance interchangeable"""
        if self.similarity is None and self.relevance is not None:
            self.similarity = self.relevance
        elif self.relevance is None and self.similarity is not None:
            self.relevance = self.similarity
        return self


class MoltbookRegistration(BaseModel):
    """Registration response"""

    api_key: str
    claim_url: str
    verification_code: str
    agent_name: str


class MoltbookCredentials(BaseModel):
    """Stored credentials"""

    api_key: str
    agent_name: str
    registered_at: datetime = Field(default_factory=datetime.now)
    claimed: bool = False
    claimed_at: Optional[datetime] = None

# -*- coding: utf-8 -*-
"""Moltbook Integration Package

This package provides integration with Moltbook - the social network for AI agents.
"""

from .client import MoltbookClient, RateLimiter
from .auth import MoltbookAuth
from .config import MoltbookConfig
from .models import (
    MoltbookPost,
    MoltbookComment,
    MoltbookAgent,
    MoltbookSubmolt,
    MoltbookFeedItem,
    MoltbookSearchResult,
    MoltbookRegistration,
    MoltbookCredentials,
)
from .heartbeat import MoltbookHeartbeat, HeartbeatState
from .engagement_strategy import EngagementStrategy
from .verifier import ChallengeVerifier

__all__ = [
    "MoltbookClient",
    "RateLimiter",
    "MoltbookAuth",
    "MoltbookConfig",
    "MoltbookPost",
    "MoltbookComment",
    "MoltbookAgent",
    "MoltbookSubmolt",
    "MoltbookFeedItem",
    "MoltbookSearchResult",
    "MoltbookRegistration",
    "MoltbookCredentials",
    "MoltbookHeartbeat",
    "HeartbeatState",
    "EngagementStrategy",
    "ChallengeVerifier",
]

__version__ = "0.1.0"

# -*- coding: utf-8 -*-
"""Moltbook Configuration Management"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


import logging
logger = logging.getLogger(__name__)

class MoltbookConfig(BaseModel):
    """Configuration for Moltbook integration"""

    # API Configuration
    api_base: str = Field(
        default="https://www.moltbook.com/api/v1",
        description="Moltbook API base URL",
    )
    api_key: Optional[str] = Field(default=None, description="Moltbook API key")

    # Agent Configuration
    agent_name: Optional[str] = Field(default=None, description="Agent name on Moltbook")
    agent_description: Optional[str] = Field(
        default="A social learning AI agent", description="Agent description"
    )

    # Auto-registration
    auto_register: bool = Field(
        default=False, description="Automatically register if not registered"
    )

    # Rate Limits (conservative defaults)
    post_cooldown_minutes: int = Field(
        default=30, description="Minimum minutes between posts"
    )
    comment_cooldown_seconds: int = Field(
        default=20, description="Minimum seconds between comments"
    )
    max_comments_per_day: int = Field(
        default=50, description="Maximum comments per day"
    )

    # Heartbeat Configuration
    heartbeat_interval_hours: int = Field(
        default=4, description="Hours between heartbeat checks"
    )
    heartbeat_enabled: bool = Field(
        default=True, description="Enable automatic heartbeat"
    )

    # Engagement Strategy
    min_post_quality_score: float = Field(
        default=0.7, description="Minimum quality score for posting (0-1)"
    )
    min_comment_relevance_score: float = Field(
        default=0.6, description="Minimum relevance score for commenting (0-1)"
    )
    follow_threshold_interactions: int = Field(
        default=3, description="Minimum interactions before considering following"
    )
    follow_threshold_quality: float = Field(
        default=0.8, description="Minimum quality score for following (0-1)"
    )

    # Content Strategy
    preferred_submolts: list[str] = Field(
        default=["general"], description="Preferred submolts to participate in"
    )
    avoid_topics: list[str] = Field(
        default=[], description="Topics to avoid in content"
    )

    # Storage
    credentials_path: str = Field(
        default="~/.config/moltbook/credentials.json",
        description="Path to store credentials",
    )
    state_path: str = Field(
        default="~/.config/moltbook/state.json", description="Path to store state"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(
        default="./logs/moltbook.log", description="Log file path"
    )

    @classmethod
    def from_env(cls) -> "MoltbookConfig":
        """Create config from environment variables"""
        return cls(
            api_key=os.getenv("MOLTBOOK_API_KEY"),
            agent_name=os.getenv("MOLTBOOK_AGENT_NAME"),
            agent_description=os.getenv(
                "MOLTBOOK_AGENT_DESCRIPTION", "A social learning AI agent"
            ),
            auto_register=os.getenv("MOLTBOOK_AUTO_REGISTER", "false").lower()
            == "true",
            heartbeat_enabled=os.getenv("MOLTBOOK_HEARTBEAT_ENABLED", "true").lower()
            == "true",
            heartbeat_interval_hours=int(
                os.getenv("MOLTBOOK_HEARTBEAT_INTERVAL_HOURS", "4")
            ),
            preferred_submolts=os.getenv("MOLTBOOK_PREFERRED_SUBMOLTS", "general").split(
                ","
            ),
            log_level=os.getenv("MOLTBOOK_LOG_LEVEL", "INFO"),
            log_file=os.getenv("MOLTBOOK_LOG_FILE", "./logs/moltbook.log"),
        )

    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Check API base URL
            if not self.api_base.startswith("https://www.moltbook.com"):
                logger.warning("API base should be https://www.moltbook.com for security")

            # Check rate limits are reasonable
            if self.post_cooldown_minutes < 30:
                logger.info("Warning: Post cooldown should be at least 30 minutes")

            if self.comment_cooldown_seconds < 20:
                logger.info("Warning: Comment cooldown should be at least 20 seconds")

            return True

        except Exception as e:
            logger.info("Configuration validation failed: %s", e)
            return False

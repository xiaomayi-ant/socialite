# -*- coding: utf-8 -*-
"""Social Memory System for AI Agent Learning"""

from .config import (
    SocialMemoryConfig,
    VectorStoreConfig,
    StructuredStoreConfig,
    GraphStoreConfig,
)
from .models import (
    SocialPost,
    SocialComment,
    SocialInteraction,
    LearningPattern,
    SocialUser,
)
from .storage_manager import SocialMemoryManager

__version__ = "0.1.0"
__all__ = [
    "SocialMemoryConfig",
    "VectorStoreConfig",
    "StructuredStoreConfig",
    "GraphStoreConfig",
    "SocialPost",
    "SocialComment",
    "SocialInteraction",
    "LearningPattern",
    "SocialUser",
    "SocialMemoryManager",
]

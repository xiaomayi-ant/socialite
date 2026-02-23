# -*- coding: utf-8 -*-
"""Unified Storage Manager for Social Memory System"""

import asyncio
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from .config import SocialMemoryConfig
from .models import (
    SocialPost,
    SocialComment,
    SocialInteraction,
    LearningPattern,
    SocialUser,
    PlatformType,
    AuthorType,
    InteractionType,
)
from .vector_store import VectorStorageManager
from .structured_store import StructuredStorageManager
from .knowledge_graph import KnowledgeGraphManager
from .recorder import SocialInteractionRecorder
from .embeddings import BGEEmbeddingModel

import logging
logger = logging.getLogger(__name__)


class SocialMemoryManager:
    """Unified manager for all storage systems in social memory"""

    def __init__(self, config: SocialMemoryConfig):
        """Initialize social memory manager

        Args:
            config: Social memory configuration
        """
        self.config = config
        self.vector_store: Optional[VectorStorageManager] = None
        self.structured_store: Optional[StructuredStorageManager] = None
        self.graph_store: Optional[KnowledgeGraphManager] = None
        self.recorder: Optional[SocialInteractionRecorder] = None
        self._initialized = False

        # Initialize embedding model
        if config.embedding_model_name:
            logger.info("Initializing embedding model: %s...", config.embedding_model_name)
            self.embedding_model = BGEEmbeddingModel(
                model_name=config.embedding_model_name,
                dimension=config.vector_store.vector_size,
            )
        else:
            logger.warning("No embedding model specified, using placeholder")
            self.embedding_model = None
        self.embedding_dimension = config.vector_store.vector_size

    async def initialize(
        self,
        agent_user_id: str = "social_agent_001",
        agent_username: str = "SocialLearner",
    ) -> bool:
        """Initialize all storage systems

        Args:
            agent_user_id: Agent's user ID
            agent_username: Agent's username

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing Social Memory System...")

            # Initialize vector storage
            if self.config.enable_vectorization:
                logger.info("Initializing vector storage...")
                self.vector_store = VectorStorageManager(self.config.vector_store)
                if not await self.vector_store.initialize():
                    logger.warning("Vector storage initialization failed")
                    self.vector_store = None

            # Initialize structured storage
            if self.config.enable_structured_search:
                logger.info("Initializing structured storage...")
                self.structured_store = StructuredStorageManager(
                    self.config.structured_store
                )
                if not await self.structured_store.initialize():
                    logger.warning("Structured storage initialization failed")
                    self.structured_store = None

            # Initialize knowledge graph
            if self.config.enable_graph_analysis:
                logger.info("Initializing knowledge graph...")
                self.graph_store = KnowledgeGraphManager(self.config.graph_store)
                if not await self.graph_store.initialize():
                    logger.warning("Knowledge graph initialization failed")
                    self.graph_store = None

            # Initialize recorder
            logger.info("Initializing interaction recorder...")
            self.recorder = SocialInteractionRecorder(agent_user_id, agent_username)

            self._initialized = True
            logger.info("Social Memory System initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize social memory system: %s", e)
            return False

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.embedding_model:
            return self.embedding_model.encode(text)
        else:
            logger.warning("Using placeholder embedding for text: %s...", text[:50])
            return [0.0] * self.embedding_dimension

    def _calculate_interaction_learning_value(
        self, interaction_data: Dict[str, Any]
    ) -> float:
        """Calculate learning value based on engagement metrics.

        Args:
            interaction_data: Interaction data with upvotes/comment_count

        Returns:
            Learning value score (0.0 to 1.0)
        """
        upvotes = interaction_data.get("upvotes", 0)
        comment_count = interaction_data.get("comment_count", 0)

        # Normalize: high engagement = high learning value
        # Scale: comments are worth 2x upvotes
        normalized = (upvotes + comment_count * 2) / 100.0
        return min(max(normalized, 0.0), 1.0)  # Clamp to [0, 1]

    async def record_interaction(
        self,
        interaction_data: Dict[str, Any],
        platform: PlatformType,
        vectorize: bool = True,
    ) -> Dict[str, Any]:
        """Record a social interaction across all storage systems

        Args:
            interaction_data: Interaction data dictionary
            platform: Platform type
            vectorize: Whether to vectorize the content

        Returns:
            Result dictionary
        """
        if not self._initialized:
            return {"success": False, "message": "Social memory not initialized"}

        try:
            interaction_type = interaction_data.get("type")

            # Record in structured storage
            if self.structured_store:
                if interaction_type == "post":
                    # Generate valid UUID for post_id if not provided
                    import uuid

                    post_id = interaction_data.get("post_id") or str(uuid.uuid4())
                    # Create author from recorder's agent info
                    author = (
                        self.recorder.create_agent_user(platform)
                        if self.recorder
                        else None
                    )

                    post_data = {
                        **interaction_data,
                        "post_id": post_id,
                        "author": author,
                    }
                    post = SocialPost(**post_data)
                    await self.structured_store.record_post(post)

                elif interaction_type == "comment":
                    # Generate valid UUID for comment_id if not provided
                    import uuid

                    comment_id = interaction_data.get("comment_id") or str(uuid.uuid4())
                    # Create author from recorder's agent info
                    author = (
                        self.recorder.create_agent_user(platform)
                        if self.recorder
                        else None
                    )

                    comment_data = {
                        **interaction_data,
                        "comment_id": comment_id,
                        "author": author,
                        "author_type": AuthorType.ME,
                    }
                    comment = SocialComment(**comment_data)
                    await self.structured_store.record_comment(comment)

            # Record in vector storage
            if self.vector_store and vectorize and self.config.enable_vectorization:
                content = interaction_data.get("content", "")
                if content:
                    embedding = self._get_embedding(content)

                    if interaction_type == "post":
                        # Generate valid UUID for post_id if not provided
                        import uuid

                        post_id = interaction_data.get("post_id") or str(uuid.uuid4())
                        author = (
                            self.recorder.create_agent_user(platform)
                            if self.recorder
                            else None
                        )
                        post_data = {
                            **interaction_data,
                            "post_id": post_id,
                            "author": author,
                        }
                        post = SocialPost(**post_data)
                        await self.vector_store.store_post(post, embedding)
                    elif interaction_type == "comment":
                        # Generate valid UUID for comment_id if not provided
                        import uuid

                        comment_id = interaction_data.get("comment_id") or str(
                            uuid.uuid4()
                        )
                        author = (
                            self.recorder.create_agent_user(platform)
                            if self.recorder
                            else None
                        )
                        comment_data = {
                            **interaction_data,
                            "comment_id": comment_id,
                            "author": author,
                            "author_type": AuthorType.ME,
                        }
                        comment = SocialComment(**comment_data)
                        await self.vector_store.store_comment(comment, embedding)

            # Record in knowledge graph
            if self.graph_store and self.config.enable_graph_analysis:
                if interaction_type == "post":
                    post = SocialPost(**interaction_data)
                    await self.graph_store.add_post_node(post)
                elif interaction_type == "comment":
                    comment = SocialComment(**interaction_data)
                    await self.graph_store.add_comment_node(comment)

            return {
                "success": True,
                "message": "Interaction recorded successfully",
                "recorded_in": {
                    "vector_store": self.vector_store is not None,
                    "structured_store": self.structured_store is not None,
                    "graph_store": self.graph_store is not None,
                },
            }

        except Exception as e:
            logger.error("Error recording interaction: %s", e)
            return {"success": False, "message": str(e)}

    async def retrieve_similar(
        self,
        query: str,
        content_type: str = "posts",
        platform: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar content from storage systems

        Args:
            query: Query text
            content_type: Type of content ("posts", "comments", "patterns")
            platform: Optional platform filter
            limit: Maximum number of results

        Returns:
            List of similar items
        """
        if not self._initialized:
            return []

        results = []

        # Retrieve from vector storage
        if self.vector_store and self.config.enable_vectorization:
            query_embedding = self._get_embedding(query)

            if content_type == "posts":
                results = await self.vector_store.find_similar_posts(
                    query=query,
                    query_embedding=query_embedding,
                    platform=platform,
                    limit=limit,
                )
            elif content_type == "comments":
                results = await self.vector_store.find_similar_comments(
                    query=query, query_embedding=query_embedding, limit=limit
                )

        return results

    async def analyze_patterns(
        self,
        platform: Optional[PlatformType] = None,
        time_period: Optional[int] = None,  # days
    ) -> Dict[str, Any]:
        """Analyze social interaction patterns

        Args:
            platform: Optional platform filter
            time_period: Optional time period in days

        Returns:
            Dictionary containing pattern analysis
        """
        if not self._initialized:
            return {}

        analysis = {
            "total_interactions": 0,
            "successful_patterns": [],
            "failed_patterns": [],
            "learning_opportunities": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Get interaction summary from recorder
        if self.recorder:
            summary = self.recorder.get_interaction_summary(platform=platform)
            analysis["total_interactions"] = summary["total_interactions"]
            analysis["breakdown"] = {
                "posts": summary["posts"],
                "comments": summary["comments"],
                "upvotes": summary.get("upvotes", summary.get("likes", 0)),
            }

        # Analyze using structured storage
        if self.structured_store:
            # Query recent posts
            posts = await self.structured_store.query_post_history(
                platform=platform.value if platform else None,
                limit=100,
            )
            analysis["recent_posts_count"] = len(posts)

            # Get engagement statistics
            engagement_stats = await self.structured_store.get_engagement_stats(
                platform=platform.value if platform else None,
                days=time_period or 30,
            )
            analysis["engagement_stats"] = engagement_stats

        # Analyze using knowledge graph
        if self.graph_store:
            # Find similar users (if agent_user_id is set)
            if self.recorder:
                communities = await self.graph_store.detect_communities(
                    min_community_size=3,
                )
                analysis["community_analysis"] = communities

        return analysis

    async def get_learning_data(
        self,
        platform: Optional[PlatformType] = None,
        min_learning_value: float = 0.5,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get learning data from recorded interactions

        Args:
            platform: Optional platform filter
            min_learning_value: Minimum learning value threshold
            limit: Maximum number of results

        Returns:
            List of learning data items
        """
        learning_data = []

        # This is a placeholder implementation
        # In production, this would query the structured storage for high-learning-value interactions
        # and combine with vector similarity search results

        if self.structured_store:
            # Query recent comments with high learning value
            # (This would need to be implemented in the structured_store module)
            pass

        if self.recorder:
            # Get interaction summary
            summary = self.recorder.get_interaction_summary(
                limit=limit, platform=platform
            )

            # Add to learning data
            for interaction in summary["interactions"]:
                learning_data.append(
                    {
                        "id": interaction["id"],
                        "type": interaction["type"],
                        "content": interaction["content"],
                        "platform": interaction["platform"],
                        "timestamp": interaction["timestamp"],
                        "learning_value": 0.5,  # Placeholder
                    }
                )

        return learning_data[:limit]

    async def create_learning_pattern(
        self, pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a learning pattern from analysis

        Args:
            pattern_data: Pattern data dictionary

        Returns:
            Result dictionary
        """
        if not self._initialized:
            return {"success": False, "message": "Social memory not initialized"}

        try:
            pattern = LearningPattern(**pattern_data)

            # Record in structured storage
            if self.structured_store:
                # This would need to be implemented
                pass

            # Record in vector storage
            if self.vector_store and self.config.enable_vectorization:
                # Vectorize trigger and response
                trigger_embedding = self._get_embedding(pattern.trigger_context)
                response_embedding = self._get_embedding(pattern.response_content)

                # Store patterns (this would need to be implemented in vector_store)
                pass

            return {
                "success": True,
                "pattern_id": pattern.pattern_id,
                "message": "Learning pattern created successfully",
            }

        except Exception as e:
            logger.error("Error creating learning pattern: %s", e)
            return {"success": False, "message": str(e)}

    def get_recorder(self) -> Optional[SocialInteractionRecorder]:
        """Get the interaction recorder

        Returns:
            Social interaction recorder or None
        """
        return self.recorder

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics from all storage systems

        Returns:
            Dictionary containing storage statistics
        """
        stats = {
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "vectorization_enabled": self.config.enable_vectorization,
                "graph_analysis_enabled": self.config.enable_graph_analysis,
                "structured_search_enabled": self.config.enable_structured_search,
            },
        }

        # Vector store stats
        if self.vector_store:
            stats["vector_store"] = {
                "posts": await self.vector_store.get_collection_stats("posts"),
                "comments": await self.vector_store.get_collection_stats("comments"),
                "patterns": await self.vector_store.get_collection_stats("patterns"),
            }

        # Recorder stats
        if self.recorder:
            stats["recorder"] = {
                "total_interactions": self.recorder.get_interaction_count(),
                "agent_user_id": self.recorder.agent_user_id,
                "agent_username": self.recorder.agent_username,
            }

        return stats

    async def close(self) -> None:
        """Close all storage connections"""
        logger.info("Closing Social Memory System...")

        if self.vector_store:
            await self.vector_store.close()

        if self.structured_store:
            await self.structured_store.close()

        if self.graph_store:
            await self.graph_store.close()

        self._initialized = False
        logger.info("Social Memory System closed")

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all storage systems

        Returns:
            Dictionary containing health status
        """
        return {
            "initialized": self._initialized,
            "vector_store": self.vector_store is not None,
            "structured_store": self.structured_store is not None,
            "graph_store": self.graph_store is not None,
            "recorder": self.recorder is not None,
        }

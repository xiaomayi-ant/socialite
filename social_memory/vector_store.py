# -*- coding: utf-8 -*-
"""Vector Storage Manager for Social Memory System"""

import logging
logger = logging.getLogger(__name__)

import asyncio
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    Range = None
    logger.warning(" qdrant-client not installed. Install with: pip install qdrant-client"
    )

from .config import VectorStoreConfig
from .models import SocialPost, SocialComment, SocialInteraction


class VectorStorageManager:
    """Manager for vector storage using Qdrant"""

    def __init__(self, config: VectorStoreConfig):
        """Initialize vector storage manager

        Args:
            config: Vector store configuration
        """
        self.config = config
        self.client: Optional[QdrantClient] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize vector storage connection

        Returns:
            bool: True if initialization successful
        """
        try:
            if QdrantClient is None:
                logger.error("Error: qdrant-client not installed")
                return False

            # Create Qdrant client based on storage strategy
            if self.config.storage_strategy == "memory":
                # In-memory mode for development
                self.client = QdrantClient(":memory:")
            else:
                logger.info(
                    "Connecting to Qdrant host=%s port=%s https=%s via_proxy=%s no_proxy=%s",
                    self.config.host,
                    self.config.port,
                    self.config.https,
                    bool(
                        os.getenv("HTTP_PROXY")
                        or os.getenv("HTTPS_PROXY")
                        or os.getenv("ALL_PROXY")
                    ),
                    os.getenv("NO_PROXY") or os.getenv("no_proxy") or "",
                )
                # Persistent mode
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    https=self.config.https,
                    api_key=self.config.api_key,
                )

            # Create collections
            await self._create_collections()

            self._initialized = True
            logger.info("Vector storage initialized: %s", self.config.collection_name)
            return True

        except Exception as e:
            logger.error("Failed to initialize vector storage: %s", e)
            return False

    async def _create_collections(self) -> None:
        """Create Qdrant collections if they don't exist"""
        if not self.client:
            return

        collections = {
            f"{self.config.collection_name}_posts": {
                "description": "Social posts vectors",
                "vector_size": self.config.vector_size,
            },
            f"{self.config.collection_name}_comments": {
                "description": "Social comments vectors",
                "vector_size": self.config.vector_size,  # Fixed: use same vector size for all collections
            },
            f"{self.config.collection_name}_patterns": {
                "description": "Learning patterns vectors",
                "vector_size": self.config.vector_size,
            },
        }

        for collection_name, collection_config in collections.items():
            try:
                # Check if collection exists
                collections_list = self.client.get_collections().collections
                collection_exists = any(
                    c.name == collection_name for c in collections_list
                )

                if not collection_exists:
                    # Create collection
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=collection_config["vector_size"],
                            distance=self._get_distance_metric(),
                        ),
                    )
                    logger.info("Created collection: %s", collection_name)
                else:
                    logger.debug("Collection already exists: %s", collection_name)

            except Exception as e:
                logger.error("Error creating collection %s: %s", collection_name, e)

    def _get_distance_metric(self):
        """Get distance metric from config"""
        metric_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        return metric_map.get(self.config.distance, Distance.COSINE)

    async def store_post(self, post: SocialPost, embedding: List[float]) -> bool:
        """Store a post and its vector

        Args:
            post: Social post object
            embedding: Post content embedding vector

        Returns:
            bool: True if storage successful
        """
        if not self._initialized or not self.client:
            logger.warning("Vector storage not initialized")
            return False

        try:
            collection_name = f"{self.config.collection_name}_posts"

            # Prepare payload with metadata
            payload = {
                "post_id": post.post_id,
                "content": post.content,
                "platform": post.platform.value,
                "author_id": post.author.user_id,
                "author_username": post.author.username,
                "timestamp": post.timestamp.isoformat(),
                "sentiment": post.sentiment.value if post.sentiment else None,
                "topics": post.topics or [],
                "engagement": {
                    "upvotes": post.engagement.upvotes,
                    "downvotes": post.engagement.downvotes,
                    "comment_count": post.engagement.comment_count,
                },
            }

            # Create point
            point = PointStruct(
                id=post.post_id,
                vector=embedding,
                payload=payload,
            )

            # Upsert point
            self.client.upsert(
                collection_name=collection_name,
                points=[point],
            )

            return True

        except Exception as e:
            logger.error("Error storing post vector: %s", e)
            return False

    async def store_comment(
        self, comment: SocialComment, embedding: List[float]
    ) -> bool:
        """Store a comment and its vector

        Args:
            comment: Social comment object
            embedding: Comment content embedding vector

        Returns:
            bool: True if storage successful
        """
        if not self._initialized or not self.client:
            logger.warning("Vector storage not initialized")
            return False

        try:
            collection_name = f"{self.config.collection_name}_comments"

            # Prepare payload
            payload = {
                "comment_id": comment.comment_id,
                "content": comment.content,
                "platform": comment.platform.value,
                "target_post_id": comment.target_post_id,
                "target_post_content": comment.target_post_content,
                "author_id": comment.author.user_id,
                "author_type": comment.author_type.value,
                "timestamp": comment.timestamp.isoformat(),
                "sentiment": comment.sentiment.value if comment.sentiment else None,
                "reply_to_comment_id": comment.reply_to_comment_id,
                "learning_value": comment.learning_value,
            }

            point = PointStruct(
                id=comment.comment_id,
                vector=embedding,
                payload=payload,
            )

            self.client.upsert(
                collection_name=collection_name,
                points=[point],
            )

            return True

        except Exception as e:
            logger.error("Error storing comment vector: %s", e)
            return False

    async def semantic_search(
        self,
        query_embedding: List[float],
        collection_type: str = "posts",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search

        Args:
            query_embedding: Query vector
            collection_type: Type of collection ("posts", "comments", "patterns")
            limit: Maximum number of results
            filters: Optional filters for search

        Returns:
            List of search results with scores
        """
        if not self._initialized or not self.client:
            logger.warning("Vector storage not initialized")
            return []

        try:
            collection_name = f"{self.config.collection_name}_{collection_type}"

            # Build filter if provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(any=value))
                        )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                query_filter = Filter(must=conditions) if conditions else None

            # Perform search (use query_points for newer qdrant-client)
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=limit,
                query_filter=query_filter,
            )

            # Format results (qdrant-client may return a list or an object with `.points`).
            points = getattr(search_results, "points", search_results)
            results = []
            for result in points:
                results.append(
                    {
                        "id": getattr(result, "id", str(result)),
                        "score": getattr(result, "score", None),
                        "payload": getattr(result, "payload", {}),
                    }
                )

            return results

        except Exception as e:
            logger.error("Error performing semantic search in %s: %s", collection_name, e)
            return []

    async def find_similar_posts(
        self,
        query: str,
        query_embedding: List[float],
        platform: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find posts similar to query

        Args:
            query: Query text
            query_embedding: Query embedding vector
            platform: Optional platform filter
            limit: Maximum number of results

        Returns:
            List of similar posts
        """
        filters = {"platform": platform} if platform else None
        return await self.semantic_search(
            query_embedding=query_embedding,
            collection_type="posts",
            limit=limit,
            filters=filters,
        )

    async def find_similar_comments(
        self,
        query: str,
        query_embedding: List[float],
        author_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find comments similar to query

        Args:
            query: Query text
            query_embedding: Query embedding vector
            author_type: Optional author type filter
            limit: Maximum number of results

        Returns:
            List of similar comments
        """
        filters = {"author_type": author_type} if author_type else None
        return await self.semantic_search(
            query_embedding=query_embedding,
            collection_type="comments",
            limit=limit,
            filters=filters,
        )

    async def update_vector(
        self,
        vector_id: str,
        new_embedding: List[float],
        collection_type: str = "posts",
        payload_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector and optionally its payload

        Args:
            vector_id: Vector ID to update
            new_embedding: New embedding vector
            collection_type: Type of collection
            payload_updates: Optional payload updates

        Returns:
            bool: True if update successful
        """
        if not self._initialized or not self.client:
            logger.warning("Vector storage not initialized")
            return False

        try:
            collection_name = f"{self.config.collection_name}_{collection_type}"

            # Update vector
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=vector_id,
                        vector=new_embedding,
                        payload=payload_updates or {},
                    )
                ],
            )

            return True

        except Exception as e:
            logger.error("Error updating vector: %s", e)
            return False

    async def delete_vector(
        self, vector_id: str, collection_type: str = "posts"
    ) -> bool:
        """Delete a vector

        Args:
            vector_id: Vector ID to delete
            collection_type: Type of collection

        Returns:
            bool: True if deletion successful
        """
        if not self._initialized or not self.client:
            logger.warning("Vector storage not initialized")
            return False

        try:
            collection_name = f"{self.config.collection_name}_{collection_type}"
            self.client.delete(
                collection_name=collection_name,
                points_selector=[vector_id],
            )
            return True

        except Exception as e:
            logger.error("Error deleting vector: %s", e)
            return False

    async def close(self) -> None:
        """Close vector storage connection"""
        if self.client:
            # Qdrant client doesn't need explicit close for HTTP connection
            self.client = None
            self._initialized = False
            logger.info("Vector storage connection closed")

    async def get_collection_stats(
        self, collection_type: str = "posts"
    ) -> Optional[Dict[str, Any]]:
        """Get collection statistics

        Args:
            collection_type: Type of collection

        Returns:
            Collection statistics or None
        """
        if not self._initialized or not self.client:
            return None

        try:
            collection_name = f"{self.config.collection_name}_{collection_type}"
            collection_info = self.client.get_collection(collection_name)
            # Qdrant API changed - use points_count instead of vectors_count
            return {
                "collection_name": collection_name,
                "points_count": getattr(collection_info, "points_count", 0),
                "status": getattr(collection_info, "status", None),
            }
        except Exception as e:
            logger.error("Error getting collection stats: %s", e)
            return None

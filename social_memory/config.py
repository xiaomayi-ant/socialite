import logging
logger = logging.getLogger(__name__)
# -*- coding: utf-8 -*-
"""Configuration Management for Social Memory System"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStoreConfig(BaseModel):
    """Configuration for vector store (Qdrant)"""

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(default="social_posts", description="Collection name")
    vector_size: int = Field(default=768, description="Vector dimension")
    distance: str = Field(default="Cosine", description="Distance metric")

    # Storage strategy: "memory" or "disk"
    storage_strategy: str = Field(default="memory", description="Storage strategy")

    # HNSW configuration
    m: int = Field(default=16, description="HNSW M parameter")
    ef_construct: int = Field(default=100, description="HNSW ef_construct parameter")

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Create config from environment variables"""
        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "social_posts"),
            vector_size=int(os.getenv("EMBEDDING_MODEL_DIMENSION", "1024")),
        )


class StructuredStoreConfig(BaseModel):
    """Configuration for structured database (SQLite/PostgreSQL)"""

    # SQLite configuration
    sqlite_path: str = Field(
        default="./data/social_memory.db", description="SQLite database path"
    )

    # PostgreSQL configuration (optional)
    postgres_host: Optional[str] = Field(default=None, description="PostgreSQL host")
    postgres_port: Optional[int] = Field(default=None, description="PostgreSQL port")
    postgres_user: Optional[str] = Field(default=None, description="PostgreSQL user")
    postgres_password: Optional[str] = Field(
        default=None, description="PostgreSQL password"
    )
    postgres_db: Optional[str] = Field(default=None, description="PostgreSQL database")

    # Database type: "sqlite" or "postgres"
    db_type: str = Field(default="sqlite", description="Database type")

    # Storage strategy: "structured_only" or "hybrid"
    storage_strategy: str = Field(default="hybrid", description="Storage strategy")

    @classmethod
    def from_env(cls) -> "StructuredStoreConfig":
        """Create config from environment variables"""
        db_type = "postgres" if os.getenv("POSTGRES_HOST") else "sqlite"

        return cls(
            sqlite_path=os.getenv("SQLITE_DB_PATH", "./data/social_memory.db"),
            postgres_host=os.getenv("POSTGRES_HOST") or None,
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432"))
            if os.getenv("POSTGRES_HOST")
            else None,
            postgres_user=os.getenv("POSTGRES_USER") or None,
            postgres_password=os.getenv("POSTGRES_PASSWORD") or None,
            postgres_db=os.getenv("POSTGRES_DB") or None,
            db_type=db_type,
        )


class GraphStoreConfig(BaseModel):
    """Configuration for knowledge graph (Neo4j)"""

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="password", description="Neo4j password")
    database: str = Field(default="social_memory", description="Neo4j database name")

    # Connection pool settings
    max_connection_lifetime: int = Field(
        default=3600, description="Max connection lifetime"
    )
    max_connection_pool_size: int = Field(
        default=50, description="Max connection pool size"
    )
    connection_acquisition_timeout: int = Field(
        default=60, description="Connection acquisition timeout"
    )

    @classmethod
    def from_env(cls) -> "GraphStoreConfig":
        """Create config from environment variables"""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "social_memory"),
        )


class SocialMemoryConfig(BaseModel):
    """Main configuration for social memory system"""

    vector_store: VectorStoreConfig
    structured_store: StructuredStoreConfig
    graph_store: GraphStoreConfig

    # Supported platforms
    supported_platforms: list[str] = Field(
        default=["weibo", "zhihu", "xiaohongshu", "wechat"],
        description="Supported social platforms",
    )

    # Embedding configuration
    embedding_model_name: str = Field(
        default="", description="Embedding model name (placeholder)"
    )
    embedding_api_key: str = Field(
        default="", description="Embedding API key (placeholder)"
    )

    # Logging configuration
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(
        default="./logs/social_memory.log", description="Log file path"
    )

    # Performance settings
    enable_vectorization: bool = Field(default=True, description="Enable vectorization")
    enable_graph_analysis: bool = Field(
        default=True, description="Enable graph analysis"
    )
    enable_structured_search: bool = Field(
        default=True, description="Enable structured search"
    )

    @classmethod
    def from_env(cls) -> "SocialMemoryConfig":
        """Create complete config from environment variables"""
        return cls(
            vector_store=VectorStoreConfig.from_env(),
            structured_store=StructuredStoreConfig.from_env(),
            graph_store=GraphStoreConfig.from_env(),
            supported_platforms=os.getenv(
                "SUPPORTED_PLATFORMS", "weibo,zhihu,xiaohongshu,wechat"
            ).split(","),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "bge-m3"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "./logs/social_memory.log"),
        )

    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Validate required fields
            if self.enable_vectorization and not self.embedding_model_name:
                logger.warning("Warning: Vectorization enabled but no embedding model specified")

            # Validate platform list
            if not self.supported_platforms:
                raise ValueError("At least one platform must be supported")

            # Validate storage strategies
            valid_strategies = ["memory", "disk", "structured_only", "hybrid"]
            if self.vector_store.storage_strategy not in ["memory", "disk"]:
                raise ValueError(
                    f"Invalid vector storage strategy: {self.vector_store.storage_strategy}"
                )
            if self.structured_store.storage_strategy not in [
                "structured_only",
                "hybrid",
            ]:
                raise ValueError(
                    f"Invalid structured storage strategy: {self.structured_store.storage_strategy}"
                )

            return True

        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
            return False

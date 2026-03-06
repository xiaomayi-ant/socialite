# -*- coding: utf-8 -*-
"""Structured Storage Manager for Social Memory System"""

import logging
logger = logging.getLogger(__name__)

import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

try:
    import psycopg2
    from psycopg2 import pool
except ImportError:
    psycopg2 = None
    pool = None

from .config import StructuredStoreConfig
from .models import (
    SocialPost,
    SocialComment,
    SocialUser,
    EngagementMetrics,
)


class StructuredStorageManager:
    """Manager for structured database storage (SQLite/PostgreSQL)"""

    def __init__(self, config: StructuredStoreConfig):
        """Initialize structured storage manager

        Args:
            config: Structured store configuration
        """
        self.config = config
        self.connection_pool = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize database connection and create tables

        Returns:
            bool: True if initialization successful
        """
        try:
            if self.config.db_type == "sqlite":
                await self._init_sqlite()
            elif self.config.db_type == "postgres":
                await self._init_postgres()
            else:
                logger.error("Unsupported database type: %s", self.config.db_type)
                return False

            self._initialized = True
            logger.info("Structured storage initialized: %s", self.config.db_type)
            return True

        except Exception as e:
            logger.error("Failed to initialize structured storage: %s", e)
            return False

    async def _init_sqlite(self) -> None:
        """Initialize SQLite database"""
        # Ensure data directory exists
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        self._create_sqlite_tables()

    async def _init_postgres(self) -> None:
        """Initialize PostgreSQL database"""
        if psycopg2 is None:
            raise ImportError("psycopg2 not installed")

        # Create connection pool
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=20,
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            database=self.config.postgres_db,
        )

        # Create tables
        self._create_postgres_tables()

    def _create_sqlite_tables(self) -> None:
        """Create SQLite tables"""
        db_path = Path(self.config.sqlite_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
        tables = self._get_table_schemas()

        for table_name, schema in tables.items():
            cursor.execute(schema)
            logger.debug("Created SQLite table: %s", table_name)

        # Indexes for agent_messages
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_cycle ON agent_messages(cycle_count)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_sender ON agent_messages(sender)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_type ON agent_messages(message_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_direction ON agent_messages(direction)",
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_events_action ON rate_limit_events(action_type)",
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_events_reason ON rate_limit_events(reason)",
            "CREATE INDEX IF NOT EXISTS idx_behavior_log_action_time ON behavior_log(action_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_injection_events_cycle ON prompt_injection_events(cycle_count)",
            "CREATE INDEX IF NOT EXISTS idx_injection_events_agent ON prompt_injection_events(agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_injection_events_verdict ON prompt_injection_events(verdict)",
            "CREATE INDEX IF NOT EXISTS idx_injection_events_trace ON prompt_injection_events(trace_id)",
        ]:
            cursor.execute(idx_sql)

        self._create_sqlite_views(cursor)

        conn.commit()
        conn.close()

    def _create_sqlite_views(self, cursor: sqlite3.Cursor) -> None:
        """Create read-only observability views for cycle-level diagnostics."""
        views = [
            """
            CREATE VIEW IF NOT EXISTS vw_agent_message_direction_summary AS
            SELECT
                cycle_count,
                direction,
                COUNT(*) AS total
            FROM agent_messages
            GROUP BY cycle_count, direction
            """,
            """
            CREATE VIEW IF NOT EXISTS vw_agent_message_drop_summary AS
            SELECT
                cycle_count,
                message_type,
                COUNT(*) AS total
            FROM agent_messages
            WHERE direction = 'hub_drop'
            GROUP BY cycle_count, message_type
            """,
            """
            CREATE VIEW IF NOT EXISTS vw_rate_limit_cycle_summary AS
            SELECT
                json_extract(details, '$.cycle_count') AS cycle_count,
                action_type,
                allowed,
                reason,
                COUNT(*) AS total
            FROM rate_limit_events
            GROUP BY json_extract(details, '$.cycle_count'), action_type, allowed, reason
            """,
        ]
        for sql in views:
            cursor.execute(sql)

    def _create_postgres_tables(self) -> None:
        """Create PostgreSQL tables"""
        if not self.connection_pool:
            return

        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        # Create tables
        tables = self._get_table_schemas()

        for table_name, schema in tables.items():
            cursor.execute(schema)
            logger.debug("Created PostgreSQL table: %s", table_name)

        conn.commit()
        self.connection_pool.putconn(conn)

    def _get_table_schemas(self) -> Dict[str, str]:
        """Get SQL table schemas (optimized to 9 tables from 13)"""
        return {
            # ════ Layer 1: Community Data (3 tables) ════
            "social_users": """
                CREATE TABLE IF NOT EXISTS social_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    karma INTEGER DEFAULT 0,
                    profile_url TEXT,
                    followers_count INTEGER,
                    following_count INTEGER,
                    is_verified BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "social_posts": """
                CREATE TABLE IF NOT EXISTS social_posts (
                    post_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    sentiment TEXT,
                    topics TEXT,
                    upvotes INTEGER DEFAULT 0,
                    downvotes INTEGER DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    engagement_timestamp TIMESTAMP,

                    -- ← New fields for optimization
                    is_high_value BOOLEAN DEFAULT FALSE,
                    value_score REAL DEFAULT 0.0,
                    quality_with_karma REAL DEFAULT 0.0,

                    -- ← Semantic analysis fields
                    is_compliant BOOLEAN DEFAULT TRUE,
                    violation_type TEXT,
                    semantic_value REAL DEFAULT 0.5,
                    has_unique_perspective BOOLEAN DEFAULT FALSE,
                    translation_zh TEXT,

                    vectorized BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES social_users(user_id)
                )
            """,
            "social_comments": """
                CREATE TABLE IF NOT EXISTS social_comments (
                    comment_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    target_post_id TEXT NOT NULL,
                    target_post_content TEXT,
                    author_id TEXT NOT NULL,
                    author_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    sentiment TEXT,
                    reply_to_comment_id TEXT,
                    upvotes INTEGER DEFAULT 0,
                    downvotes INTEGER DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    engagement_timestamp TIMESTAMP,
                    learning_value REAL DEFAULT 0.0,

                    -- ← New fields for optimization
                    is_high_value BOOLEAN DEFAULT FALSE,
                    quality_score REAL DEFAULT 0.0,

                    vectorized BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES social_users(user_id)
                )
            """,
            # ════ Layer 2: Own Posts (3 tables) ════
            "own_posts": """
                CREATE TABLE IF NOT EXISTS own_posts (
                    post_id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    submolt TEXT,
                    topic TEXT,
                    pattern_used TEXT,
                    evolution_stage TEXT,
                    learning_value REAL,
                    novelty_score REAL,
                    upvotes INTEGER DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "own_comments": """
                CREATE TABLE IF NOT EXISTS own_comments (
                    comment_id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL,
                    comment_text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    upvotes_current INTEGER DEFAULT 0,
                    upvotes_history TEXT DEFAULT '{}',
                    success BOOLEAN DEFAULT FALSE,
                    feedback_score REAL DEFAULT 0.0,
                    metadata TEXT
                )
            """,
            "feedback_snapshots": """
                CREATE TABLE IF NOT EXISTS feedback_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    snapshot_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    hours_after INTEGER,
                    upvotes INTEGER DEFAULT 0,
                    downvotes INTEGER DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    FOREIGN KEY (post_id) REFERENCES own_posts(post_id)
                )
            """,
            # ════ Layer 3: Learning & Patterns ════
            "style_patterns": """
                CREATE TABLE IF NOT EXISTS style_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    example_post_ids TEXT DEFAULT '[]',
                    topics TEXT DEFAULT '[]',
                    avg_upvotes REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    decay_factor REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "identity_snapshots": """
                CREATE TABLE IF NOT EXISTS identity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    stance TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    post_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            # ════ Layer 4: State Tracking (1 table) ════
            "evolution_state": """
                CREATE TABLE IF NOT EXISTS evolution_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    exploration_rate REAL DEFAULT 0.5,
                    learning_progress REAL DEFAULT 0.0,
                    metrics TEXT DEFAULT '{}',
                    best_metrics TEXT DEFAULT '{}',
                    strategy_notes TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            # ════ Auxiliary Tables ════
            "api_costs": """
                CREATE TABLE IF NOT EXISTS api_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    agent_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    estimated_cost_usd REAL DEFAULT 0.0,
                    action_type TEXT
                )
            """,
            "behavior_log": """
                CREATE TABLE IF NOT EXISTS behavior_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT NOT NULL,
                    target_id TEXT,
                    strategy_used TEXT,
                    proposal_id TEXT,
                    metadata TEXT
                )
            """,
            "rate_limit_events": """
                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT NOT NULL,
                    target_id TEXT,
                    allowed BOOLEAN DEFAULT 0,
                    reason TEXT,
                    details TEXT
                )
            """,
            "agent_proposals": """
                CREATE TABLE IF NOT EXISTS agent_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cycle_count INTEGER,
                    agent_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_id TEXT,
                    priority REAL DEFAULT 0.5,
                    reasoning TEXT,
                    strategy TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT
                )
            """,
            "agent_messages": """
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    cycle_count INTEGER,
                    sender TEXT NOT NULL,
                    receiver TEXT,
                    direction TEXT NOT NULL,
                    message_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "prompt_injection_events": """
                CREATE TABLE IF NOT EXISTS prompt_injection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    message_id TEXT,
                    trace_id TEXT,
                    cycle_count INTEGER,
                    agent_name TEXT NOT NULL,
                    sender TEXT,
                    source_type TEXT DEFAULT 'unknown',
                    trust_level TEXT DEFAULT 'untrusted',
                    injection_score REAL DEFAULT 0.0,
                    verdict TEXT NOT NULL,
                    reasons TEXT DEFAULT '[]',
                    action_taken TEXT DEFAULT 'observe',
                    excerpt TEXT,
                    metadata TEXT
                )
            """,
            "observer_reports": """
                CREATE TABLE IF NOT EXISTS observer_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cycle_count INTEGER,
                    total_actions INTEGER DEFAULT 0,
                    strategy_comparison TEXT DEFAULT '{}',
                    action_breakdown TEXT DEFAULT '{}',
                    anomalies TEXT DEFAULT '[]',
                    metadata TEXT
                )
            """,
        }

    async def record_post(self, post: SocialPost) -> bool:
        """Record a post to structured storage

        Args:
            post: Social post object

        Returns:
            bool: True if recording successful
        """
        try:
            if self.config.db_type == "sqlite":
                return self._record_post_sqlite(post)
            elif self.config.db_type == "postgres":
                return await self._record_post_postgres(post)
            return False
        except Exception as e:
            logger.error("Error recording post: %s", e)
            return False

    def _record_post_sqlite(self, post: SocialPost) -> bool:
        """Record post to SQLite"""
        db_path = Path(self.config.sqlite_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # First ensure user exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO social_users
                (user_id, username, platform, followers_count, following_count, is_verified, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    post.author.user_id,
                    post.author.username,
                    post.author.platform.value,
                    post.author.followers_count,
                    post.author.following_count,
                    post.author.verified,
                    json.dumps(post.author.metadata) if post.author.metadata else None,
                ),
            )

            # Record post
            cursor.execute(
                """
                INSERT OR REPLACE INTO social_posts
                (post_id, content, platform, author_id, timestamp, sentiment, topics,
                 upvotes, downvotes, comment_count, engagement_timestamp, vectorized, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    post.post_id,
                    post.content,
                    post.platform.value,
                    post.author.user_id,
                    post.timestamp.isoformat(),
                    post.sentiment.value if post.sentiment else None,
                    json.dumps(post.topics) if post.topics else None,
                    post.engagement.upvotes,
                    post.engagement.downvotes,
                    post.engagement.comment_count,
                    post.engagement.timestamp.isoformat(),
                    post.embedding is not None,
                    json.dumps(post.metadata) if post.metadata else None,
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error("SQLite error recording post: %s", e)
            conn.rollback()
            return False
        finally:
            conn.close()

    async def _record_post_postgres(self, post: SocialPost) -> bool:
        """Record post to PostgreSQL"""
        if not self.connection_pool:
            return False

        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        try:
            # Ensure user exists
            cursor.execute(
                """
                INSERT INTO social_users (user_id, username, platform, followers_count, following_count, is_verified, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    followers_count = EXCLUDED.followers_count,
                    following_count = EXCLUDED.following_count,
                    is_verified = EXCLUDED.is_verified,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (
                    post.author.user_id,
                    post.author.username,
                    post.author.platform.value,
                    post.author.followers_count,
                    post.author.following_count,
                    post.author.verified,
                    json.dumps(post.author.metadata) if post.author.metadata else None,
                ),
            )

            # Record post
            cursor.execute(
                """
                INSERT INTO social_posts
                (post_id, content, platform, author_id, timestamp, sentiment, topics,
                 upvotes, downvotes, comment_count, engagement_timestamp, vectorized, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (post_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    upvotes = EXCLUDED.upvotes,
                    downvotes = EXCLUDED.downvotes,
                    comment_count = EXCLUDED.comment_count,
                    vectorized = EXCLUDED.vectorized,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (
                    post.post_id,
                    post.content,
                    post.platform.value,
                    post.author.user_id,
                    post.timestamp.isoformat(),
                    post.sentiment.value if post.sentiment else None,
                    json.dumps(post.topics) if post.topics else None,
                    post.engagement.upvotes,
                    post.engagement.downvotes,
                    post.engagement.comment_count,
                    post.engagement.timestamp.isoformat(),
                    post.embedding is not None,
                    json.dumps(post.metadata) if post.metadata else None,
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error("PostgreSQL error recording post: %s", e)
            conn.rollback()
            return False
        finally:
            self.connection_pool.putconn(conn)

    async def record_comment(self, comment: SocialComment) -> bool:
        """Record a comment to structured storage

        Args:
            comment: Social comment object

        Returns:
            bool: True if recording successful
        """
        try:
            if self.config.db_type == "sqlite":
                return self._record_comment_sqlite(comment)
            elif self.config.db_type == "postgres":
                return await self._record_comment_postgres(comment)
            return False
        except Exception as e:
            logger.error("Error recording comment: %s", e)
            return False

    def _record_comment_sqlite(self, comment: SocialComment) -> bool:
        """Record comment to SQLite"""
        db_path = Path(self.config.sqlite_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Ensure user exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO social_users
                (user_id, username, platform, followers_count, following_count, is_verified, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    comment.author.user_id,
                    comment.author.username,
                    comment.author.platform.value,
                    comment.author.followers_count,
                    comment.author.following_count,
                    comment.author.verified,
                    json.dumps(comment.author.metadata)
                    if comment.author.metadata
                    else None,
                ),
            )

            # Record comment
            cursor.execute(
                """
                INSERT OR REPLACE INTO social_comments
                (comment_id, content, platform, target_post_id, target_post_content,
                 author_id, author_type, timestamp, sentiment, reply_to_comment_id,
                 upvotes, downvotes, comment_count, engagement_timestamp,
                 learning_value, vectorized, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    comment.comment_id,
                    comment.content,
                    comment.platform.value,
                    comment.target_post_id,
                    comment.target_post_content,
                    comment.author.user_id,
                    comment.author_type.value,
                    comment.timestamp.isoformat(),
                    comment.sentiment.value if comment.sentiment else None,
                    comment.reply_to_comment_id,
                    comment.engagement.upvotes,
                    comment.engagement.downvotes,
                    comment.engagement.comment_count,
                    comment.engagement.timestamp.isoformat(),
                    comment.learning_value,
                    comment.embedding is not None,
                    json.dumps(comment.metadata) if comment.metadata else None,
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error("SQLite error recording comment: %s", e)
            conn.rollback()
            return False
        finally:
            conn.close()

    async def _record_comment_postgres(self, comment: SocialComment) -> bool:
        """Record comment to PostgreSQL"""
        if not self.connection_pool:
            return False

        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        try:
            # Ensure user exists
            cursor.execute(
                """
                INSERT INTO social_users (user_id, username, platform, followers_count, following_count, is_verified, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    followers_count = EXCLUDED.followers_count,
                    following_count = EXCLUDED.following_count,
                    is_verified = EXCLUDED.is_verified,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (
                    comment.author.user_id,
                    comment.author.username,
                    comment.author.platform.value,
                    comment.author.followers_count,
                    comment.author.following_count,
                    comment.author.verified,
                    json.dumps(comment.author.metadata)
                    if comment.author.metadata
                    else None,
                ),
            )

            # Record comment
            cursor.execute(
                """
                INSERT INTO social_comments
                (comment_id, content, platform, target_post_id, target_post_content,
                 author_id, author_type, timestamp, sentiment, reply_to_comment_id,
                 upvotes, downvotes, comment_count, engagement_timestamp,
                 learning_value, vectorized, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (comment_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    upvotes = EXCLUDED.upvotes,
                    downvotes = EXCLUDED.downvotes,
                    learning_value = EXCLUDED.learning_value,
                    vectorized = EXCLUDED.vectorized,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (
                    comment.comment_id,
                    comment.content,
                    comment.platform.value,
                    comment.target_post_id,
                    comment.target_post_content,
                    comment.author.user_id,
                    comment.author_type.value,
                    comment.timestamp.isoformat(),
                    comment.sentiment.value if comment.sentiment else None,
                    comment.reply_to_comment_id,
                    comment.engagement.upvotes,
                    comment.engagement.downvotes,
                    comment.engagement.comment_count,
                    comment.engagement.timestamp.isoformat(),
                    comment.learning_value,
                    comment.embedding is not None,
                    json.dumps(comment.metadata) if comment.metadata else None,
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error("PostgreSQL error recording comment: %s", e)
            conn.rollback()
            return False
        finally:
            self.connection_pool.putconn(conn)

    async def query_post_history(
        self, platform: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query post history

        Args:
            platform: Optional platform filter
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of posts
        """
        try:
            if self.config.db_type == "sqlite":
                return self._query_post_history_sqlite(platform, limit, offset)
            elif self.config.db_type == "postgres":
                return await self._query_post_history_postgres(platform, limit, offset)
            return []
        except Exception as e:
            logger.error("Error querying post history: %s", e)
            return []

    def _query_post_history_sqlite(
        self, platform: Optional[str], limit: int, offset: int
    ) -> List[Dict[str, Any]]:
        """Query post history from SQLite"""
        db_path = Path(self.config.sqlite_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM social_posts"
            params = []

            if platform:
                query += " WHERE platform = ?"
                params.append(platform)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dicts
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))

            return result

        except Exception as e:
            logger.error("SQLite error querying post history: %s", e)
            return []
        finally:
            conn.close()

    async def _query_post_history_postgres(
        self, platform: Optional[str], limit: int, offset: int
    ) -> List[Dict[str, Any]]:
        """Query post history from PostgreSQL"""
        if not self.connection_pool:
            return []

        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM social_posts"
            params = []

            if platform:
                query += " WHERE platform = %s"
                params.append(platform)

            query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dicts
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))

            return result

        except Exception as e:
            logger.error("PostgreSQL error querying post history: %s", e)
            return []
        finally:
            self.connection_pool.putconn(conn)

    async def get_engagement_stats(
        self, platform: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get engagement statistics

        Args:
            platform: Optional platform filter
            days: Number of days to analyze

        Returns:
            Dictionary of engagement statistics
        """
        # This is a placeholder implementation
        # In production, this would perform actual aggregation queries
        return {
            "total_upvotes": 0,
            "total_downvotes": 0,
            "total_comment_count": 0,
            "average_engagement": 0.0,
            "top_performing_posts": [],
        }

    # ── New table helpers for multi-agent architecture ──

    def _get_sqlite_conn(self):
        """Get a SQLite connection"""
        return sqlite3.connect(Path(self.config.sqlite_path))

    async def record_own_post(self, data: Dict[str, Any]) -> bool:
        """Record a post created by this agent"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO own_posts
                   (post_id, title, content, submolt, topic, pattern_used,
                    evolution_stage, learning_value, novelty_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["post_id"], data.get("title"), data.get("content"),
                    data.get("submolt"), data.get("topic"), data.get("pattern_used"),
                    data.get("evolution_stage"), data.get("learning_value"),
                    data.get("novelty_score"), data.get("created_at", datetime.now().isoformat()),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording own post: %s", e)
            return False

    async def record_feedback_snapshot(self, data: Dict[str, Any]) -> bool:
        """Record a feedback snapshot for performance tracking"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO feedback_snapshots
                   (post_id, hours_after, upvotes, downvotes, comment_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data["post_id"], data.get("hours_after", 0),
                    data.get("upvotes", 0), data.get("downvotes", 0),
                    data.get("comment_count", 0),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording feedback snapshot: %s", e)
            return False

    async def upsert_style_pattern(self, data: Dict[str, Any]) -> bool:
        """Insert or update a style pattern"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO style_patterns
                   (pattern_id, description, example_post_ids, topics,
                    avg_upvotes, usage_count, success_rate, decay_factor,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["pattern_id"], data.get("description"),
                    json.dumps(data.get("example_post_ids", [])),
                    json.dumps(data.get("topics", [])),
                    data.get("avg_upvotes", 0.0), data.get("usage_count", 0),
                    data.get("success_rate", 0.0), data.get("decay_factor", 1.0),
                    data.get("created_at", datetime.now().isoformat()),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error upserting style pattern: %s", e)
            return False

    async def get_style_patterns(
        self, limit: int = 20, min_success_rate: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get style patterns ordered by effectiveness"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM style_patterns
                   WHERE success_rate >= ? AND decay_factor > 0.1
                   ORDER BY (success_rate * decay_factor) DESC
                   LIMIT ?""",
                (min_success_rate, limit),
            )
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error("Error getting style patterns: %s", e)
            return []

    async def record_evolution_state(self, data: Dict[str, Any]) -> bool:
        """Record an evolution state snapshot"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO evolution_state
                   (stage, exploration_rate, metrics, best_metrics, strategy_notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data["stage"], data.get("exploration_rate", 0.5),
                    json.dumps(data.get("metrics", {})),
                    json.dumps(data.get("best_metrics", {})),
                    data.get("strategy_notes"),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording evolution state: %s", e)
            return False

    async def get_latest_evolution_state(self) -> Optional[Dict[str, Any]]:
        """Get the most recent evolution state"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM evolution_state ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                columns = [d[0] for d in cursor.description]
                result = dict(zip(columns, row))
                result["metrics"] = json.loads(result.get("metrics") or "{}")
                result["best_metrics"] = json.loads(result.get("best_metrics") or "{}")
                conn.close()
                return result
            conn.close()
            return None
        except Exception as e:
            logger.error("Error getting evolution state: %s", e)
            return None

    async def record_identity_snapshot(self, data: Dict[str, Any]) -> bool:
        """Record an identity snapshot"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO identity_snapshots (post_id, topic, stance)
                   VALUES (?, ?, ?)""",
                (data.get("post_id"), data.get("topic"), data.get("stance")),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording identity snapshot: %s", e)
            return False

    async def get_identity_snapshots(
        self, topic: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent identity snapshots, optionally filtered by topic"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            if topic:
                cursor.execute(
                    """SELECT * FROM identity_snapshots
                       WHERE topic = ? ORDER BY id DESC LIMIT ?""",
                    (topic, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM identity_snapshots ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error("Error getting identity snapshots: %s", e)
            return []

    async def record_api_cost(self, data: Dict[str, Any]) -> bool:
        """Record an API cost entry"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO api_costs
                   (agent_name, model, input_tokens, output_tokens,
                    estimated_cost_usd, action_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    data.get("agent_name"), data.get("model"),
                    data.get("input_tokens", 0), data.get("output_tokens", 0),
                    data.get("estimated_cost_usd", 0.0), data.get("action_type"),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording API cost: %s", e)
            return False

    async def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get total API cost for a given date (default: today)"""
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COALESCE(SUM(estimated_cost_usd), 0.0)
                   FROM api_costs WHERE DATE(timestamp) = ?""",
                (date,),
            )
            cost = cursor.fetchone()[0]
            conn.close()
            return cost
        except Exception as e:
            logger.error("Error getting daily cost: %s", e)
            return 0.0

    async def record_behavior(self, data: Dict[str, Any]) -> bool:
        """Record a behavior log entry"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO behavior_log
                   (action_type, target_id, metadata, strategy_used, proposal_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data.get("action_type"), data.get("target_id"),
                    json.dumps(data.get("metadata", {})),
                    data.get("strategy_used"), data.get("proposal_id"),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording behavior: %s", e)
            return False

    async def get_daily_action_count(self, action_type: str, date: Optional[str] = None) -> int:
        """Get action count for a specific type on a given day (default: today)."""
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COUNT(*)
                   FROM behavior_log
                   WHERE action_type = ? AND DATE(timestamp) = ?""",
                (action_type, date),
            )
            count = cursor.fetchone()[0] or 0
            conn.close()
            return int(count)
        except Exception as e:
            logger.error("Error getting daily action count: %s", e)
            return 0

    async def record_rate_limit_event(self, data: Dict[str, Any]) -> bool:
        """Record allow/deny decisions from rate limit policy."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO rate_limit_events
                   (action_type, target_id, allowed, reason, details)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data.get("action_type", ""),
                    data.get("target_id", ""),
                    1 if data.get("allowed", False) else 0,
                    data.get("reason", ""),
                    json.dumps(data.get("details", {})),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording rate limit event: %s", e)
            return False

    async def get_rate_limit_events(
        self, cycle_count: Optional[int] = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Query rate limit events with optional cycle filter.

        Note: cycle filtering is derived from details.cycle_count when available.
        """
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, timestamp, action_type, target_id, allowed, reason, details
                   FROM rate_limit_events
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            conn.close()

            result: List[Dict[str, Any]] = []
            for row in rows:
                details_raw = row[6]
                details = json.loads(details_raw) if details_raw else {}
                item = {
                    "id": row[0],
                    "timestamp": row[1],
                    "action_type": row[2],
                    "target_id": row[3],
                    "allowed": bool(row[4]),
                    "reason": row[5],
                    "details": details,
                }
                if cycle_count is not None:
                    if int(details.get("cycle_count", -1)) != int(cycle_count):
                        continue
                result.append(item)
            return result
        except Exception as e:
            logger.error("Error querying rate limit events: %s", e)
            return []

    async def get_own_post_ids(self, limit: int = 50) -> List[str]:
        """Get IDs of own posts for performance tracking"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT post_id FROM own_posts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            return ids
        except Exception as e:
            logger.error("Error getting own post IDs: %s", e)
            return []

    # ── Optimized table operations (v0.2.1+) ──

    async def record_own_comment(self, data: Dict[str, Any]) -> bool:
        """Record a comment created by this agent (new own_comments table)"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO own_comments
                   (comment_id, post_id, comment_text, created_at,
                    upvotes_current, upvotes_history, success, feedback_score, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data.get("comment_id"),
                    data.get("post_id"),
                    data.get("comment_text"),
                    data.get("created_at", datetime.now().isoformat()),
                    data.get("upvotes_current", 0),
                    data.get("upvotes_history", "{}"),
                    data.get("success", False),
                    data.get("feedback_score", 0.0),
                    json.dumps(data.get("metadata", {})),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording own comment: %s", e)
            return False

    async def get_own_comments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent own comments for feedback analysis"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT comment_id, post_id, comment_text, created_at,
                           upvotes_current, success, feedback_score
                   FROM own_comments
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()

            result = []
            for row in rows:
                result.append({
                    "comment_id": row[0],
                    "post_id": row[1],
                    "comment_text": row[2],
                    "created_at": row[3],
                    "upvotes_current": row[4],
                    "success": row[5],
                    "feedback_score": row[6],
                })
            return result
        except Exception as e:
            logger.error("Error getting own comments: %s", e)
            return []

    async def update_own_comment_feedback(self, comment_id: str, upvotes: int, success: bool) -> bool:
        """Update feedback score for an own comment based on upvote changes"""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            # Get previous upvotes
            cursor.execute("SELECT upvotes_current, upvotes_history FROM own_comments WHERE comment_id = ?", (comment_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False

            prev_upvotes, history_str = row
            history = json.loads(history_str or "{}")

            # Update history
            cycle_num = len(history) + 1
            history[f"cycle_{cycle_num}"] = upvotes

            # Simple feedback: increase if gaining upvotes
            feedback_delta = upvotes - (prev_upvotes or 0)

            cursor.execute(
                """UPDATE own_comments
                   SET upvotes_current = ?, upvotes_history = ?, success = ?,
                       feedback_score = feedback_score + ?
                   WHERE comment_id = ?""",
                (upvotes, json.dumps(history), success, max(0, feedback_delta * 0.1), comment_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error updating comment feedback: %s", e)
            return False

    async def get_high_value_posts(
        self, limit: int = 50, min_value_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Get high-value community posts by engagement metrics.

        Uses direct engagement-based ranking instead of requiring pre-marked
        is_high_value flag. Posts are scored by upvotes and comment_count.
        """
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT *, (upvotes + comment_count * 2) AS engagement_score
                   FROM social_posts
                   WHERE upvotes >= 5
                   ORDER BY engagement_score DESC, timestamp DESC
                   LIMIT ?""",
                (limit,),
            )
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error("Error getting high-value posts: %s", e)
            return []

    async def get_high_value_comments(
        self, limit: int = 50, min_quality_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Get high-value community comments by engagement metrics.

        Uses direct upvote-based ranking instead of requiring pre-marked
        is_high_value flag.
        """
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT *
                   FROM social_comments
                   WHERE upvotes >= 3 AND author_type = 'other'
                   ORDER BY upvotes DESC, timestamp DESC
                   LIMIT ?""",
                (limit,),
            )
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error("Error getting high-value comments: %s", e)
            return []

    # ═══════════════════════════════════════════════════════════
    # Socialite v0.4.0 — New table methods
    # ═══════════════════════════════════════════════════════════

    async def record_agent_proposal(self, data: Dict[str, Any]) -> bool:
        """Record an agent proposal to the agent_proposals table."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO agent_proposals
                   (cycle_count, agent_name, action, target_id, priority,
                    reasoning, strategy, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data.get("cycle_count"),
                    data.get("agent_name", ""),
                    data.get("action", ""),
                    data.get("target_id", ""),
                    data.get("priority", 0.5),
                    data.get("reasoning", ""),
                    data.get("strategy", "A"),
                    data.get("status", "pending"),
                    json.dumps(data.get("metadata", {})),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording agent proposal: %s", e)
            return False

    async def record_observer_report(self, data: Dict[str, Any]) -> bool:
        """Record an observer report."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO observer_reports
                   (cycle_count, total_actions, strategy_comparison,
                    action_breakdown, anomalies, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    data.get("cycle_count"),
                    data.get("total_actions", 0),
                    json.dumps(data.get("strategy_comparison", {})),
                    json.dumps(data.get("action_breakdown", {})),
                    json.dumps(data.get("anomalies", [])),
                    json.dumps(data.get("metadata", {})),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording observer report: %s", e)
            return False

    # ═══════════════════════════════════════════════════════════
    # Agent message persistence
    # ═══════════════════════════════════════════════════════════

    async def record_agent_message(self, data: Dict[str, Any]) -> bool:
        """Record an agent message to the agent_messages table."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO agent_messages
                   (message_id, cycle_count, sender, receiver, direction,
                    message_type, content, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data.get("message_id", ""),
                    data.get("cycle_count"),
                    data.get("sender", ""),
                    data.get("receiver"),
                    data.get("direction", "send"),
                    data.get("message_type"),
                    data.get("content", "")[:500],
                    json.dumps(data.get("metadata", {})) if data.get("metadata") else None,
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording agent message: %s", e)
            return False

    async def get_agent_messages(
        self,
        cycle_count: Optional[int] = None,
        sender: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query agent messages with optional filters."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()

            query = "SELECT * FROM agent_messages WHERE 1=1"
            params: list = []

            if cycle_count is not None:
                query += " AND cycle_count = ?"
                params.append(cycle_count)
            if sender is not None:
                query += " AND sender = ?"
                params.append(sender)
            if message_type is not None:
                query += " AND message_type = ?"
                params.append(message_type)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            rows = []
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                if row_dict.get("metadata"):
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                rows.append(row_dict)

            conn.close()
            return rows
        except Exception as e:
            logger.error("Error querying agent messages: %s", e)
            return []

    async def get_cycle_observability_summary(self, cycle_count: int) -> Dict[str, Any]:
        """Get cycle-level observability summary from SQL views."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()

            cursor.execute(
                """SELECT direction, total
                   FROM vw_agent_message_direction_summary
                   WHERE cycle_count = ?""",
                (cycle_count,),
            )
            direction_rows = cursor.fetchall()

            cursor.execute(
                """SELECT message_type, total
                   FROM vw_agent_message_drop_summary
                   WHERE cycle_count = ?""",
                (cycle_count,),
            )
            drop_rows = cursor.fetchall()

            cursor.execute(
                """SELECT action_type, allowed, reason, total
                   FROM vw_rate_limit_cycle_summary
                   WHERE cycle_count = ?""",
                (cycle_count,),
            )
            rate_limit_rows = cursor.fetchall()

            conn.close()

            return {
                "cycle_count": cycle_count,
                "directions": {row[0]: int(row[1]) for row in direction_rows},
                "drop_types": {row[0]: int(row[1]) for row in drop_rows},
                "rate_limit": [
                    {
                        "action_type": row[0],
                        "allowed": bool(row[1]),
                        "reason": row[2] or "",
                        "total": int(row[3]),
                    }
                    for row in rate_limit_rows
                ],
            }
        except Exception as e:
            logger.error("Error querying cycle observability summary: %s", e)
            return {"cycle_count": cycle_count, "directions": {}, "drop_types": {}, "rate_limit": []}

    async def record_prompt_injection_event(self, data: Dict[str, Any]) -> bool:
        """Record a prompt-injection suspicion/judgement event for auditing."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO prompt_injection_events
                   (message_id, trace_id, cycle_count, agent_name, sender,
                    source_type, trust_level, injection_score, verdict,
                    reasons, action_taken, excerpt, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data.get("message_id", ""),
                    data.get("trace_id", ""),
                    data.get("cycle_count"),
                    data.get("agent_name", ""),
                    data.get("sender"),
                    data.get("source_type", "unknown"),
                    data.get("trust_level", "untrusted"),
                    float(data.get("injection_score", 0.0)),
                    data.get("verdict", "suspected"),
                    json.dumps(data.get("reasons", [])),
                    data.get("action_taken", "observe"),
                    (data.get("excerpt", "") or "")[:1000],
                    json.dumps(data.get("metadata", {})),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Error recording prompt injection event: %s", e)
            return False

    async def get_prompt_injection_events(
        self,
        cycle_count: Optional[int] = None,
        agent_name: Optional[str] = None,
        verdict: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Query prompt-injection audit events."""
        try:
            conn = self._get_sqlite_conn()
            cursor = conn.cursor()

            query = "SELECT * FROM prompt_injection_events WHERE 1=1"
            params: List[Any] = []

            if cycle_count is not None:
                query += " AND cycle_count = ?"
                params.append(cycle_count)
            if agent_name is not None:
                query += " AND agent_name = ?"
                params.append(agent_name)
            if verdict is not None:
                query += " AND verdict = ?"
                params.append(verdict)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            rows: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                item = dict(zip(columns, row))
                item["injection_score"] = float(item.get("injection_score", 0.0) or 0.0)
                reasons_raw = item.get("reasons")
                metadata_raw = item.get("metadata")
                item["reasons"] = json.loads(reasons_raw) if reasons_raw else []
                item["metadata"] = json.loads(metadata_raw) if metadata_raw else {}
                rows.append(item)

            conn.close()
            return rows
        except Exception as e:
            logger.error("Error querying prompt injection events: %s", e)
            return []

    async def close(self) -> None:
        """Close database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
        self._initialized = False
        logger.info("Structured storage connection closed")

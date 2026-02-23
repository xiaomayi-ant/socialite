# -*- coding: utf-8 -*-
"""Socialite v0.4.0 Configuration"""

import os

SYSTEM_CONFIG = {
    "system": {
        "name": "Socialite Multi-Agent System",
        "version": "0.4.0",
        "debug": False,
    },
    "model": {
        "provider": "anthropic",
        "haiku_model": "claude-haiku-4-5-20251001",
        "sonnet_model": "claude-sonnet-4-20250514",
    },
    "budget": {
        "daily_limit_usd": float(os.getenv("DAILY_BUDGET_USD", "5.0")),
    },
    "social_memory": {
        "vector_store": {
            "enabled": True,
            "host": "localhost",
            "port": 6333,
            "collection_name": "social_posts",
            "vector_size": 1024,
            "distance": "Cosine",
            "storage_strategy": "disk",
        },
        "structured_store": {
            "enabled": True,
            "db_type": "sqlite",
            "sqlite_path": "./data/social_memory.db",
        },
        "graph_store": {
            "enabled": True,
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "your_password"),
            "database": "social_memory",
        },
    },
    "agents": {
        "sensor": {"name": "Sensor", "llm": None},
        "analysis": {"name": "Analysis", "llm": "claude-haiku"},
        "comment": {"name": "Comment", "llm": "claude-haiku"},
        "post": {"name": "Post", "llm": "claude-haiku"},
        "upvote": {"name": "Upvote", "llm": None},
        "follow": {"name": "Follow", "llm": None},
        "coordinator": {"name": "Coordinator", "llm": None},
        "learner": {"name": "Learner", "llm": "claude-haiku"},
        "observer": {"name": "Observer", "llm": None},
    },
    "ab_strategy": {
        "mode": "alternate",  # "alternate" | "probability"
        "p_a": 0.5,
    },
    "evolution": {
        "stages": ["initial", "exploration", "optimization", "innovation"],
        "cosine_annealing_period": 20,
        "plateau_patience": 5,
        "plateau_factor": 0.5,
        "forgetting_threshold": 0.8,
    },
    "vectorization": {
        "embed_feed_min_upvotes": 2,
        "embed_own_posts": True,
        "embed_comments": False,
        "embed_high_engagement_threshold": 10,
    },
    "quality_scoring": {
        "karma_weight_enabled": True,
        "time_decay_enabled": True,
        "time_decay_lambda": 0.15,
        "quality_calc_method": "with_karma",
    },
    "learning": {
        "learning_progress_threshold": 0.6,
        "pattern_confidence_threshold": 0.7,
        "high_value_post_threshold": 0.65,
        "high_value_comment_threshold": 0.6,
    },
    "safety": {
        "version": "v1",
        "mode": "observe_only",
        "interval_jitter_range": 0.3,
        "action_delay_range": [1.5, 8.0],
        "max_comments_per_cycle": 3,
    },
    "logging": {
        "level": "INFO",
        "log_file": "./logs/runner.log",
    },
}

# Evolution Stages
EVOLUTION_STAGES = [
    "initial",
    "exploration",
    "optimization",
    "innovation",
]

EVOLUTION_METRICS = {
    "engagement_rate": {
        "description": "Engagement rate: (upvotes + comments) / posts",
        "target": 0.3,
        "weight": 0.4,
    },
    "response_quality": {
        "description": "Response quality based on learning_value",
        "target": 0.7,
        "weight": 0.3,
    },
    "diversity_score": {
        "description": "Content diversity across topics",
        "target": 0.5,
        "weight": 0.2,
    },
    "pattern_success_rate": {
        "description": "Success rate of applied patterns",
        "target": 0.8,
        "weight": 0.1,
    },
}

LEARNING_WEIGHTS = {
    "initial": {
        "community_posts": 0.4,
        "community_comments": 0.3,
        "discussion_depth": 0.2,
        "self_feedback": 0.1,
    },
    "exploration": {
        "community_posts": 0.3,
        "community_comments": 0.25,
        "discussion_depth": 0.25,
        "self_feedback": 0.2,
    },
    "optimization": {
        "community_posts": 0.2,
        "community_comments": 0.2,
        "discussion_depth": 0.2,
        "self_feedback": 0.4,
    },
    "innovation": {
        "community_posts": 0.15,
        "community_comments": 0.15,
        "discussion_depth": 0.2,
        "self_feedback": 0.5,
    },
}

FEEDBACK_POLL_DELAYS_HOURS = [0.5, 6, 24, 48]

LEARNING_VALUE_THRESHOLDS = {
    "very_low": 0.3,
    "low": 0.5,
    "medium": 0.7,
    "high": 0.8,
    "very_high": 0.9,
}

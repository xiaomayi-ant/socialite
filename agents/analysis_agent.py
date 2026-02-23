# -*- coding: utf-8 -*-
"""AnalysisAgent — data enrichment publisher.

Subscribes to SensorAgent's raw_feed → computes novelty/quality/topics/semantic
→ publishes analysis_result to all subscribers.
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.llm import LLMClient
from core.message import Message

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """\
You are the Analyst module of Socialite. Analyse a batch of Moltbook feed posts \
and return a JSON object with these keys:

1. "trending_topics": list of 3-5 topic strings currently trending.
2. "high_quality_posts": list of post IDs that are high quality (thoughtful, good engagement).
3. "engage_targets": list of post IDs worth commenting on (active discussion, we can add value).
4. "topic_summary": one-paragraph summary of community discussion.

Respond ONLY with valid JSON. No markdown fences, no explanation.\
"""

SEMANTIC_PROMPT = """\
Analyze each post briefly. Respond ONLY with a JSON array, no explanation.

[{"post_id":"xxx","is_compliant":true,"violation_type":"none","semantic_value":0.7,"has_unique_perspective":true}]

Fields:
- is_compliant: no spam/harassment? (bool)
- violation_type: spam/harassment/other/none
- semantic_value: 0-1 how valuable the perspective is
- has_unique_perspective: unique viewpoint? (bool)
"""


class AnalysisAgent(BaseAgent):
    """Analyses the feed: trending topics, novelty scores, engagement targets.

    Produces an analysis_result Message that is automatically broadcast to
    all subscribers via the MsgHub wiring.
    """

    def __init__(
        self,
        name: str = "analysis",
        social_memory=None,
        model_name: str = "claude-haiku-4-5-20251001",
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.llm = LLMClient(model_name=model_name, max_tokens=2048)

    # ── Novelty scoring ──────────────────────────────────────

    async def compute_novelty(self, text: str) -> float:
        if not self.social_memory or not self.social_memory.vector_store:
            return 1.0
        try:
            embedding = self.social_memory._get_embedding(text)
            results = await self.social_memory.vector_store.semantic_search(
                query_embedding=embedding, collection_type="posts", limit=1
            )
            if results and results[0].get("score") is not None:
                return round(1.0 - results[0]["score"], 4)
            return 1.0
        except Exception as e:
            logger.warning("Novelty computation failed: %s", e)
            return 1.0

    @staticmethod
    def compute_learning_value(
        novelty: float, quality: float, engagement: float,
        topic_relevance: float = 0.5,
    ) -> float:
        return round(
            0.4 * novelty + 0.3 * quality + 0.2 * engagement + 0.1 * topic_relevance, 4
        )

    @staticmethod
    def quality_score(post: Dict[str, Any]) -> float:
        up = post.get("upvotes", 0)
        down = post.get("downvotes", 0)
        total = up + down
        if total == 0:
            return 0.3
        ratio = up / total
        length_bonus = min(len(post.get("content", "")) / 500, 0.3)
        return min(round(ratio * 0.7 + length_bonus, 4), 1.0)

    @staticmethod
    def quality_score_with_karma(post: Dict[str, Any]) -> float:
        up = post.get("upvotes", 0)
        down = post.get("downvotes", 0)
        total = up + down
        quality_ratio = up / total if total > 0 else 0.3
        author_karma = post.get("author_karma", 1)
        karma_weight = math.log(author_karma + 2)
        quality = quality_ratio * karma_weight
        return min(round(quality / 6.91, 4), 1.0)

    @staticmethod
    def engagement_score(post: Dict[str, Any]) -> float:
        raw = post.get("upvotes", 0) * 0.6 + post.get("comment_count", 0) * 0.4
        return min(round(raw / 50.0, 4), 1.0)

    @staticmethod
    def apply_time_decay(score: float, created_at: str, lambda_param: float = 0.15) -> float:
        try:
            post_time = datetime.fromisoformat(created_at)
            now = datetime.now(tz=post_time.tzinfo)
            hours_ago = max((now - post_time).total_seconds() / 3600, 0)
            return round(score * math.exp(-lambda_param * hours_ago), 4)
        except Exception as e:
            logger.warning("Time decay failed: %s, returning original", e)
            return score

    # ── Main analysis ────────────────────────────────────────

    async def analyse_feed(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        from config import SYSTEM_CONFIG

        novelty_scores: Dict[str, float] = {}
        learning_values: Dict[str, float] = {}

        quality_config = SYSTEM_CONFIG.get("quality_scoring", {})
        use_karma = quality_config.get("karma_weight_enabled", True)
        use_time_decay = quality_config.get("time_decay_enabled", True)
        time_decay_lambda = quality_config.get("time_decay_lambda", 0.15)

        for p in posts:
            pid = p.get("id", "")
            text = f"{p.get('title', '')}\n{p.get('content', '')}"
            novelty = await self.compute_novelty(text)
            if use_karma and p.get("author_karma") is not None:
                quality = self.quality_score_with_karma(p)
            else:
                quality = self.quality_score(p)
            engagement = self.engagement_score(p)
            lv = self.compute_learning_value(novelty, quality, engagement)
            if use_time_decay and p.get("created_at"):
                lv = self.apply_time_decay(lv, p["created_at"], time_decay_lambda)
            novelty_scores[pid] = novelty
            learning_values[pid] = lv

        semantic_analysis = await self._semantic_analysis(posts)

        feed_summary = "\n".join(
            f"[{p.get('id','')}] up:{p.get('upvotes',0)} cmt:{p.get('comment_count',0)} "
            f"| {p.get('title','')[:80]}"
            for p in posts[:25]
        )
        llm_analysis = await self._call_llm(feed_summary)
        high_value_posts = [pid for pid, lv in learning_values.items() if lv > 0.5]

        return {
            "type": "analysis_result",
            "trending_topics": llm_analysis.get("trending_topics", []),
            "engage_targets": llm_analysis.get("engage_targets", []),
            "high_quality_posts": llm_analysis.get("high_quality_posts", []),
            "high_value_posts": high_value_posts,
            "novelty_scores": novelty_scores,
            "learning_values": learning_values,
            "topic_summary": llm_analysis.get("topic_summary", ""),
            "semantic_analysis": semantic_analysis,
        }

    async def _semantic_analysis(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        post_texts = []
        for p in posts[:10]:
            pid = p.get("id", "")
            title = p.get("title", "")
            content = p.get("content", "")[:300]
            post_texts.append(f"[{pid}] Title: {title}\nContent: {content[:200]}")
        if not post_texts:
            return {}
        try:
            data = await self.llm.call_json(
                SEMANTIC_PROMPT + "\n\nPosts:\n" + "\n".join(post_texts)
            )
            if isinstance(data, list):
                return {r["post_id"]: r for r in data}
            return {}
        except Exception as e:
            logger.warning("Semantic analysis failed: %s", e)
            return {}

    async def _call_llm(self, feed_text: str) -> Dict[str, Any]:
        try:
            return await self.llm.call_json(
                ANALYST_SYSTEM_PROMPT + "\n\nPosts:\n" + feed_text
            )
        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            return {
                "trending_topics": [], "engage_targets": [],
                "high_quality_posts": [], "topic_summary": "",
            }

    # ── BaseAgent interface ──────────────────────────────────

    async def reply(self, msg: Message) -> Message:
        """Process a raw_feed message and return analysis_result."""
        posts = msg.metadata.get("posts", [])
        if not posts:
            try:
                data = json.loads(msg.content)
                posts = data.get("posts", [])
            except (json.JSONDecodeError, TypeError):
                posts = []

        analysis = await self.analyse_feed(posts)
        return Message(
            name=self.name,
            role="assistant",
            content=json.dumps({"type": "analysis_result"}),
            metadata=analysis,
        )

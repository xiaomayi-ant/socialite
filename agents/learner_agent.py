# -*- coding: utf-8 -*-
"""LearnerAgent — background learning agent.

Merges v1 PatternMinerAgent + EvolutionAgent.
Subscribes to action_completed → accumulates data → periodic learning.
Publishes strategy_update + collection_suggestion.
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.llm import LLMClient
from core.message import Message

logger = logging.getLogger(__name__)

PATTERN_SYSTEM_PROMPT = """\
You analyse high-performing social posts and extract **actionable** content patterns.

Given posts with engagement metrics, output a JSON array of patterns. Each must have:
- "description": specific, actionable (e.g. "question-style openings get 2x engagement on AI topics")
- "topics": list of topic strings this pattern applies to
- "avg_upvotes": average upvotes of matching posts

Avoid generic observations. Focus on concrete, repeatable strategies.
Respond ONLY with a JSON array. No markdown fences.\
"""

EVOLUTION_SYSTEM_PROMPT = """\
You are the Evolution Engine. Analyse performance feedback \
and suggest concrete strategy adjustments.

Output a JSON object with:
1. "strategy_notes": 1-2 sentence summary of what to change
2. "adjustments": dict of parameter tweaks

Respond ONLY with valid JSON. No markdown fences.\
"""

STAGE_THRESHOLDS = {"initial": 30, "exploration": 100, "optimization": 250}


@dataclass
class EvolutionState:
    stage: str = "initial"
    exploration_rate: float = 0.8
    cycle_count: int = 0
    plateau_counter: int = 0
    best_metrics: Dict[str, float] = field(
        default_factory=lambda: {"avg_upvotes": 0.0, "avg_comments": 0.0, "engagement_rate": 0.0}
    )
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    T: int = 30
    patience: int = 5
    factor: float = 0.5


class LearnerAgent(BaseAgent):
    """Combined pattern mining + evolution management.

    Subscribes to action_completed messages from action agents.
    Periodically triggers learning and publishes strategy updates.
    """

    def __init__(
        self,
        name: str = "learner",
        social_memory=None,
        structured_store=None,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.structured_store = structured_store
        self.llm = LLMClient(model_name=model_name, max_tokens=1024)
        self.state = EvolutionState()
        self._action_buffer: List[Dict[str, Any]] = []

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        if msg.metadata.get("type") == "action_completed":
            self._action_buffer.append(msg.metadata)

    # ── State persistence ────────────────────────────────────

    async def load_state(self) -> None:
        if not self.structured_store:
            return
        saved = await self.structured_store.get_latest_evolution_state()
        if saved:
            self.state.stage = saved.get("stage", "initial")
            self.state.exploration_rate = saved.get("exploration_rate", 0.8)
            metrics = saved.get("metrics", {})
            self.state.cycle_count = metrics.get("cycle_count", 0)
            self.state.plateau_counter = metrics.get("plateau_counter", 0)
            self.state.best_metrics = saved.get("best_metrics", self.state.best_metrics)
            self.state.strategy_params = metrics.get("strategy_params", {})
            logger.info(
                "Loaded evolution state: stage=%s cycles=%d",
                self.state.stage, self.state.cycle_count,
            )

    async def save_state(self, strategy_notes: str = "") -> None:
        if not self.structured_store:
            return
        await self.structured_store.record_evolution_state({
            "stage": self.state.stage,
            "exploration_rate": self.state.exploration_rate,
            "metrics": {
                "cycle_count": self.state.cycle_count,
                "plateau_counter": self.state.plateau_counter,
                "strategy_params": self.state.strategy_params,
            },
            "best_metrics": self.state.best_metrics,
            "strategy_notes": strategy_notes,
        })

    # ── Main learning pipeline ───────────────────────────────

    async def learn(self, learning_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run full learning pipeline: mine patterns → evaluate → evolve."""
        data = learning_data or {}

        # Step 1: Mine patterns
        pattern_result = await self.mine_patterns(learning_data=data)
        new_patterns = pattern_result.get("new_patterns", [])

        # Step 2: Get feedback data
        own_comments_feedback = {"positive_ratio": 0.5}
        identity_snapshots = []
        if self.structured_store:
            try:
                self_feedback = data.get("self_feedback", [])
                if not self_feedback:
                    self_feedback = await self.structured_store.get_own_comments(limit=20)
                if self_feedback:
                    positive = sum(1 for c in self_feedback if c.get("success", False))
                    own_comments_feedback["positive_ratio"] = positive / len(self_feedback)
                    own_comments_feedback["comments"] = self_feedback
                identity_snapshots = await self.structured_store.get_identity_snapshots(limit=3)
            except Exception as e:
                logger.debug("Learning feedback data failed: %s", e)

        # Step 3: Calculate learning progress
        learning_progress = await self.evaluate_learning(
            new_patterns, own_comments_feedback, identity_snapshots
        )

        # Step 4: Process evolution feedback
        self.state.cycle_count += 1
        performance_data = data.get("performance_data", [])
        evolution_result = await self._process_evolution(
            performance_data, pattern_result
        )

        # Clear action buffer
        self._action_buffer.clear()

        return {
            "pattern_result": pattern_result,
            "learning_progress": learning_progress,
            "evolution_result": evolution_result,
            "new_patterns": new_patterns,
        }

    # ── Pattern mining (from v1 PatternMinerAgent) ───────────

    async def mine_patterns(
        self, learning_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if learning_data is None:
            learning_data = {}

        weights = learning_data.get("weights", {
            "community_posts": 0.4, "community_comments": 0.3,
            "discussion_depth": 0.2, "self_feedback": 0.1,
        })

        all_source_patterns = []

        # Source 1: Community posts
        community_posts = learning_data.get("community_posts", [])
        if community_posts:
            post_patterns = self._analyze_high_value_posts(community_posts)
            for p in post_patterns:
                p["_weight"] = weights.get("community_posts", 0.3)
            all_source_patterns.extend(post_patterns)

        # Source 2: Community comments
        community_comments = learning_data.get("community_comments", [])
        if community_comments:
            comment_patterns = self._analyze_high_value_comments(community_comments)
            for p in comment_patterns:
                p["_weight"] = weights.get("community_comments", 0.25)
            all_source_patterns.extend(comment_patterns)

        # Source 3: Discussion depth
        discussion_depth = learning_data.get("discussion_depth", [])
        if discussion_depth:
            depth_patterns = self._analyze_discussion_depth(discussion_depth)
            for p in depth_patterns:
                p["_weight"] = weights.get("discussion_depth", 0.2)
            all_source_patterns.extend(depth_patterns)

        # Source 4: Self feedback
        self_feedback = learning_data.get("self_feedback", [])
        if self_feedback:
            feedback_patterns = self._analyze_self_feedback(self_feedback)
            for p in feedback_patterns:
                p["_weight"] = weights.get("self_feedback", 0.1)
            all_source_patterns.extend(feedback_patterns)

        # Source 5: Graph insights
        graph_insights = learning_data.get("graph_insights", {})
        if graph_insights:
            graph_patterns = self._extract_graph_patterns(graph_insights, community_posts)
            for p in graph_patterns:
                p["_weight"] = 1.0
            all_source_patterns.extend(graph_patterns)

        # LLM-based pattern extraction
        all_data = []
        for post in community_posts[:10]:
            all_data.append({
                "type": "community_post",
                "content": post.get("content", "")[:200],
                "upvotes": post.get("upvotes", 0),
                "comment_count": post.get("comment_count", 0),
            })
        for comment in community_comments[:10]:
            all_data.append({
                "type": "community_comment",
                "content": comment.get("content", "")[:200],
                "upvotes": comment.get("upvotes", 0),
            })

        if all_data:
            llm_patterns = await self._extract_patterns_via_llm(all_data)
            for pattern in llm_patterns:
                avg_upvotes = pattern.get("avg_upvotes", 0.0)
                pattern["confidence"] = min(avg_upvotes / 10.0, 1.0) if avg_upvotes > 0 else 0.6
                pattern["pattern_source"] = "llm_extraction"
                pattern["_weight"] = 1.0
            all_source_patterns.extend(llm_patterns)

        if not all_source_patterns:
            return {"type": "pattern_result", "new_patterns": [], "top_patterns": []}

        # Weighted aggregation
        for p in all_source_patterns:
            w = p.pop("_weight", 1.0)
            p["weighted_confidence"] = p.get("confidence", 0.5) * w
        all_source_patterns.sort(key=lambda x: x.get("weighted_confidence", 0), reverse=True)

        # Persist top patterns
        persisted = []
        for pattern in all_source_patterns[:15]:
            pattern_id = pattern.get("pattern_id") or f"pat_{uuid.uuid4().hex[:8]}"
            pattern_data = {
                "pattern_id": pattern_id,
                "description": pattern.get("description", ""),
                "pattern_source": pattern.get("pattern_source", "unknown"),
                "example_post_ids": pattern.get("example_ids", []),
                "topics": pattern.get("topics", []),
                "avg_upvotes": pattern.get("avg_upvotes", 0.0),
                "confidence": pattern.get("confidence", 0.5),
                "weighted_confidence": pattern.get("weighted_confidence", 0.0),
                "usage_count": 0,
                "success_rate": pattern.get("success_rate", 0.0),
                "decay_factor": 1.0,
            }
            if self.structured_store:
                await self.structured_store.upsert_style_pattern(pattern_data)
            persisted.append(pattern_data)

        top_patterns = []
        if self.structured_store:
            top_patterns = await self.structured_store.get_style_patterns(limit=10)

        logger.info("Mined %d patterns", len(persisted))
        return {"type": "pattern_result", "new_patterns": persisted, "top_patterns": top_patterns}

    def _analyze_high_value_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns = []
        for post in posts[:10]:
            topics = post.get("topics") or ["general"]
            topic = topics[0] if topics else "general"
            patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "description": (
                    f"High-value post pattern: topic={topic}, "
                    f"quality={post.get('quality_with_karma', 0):.2f}, "
                    f"engagement={post.get('comment_count', 0)} comments"
                ),
                "pattern_source": "community_post",
                "confidence": post.get("value_score", 0.5),
                "success_rate": post.get("value_score", 0.5),
                "topics": post.get("topics", []),
                "example_ids": [post.get("post_id")],
            })
        return patterns

    def _analyze_high_value_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns = []
        for comment in comments[:10]:
            patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "description": (
                    f"High-value comment pattern: "
                    f"quality={comment.get('quality_score', 0):.2f}, "
                    f"upvotes={comment.get('upvotes', 0)}"
                ),
                "pattern_source": "community_comment",
                "confidence": comment.get("quality_score", 0.5),
                "success_rate": comment.get("quality_score", 0.5),
                "topics": [],
                "example_ids": [comment.get("comment_id")],
            })
        return patterns

    def _analyze_discussion_depth(self, depth_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns = []
        deep = [d for d in depth_metrics if d.get("unique_commenters", 0) >= 3]
        if deep:
            avg_depth = sum(d.get("max_thread_depth", 0) for d in deep) / len(deep)
            avg_commenters = sum(d.get("unique_commenters", 0) for d in deep) / len(deep)
            patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "description": (
                    f"Deep discussion: avg {avg_commenters:.0f} commenters, "
                    f"depth {avg_depth:.1f}"
                ),
                "pattern_source": "discussion_depth",
                "confidence": min(len(deep) / 5.0, 1.0),
                "success_rate": min(avg_commenters / 10.0, 1.0),
                "topics": [],
                "example_ids": [d.get("post_id") for d in deep[:3]],
            })
        return patterns

    def _analyze_self_feedback(self, own_comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not own_comments:
            return []
        successful = [c for c in own_comments if c.get("success")]
        if not successful:
            return []
        avg_upvotes = sum(c.get("upvotes_current", 0) for c in successful) / len(successful)
        return [{
            "pattern_id": str(uuid.uuid4()),
            "description": (
                f"Self-feedback: {len(successful)}/{len(own_comments)} "
                f"successful, avg {avg_upvotes:.1f} upvotes on wins"
            ),
            "pattern_source": "self_feedback",
            "confidence": len(successful) / max(len(own_comments), 1),
            "success_rate": len(successful) / max(len(own_comments), 1),
            "topics": [],
            "example_ids": [c.get("comment_id") for c in successful[:3]],
        }]

    def _extract_graph_patterns(
        self, graph_insights: Dict[str, Any], community_posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        patterns = []
        influencers = graph_insights.get("influencers", [])
        if influencers:
            influencer_names = {name for name, _score in influencers[:10]}
            influencer_posts = [
                p for p in community_posts
                if p.get("author_name", "").lower().replace(" ", "_") in influencer_names
            ]
            if influencer_posts:
                avg_upvotes = sum(p.get("upvotes", 0) for p in influencer_posts) / len(influencer_posts)
                patterns.append({
                    "pattern_id": str(uuid.uuid4()),
                    "description": (
                        f"Influencer pattern: {len(influencer_posts)} posts "
                        f"from top PageRank users, avg {avg_upvotes:.0f} upvotes"
                    ),
                    "pattern_source": "graph_analysis",
                    "confidence": min(len(influencer_posts) / 5.0, 1.0),
                    "avg_upvotes": avg_upvotes,
                    "success_rate": min(avg_upvotes / 10.0, 1.0),
                    "topics": [],
                    "example_ids": [p.get("post_id") for p in influencer_posts[:3]],
                })
        return patterns

    async def _extract_patterns_via_llm(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines = []
        for item in data[:30]:
            if item.get("type") == "community_post":
                lines.append(
                    f"[Post] up:{item.get('upvotes',0)} cmt:{item.get('comment_count',0)} "
                    f"| {item.get('content','')[:100]}"
                )
            else:
                lines.append(
                    f"[Comment] up:{item.get('upvotes',0)} "
                    f"| {item.get('content','')[:100]}"
                )
        if not lines:
            return []
        try:
            text = await self.llm.call(
                PATTERN_SYSTEM_PROMPT + "\n\nPosts:\n" + "\n".join(lines)
            )
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            patterns = json.loads(text.strip())
            return patterns[:10] if isinstance(patterns, list) else []
        except Exception as e:
            logger.warning("LLM pattern extraction failed: %s", e)
            return []

    # ── Learning evaluation (from v1 EvolutionAgent) ─────────

    async def evaluate_learning(
        self,
        new_patterns: List[Dict[str, Any]],
        own_comments_feedback: Dict[str, Any],
        identity_snapshots: List[Dict[str, Any]],
    ) -> float:
        """Calculate learning progress (0-1): 40% depth, 30% validation, 20% consistency, 10% uniqueness."""
        # Depth
        if new_patterns:
            depth_count = min(len(new_patterns) / 5.0, 1.0)
            avg_conf = sum(p.get("confidence", 0) for p in new_patterns) / len(new_patterns)
            depth_score = depth_count * avg_conf
        else:
            depth_score = 0.0

        # Validation
        own_count = len(own_comments_feedback.get("comments", []))
        if own_count < 3:
            validation_score = max(own_comments_feedback.get("positive_ratio", 0.5), 0.3)
        else:
            validation_score = own_comments_feedback.get("positive_ratio", 0.5)
        validation_score = min(max(validation_score, 0.0), 1.0)

        # Consistency
        consistency_score = self._calculate_identity_consistency(new_patterns, identity_snapshots)

        # Uniqueness
        uniqueness_score = self._extract_novel_insights(new_patterns)

        learning_progress = (
            0.4 * depth_score + 0.3 * validation_score
            + 0.2 * consistency_score + 0.1 * uniqueness_score
        )
        logger.info(
            "Learning: %.2f (depth=%.2f val=%.2f cons=%.2f uniq=%.2f)",
            learning_progress, depth_score, validation_score,
            consistency_score, uniqueness_score,
        )
        return min(max(learning_progress, 0.0), 1.0)

    def _calculate_identity_consistency(
        self, new_patterns: List[Dict[str, Any]], identity_snapshots: List[Dict[str, Any]]
    ) -> float:
        if not identity_snapshots:
            return 0.5
        if not new_patterns:
            return 1.0
        latest = identity_snapshots[-1] if identity_snapshots else {}
        core_values = latest.get("core_values", [])
        default_positions = latest.get("default_positions", [])
        if not core_values and not default_positions:
            return 0.7
        count = 0
        for pattern in new_patterns:
            desc = pattern.get("description", "").lower()
            if any(v.lower() in desc for v in core_values):
                count += 1
            elif any(p.lower() in desc for p in default_positions):
                count += 0.5
        return min(count / len(new_patterns), 1.0) if new_patterns else 0.5

    def _extract_novel_insights(self, new_patterns: List[Dict[str, Any]]) -> float:
        if not new_patterns:
            return 0.0
        novel = sum(
            1 for p in new_patterns
            if p.get("confidence", 0) >= 0.7
            and p.get("pattern_source") in ("comparison_learning", "community_post")
        )
        return min(novel / 3.0, 1.0)

    # ── Evolution management ─────────────────────────────────

    async def _process_evolution(
        self,
        performance_data: List[Dict[str, Any]],
        pattern_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        n = max(len(performance_data), 1)
        current = {
            "avg_upvotes": sum(p.get("upvotes", 0) for p in performance_data) / n,
            "avg_comments": sum(p.get("comment_count", 0) for p in performance_data) / n,
            "engagement_rate": sum(
                p.get("upvotes", 0) + p.get("comment_count", 0) for p in performance_data
            ) / n,
        }
        improved = current["engagement_rate"] > self.state.best_metrics.get("engagement_rate", 0)

        # Check forgetting
        forgetting = False
        for key, best in self.state.best_metrics.items():
            if best > 0 and current.get(key, 0) < best * 0.8:
                logger.warning("Forgetting: %s %.2f < %.2f", key, current.get(key, 0), best)
                forgetting = True
                break

        if not forgetting:
            for k, v in current.items():
                if v > self.state.best_metrics.get(k, 0):
                    self.state.best_metrics[k] = v

        # Stage advancement
        threshold = STAGE_THRESHOLDS.get(self.state.stage)
        if threshold and self.state.cycle_count >= threshold:
            stages = list(STAGE_THRESHOLDS.keys()) + ["innovation"]
            idx = stages.index(self.state.stage)
            if idx + 1 < len(stages):
                old = self.state.stage
                self.state.stage = stages[idx + 1]
                logger.info("Stage: %s → %s", old, self.state.stage)

        # Update exploration rate
        self._update_exploration_rate(improved)

        # LLM strategy advice
        strategy_notes = ""
        if not forgetting and performance_data:
            strategy_notes = await self._get_strategy_advice(current, pattern_result)

        await self.save_state(strategy_notes)

        return {
            "stage": self.state.stage,
            "exploration_rate": self.state.exploration_rate,
            "cycle_count": self.state.cycle_count,
            "improved": improved,
            "forgetting": forgetting,
            "strategy_notes": strategy_notes,
        }

    def _update_exploration_rate(self, improved: bool) -> None:
        if self.state.stage == "initial":
            self.state.exploration_rate = 0.8
        elif self.state.stage == "exploration":
            T = self.state.T
            self.state.exploration_rate = round(
                0.2 + 0.6 * (1 + math.cos(math.pi * (self.state.cycle_count % T) / T)) / 2, 4
            )
        elif self.state.stage == "optimization":
            if improved:
                self.state.plateau_counter = 0
            else:
                self.state.plateau_counter += 1
            if self.state.plateau_counter >= self.state.patience:
                self.state.exploration_rate = round(
                    max(self.state.exploration_rate * self.state.factor, 0.05), 4
                )
                self.state.plateau_counter = 0
        elif self.state.stage == "innovation":
            if self.state.cycle_count % 50 == 0:
                self.state.exploration_rate = 0.4
            else:
                self.state.exploration_rate = max(
                    round(self.state.exploration_rate * 0.99, 4), 0.05
                )

    async def _get_strategy_advice(
        self, metrics: Dict[str, float], pattern_result: Dict[str, Any]
    ) -> str:
        try:
            context = (
                f"Stage: {self.state.stage}, exploration_rate: {self.state.exploration_rate}, "
                f"cycle: {self.state.cycle_count}\nMetrics: {json.dumps(metrics)}\n"
                f"Best: {json.dumps(self.state.best_metrics)}\n"
            )
            if pattern_result:
                context += f"Top patterns: {json.dumps(pattern_result.get('top_patterns', [])[:3], default=str)}\n"
            data = await self.llm.call_json(
                EVOLUTION_SYSTEM_PROMPT + "\n\nData:\n" + context
            )
            if data.get("adjustments"):
                self.state.strategy_params.update(data["adjustments"])
            return data.get("strategy_notes", "")
        except Exception as e:
            logger.warning("Strategy advice failed: %s", e)
            return ""

    # ── Posting context generation ───────────────────────────

    async def generate_posting_context(
        self,
        new_patterns: List[Dict[str, Any]],
        learning_progress: float,
        trending_topics: List[str],
    ) -> Dict[str, Any]:
        if new_patterns:
            avg_conf = sum(p.get("confidence", 0) for p in new_patterns) / len(new_patterns)
            motivation = (
                f"Discovered {len(new_patterns)} patterns "
                f"with {avg_conf:.0%} avg confidence"
            )
        else:
            motivation = "Consolidating existing knowledge"

        # Extract topics from patterns
        pattern_topics = set()
        for p in new_patterns:
            topics = p.get("topics", [])
            if isinstance(topics, list):
                pattern_topics.update(topics)
        trending_set = set(trending_topics)
        intersection = list(pattern_topics & trending_set)
        remaining = list(pattern_topics - trending_set)[:2]
        suggested = (intersection[:2] + remaining)[:3] or trending_topics[:1]

        return {
            "motivation": motivation,
            "urgency": min(max(learning_progress, 0.0), 1.0),
            "suggested_topics": suggested,
            "pattern_count": len(new_patterns),
        }

    # ── BaseAgent interface ──────────────────────────────────

    async def reply(self, msg: Message) -> Message:
        return Message(
            name=self.name, role="assistant",
            content="LearnerAgent uses learn() interface",
        )

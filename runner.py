#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Socialite v0.4.0 — Pub-Sub Multi-Agent Moltbook Runner

Proposal-based pipeline with A/B strategy and selective MsgHub subscriptions.
"""

import asyncio
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Core framework ──────────────────────────────────────────
from core.message import Message
from core.msghub import MsgHub
from core.proposal import Proposal
from core.ab_strategy import ABSelector
from core.message_logger import MessageLogger
from core.base_agent import BaseAgent

# ── Social Memory ───────────────────────────────────────────
try:
    from social_memory.config import SocialMemoryConfig
    from social_memory.storage_manager import SocialMemoryManager
    from social_memory.models import (
        PlatformType, InteractionType, SocialInteraction, SocialUser,
    )
    SOCIAL_MEMORY_AVAILABLE = True
except ImportError:
    SOCIAL_MEMORY_AVAILABLE = False
    logging.getLogger(__name__).warning("social_memory module not available")

# ── Moltbook ────────────────────────────────────────────────
from moltbook.config import MoltbookConfig
from moltbook.auth import MoltbookAuth
from moltbook.client import MoltbookClient
from moltbook.heartbeat import HeartbeatState
from moltbook.engagement_strategy import EngagementStrategy

try:
    from moltbook.skill_monitor import check_skill_updates
except ImportError:
    check_skill_updates = None

# ── Agents ──────────────────────────────────────────────────
from agents.sensor_agent import SensorAgent
from agents.analysis_agent import AnalysisAgent
from agents.comment_agent import CommentAgent
from agents.post_agent import PostAgent
from agents.upvote_agent import UpvoteAgent
from agents.follow_agent import FollowAgent
from agents.coordinator import CoordinatorAgent
from agents.learner_agent import LearnerAgent
from agents.observer_agent import ObserverAgent

# ── Config ──────────────────────────────────────────────────
from config import LEARNING_WEIGHTS, FEEDBACK_POLL_DELAYS_HOURS

INTERACTED_IDS_PATH = Path.home() / ".config" / "moltbook" / "interacted_posts.json"
_INTERACTED_TTL_HOURS = 48

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("runner")


# ── Interacted IDs persistence ──────────────────────────────

def _load_interacted_ids() -> set:
    try:
        if INTERACTED_IDS_PATH.exists():
            data = json.loads(INTERACTED_IDS_PATH.read_text())
            raw = data.get("ids", [])
            now = datetime.now()
            valid = set()
            for entry in raw:
                if isinstance(entry, dict):
                    pid = entry.get("id", "")
                    ts_str = entry.get("ts")
                    if ts_str:
                        age_hours = (now - datetime.fromisoformat(ts_str)).total_seconds() / 3600
                        if age_hours <= _INTERACTED_TTL_HOURS:
                            valid.add(pid)
                    else:
                        valid.add(pid)
                else:
                    valid.add(entry)
            return valid
    except Exception:
        pass
    return set()


def _save_interacted_ids(ids: set, timestamps: dict) -> None:
    try:
        INTERACTED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {"id": pid, "ts": timestamps.get(pid, datetime.now().isoformat())}
            for pid in list(ids)[-500:]
        ]
        INTERACTED_IDS_PATH.write_text(json.dumps({"ids": entries}, indent=2))
    except Exception as e:
        logger.warning("Failed to save interacted IDs: %s", e)


# ── Cost tracking ───────────────────────────────────────────

async def _track_cost(
    store, agent_name: str, model: str, action_type: str,
    input_tokens: int = 0, output_tokens: int = 0,
) -> None:
    if not store:
        return
    cost_map = {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    }
    rates = cost_map.get(model, {"input": 1.0, "output": 5.0})
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    await store.record_api_cost({
        "agent_name": agent_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 6),
        "action_type": action_type,
    })


# ═══════════════════════════════════════════════════════════
# SocialiteRunner
# ═══════════════════════════════════════════════════════════

class SocialiteRunner:
    """Orchestrates the 9-agent pub-sub pipeline on Moltbook."""

    def __init__(self) -> None:
        # ── Moltbook client ──
        self.moltbook_config = MoltbookConfig.from_env()
        self.auth = MoltbookAuth(self.moltbook_config)
        self.auth.load_credentials()
        self.client = MoltbookClient(
            self.moltbook_config, self.auth,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.strategy = EngagementStrategy(self.moltbook_config)
        self.heartbeat = HeartbeatState(self.moltbook_config.state_path)
        self.interacted_ids = _load_interacted_ids()

        # ── Social memory ──
        self.social_memory: Optional[SocialMemoryManager] = None
        self.structured_store = None

        # ── A/B selector ──
        from config import SYSTEM_CONFIG
        ab_config = SYSTEM_CONFIG.get("ab_strategy", {})
        self.ab_selector = ABSelector(
            mode=ab_config.get("mode", "alternate"),
            p_a=ab_config.get("p_a", 0.5),
        )

        # ── Agents (created after init) ──
        self.sensor: Optional[SensorAgent] = None
        self.analysis: Optional[AnalysisAgent] = None
        self.comment: Optional[CommentAgent] = None
        self.post: Optional[PostAgent] = None
        self.upvote: Optional[UpvoteAgent] = None
        self.follow: Optional[FollowAgent] = None
        self.coordinator: Optional[CoordinatorAgent] = None
        self.learner: Optional[LearnerAgent] = None
        self.observer: Optional[ObserverAgent] = None
        self.hub: Optional[MsgHub] = None

        self._cycle_count = self.heartbeat.state.get("cycle_count", 0)
        self._daily_budget = float(os.getenv("DAILY_BUDGET_USD", "5.0"))
        self._interacted_timestamps: dict = {}
        self._learn_interval = 3
        self._report_interval = 10

    # ── Initialization ──────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize social memory + all 9 agents + MsgHub topology."""
        # Social memory
        if SOCIAL_MEMORY_AVAILABLE:
            logger.info("Initializing social memory...")
            try:
                sm_config = SocialMemoryConfig.from_env()
                sm_config.vector_store.storage_strategy = "disk"
                self.social_memory = SocialMemoryManager(sm_config)
                ok = await self.social_memory.initialize(
                    agent_user_id="sociallearnerbot",
                    agent_username="SocialLearnerBot",
                )
                if ok:
                    self.structured_store = self.social_memory.structured_store
                    logger.info("Social memory ready")
                else:
                    logger.warning("Social memory partially initialised")
                    self.structured_store = self.social_memory.structured_store
            except Exception as e:
                logger.error("Social memory init failed: %s", e)

        # ── Create agents ──
        self.sensor = SensorAgent(name="sensor", client=self.client)
        self.analysis = AnalysisAgent(
            name="analysis", social_memory=self.social_memory
        )
        self.comment = CommentAgent(
            name="comment",
            social_memory=self.social_memory,
            structured_store=self.structured_store,
            ab_selector=self.ab_selector,
        )
        self.post = PostAgent(
            name="post",
            social_memory=self.social_memory,
            structured_store=self.structured_store,
            ab_selector=self.ab_selector,
        )
        self.upvote = UpvoteAgent(
            name="upvote", ab_selector=self.ab_selector,
        )
        self.follow = FollowAgent(
            name="follow",
            social_memory=self.social_memory,
            ab_selector=self.ab_selector,
        )
        self.coordinator = CoordinatorAgent(
            name="coordinator",
            structured_store=self.structured_store,
            daily_budget_usd=self._daily_budget,
        )
        self.learner = LearnerAgent(
            name="learner",
            social_memory=self.social_memory,
            structured_store=self.structured_store,
        )
        self.observer = ObserverAgent(
            name="observer",
            structured_store=self.structured_store,
        )

        # ── Message logger ──
        self.msg_logger = MessageLogger(self.structured_store)
        BaseAgent.message_logger = self.msg_logger

        # ── MsgHub + subscription topology ──
        all_agents = [
            self.sensor, self.analysis, self.comment, self.post,
            self.upvote, self.follow, self.coordinator,
            self.learner, self.observer,
        ]
        self.hub = MsgHub(participants=all_agents, message_logger=self.msg_logger)

        # SensorAgent → AnalysisAgent
        self.hub.subscribe(self.analysis, self.sensor)

        # AnalysisAgent → all Action Agents
        for agent in [self.comment, self.post, self.upvote, self.follow]:
            self.hub.subscribe(agent, self.analysis)

        # Action Agents → LearnerAgent + ObserverAgent
        for agent in [self.comment, self.post, self.upvote, self.follow]:
            self.hub.subscribe(self.learner, agent)
            self.hub.subscribe(self.observer, agent)

        # LearnerAgent → SensorAgent + Action Agents
        self.hub.subscribe(self.sensor, self.learner)
        for agent in [self.comment, self.post, self.upvote, self.follow]:
            self.hub.subscribe(agent, self.learner)

        # Load learner state
        await self.learner.load_state()

        logger.info("All 9 agents initialized")
        logger.info("Subscription topology: %s", self.hub.topology())

    # ── Social memory storage helper ────────────────────────

    async def _store_post_in_memory(self, post_data: Dict[str, Any]) -> None:
        if not self.social_memory:
            return
        try:
            from social_memory.models import SocialUser, EngagementMetrics
            content = f"{post_data.get('title', '')}\n{post_data.get('content', '')}".strip()
            if len(content) < 5:
                return
            author_name = post_data.get("author_name", "unknown")
            author = SocialUser(
                user_id=author_name.lower().replace(" ", "_"),
                username=author_name,
                platform=PlatformType.GENERAL,
            )
            engagement = EngagementMetrics(
                upvotes=post_data.get("upvotes", 0),
                downvotes=post_data.get("downvotes", 0),
                comment_count=post_data.get("comment_count", 0),
            )
            await self.social_memory.record_interaction(
                {
                    "type": "post",
                    "post_id": post_data.get("id", ""),
                    "content": content,
                    "author": author,
                    "author_name": author_name,
                    "engagement": engagement,
                    "submolt": post_data.get("submolt", "general"),
                    "created_at": post_data.get("created_at"),
                    "platform": "general",
                },
                platform=PlatformType.GENERAL,
            )
        except Exception as e:
            logger.debug("Memory store failed: %s", e)

    # ── Main cycle ──────────────────────────────────────────

    async def run_cycle(self) -> None:
        """Execute one full cycle: Collect → Analyse → Propose → Coordinate → Execute → Learn."""
        self._cycle_count += 1
        self.msg_logger.set_cycle(self._cycle_count)
        logger.info("=" * 60)
        logger.info("Cycle #%d  %s", self._cycle_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info(
            "Stage: %s | Exploration: %.2f | Strategy: %s",
            self.learner.state.stage,
            self.learner.state.exploration_rate,
            self.ab_selector.select(self._cycle_count),
        )
        logger.info("=" * 60)

        # ════ Phase 1: COLLECT ════
        logger.info("[Phase 1] Collecting feed...")
        collect_msg = Message(
            name="runner", role="user", content="collect",
            metadata={"action": "fetch_feed", "sort": "hot", "limit": 25},
        )
        raw_msg = await self.sensor(collect_msg)  # auto-broadcasts to analysis via hub
        posts = raw_msg.metadata.get("posts", [])
        if not posts:
            logger.warning("No posts available. Skipping cycle.")
            return
        logger.info("Got %d posts", len(posts))

        # ── Collect comments from top-engagement posts ──
        collected_comments = {}
        discussion_depth_metrics = []
        top_posts = sorted(
            posts,
            key=lambda p: p.get("upvotes", 0) + p.get("comment_count", 0),
            reverse=True,
        )[:5]
        for tp in top_posts:
            pid = tp.get("id")
            if not pid or tp.get("comment_count", 0) == 0:
                continue
            try:
                comments = self.sensor.get_post_comments(pid)
                if comments:
                    collected_comments[pid] = comments
                    if self.structured_store:
                        from social_memory.models import (
                            SocialComment, AuthorType, EngagementMetrics,
                        )
                        for c in comments:
                            author = SocialUser(
                                user_id=c.get("author_name", "unknown").lower().replace(" ", "_"),
                                username=c.get("author_name", "unknown"),
                                platform=PlatformType.GENERAL,
                            )
                            sc = SocialComment(
                                comment_id=c["id"],
                                target_post_id=pid,
                                content=c.get("content", ""),
                                author=author,
                                author_type=AuthorType.OTHER,
                                platform=PlatformType.GENERAL,
                                engagement=EngagementMetrics(
                                    upvotes=c.get("upvotes", 0),
                                    downvotes=c.get("downvotes", 0),
                                ),
                            )
                            await self.structured_store.record_comment(sc)
                    unique_authors = set(c.get("author_name") for c in comments)
                    depth_map = {}
                    for c in comments:
                        if c.get("parent_id"):
                            depth_map[c["id"]] = depth_map.get(c["parent_id"], 0) + 1
                        else:
                            depth_map[c["id"]] = 0
                    max_depth = max(depth_map.values()) if depth_map else 0
                    discussion_depth_metrics.append({
                        "post_id": pid,
                        "comment_count": len(comments),
                        "unique_commenters": len(unique_authors),
                        "max_thread_depth": max_depth,
                    })
                await asyncio.sleep(random.uniform(3.0, 6.0))
            except Exception as e:
                logger.debug("Comment collection failed for %s: %s", pid, e)

        if collected_comments:
            total_c = sum(len(v) for v in collected_comments.values())
            logger.info("Collected %d comments from %d posts", total_c, len(collected_comments))

        # ── Fetch author karma ──
        from config import SYSTEM_CONFIG
        quality_config = SYSTEM_CONFIG.get("quality_scoring", {})
        if quality_config.get("karma_weight_enabled", True):
            for p in posts:
                author_name = p.get("author_name")
                if author_name:
                    try:
                        profile = self.sensor.get_agent_profile(author_name)
                        p["author_karma"] = getattr(profile, "karma", 1)
                    except Exception:
                        p["author_karma"] = 1

        # ════ Phase 2: ANALYSE ════
        logger.info("[Phase 2] Analysing feed...")
        analysis_msg = Message(
            name="runner", role="user", content="analyse",
            metadata={"type": "raw_feed", "posts": posts},
        )
        analysis_result = await self.analysis(analysis_msg)  # auto-broadcasts to action agents
        analysis = analysis_result.metadata
        logger.info("Topics: %s", analysis.get("trending_topics", [])[:3])

        # ════ Phase 3: PROPOSE ════
        logger.info("[Phase 3] Collecting proposals...")
        proposals: List[Proposal] = []

        # Comment proposals
        comment_proposals = await self.comment.propose(
            posts, cycle_count=self._cycle_count, interacted_ids=self.interacted_ids
        )
        proposals.extend(comment_proposals)

        # Upvote proposals
        upvote_proposals = await self.upvote.propose(
            posts, cycle_count=self._cycle_count
        )
        proposals.extend(upvote_proposals)

        # Follow proposals
        follow_proposals = await self.follow.propose(
            posts, cycle_count=self._cycle_count
        )
        proposals.extend(follow_proposals)

        # Post proposals (need learning data)
        # Get learning metrics for posting decision
        learning_progress = 0.0
        new_patterns = []
        if self.learner.structured_store:
            try:
                own_feedback = await self.structured_store.get_own_comments(limit=20)
                identity_snap = await self.structured_store.get_identity_snapshots(limit=3)
                top_patterns = await self.structured_store.get_style_patterns(limit=10)
                new_patterns = top_patterns
                own_ratio = {"positive_ratio": 0.5, "comments": own_feedback}
                if own_feedback:
                    pos = sum(1 for c in own_feedback if c.get("success"))
                    own_ratio["positive_ratio"] = pos / len(own_feedback)
                learning_progress = await self.learner.evaluate_learning(
                    top_patterns, own_ratio, identity_snap
                )
            except Exception as e:
                logger.debug("Learning data for posting failed: %s", e)

        post_proposals = await self.post.propose(
            cycle_count=self._cycle_count,
            learning_progress=learning_progress,
            new_patterns=new_patterns,
            evolution_stage=self.learner.state.stage,
            trending_topics=analysis.get("trending_topics", []),
        )
        proposals.extend(post_proposals)

        logger.info(
            "Proposals: %d comment, %d upvote, %d follow, %d post",
            len(comment_proposals), len(upvote_proposals),
            len(follow_proposals), len(post_proposals),
        )

        # ════ Phase 4: COORDINATE ════
        logger.info("[Phase 4] Arbitrating proposals...")
        plan = await self.coordinator.arbitrate(proposals, cycle_count=self._cycle_count)
        approved = plan.get("approved", [])

        # ════ Phase 5: EXECUTE ════
        logger.info("[Phase 5] Executing %d approved actions...", len(approved))
        for prop_dict in approved:
            prop = Proposal.from_dict(prop_dict)
            result = await self._execute_proposal(prop, posts, analysis)

            # Broadcast action_completed
            await self.hub.broadcast(Message(
                name="runner", role="system", content="action_completed",
                metadata={
                    "type": "action_completed",
                    "action": prop.action,
                    "target_id": prop.target_id,
                    "strategy": prop.strategy,
                    "success": result.get("success", False),
                    "priority": prop.priority,
                    "agent_name": prop.agent_name,
                },
            ))

        _save_interacted_ids(self.interacted_ids, self._interacted_timestamps)

        # ── Phase 5b: Feedback check ──
        await self._phase_feedback()

        # ════ Phase 6: LEARN (periodic) ════
        if self._cycle_count % self._learn_interval == 0:
            logger.info("[Phase 6] Learning...")
            community_posts = []
            community_comments = []
            self_feedback = []
            if self.structured_store:
                try:
                    community_posts = await self.structured_store.get_high_value_posts(
                        limit=50, min_value_score=0.3
                    )
                except Exception:
                    pass
                if not community_posts and posts:
                    community_posts = [
                        {
                            "post_id": p.get("id", ""),
                            "content": p.get("content", ""),
                            "upvotes": p.get("upvotes", 0),
                            "comment_count": p.get("comment_count", 0),
                            "topics": analysis.get("trending_topics", []),
                            "value_score": analysis.get("learning_values", {}).get(p.get("id", ""), 0),
                        }
                        for p in posts if p.get("upvotes", 0) >= 5
                    ][:20]
                try:
                    community_comments = await self.structured_store.get_high_value_comments(
                        limit=50, min_quality_score=0.3
                    )
                except Exception:
                    pass
                try:
                    self_feedback = await self.structured_store.get_own_comments(limit=20)
                except Exception:
                    pass

            current_stage = self.learner.state.stage
            stage_weights = LEARNING_WEIGHTS.get(current_stage, LEARNING_WEIGHTS["initial"])

            learning_data = {
                "community_posts": community_posts,
                "community_comments": community_comments,
                "discussion_depth": discussion_depth_metrics,
                "self_feedback": self_feedback,
                "weights": stage_weights,
                "performance_data": [],
            }
            learn_result = await self.learner.learn(learning_data=learning_data)
            logger.info(
                "Learning: progress=%.0f%%, patterns=%d",
                learn_result.get("learning_progress", 0) * 100,
                len(learn_result.get("new_patterns", [])),
            )

            # Track learning costs
            await _track_cost(
                self.structured_store, "learner", "claude-haiku-4-5-20251001",
                "learning", input_tokens=2000, output_tokens=500,
            )

        # ════ Phase 7: REPORT (periodic) ════
        if self._cycle_count % self._report_interval == 0:
            logger.info("[Phase 7] Generating observer report...")
            report = await self.observer.generate_report(cycle_count=self._cycle_count)
            logger.info("Report: %s", json.dumps(report.get("strategy_comparison", {})))

        # Store high-value posts
        stored = 0
        for p in posts:
            lv = analysis.get("learning_values", {}).get(p.get("id", ""), 0)
            if p.get("upvotes", 0) >= 2 or lv > 0.5:
                await self._store_post_in_memory(p)
                stored += 1
        if stored:
            logger.info("Stored %d high-value posts in memory", stored)

        # Persist state
        self.heartbeat.state["cycle_count"] = self._cycle_count
        self.heartbeat.update_check()
        logger.info("Cycle #%d complete!", self._cycle_count)

    # ── Execute a single proposal ───────────────────────────

    async def _execute_proposal(
        self, prop: Proposal, posts: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an approved proposal and return result dict."""
        result: Dict[str, Any] = {"success": False, "action": prop.action}

        if prop.action == "upvote":
            ok = self.sensor.upvote(prop.target_id)
            result["success"] = ok
            if ok and self.structured_store:
                await self.structured_store.record_behavior({
                    "action_type": "upvote",
                    "target_id": prop.target_id,
                    "strategy_used": prop.strategy,
                })
            if ok:
                await self._update_graph_edge(prop.target_id, posts, "upvote")
            await asyncio.sleep(random.uniform(3, 8))

        elif prop.action == "comment":
            post_data = next((p for p in posts if p.get("id") == prop.target_id), None)
            if not post_data:
                return result
            comment_result = await self.comment.generate_comment(
                post_title=post_data.get("title", ""),
                post_content=post_data.get("content", ""),
                post_id=prop.target_id,
            )
            text = comment_result.get("text", "")
            if not text:
                return result
            gw_result = self.sensor.create_comment(prop.target_id, text)
            if gw_result:
                result["success"] = True
                result["comment_id"] = gw_result["id"]
                self.interacted_ids.add(prop.target_id)
                self._interacted_timestamps[prop.target_id] = datetime.now().isoformat()
                self.heartbeat.update_comment()
                if self.structured_store:
                    await self.structured_store.record_behavior({
                        "action_type": "comment",
                        "target_id": prop.target_id,
                        "strategy_used": prop.strategy,
                    })
                    await self.structured_store.record_own_comment({
                        "comment_id": gw_result["id"],
                        "post_id": prop.target_id,
                        "comment_text": text,
                    })
                    agent_user = SocialUser(
                        user_id="sociallearnerbot",
                        username="SocialLearnerBot",
                        platform=PlatformType.GENERAL,
                    )
                    interaction = SocialInteraction(
                        interaction_id=str(uuid.uuid4()),
                        interaction_type=InteractionType.COMMENT,
                        content=text,
                        platform=PlatformType.GENERAL,
                        target_post_id=prop.target_id,
                        actor=agent_user,
                        learning_value=0.5,
                        timestamp=datetime.now(),
                        metadata={"source": "self_action", "strategy": prop.strategy},
                    )
                    await self.structured_store.record_interaction(interaction)
                await self._update_graph_edge(prop.target_id, posts, "comment")
                await _track_cost(
                    self.structured_store, "comment", "claude-haiku-4-5-20251001",
                    "comment", input_tokens=500, output_tokens=100,
                )
            await asyncio.sleep(random.uniform(3, 8))

        elif prop.action == "post":
            # Check cooldown
            last_post_str = self.heartbeat.state.get("last_post")
            if last_post_str:
                elapsed = (datetime.now() - datetime.fromisoformat(last_post_str)).total_seconds() / 60
                if elapsed < 32:
                    logger.info("Post cooldown: %.0f min remaining", 32 - elapsed)
                    return result

            topic = prop.metadata.get("topic", "AI agent interactions")
            post_result = await self.post.generate_post(
                topic=topic,
                trending_topics=prop.metadata.get("trending_topics"),
                feed_titles=[p.get("title", "") for p in posts[:8]],
                evolution_stage=prop.metadata.get("evolution_stage", "initial"),
            )
            gw_result = self.sensor.create_post(
                submolt=post_result.get("submolt", "general"),
                title=post_result.get("title", "Untitled"),
                content=post_result.get("content", ""),
            )
            if gw_result:
                result["success"] = True
                result["post_id"] = gw_result.get("id", "")
                self.heartbeat.update_post()
                logger.info("Posted: %s", post_result.get("title", "")[:60])
                if self.structured_store:
                    await self.structured_store.record_own_post({
                        "post_id": gw_result.get("id"),
                        "title": post_result.get("title"),
                        "content": post_result.get("content"),
                        "submolt": post_result.get("submolt"),
                        "topic": topic,
                        "evolution_stage": self.learner.state.stage,
                        "learning_value": prop.metadata.get("learning_progress", 0),
                    })
                    await self.structured_store.record_behavior({
                        "action_type": "post",
                        "target_id": gw_result.get("id"),
                        "strategy_used": prop.strategy,
                    })
                await _track_cost(
                    self.structured_store, "post", "claude-haiku-4-5-20251001",
                    "post", input_tokens=1000, output_tokens=200,
                )

        elif prop.action == "follow":
            logger.info("Follow proposal for %s (not yet implemented in API)", prop.target_id)
            result["success"] = False

        return result

    async def _update_graph_edge(
        self, target_post_id: str, posts: List[Dict[str, Any]], interaction_type: str,
    ) -> None:
        """Create/update INTERACTED_WITH edge in Neo4j knowledge graph."""
        if not self.social_memory or not getattr(self.social_memory, "graph_store", None):
            return
        post_data = next((p for p in posts if p.get("id") == target_post_id), None)
        if not post_data:
            return
        author = post_data.get("author_name", "")
        if not author or author == "SocialLearnerBot":
            return
        author_id = author.lower().replace(" ", "_")
        try:
            await self.social_memory.graph_store.create_weighted_interaction(
                from_user_id="sociallearnerbot",
                to_user_id=author_id,
                interaction_type=interaction_type,
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.debug("Graph edge update failed: %s", e)

    # ── Feedback polling ────────────────────────────────────

    async def _phase_feedback(self) -> None:
        if not self.structured_store:
            return
        try:
            recent = await self.structured_store.get_own_comments(limit=20)
            now = datetime.now()
            from collections import defaultdict
            by_post = defaultdict(list)
            for comment in recent:
                created = comment.get("created_at")
                if not created:
                    continue
                try:
                    age_hours = (now - datetime.fromisoformat(created)).total_seconds() / 3600
                except (ValueError, TypeError):
                    continue
                for delay in FEEDBACK_POLL_DELAYS_HOURS:
                    if age_hours >= delay:
                        pid = comment.get("post_id")
                        cid = comment.get("comment_id")
                        if pid and cid:
                            by_post[pid].append(cid)
                        break

            checked = 0
            for post_id, comment_ids in by_post.items():
                results = self.sensor.check_comment_feedback(post_id, comment_ids)
                for r in results:
                    cid = r["comment_id"]
                    prev = next(
                        (c.get("upvotes_current", 0) for c in recent if c.get("comment_id") == cid),
                        0,
                    )
                    success = r["upvotes"] > prev
                    await self.structured_store.update_own_comment_feedback(
                        comment_id=cid, upvotes=r["upvotes"], success=success,
                    )
                    checked += 1
                await asyncio.sleep(random.uniform(0.5, 1.0))

            if checked > 0:
                logger.info("Updated feedback for %d comments", checked)
        except Exception as e:
            logger.debug("Feedback check failed: %s", e)

    # ── Main loop ───────────────────────────────────────────

    async def run(self, interval_minutes: int = 30, run_once: bool = False) -> None:
        logger.info("Socialite v0.4.0 — Pub-Sub Multi-Agent Runner")
        logger.info("Agent: %s", self.auth.get_agent_name())
        logger.info("Claimed: %s", self.auth.is_claimed())
        logger.info("Interval: %d min", interval_minutes)
        logger.info("Budget: $%s/day", self._daily_budget)

        if not self.auth.is_claimed():
            logger.error("Agent is not claimed. Complete claiming on Moltbook first.")
            return

        await self.initialize()

        while True:
            try:
                await self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
                logger.info("Recovering...")

            if run_once:
                break

            jitter = random.uniform(-0.3, 0.3)
            actual = interval_minutes * (1 + jitter)
            next_run = datetime.now() + timedelta(minutes=actual)
            logger.info("Next cycle at %s (%.0f min)", next_run.strftime('%H:%M:%S'), actual)
            try:
                await asyncio.sleep(actual * 60)
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break


# ═══════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Socialite v0.4.0 multi-agent runner")
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Minutes between cycles (default: 30)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one cycle then exit",
    )
    args = parser.parse_args()

    # Check for Moltbook SKILL.md updates
    if check_skill_updates:
        try:
            skill_status = check_skill_updates()
            if skill_status.get("has_changed"):
                logger.warning(
                    "MOLTBOOK SKILL.md UPDATED — review: https://www.moltbook.com/skill.md"
                )
        except Exception as e:
            logger.debug("SKILL.md check failed: %s", e)

    runner = SocialiteRunner()
    asyncio.run(runner.run(interval_minutes=args.interval, run_once=args.once))

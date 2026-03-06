#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Socialite v1.0.0 — Autonomous Multi-Agent Launcher

Agents run autonomously with bidirectional communication via MsgHub.
Runner only initializes agents, wires the full-mesh hub, and launches
all agent run loops as concurrent tasks.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Core framework ──────────────────────────────────────────
from core.message import Message
from core.msghub import MsgHub
from core.ab_strategy import ABSelector
from core.message_logger import MessageLogger
from core.base_agent import BaseAgent
from core.rate_limit import RateLimitPolicy

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

def setup_logging() -> None:
    """Configure logging with console + rotating file handlers."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    console_fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    file_fmt = "%(asctime)s [%(name)s] %(levelname)-8s %(message)s"

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(console_fmt))
    root.addHandler(console)

    # Main file handler — 10 MB × 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "runner.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(file_fmt))
    root.addHandler(file_handler)

    # Error-only file handler — 5 MB × 3 backups
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(file_fmt))
    root.addHandler(error_handler)


setup_logging()
logger = logging.getLogger("runner")


# ═══════════════════════════════════════════════════════════
# SocialiteRunner — Autonomous Agent Launcher
# ═══════════════════════════════════════════════════════════

class SocialiteRunner:
    """Initializes 9 agents, wires full-mesh hub, launches autonomous loops."""

    def __init__(self) -> None:
        # ── Moltbook client ──
        self.moltbook_config = MoltbookConfig.from_env()
        self.auth = MoltbookAuth(self.moltbook_config)
        self.auth.load_credentials()
        self.client = MoltbookClient(
            self.moltbook_config, self.auth,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.strategy = EngagementStrategy(self.moltbook_config)
        self.heartbeat = HeartbeatState(self.moltbook_config.state_path)

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

        self._daily_budget = float(os.getenv("DAILY_BUDGET_USD", "5.0"))
        self._agents: List[BaseAgent] = []
        self._tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self.rate_limits: Optional[RateLimitPolicy] = None

    # ── Initialization ──────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize social memory + all 9 agents + full-mesh MsgHub."""
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
        self.sensor = SensorAgent(
            name="sensor", client=self.client,
            structured_store=self.structured_store,
        )
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

        # ── MsgHub: full-mesh bidirectional ──
        self._agents = [
            self.sensor, self.analysis, self.comment, self.post,
            self.upvote, self.follow, self.coordinator,
            self.learner, self.observer,
        ]
        round_budget = int(os.getenv("ROUND_DIALOGUE_BUDGET", "0"))
        self.hub = MsgHub(
            participants=self._agents,
            message_logger=self.msg_logger,
            round_budget=round_budget,
        )
        self.hub.connect_all()

        # Inject hub into all agents
        for agent in self._agents:
            agent.set_hub(self.hub)

        # Load learner state
        await self.learner.load_state()

        # Inject rate limit policy into sensor execution gateway.
        self.rate_limits = RateLimitPolicy(
            structured_store=self.structured_store,
            post_cooldown_minutes=self.moltbook_config.post_cooldown_minutes,
            comment_cooldown_seconds=self.moltbook_config.comment_cooldown_seconds,
            max_comments_per_day=self.moltbook_config.max_comments_per_day,
        )
        self.sensor.set_execution_context(
            rate_limits=self.rate_limits,
            heartbeat=self.heartbeat,
        )

        # ── Inject cost recorder into LLM clients ──
        if self.structured_store:
            store = self.structured_store

            def _make_cost_recorder(agent_name: str) -> callable:
                def recorder(usage: dict) -> None:
                    import asyncio
                    coro = store.record_api_cost({
                        "agent_name": agent_name,
                        "model": usage.get("model", ""),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "estimated_cost_usd": usage.get("estimated_cost_usd", 0.0),
                        "action_type": usage.get("provider", ""),
                    })
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(coro)
                    except RuntimeError:
                        pass
                return recorder

            for agent in self._agents:
                if hasattr(agent, "llm") and agent.llm is not None:
                    agent.llm.cost_recorder = _make_cost_recorder(agent.name)
            logger.info("Cost recorder injected into LLM-enabled agents")

        logger.info("All 9 agents initialized with full-mesh hub")
        logger.info("Topology edges: %d", sum(len(v) for v in self.hub.topology().values()))
        if round_budget > 0:
            logger.info("Round dialogue budget enabled: %d", round_budget)

    # ── Launch ───────────────────────────────────────────────

    async def run(self, run_once: bool = False) -> None:
        """Launch all agents as concurrent autonomous tasks."""
        logger.info("Socialite v1.0.0 — Autonomous Multi-Agent System")
        logger.info("Agent: %s", self.auth.get_agent_name())
        logger.info("Claimed: %s", self.auth.is_claimed())
        logger.info("Budget: $%s/day", self._daily_budget)

        if not self.auth.is_claimed():
            logger.error("Agent is not claimed. Complete claiming on Moltbook first.")
            return

        await self.initialize()

        if run_once:
            logger.info("Running single-shot mode...")
            await self._run_once()
            return

        self._install_signal_handlers()

        # Continuous mode: launch autonomous agent loops only.
        # Passive agents can still react via observe() without a run loop.
        logger.info("Launching autonomous agent loops...")
        active_agents: List[BaseAgent] = [
            agent for agent in self._agents
            if agent.__class__.run is not BaseAgent.run
        ]

        if not active_agents:
            logger.warning("No autonomous run loops found; exiting.")
            return

        for agent in active_agents:
            task = asyncio.create_task(agent.start(), name=f"agent-{agent.name}")
            self._tasks.append(task)

        logger.info("Launched %d autonomous agent tasks", len(self._tasks))

        stop_waiter = asyncio.create_task(self._stop_event.wait(), name="runner-stop")
        try:
            done, _pending = await asyncio.wait(
                [*self._tasks, stop_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_waiter in done:
                logger.info("Stop signal received, shutting down...")
            else:
                # At least one agent task exited unexpectedly.
                for task in done:
                    if task is stop_waiter:
                        continue
                    exc = task.exception()
                    if exc:
                        logger.error("Agent task crashed: %s", exc, exc_info=exc)
                    else:
                        logger.warning("Agent task exited unexpectedly: %s", task.get_name())
                self._stop_event.set()
        finally:
            if not stop_waiter.done():
                stop_waiter.cancel()
                await asyncio.gather(stop_waiter, return_exceptions=True)
            await self._shutdown()

    async def _run_once(self) -> None:
        """Single-shot: inject one raw_feed event and wait for message chain."""
        if not self.sensor or not self.hub:
            logger.error("Runner not initialized")
            return
        self.hub.reset_round()

        next_cycle = int(self.heartbeat.state.get("cycle_count", 0)) + 1
        self.msg_logger.set_cycle(next_cycle)
        logger.info("Single-shot cycle #%d", next_cycle)

        posts = self.sensor.fetch_feed(sort="hot", limit=self.sensor.fetch_limit)
        if not posts:
            logger.warning("No posts fetched in single-shot cycle")
            return

        raw_msg = Message(
            name="sensor",
            role="assistant",
            content=json.dumps({"type": "raw_feed", "count": len(posts)}),
            metadata={"type": "raw_feed", "posts": posts, "count": len(posts)},
        )
        # Trigger the autonomous message chain:
        # raw_feed -> analysis_result -> proposals -> approvals -> exec_command -> action_completed
        await self.hub.broadcast(raw_msg)

        # Log communication stats
        stats = self.hub.round_stats()
        logger.info("Communication stats: %s", json.dumps(stats, indent=2))

        self.heartbeat.state["cycle_count"] = next_cycle
        self.heartbeat.update_check()

        logger.info("Single-shot complete")

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: self._request_stop(s))
            except NotImplementedError:
                # add_signal_handler is not available on all platforms/event loops.
                logger.debug("Signal handlers unavailable for %s", sig)

    def _request_stop(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s", sig.name)
        self._stop_event.set()

    async def _shutdown(self) -> None:
        """Gracefully stop all agents."""
        logger.info("Shutting down...")
        self._stop_event.set()
        for agent in self._agents:
            agent.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("All agents stopped")


# ═══════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Socialite v1.0.0 autonomous agent launcher")
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
    asyncio.run(runner.run(run_once=args.once))

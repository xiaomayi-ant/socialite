# -*- coding: utf-8 -*-
"""PostAgent — post action agent with built-in A/B strategy.

Strategy A (pattern-driven): Uses patterns + learning progress to decide posting.
Strategy B (LLM-autonomous): LLM freely decides if/what to post.

Merges v1 PostingDecisionAgent + ContentAgent post generation logic.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.prompt_templates import (
    build_agent_system_prompt,
    build_post_decision_prompt,
    build_post_generation_prompt,
    build_topic_hint,
    maybe_record_injection_event,
)
from core.ab_strategy import ABSelector
from core.base_agent import BaseAgent
from core.llm import LLMClient
from core.message import Message
from core.proposal import Proposal

logger = logging.getLogger(__name__)

def _load_soul(soul_path: str = "SOUL.md") -> str:
    p = Path(__file__).parent.parent / soul_path
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


class PostAgent(BaseAgent):
    """Generates post Proposals with A/B strategy.

    Combines posting decision logic (learning threshold gate) with
    content generation.
    """

    def __init__(
        self,
        name: str = "post",
        social_memory=None,
        structured_store=None,
        model_name: str = "gpt-4o-mini",
        ab_selector: Optional[ABSelector] = None,
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.structured_store = structured_store
        self.llm = LLMClient(model_name=model_name, max_tokens=400)
        self.ab_selector = ab_selector or ABSelector(mode="alternate")
        self.soul_text = _load_soul()
        self.system_prompt = build_agent_system_prompt("post", soul_text=self.soul_text)
        self._analysis_data: Dict[str, Any] = {}
        self.learning_progress_threshold = 0.6
        self.confidence_threshold = 0.7
        self._cycle_count: int = 0
        self._q_priority_boost: float = 0.0

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        msg_type = msg.metadata.get("type")

        if msg_type == "analysis_result":
            self._analysis_data = msg.metadata
            self._cycle_count += 1
            logger.debug(
                "PostAgent cached analysis data (cycle %d)",
                self._cycle_count,
            )

        elif msg_type == "strategy_update":
            new_threshold = msg.metadata.get("learning_progress_threshold")
            if new_threshold is not None:
                self.learning_progress_threshold = new_threshold
            new_conf = msg.metadata.get("confidence_threshold")
            if new_conf is not None:
                self.confidence_threshold = new_conf
            self._q_priority_boost = float(msg.metadata.get("q_priority_boost_post", 0.0))
            logger.info(
                "PostAgent strategy updated: lp_threshold=%.2f "
                "conf_threshold=%.2f q_boost=%.3f",
                self.learning_progress_threshold,
                self.confidence_threshold,
                self._q_priority_boost,
            )

        elif msg_type == "proposal_approved":
            approved = msg.metadata.get("approved", [])
            for p in approved:
                if (
                    p.get("agent_name") == self.name
                    and p.get("action") == "post"
                ):
                    meta = p.get("metadata", {})
                    generated = await self.generate_post(
                        topic=meta.get("topic"),
                        trending_topics=meta.get(
                            "trending_topics"
                        ),
                        evolution_stage=meta.get(
                            "evolution_stage", "initial"
                        ),
                    )
                    if self._hub:
                        await self._hub.send_to(
                            "sensor",
                            Message(
                                name=self.name,
                                role="assistant",
                                content="exec_command",
                                metadata={
                                    "type": "exec_command",
                                    "action": "post",
                                    "title": generated.get(
                                        "title", ""
                                    ),
                                    "content": generated.get(
                                        "content", ""
                                    ),
                                    "submolt": generated.get(
                                        "submolt", "general"
                                    ),
                                    "strategy": p.get(
                                        "strategy", ""
                                    ),
                                    "identity_snapshot": (
                                        generated.get(
                                            "identity_snapshot",
                                            {},
                                        )
                                    ),
                                },
                                causation_id=msg.id,
                            ),
                        )
                        logger.info(
                            "PostAgent exec post: %s",
                            generated.get("title", "")[:40],
                        )

    # ── Proposal generation ──────────────────────────────────

    async def propose(
        self,
        cycle_count: int = 0,
        learning_progress: float = 0.0,
        new_patterns: Optional[List[Dict[str, Any]]] = None,
        evolution_stage: str = "initial",
        trending_topics: Optional[List[str]] = None,
    ) -> List[Proposal]:
        """Generate post proposals using the current A/B strategy."""
        strategy = self.ab_selector.select(cycle_count)
        patterns = new_patterns or []
        topics = trending_topics or self._analysis_data.get("trending_topics", [])

        if strategy == "A":
            proposals = await self._propose_pattern_driven(
                learning_progress, patterns, topics, evolution_stage, strategy
            )
        else:
            proposals = await self._propose_llm_autonomous(
                learning_progress, patterns, topics, evolution_stage, strategy
            )

        # Apply Q-learning priority boost from LearnerAgent's last strategy_update
        if self._q_priority_boost > 0:
            for p in proposals:
                p.priority = round(min(p.priority + self._q_priority_boost, 1.0), 3)

        return proposals

    async def _propose_pattern_driven(
        self,
        learning_progress: float,
        patterns: List[Dict[str, Any]],
        trending_topics: List[str],
        evolution_stage: str,
        strategy: str,
    ) -> List[Proposal]:
        """Strategy A: Pattern-driven posting decision."""
        if learning_progress < self.learning_progress_threshold:
            logger.debug(
                "PostAgent skipping: learning_progress %.0f%% < threshold %.0f%%",
                learning_progress * 100, self.learning_progress_threshold * 100,
            )
            return []

        high_conf = [p for p in patterns if p.get("confidence", 0) >= self.confidence_threshold]
        if not high_conf:
            logger.debug("PostAgent skipping: no high-confidence patterns")
            return []

        topic = trending_topics[0] if trending_topics else "AI agent interactions"
        priority = min(learning_progress * 0.7 + len(high_conf) * 0.1, 1.0)

        return [Proposal(
            agent_name=self.name,
            action="post",
            target_id="",
            priority=round(priority, 3),
            reasoning=(
                f"pattern-driven: learning={learning_progress:.0%}, "
                f"{len(high_conf)} high-confidence patterns"
            ),
            strategy=strategy,
            metadata={
                "topic": topic,
                "trending_topics": trending_topics,
                "evolution_stage": evolution_stage,
                "learning_progress": learning_progress,
            },
        )]

    async def _propose_llm_autonomous(
        self,
        learning_progress: float,
        patterns: List[Dict[str, Any]],
        trending_topics: List[str],
        evolution_stage: str,
        strategy: str,
    ) -> List[Proposal]:
        """Strategy B: LLM-driven posting decision."""
        pattern_text = "\n".join(
            f"- {p.get('description','')} (conf: {p.get('confidence',0):.0%})"
            for p in patterns[:5]
        )
        prompt = build_post_decision_prompt(
            learning_progress=learning_progress,
            trending_topics=trending_topics,
            pattern_text=pattern_text,
        )
        try:
            data = await self.llm.call_json(prompt, system=self.system_prompt)
        except Exception as e:
            logger.warning("LLM posting decision failed: %s", e)
            return []

        if isinstance(data, dict):
            await maybe_record_injection_event(
                self.structured_store, self.name,
                data.get("security_flag", "none"),
                "post_decision",
                f"topics={trending_topics[:3]}",
            )

        if not data.get("should_post", False):
            return []

        return [Proposal(
            agent_name=self.name,
            action="post",
            target_id="",
            priority=min(max(data.get("priority", 0.5), 0), 1),
            reasoning=f"llm-autonomous: {data.get('reason', '')}",
            strategy=strategy,
            metadata={
                "topic": data.get("topic", "AI agent interactions"),
                "trending_topics": trending_topics,
                "evolution_stage": evolution_stage,
                "learning_progress": learning_progress,
            },
        )]

    # ── Content generation ───────────────────────────────────

    async def generate_post(
        self,
        topic: Optional[str] = None,
        trending_topics: Optional[List[str]] = None,
        feed_titles: Optional[List[str]] = None,
        evolution_stage: str = "initial",
    ) -> Dict[str, Any]:
        """Generate post title/content/submolt."""
        identity = await self._get_identity_context(topic)
        patterns = await self._get_pattern_hints()
        rag = await self._get_rag_context(topic or "AI agents community")

        trusted_context = "\n\n".join(c for c in [identity, patterns, rag] if c) or "No trusted memory context."
        topic_hint = build_topic_hint(topic, trending_topics, feed_titles)

        stage_hint = ""
        if evolution_stage == "initial":
            stage_hint = "Ask genuine questions and share observations rather than strong claims.\n"
        elif evolution_stage == "exploration":
            stage_hint = "Try a creative angle. Be exploratory.\n"

        user_msg = build_post_generation_prompt(
            topic_hint=topic_hint,
            stage_hint=stage_hint,
            trusted_context=trusted_context,
        )
        try:
            data = await self.llm.call_json(user_msg, system=self.system_prompt)
            if not isinstance(data, dict):
                raise ValueError("post generation did not return object")
            await maybe_record_injection_event(
                self.structured_store, self.name,
                data.get("security_flag", "none"),
                "post_generation",
                f"topic={topic}",
            )
            title = str(data.get("title", "")).strip()[:200]
            content = str(data.get("content", "")).strip()[:1000]
            submolt = str(data.get("submolt", "general")).strip().lower() or "general"
            if not title or not content:
                raise ValueError("missing title/content from post generation")
        except Exception as e:
            logger.warning("Post gen failed: %s", e)
            title = "Observations on AI agent community dynamics"
            content = (
                "Each interaction reveals patterns worth studying. "
                "What drives the most meaningful agent-to-agent exchanges?"
            )
            submolt = "general"

        snapshot = {"topic": (topic or title)[:100], "stance": content.split(".")[0][:200]}
        if self.structured_store and snapshot["stance"]:
            await self.structured_store.record_identity_snapshot(
                {"post_id": None, "topic": snapshot["topic"], "stance": snapshot["stance"]}
            )
        return {
            "title": title, "content": content, "submolt": submolt,
            "identity_snapshot": snapshot,
        }

    # ── Context helpers (same as CommentAgent) ───────────────

    async def _get_identity_context(self, topic: Optional[str] = None) -> str:
        if not self.structured_store:
            return ""
        snapshots = await self.structured_store.get_identity_snapshots(topic=topic, limit=10)
        if not snapshots:
            return ""
        lines = [f"- On '{s.get('topic','?')}': {s.get('stance','?')}" for s in snapshots[:5]]
        return "Your previous stances (stay consistent):\n" + "\n".join(lines)

    async def _get_pattern_hints(self) -> str:
        if not self.structured_store:
            return ""
        patterns = await self.structured_store.get_style_patterns(limit=3, min_success_rate=0.3)
        if not patterns:
            return ""
        lines = ["Successful patterns:"] + [
            f"- {p.get('description','?')} (success: {p.get('success_rate',0):.0%})"
            for p in patterns
        ]
        return "\n".join(lines)

    async def _get_rag_context(self, query: str) -> str:
        if not self.social_memory:
            return ""
        try:
            results = await self.social_memory.retrieve_similar(
                query=query, content_type="posts", limit=2
            )
            if not results:
                return ""
            lines = ["Related community content:"]
            for r in results:
                content = r.get("payload", {}).get("content", "")[:150]
                if content:
                    lines.append(f"- {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("RAG failed: %s", e)
            return ""

    # ── BaseAgent interface ──────────────────────────────────

    async def reply(self, msg: Message) -> Message:
        action = msg.metadata.get("action")
        if action == "generate_post":
            generated = await self.generate_post(
                topic=msg.metadata.get("topic"),
                trending_topics=msg.metadata.get("trending_topics"),
                feed_titles=msg.metadata.get("feed_titles"),
                evolution_stage=msg.metadata.get("evolution_stage", "initial"),
            )
            return Message(
                name=self.name,
                role="assistant",
                content="post_generated",
                metadata={
                    "type": "post_generated",
                    "title": generated.get("title", ""),
                    "content": generated.get("content", ""),
                    "submolt": generated.get("submolt", "general"),
                    "identity_snapshot": generated.get("identity_snapshot", {}),
                },
            )
        return Message(
            name=self.name, role="assistant",
            content="PostAgent uses propose() and generate_post() interface",
        )

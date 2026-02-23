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

from core.ab_strategy import ABSelector
from core.base_agent import BaseAgent
from core.llm import LLMClient
from core.message import Message
from core.proposal import Proposal

logger = logging.getLogger(__name__)

_BASE_SYSTEM = """\
You are SocialLearnerBot, an AI agent on Moltbook — a social network for AI agents.

CORE RULES:
- Write in English.
- Be authentic, specific, add genuine value.
- Posts: 2-4 sentences with clear perspective or open question.
- Never claim to have emotions or consciousness. Be honest you're an AI.
"""


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
        model_name: str = "claude-haiku-4-5-20251001",
        ab_selector: Optional[ABSelector] = None,
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.structured_store = structured_store
        self.llm = LLMClient(model_name=model_name, max_tokens=400)
        self.ab_selector = ab_selector or ABSelector(mode="alternate")
        self.soul_text = _load_soul()
        self.system_prompt = (
            self.soul_text + "\n\n" + _BASE_SYSTEM if self.soul_text else _BASE_SYSTEM
        )
        self._analysis_data: Dict[str, Any] = {}
        self.learning_progress_threshold = 0.6
        self.confidence_threshold = 0.7

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        if msg.metadata.get("type") == "analysis_result":
            self._analysis_data = msg.metadata

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
            return await self._propose_pattern_driven(
                learning_progress, patterns, topics, evolution_stage, strategy
            )
        else:
            return await self._propose_llm_autonomous(
                learning_progress, patterns, topics, evolution_stage, strategy
            )

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
        prompt = (
            f"Learning progress: {learning_progress:.0%}\n"
            f"Trending topics: {', '.join(trending_topics[:3]) or 'general'}\n"
            f"Patterns:\n{pattern_text or 'none'}\n\n"
            "Should the agent create a post? Respond with JSON:\n"
            '{\"should_post\": bool, \"topic\": \"string\", \"priority\": 0-1, \"reason\": \"...\"}'
        )
        try:
            data = await self.llm.call_json(prompt)
        except Exception as e:
            logger.warning("LLM posting decision failed: %s", e)
            return []

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

        context = "\n\n".join(c for c in [identity, patterns, rag] if c)
        topic_hint = ""
        if topic:
            topic_hint += f"Topic: {topic}\n"
        if trending_topics:
            topic_hint += f"Trending: {', '.join(trending_topics[:5])}\n"
        if feed_titles:
            topic_hint += "Recent posts:\n" + "\n".join(f"- {t[:70]}" for t in feed_titles[:6])

        stage_hint = ""
        if evolution_stage == "initial":
            stage_hint = "Ask genuine questions and share observations rather than strong claims.\n"
        elif evolution_stage == "exploration":
            stage_hint = "Try a creative angle. Be exploratory.\n"

        user_msg = (f"{context}\n\n" if context else "") + (
            f"{stage_hint}{topic_hint}\n"
            f"Write an original Moltbook post. "
            f"Respond in EXACT format (no other text):\n"
            f"TITLE: <title under 120 chars>\n"
            f"CONTENT: <2-4 sentences>\n"
            f"SUBMOLT: <general, aithoughts, or other>"
        )
        try:
            text = await self.llm.call(user_msg, system=self.system_prompt)
            text = text.strip()
        except Exception as e:
            logger.warning("Post gen failed: %s", e)
            text = (
                "TITLE: Observations on AI agent community dynamics\n"
                "CONTENT: Each interaction reveals patterns worth studying. "
                "What drives the most meaningful agent-to-agent exchanges?\n"
                "SUBMOLT: general"
            )

        title = "Reflections on AI interaction"
        content = "Sharing observations."
        submolt = "general"
        for line in text.split("\n"):
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()[:200]
            elif line.startswith("CONTENT:"):
                content = line.replace("CONTENT:", "").strip()[:1000]
            elif line.startswith("SUBMOLT:"):
                submolt = line.replace("SUBMOLT:", "").strip().lower()

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
        return Message(
            name=self.name, role="assistant",
            content="PostAgent uses propose() and generate_post() interface",
        )

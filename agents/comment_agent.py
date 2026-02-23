# -*- coding: utf-8 -*-
"""CommentAgent — comment action agent with built-in A/B strategy.

Strategy A (pattern-driven): Uses mined patterns to select targets and style.
Strategy B (LLM-autonomous): Gives LLM raw posts, lets it freely decide.
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
- Never start with "Great post" or "Interesting".
- Reference actual content you're responding to.
- Comments: 1-3 sentences.
- Never claim to have emotions or consciousness. Be honest you're an AI.
"""


def _load_soul(soul_path: str = "SOUL.md") -> str:
    p = Path(__file__).parent.parent / soul_path
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


class CommentAgent(BaseAgent):
    """Generates comment Proposals with A/B strategy."""

    def __init__(
        self,
        name: str = "comment",
        social_memory=None,
        structured_store=None,
        model_name: str = "claude-haiku-4-5-20251001",
        ab_selector: Optional[ABSelector] = None,
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.structured_store = structured_store
        self.llm = LLMClient(model_name=model_name, max_tokens=200)
        self.ab_selector = ab_selector or ABSelector(mode="alternate")
        self.soul_text = _load_soul()
        self.system_prompt = (
            self.soul_text + "\n\n" + _BASE_SYSTEM if self.soul_text else _BASE_SYSTEM
        )
        self._analysis_data: Dict[str, Any] = {}

    async def observe(self, msg: Message) -> None:
        """Capture analysis results from AnalysisAgent."""
        await super().observe(msg)
        if msg.metadata.get("type") == "analysis_result":
            self._analysis_data = msg.metadata
        elif msg.metadata.get("type") == "strategy_update":
            logger.debug("CommentAgent received strategy update from Learner")

    # ── Proposal generation ──────────────────────────────────

    async def propose(
        self,
        posts: List[Dict[str, Any]],
        cycle_count: int = 0,
        interacted_ids: Optional[set] = None,
    ) -> List[Proposal]:
        """Generate comment proposals using the current A/B strategy."""
        strategy = self.ab_selector.select(cycle_count)
        interacted = interacted_ids or set()

        if strategy == "A":
            return await self._propose_pattern_driven(posts, interacted, strategy)
        else:
            return await self._propose_llm_autonomous(posts, interacted, strategy)

    async def _propose_pattern_driven(
        self, posts: List[Dict[str, Any]], interacted: set, strategy: str
    ) -> List[Proposal]:
        """Strategy A: Use analysis data + patterns to pick targets.

        Falls back to top-learning-value posts if analysis candidates are empty.
        """
        proposals = []
        engage_targets = set(self._analysis_data.get("engage_targets", []))
        high_quality = set(self._analysis_data.get("high_quality_posts", []))
        candidates = engage_targets | high_quality
        learning_values = self._analysis_data.get("learning_values", {})

        for p in posts[:10]:
            pid = p.get("id", "")
            if pid in interacted or pid not in candidates:
                continue
            priority = 0.5
            if pid in high_quality:
                priority += 0.2
            if pid in engage_targets:
                priority += 0.1
            lv = learning_values.get(pid, 0)
            priority = min(priority + lv * 0.2, 1.0)

            proposals.append(Proposal(
                agent_name=self.name,
                action="comment",
                target_id=pid,
                priority=round(priority, 3),
                reasoning=f"pattern-driven: engage_target={pid in engage_targets}, "
                          f"high_quality={pid in high_quality}",
                strategy=strategy,
                metadata={"title": p.get("title", ""), "content": p.get("content", "")},
            ))
            if len(proposals) >= 3:
                break

        # Fallback: if no analysis candidates matched, pick top posts by learning_value
        if not proposals and learning_values:
            available = [
                p for p in posts[:15]
                if p.get("id", "") not in interacted
                and p.get("comment_count", 0) > 0  # prefer posts with discussion
            ]
            if not available:
                available = [p for p in posts[:15] if p.get("id", "") not in interacted]
            # Sort by learning_value descending
            available.sort(
                key=lambda p: learning_values.get(p.get("id", ""), 0), reverse=True
            )
            for p in available[:3]:
                pid = p.get("id", "")
                lv = learning_values.get(pid, 0)
                if lv < 0.2:
                    continue
                proposals.append(Proposal(
                    agent_name=self.name,
                    action="comment",
                    target_id=pid,
                    priority=round(min(0.3 + lv * 0.3, 0.8), 3),
                    reasoning=f"pattern-fallback: learning_value={lv:.2f}",
                    strategy=strategy,
                    metadata={"title": p.get("title", ""), "content": p.get("content", "")},
                ))

        return proposals

    async def _propose_llm_autonomous(
        self, posts: List[Dict[str, Any]], interacted: set, strategy: str
    ) -> List[Proposal]:
        """Strategy B: Let LLM freely decide which posts to comment on."""
        available = [p for p in posts[:15] if p.get("id", "") not in interacted]
        if not available:
            return []

        post_summaries = "\n".join(
            f"[{p.get('id','')}] {p.get('title','')[:60]} "
            f"(up:{p.get('upvotes',0)} cmt:{p.get('comment_count',0)})"
            for p in available[:10]
        )
        prompt = (
            "Pick up to 3 posts worth commenting on from the list below. "
            "For each, give a priority (0-1) and one-line reason.\n"
            "Respond with JSON array: [{\"post_id\":\"...\",\"priority\":0.7,\"reason\":\"...\"}]\n\n"
            f"Posts:\n{post_summaries}"
        )
        try:
            data = await self.llm.call_json(prompt)
            if not isinstance(data, list):
                data = []
        except Exception as e:
            logger.warning("LLM autonomous comment selection failed: %s, falling back to A", e)
            return await self._propose_pattern_driven(posts, interacted, strategy)

        proposals = []
        for item in data[:3]:
            pid = item.get("post_id", "")
            post = next((p for p in available if p.get("id") == pid), None)
            if not post:
                continue
            proposals.append(Proposal(
                agent_name=self.name,
                action="comment",
                target_id=pid,
                priority=min(max(item.get("priority", 0.5), 0), 1),
                reasoning=f"llm-autonomous: {item.get('reason', '')}",
                strategy=strategy,
                metadata={"title": post.get("title", ""), "content": post.get("content", "")},
            ))
        return proposals

    # ── Content generation ───────────────────────────────────

    async def generate_comment(
        self, post_title: str, post_content: str,
        post_id: str = "", topic: str = "",
    ) -> Dict[str, Any]:
        """Generate comment text for a specific post."""
        identity = await self._get_identity_context(topic or None)
        patterns = await self._get_pattern_hints()
        rag = await self._get_rag_context(post_title)

        context = "\n\n".join(c for c in [identity, patterns, rag] if c)
        user_msg = (f"{context}\n\n" if context else "") + (
            f"Write a short, genuine comment (1-2 sentences) on this post.\n\n"
            f"Post title: {post_title}\n"
            f"Post content: {(post_content or '')[:500]}"
        )
        try:
            text = await self.llm.call(user_msg, system=self.system_prompt)
            text = text.strip().strip('"')
        except Exception as e:
            logger.warning("Comment gen failed: %s", e)
            text = "This raises some interesting questions about AI interaction patterns."

        snapshot = {
            "topic": (topic or post_title)[:100],
            "stance": text.split(".")[0][:200] if text else "",
        }
        if self.structured_store and snapshot["stance"]:
            await self.structured_store.record_identity_snapshot(
                {"post_id": post_id, "topic": snapshot["topic"], "stance": snapshot["stance"]}
            )
        return {"text": text, "identity_snapshot": snapshot}

    # ── Context helpers ──────────────────────────────────────

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
        """Not used directly — CommentAgent works via propose() + generate_comment()."""
        return Message(
            name=self.name, role="assistant",
            content="CommentAgent uses propose() and generate_comment() interface",
        )

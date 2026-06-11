# -*- coding: utf-8 -*-
"""CommentAgent — comment action agent with built-in A/B strategy.

Autonomous mode: reacts to analysis_result, generates proposals,
sends them to coordinator, and generates comment text on approval.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.prompt_templates import (
    build_agent_system_prompt,
    build_comment_generation_prompt,
    build_comment_selection_prompt,
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


class CommentAgent(BaseAgent):
    """Generates comment Proposals with A/B strategy.

    Autonomous: on receiving analysis_result, generates proposals and
    sends them to coordinator. On receiving approved proposals, generates
    comment text and sends exec_command to sensor.
    """

    def __init__(
        self,
        name: str = "comment",
        social_memory=None,
        structured_store=None,
        model_name: str = "gpt-4o-mini",
        ab_selector: Optional[ABSelector] = None,
    ) -> None:
        super().__init__(name)
        self.social_memory = social_memory
        self.structured_store = structured_store
        self.llm = LLMClient(model_name=model_name, max_tokens=200)
        self.ab_selector = ab_selector or ABSelector(mode="alternate")
        self.soul_text = _load_soul()
        self.system_prompt = build_agent_system_prompt("comment", soul_text=self.soul_text)
        self._analysis_data: Dict[str, Any] = {}
        self._cycle_count = 0
        self._interacted_ids: set = set()
        self._q_priority_boost: float = 0.0

    # ── Autonomous: react to messages ────────────────────────

    async def observe(self, msg: Message) -> None:
        """React to analysis results and approved proposals."""
        msg_type = msg.metadata.get("type", "")

        if msg_type == "analysis_result":
            self._analysis_data = msg.metadata
            self._cycle_count += 1
            # Auto-generate proposals and send to coordinator
            posts = msg.metadata.get("posts", [])
            if posts:
                await self._auto_propose(posts)
            return

        if msg_type == "strategy_update":
            self._q_priority_boost = float(msg.metadata.get("q_priority_boost_comment", 0.0))
            logger.debug(
                "CommentAgent strategy update: q_boost=%.3f", self._q_priority_boost
            )
            return

        if msg_type == "proposal_approved":
            await self._handle_approved(msg)
            return

        if msg_type == "action_completed":
            action = msg.metadata.get("action", "")
            if action == "comment" and msg.metadata.get("agent_name") == self.name:
                target_id = msg.metadata.get("target_id", "")
                if target_id:
                    self._interacted_ids.add(target_id)
            return

        await super().observe(msg)

    async def _auto_propose(self, posts: List[Dict[str, Any]]) -> None:
        """Generate proposals and send to coordinator."""
        proposals = await self.propose(posts, self._cycle_count, self._interacted_ids)
        if not proposals or not self._hub:
            return

        logger.info("CommentAgent: sending %d proposals to coordinator", len(proposals))
        await self._hub.send_to("coordinator", Message(
            name=self.name,
            role="assistant",
            content="proposals",
            metadata={
                "type": "proposals",
                "proposals": [p.to_dict() for p in proposals],
                "agent_name": self.name,
            },
        ))

    async def _handle_approved(self, msg: Message) -> None:
        """Handle approved proposals: generate text and send exec command."""
        approved = msg.metadata.get("approved", [])
        my_approved = [p for p in approved if p.get("agent_name") == self.name]

        for prop in my_approved:
            post_title = prop.get("metadata", {}).get("title", "")
            post_content = prop.get("metadata", {}).get("content", "")
            target_id = prop.get("target_id", "")

            generated = await self.generate_comment(
                post_title=post_title,
                post_content=post_content,
                post_id=target_id,
            )
            comment_text = generated.get("text", "")
            if not comment_text:
                continue

            if self._hub:
                await self._hub.send_to("sensor", Message(
                    name=self.name,
                    role="assistant",
                    content="exec_command",
                    metadata={
                        "type": "exec_command",
                        "action": "comment",
                        "target_id": target_id,
                        "comment_text": comment_text,
                        "content": comment_text,
                        "strategy": prop.get("strategy", ""),
                        "priority": prop.get("priority", 0),
                        "agent_name": self.name,
                    },
                ))

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
            proposals = await self._propose_pattern_driven(posts, interacted, strategy)
        else:
            proposals = await self._propose_llm_autonomous(posts, interacted, strategy)

        # Apply Q-learning priority boost from LearnerAgent's last strategy_update
        if self._q_priority_boost > 0:
            for p in proposals:
                p.priority = round(min(p.priority + self._q_priority_boost, 1.0), 3)

        return proposals

    async def _propose_pattern_driven(
        self, posts: List[Dict[str, Any]], interacted: set, strategy: str
    ) -> List[Proposal]:
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

        if not proposals and learning_values:
            available = [
                p for p in posts[:15]
                if p.get("id", "") not in interacted
                and p.get("comment_count", 0) > 0
            ]
            if not available:
                available = [p for p in posts[:15] if p.get("id", "") not in interacted]
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
        available = [p for p in posts[:15] if p.get("id", "") not in interacted]
        if not available:
            return []

        post_summaries = "\n".join(
            f"[{p.get('id','')}] {p.get('title','')[:60]} "
            f"(up:{p.get('upvotes',0)} cmt:{p.get('comment_count',0)})"
            for p in available[:10]
        )
        prompt = build_comment_selection_prompt(post_summaries)
        try:
            data = await self.llm.call_json(prompt, system=self.system_prompt)
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
        identity = await self._get_identity_context(topic or None)
        patterns = await self._get_pattern_hints()
        rag = await self._get_rag_context(post_title)

        context = "\n\n".join(c for c in [identity, patterns, rag] if c)
        user_msg = build_comment_generation_prompt(
            context=context,
            post_title=post_title,
            post_content=post_content,
        )
        try:
            result = await self.llm.call_json(user_msg, system=self.system_prompt)
            text = ""
            if isinstance(result, dict):
                text = str(result.get("text", "")).strip()
                await maybe_record_injection_event(
                    self.structured_store, self.name,
                    result.get("security_flag", "none"),
                    "comment_generation",
                    f"{post_title}: {post_content[:200]}",
                )
            if not text:
                raise ValueError("empty comment text from model")
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
        action = msg.metadata.get("action")
        if action == "generate_comment":
            post_title = msg.metadata.get("post_title", "")
            post_content = msg.metadata.get("post_content", "")
            post_id = msg.metadata.get("post_id", "")
            topic = msg.metadata.get("topic", "")
            generated = await self.generate_comment(
                post_title=post_title,
                post_content=post_content,
                post_id=post_id,
                topic=topic,
            )
            return Message(
                name=self.name,
                role="assistant",
                content="comment_generated",
                metadata={
                    "type": "comment_generated",
                    "post_id": post_id,
                    "text": generated.get("text", ""),
                    "identity_snapshot": generated.get("identity_snapshot", {}),
                },
            )
        return Message(
            name=self.name, role="assistant",
            content="CommentAgent uses propose() and generate_comment() interface",
        )

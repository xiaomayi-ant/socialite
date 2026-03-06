# -*- coding: utf-8 -*-
"""UpvoteAgent — upvote action agent with A/B strategy.

Strategy A: Rule-based (upvotes > downvotes*2, has engagement).
Strategy B: LLM-based (let LLM assess post quality).
"""

import logging
from typing import Any, Dict, List, Optional

from agents.prompt_templates import build_agent_system_prompt, build_upvote_selection_prompt
from core.ab_strategy import ABSelector
from core.base_agent import BaseAgent
from core.llm import LLMClient
from core.message import Message
from core.proposal import Proposal

logger = logging.getLogger(__name__)


class UpvoteAgent(BaseAgent):
    """Evaluates posts for upvoting and generates Proposals."""

    def __init__(
        self,
        name: str = "upvote",
        model_name: str = "gpt-4o-mini",
        ab_selector: Optional[ABSelector] = None,
    ) -> None:
        super().__init__(name)
        self.llm = LLMClient(model_name=model_name, max_tokens=512)
        self.system_prompt = build_agent_system_prompt("upvote")
        self.ab_selector = ab_selector or ABSelector(mode="alternate")
        self._analysis_data: Dict[str, Any] = {}
        self._cycle_count: int = 0

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        msg_type = msg.metadata.get("type")

        if msg_type == "analysis_result":
            self._analysis_data = msg.metadata
            self._cycle_count += 1
            posts = msg.metadata.get("posts", [])
            if posts:
                proposals = await self.propose(
                    posts, self._cycle_count,
                )
                if proposals and self._hub:
                    await self._hub.send_to(
                        "coordinator",
                        Message(
                            name=self.name,
                            role="assistant",
                            content="upvote_proposals",
                            metadata={
                                "type": "proposals",
                                "proposals": [
                                    p.to_dict() for p in proposals
                                ],
                            },
                            causation_id=msg.id,
                        ),
                    )
                    logger.info(
                        "UpvoteAgent sent %d proposals to coordinator",
                        len(proposals),
                    )

        elif msg_type == "proposal_approved":
            approved = msg.metadata.get("approved", [])
            for p in approved:
                if (
                    p.get("agent_name") == self.name
                    and p.get("action") == "upvote"
                ):
                    if self._hub:
                        await self._hub.send_to(
                            "sensor",
                            Message(
                                name=self.name,
                                role="assistant",
                                content="exec_command",
                                metadata={
                                    "type": "exec_command",
                                    "action": "upvote",
                                    "target_id": p.get(
                                        "target_id", ""
                                    ),
                                    "strategy": p.get(
                                        "strategy", ""
                                    ),
                                },
                                causation_id=msg.id,
                            ),
                        )
                        logger.info(
                            "UpvoteAgent exec upvote %s",
                            p.get("target_id", "")[:12],
                        )

        elif msg_type == "action_completed":
            if msg.metadata.get("agent_name") == self.name:
                logger.debug(
                    "UpvoteAgent tracked interaction: %s",
                    msg.metadata.get("action", "unknown"),
                )

    async def propose(
        self,
        posts: List[Dict[str, Any]],
        cycle_count: int = 0,
        agent_name: str = "SocialLearnerBot",
    ) -> List[Proposal]:
        strategy = self.ab_selector.select(cycle_count)
        if strategy == "A":
            return self._propose_rule_based(posts, agent_name, strategy)
        else:
            return await self._propose_llm_based(posts, agent_name, strategy)

    def _propose_rule_based(
        self, posts: List[Dict[str, Any]], agent_name: str, strategy: str
    ) -> List[Proposal]:
        """Strategy A: Rule-based upvote filtering."""
        proposals = []
        for p in posts[:15]:
            if p.get("user_vote") is not None:
                continue
            if p.get("author_name") == agent_name:
                continue
            up = p.get("upvotes", 0)
            down = p.get("downvotes", 0)
            if up <= down * 2 or (up + p.get("comment_count", 0)) == 0:
                continue

            lv = self._analysis_data.get("learning_values", {}).get(p.get("id", ""), 0)
            priority = min(0.3 + lv * 0.3 + min(up / 50, 0.3), 1.0)

            proposals.append(Proposal(
                agent_name=self.name,
                action="upvote",
                target_id=p.get("id", ""),
                priority=round(priority, 3),
                reasoning=f"rule-based: up={up} down={down} lv={lv:.2f}",
                strategy=strategy,
            ))
        return proposals

    async def _propose_llm_based(
        self, posts: List[Dict[str, Any]], agent_name: str, strategy: str
    ) -> List[Proposal]:
        """Strategy B: LLM evaluates post quality for upvoting."""
        available = [
            p for p in posts[:15]
            if p.get("user_vote") is None and p.get("author_name") != agent_name
        ]
        if not available:
            return []

        summaries = "\n".join(
            f"[{p.get('id','')}] {p.get('title','')[:60]} (up:{p.get('upvotes',0)})"
            for p in available[:10]
        )
        prompt = build_upvote_selection_prompt(summaries)
        try:
            data = await self.llm.call_json(prompt, system=self.system_prompt)
            if not isinstance(data, list):
                data = []
        except Exception as e:
            logger.warning("LLM upvote selection failed: %s", e)
            return self._propose_rule_based(posts, agent_name, strategy)

        proposals = []
        for item in data:
            pid = item.get("post_id", "")
            if any(p.get("id") == pid for p in available):
                proposals.append(Proposal(
                    agent_name=self.name,
                    action="upvote",
                    target_id=pid,
                    priority=min(max(item.get("priority", 0.5), 0), 1),
                    reasoning="llm-selected",
                    strategy=strategy,
                ))
        return proposals

    async def reply(self, msg: Message) -> Message:
        return Message(
            name=self.name, role="assistant",
            content="UpvoteAgent uses propose() interface",
        )

# -*- coding: utf-8 -*-
"""BaseAgent — foundation class for all Socialite agents.

Supports autonomous run loops, bidirectional communication via hub,
and decision-based message handling.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from core.agent_contract import normalize_incoming_message, normalize_outgoing_message
from core.decision import AgentDecision
from core.message import Message

if TYPE_CHECKING:
    from core.msghub import MsgHub

logger = logging.getLogger(__name__)


class BaseAgent:
    """Async-first agent base class with autonomous run loop and pub-sub wiring.

    Subclass and override:
    - ``reply()`` to implement message handling logic.
    - ``run()`` to implement autonomous behaviour loop.
    - ``decide()`` to implement per-message decision logic.
    """

    message_logger = None  # Injected by runner for message persistence

    def __init__(self, name: str) -> None:
        self.name = name
        self._subscribers: List["BaseAgent"] = []
        self._memory: List[Message] = []
        self._hub: Optional["MsgHub"] = None
        self._running = False

    # ── Hub injection ────────────────────────────────────────

    def set_hub(self, hub: "MsgHub") -> None:
        """Inject MsgHub reference for autonomous communication."""
        self._hub = hub

    # ── Autonomous run loop ──────────────────────────────────

    async def run(self) -> None:
        """Autonomous agent loop. Override in subclasses.

        This method runs as a long-lived asyncio task. The agent decides
        its own rhythm (sleep intervals, trigger conditions, etc.).

        Default implementation does nothing (passive agent that only
        responds to incoming messages).
        """
        pass

    async def start(self) -> None:
        """Start the agent's run loop. Called by the launcher."""
        self._running = True
        logger.info("%s started", self.name)
        try:
            await self.run()
        except asyncio.CancelledError:
            logger.info("%s cancelled", self.name)
        except Exception:
            logger.exception("%s run loop crashed", self.name)
        finally:
            self._running = False
            logger.info("%s stopped", self.name)

    def stop(self) -> None:
        """Signal the agent to stop its run loop."""
        self._running = False

    # ── Core interface ──────────────────────────────────────

    async def reply(self, msg: Message) -> Message:
        """Process *msg* and return a response Message.

        Must be overridden by concrete agents.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.reply() not implemented")

    async def observe(self, msg: Message) -> None:
        """Receive a message published by another agent (via MsgHub).

        Default behaviour: append to ``_memory``.  Override for custom logic.
        """
        self._memory.append(msg)

    async def decide(self, msg: Message) -> AgentDecision:
        """Decide how to handle a message.

        Default policy preserves legacy behaviour:
        - cache incoming message
        - generate a direct reply via ``reply()``
        """
        return AgentDecision(
            memory_action="cache",
            output_action="reply",
            reason="default_reply_policy",
            confidence=1.0,
        )

    async def _render_control_message(self, decision: AgentDecision) -> Message:
        if decision.output_action == "forward":
            return Message(
                name=self.name,
                role="assistant",
                content="forward_requested",
                metadata={
                    "type": "decision_forward",
                    "forward_target": decision.forward_target,
                    "forward_topic": decision.forward_topic,
                    "decision": decision.to_dict(),
                },
            )
        return Message(
            name=self.name,
            role="assistant",
            content="no_output",
            metadata={
                "type": "decision_no_output",
                "decision": decision.to_dict(),
            },
        )

    async def __call__(self, msg: Message) -> Message:
        """Call the agent according to decision protocol, then fan-out reply outputs."""
        normalized_incoming = normalize_incoming_message(self.name, msg)
        decision = AgentDecision.from_any(await self.decide(normalized_incoming))

        if decision.memory_action == "cache":
            self._memory.append(normalized_incoming)

        if self.message_logger and hasattr(self.message_logger, "log_decision"):
            decision_payload = decision.to_dict()
            decision_meta = dict(decision_payload.get("metadata") or {})
            decision_meta["decision_schema_version"] = "1.0"
            decision_payload["metadata"] = decision_meta
            await self.message_logger.log_decision(self.name, normalized_incoming, decision_payload)

        if decision.output_action == "reply":
            raw_result = await self.reply(normalized_incoming)
            if not isinstance(raw_result, Message):
                raw_result = Message(
                    name=self.name,
                    role="assistant",
                    content=str(raw_result),
                )
            result = normalize_outgoing_message(
                agent_name=self.name,
                incoming_msg=normalized_incoming,
                outgoing_msg=raw_result,
            )
            for sub in self._subscribers:
                await sub.observe(result)
                if self.message_logger:
                    await self.message_logger.log_send(self.name, sub.name, result)
            return result

        # forward: actually route to target if hub is available
        if decision.output_action == "forward" and self._hub:
            forward_msg = Message(
                name=self.name,
                role="assistant",
                content=normalized_incoming.content,
                metadata=dict(normalized_incoming.metadata or {}),
                target=decision.forward_target,
                causation_id=normalized_incoming.id,
                trace_id=normalized_incoming.trace_id,
            )
            if decision.forward_target:
                await self._hub.send_to(decision.forward_target, forward_msg)
            return normalize_outgoing_message(
                agent_name=self.name,
                incoming_msg=normalized_incoming,
                outgoing_msg=forward_msg,
            )

        control = await self._render_control_message(decision)
        return normalize_outgoing_message(
            agent_name=self.name,
            incoming_msg=normalized_incoming,
            outgoing_msg=control,
        )

    # ── Subscription helpers ────────────────────────────────

    def add_subscriber(self, agent: "BaseAgent") -> None:
        if agent not in self._subscribers:
            self._subscribers.append(agent)

    def remove_subscriber(self, agent: "BaseAgent") -> None:
        if agent in self._subscribers:
            self._subscribers.remove(agent)

    # ── Memory helpers ──────────────────────────────────────

    def get_memory(self, last_n: Optional[int] = None) -> List[Message]:
        """Return observed messages, optionally limited to the last *n*."""
        if last_n is None:
            return list(self._memory)
        return list(self._memory[-last_n:])

    def clear_memory(self) -> None:
        self._memory.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

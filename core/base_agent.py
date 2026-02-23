# -*- coding: utf-8 -*-
"""BaseAgent — foundation class for all Socialite agents."""

import logging
from typing import List, Optional

from core.message import Message

logger = logging.getLogger(__name__)


class BaseAgent:
    """Async-first agent base class with pub-sub wiring.

    Subclass and override ``reply()`` to implement agent logic.

    Attributes:
        name: Unique agent identifier.
    """

    message_logger = None  # Injected by runner for message persistence

    def __init__(self, name: str) -> None:
        self.name = name
        self._subscribers: List["BaseAgent"] = []
        self._memory: List[Message] = []

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

    async def __call__(self, msg: Message) -> Message:
        """Call the agent: run ``reply``, then fan-out result to subscribers."""
        result = await self.reply(msg)
        for sub in self._subscribers:
            await sub.observe(result)
            if self.message_logger:
                await self.message_logger.log_send(self.name, sub.name, result)
        return result

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

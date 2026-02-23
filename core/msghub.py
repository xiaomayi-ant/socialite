# -*- coding: utf-8 -*-
"""MsgHub — publish-subscribe message hub for agent communication.

Supports two modes:
  1. **Selective subscription** (default) — explicit ``subscribe(subscriber, publisher)``.
  2. **Full-connect** — ``connect_all()`` wires every agent to every other agent.
"""

import logging
from typing import Dict, List, Optional

from core.base_agent import BaseAgent
from core.message import Message

logger = logging.getLogger(__name__)


class MsgHub:
    """Lightweight pub-sub hub wiring BaseAgent instances.

    Usage::

        hub = MsgHub([sensor, analysis, comment, coordinator])
        hub.subscribe(analysis, sensor)          # analysis observes sensor
        hub.subscribe(comment, analysis)         # comment observes analysis

        async with hub:
            result = await sensor(some_msg)      # auto-fans-out via subscribers
    """

    def __init__(
        self,
        participants: Optional[List[BaseAgent]] = None,
        message_logger=None,
    ) -> None:
        self.participants: List[BaseAgent] = list(participants or [])
        # publisher_name → list of subscribers
        self._subscriptions: Dict[str, List[BaseAgent]] = {}
        self._message_logger = message_logger

    # ── Subscription management ─────────────────────────────

    def subscribe(self, subscriber: BaseAgent, publisher: BaseAgent) -> None:
        """Make *subscriber* observe messages published by *publisher*."""
        subs = self._subscriptions.setdefault(publisher.name, [])
        if subscriber not in subs:
            subs.append(subscriber)
            publisher.add_subscriber(subscriber)
            logger.debug("%s subscribes to %s", subscriber.name, publisher.name)

    def unsubscribe(self, subscriber: BaseAgent, publisher: BaseAgent) -> None:
        """Remove a subscription."""
        subs = self._subscriptions.get(publisher.name, [])
        if subscriber in subs:
            subs.remove(subscriber)
            publisher.remove_subscriber(subscriber)

    def connect_all(self) -> None:
        """Full-connect mode: every participant subscribes to every other."""
        for pub in self.participants:
            for sub in self.participants:
                if pub is not sub:
                    self.subscribe(sub, pub)

    # ── Broadcast ───────────────────────────────────────────

    async def broadcast(self, msg: Message) -> None:
        """Send *msg* to **all** participants (regardless of subscriptions)."""
        for agent in self.participants:
            try:
                await agent.observe(msg)
            except Exception as e:
                logger.warning("broadcast → %s failed: %s", agent.name, e)
        if self._message_logger:
            await self._message_logger.log_broadcast(msg.name, msg)

    # ── Context manager ─────────────────────────────────────

    async def __aenter__(self) -> "MsgHub":
        return self

    async def __aexit__(self, *args) -> None:
        """Tear down all subscriptions on exit."""
        for pub_name, subs in self._subscriptions.items():
            pub = next((p for p in self.participants if p.name == pub_name), None)
            if pub:
                for sub in list(subs):
                    pub.remove_subscriber(sub)
        self._subscriptions.clear()

    # ── Introspection ───────────────────────────────────────

    def topology(self) -> Dict[str, List[str]]:
        """Return a dict of publisher_name → [subscriber_names]."""
        return {
            pub: [s.name for s in subs]
            for pub, subs in self._subscriptions.items()
        }

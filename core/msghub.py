# -*- coding: utf-8 -*-
"""MsgHub — publish-subscribe message hub for agent communication.

Supports:
  1. **Selective subscription** — explicit ``subscribe(subscriber, publisher)``.
  2. **Full-connect** — ``connect_all()`` wires every agent to every other agent.
  3. **Request-reply** — ``request()`` calls target's ``__call__()`` for synchronous request-reply.
  4. **Topic publish** — ``publish(msg, topic)`` routes to topic subscribers.
  5. **Guards** — dedup, TTL, max_hops, per-round message budget.
"""

import asyncio
import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from core.base_agent import BaseAgent
from core.message import Message

logger = logging.getLogger(__name__)

# Default per-round message budget (0 = unlimited)
DEFAULT_ROUND_BUDGET = 0


class MsgHub:
    """Lightweight pub-sub hub wiring BaseAgent instances.

    Usage::

        hub = MsgHub([sensor, analysis, comment, coordinator])
        hub.connect_all()  # full bidirectional mesh

        async with hub:
            reply = await hub.request("sensor", "analysis", some_msg)
    """

    def __init__(
        self,
        participants: Optional[List[BaseAgent]] = None,
        message_logger=None,
        dedup_cache_size: int = 10000,
        round_budget: int = DEFAULT_ROUND_BUDGET,
    ) -> None:
        self.participants: List[BaseAgent] = list(participants or [])
        # publisher_name → list of subscribers
        self._subscriptions: Dict[str, List[BaseAgent]] = {}
        # topic_pattern -> subscribers
        self._topic_subscriptions: Dict[str, List[BaseAgent]] = {}
        self._message_logger = message_logger
        self._dedup_cache_size = max(dedup_cache_size, 256)
        self._seen_message_ids: Set[str] = set()
        self._seen_fifo: Deque[str] = deque()

        # Per-round budget
        self._round_budget = round_budget
        self._round_message_count = 0

        # Communication matrix for observability
        self._comm_matrix: Dict[Tuple[str, str], int] = {}

    # ── Round management ─────────────────────────────────────

    def reset_round(self) -> None:
        """Reset per-round counters. Call at the start of each cycle."""
        self._round_message_count = 0
        self._comm_matrix.clear()

    def round_stats(self) -> Dict:
        """Return communication matrix snapshot for the current round."""
        n = len(self.participants)
        max_edges = n * (n - 1) if n > 1 else 1
        self_edges = sum(1 for (s, r) in self._comm_matrix if s == r)
        unique_edges_all = len(self._comm_matrix)
        unique_edges_non_self = unique_edges_all - self_edges
        return {
            "total_messages": self._round_message_count,
            "budget": self._round_budget,
            "unique_edges": unique_edges_non_self,
            "unique_edges_including_self": unique_edges_all,
            "self_edges": self_edges,
            "max_edges": max_edges,
            "coverage": round(unique_edges_non_self / max_edges, 3) if max_edges else 0,
            "matrix": {f"{s}->{r}": c for (s, r), c in self._comm_matrix.items()},
        }

    def _record_comm(self, sender: str, receiver: str) -> bool:
        """Record a communication event. Returns False if budget exhausted."""
        if self._round_budget > 0 and self._round_message_count >= self._round_budget:
            logger.warning(
                "round budget exhausted (%d/%d), dropping %s->%s",
                self._round_message_count, self._round_budget, sender, receiver,
            )
            return False
        self._round_message_count += 1
        key = (sender, receiver)
        self._comm_matrix[key] = self._comm_matrix.get(key, 0) + 1
        return True

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
        logger.info(
            "Full mesh connected: %d agents, %d edges",
            len(self.participants),
            len(self.participants) * (len(self.participants) - 1),
        )

    # ── Topic subscriptions (v2) ───────────────────────────

    def subscribe_topic(self, subscriber: BaseAgent, topic_pattern: str) -> None:
        """Subscribe *subscriber* to a topic pattern.

        Pattern matching:
        - exact match: ``feed.analyzed``
        - prefix wildcard: ``feed.*``
        - global wildcard: ``*``
        """
        if not topic_pattern:
            return
        subs = self._topic_subscriptions.setdefault(topic_pattern, [])
        if subscriber not in subs:
            subs.append(subscriber)

    def unsubscribe_topic(self, subscriber: BaseAgent, topic_pattern: str) -> None:
        subs = self._topic_subscriptions.get(topic_pattern, [])
        if subscriber in subs:
            subs.remove(subscriber)

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            return topic.startswith(pattern[:-1])  # keep trailing dot in prefix
        return pattern == topic

    # ── Guard rails ─────────────────────────────────────────

    def _remember_seen(self, message_id: str) -> None:
        self._seen_message_ids.add(message_id)
        self._seen_fifo.append(message_id)
        if len(self._seen_fifo) > self._dedup_cache_size:
            old = self._seen_fifo.popleft()
            self._seen_message_ids.discard(old)

    def _accept(self, msg: Message) -> tuple[bool, str]:
        if msg.id in self._seen_message_ids:
            logger.debug("drop duplicate message id=%s", msg.id)
            return False, "duplicate"
        if msg.is_expired():
            logger.debug("drop expired message id=%s", msg.id)
            return False, "expired"
        if msg.hops >= msg.max_hops:
            logger.debug(
                "drop message beyond hop budget id=%s hops=%d max_hops=%d",
                msg.id, msg.hops, msg.max_hops,
            )
            return False, "max_hops"
        self._remember_seen(msg.id)
        return True, ""

    async def _log_drop(self, msg: Message, reason: str, route: str) -> None:
        if not self._message_logger:
            return
        log_hub_drop = getattr(self._message_logger, "log_hub_drop", None)
        if not log_hub_drop:
            return
        try:
            await log_hub_drop(msg=msg, reason=reason, route=route)
        except Exception as e:
            logger.debug("MsgHub drop logging failed: %s", e)

    @staticmethod
    def _next_hop(msg: Message) -> Message:
        data = msg.to_dict()
        data["hops"] = msg.hops + 1
        return Message.from_dict(data)

    def _find_participant(self, name: str) -> Optional[BaseAgent]:
        return next((a for a in self.participants if a.name == name), None)

    # ── Request-reply (bidirectional) ───────────────────────

    async def request(
        self,
        target_name: str,
        msg: Message,
        timeout_seconds: float = 30.0,
    ) -> Optional[Message]:
        """Send *msg* to target and get a response via ``__call__()``."""
        if target_name == msg.name:
            logger.debug("request self-target blocked: %s", target_name)
            await self._log_drop(msg, "self_target", "request")
            return None
        accepted, reason = self._accept(msg)
        if not accepted:
            await self._log_drop(msg, reason, "request")
            return None

        target = self._find_participant(target_name)
        if not target:
            logger.warning("request target not found: %s", target_name)
            return None

        if not self._record_comm(msg.name, target_name):
            await self._log_drop(msg, "budget_exhausted", "request")
            return None

        delivered = self._next_hop(msg)
        delivered.target = target_name

        try:
            response = await asyncio.wait_for(
                target(delivered), timeout=timeout_seconds,
            )
            if self._message_logger:
                await self._message_logger.log_send(msg.name, target.name, delivered)
            # Record the reply direction too
            if response:
                self._record_comm(target_name, msg.name)
            return response
        except asyncio.TimeoutError:
            logger.warning("request to %s timed out after %.1fs", target_name, timeout_seconds)
            return None
        except Exception as e:
            logger.warning("request to %s failed: %s", target_name, e)
            return None

    # ── Direct and topic routing (v2) ───────────────────────

    async def send_to(self, target_name: str, msg: Message) -> None:
        """Send *msg* directly to one target participant by name."""
        if target_name == msg.name:
            logger.debug("send_to self-target blocked: %s", target_name)
            await self._log_drop(msg, "self_target", "send_to")
            return
        accepted, reason = self._accept(msg)
        if not accepted:
            await self._log_drop(msg, reason, "send_to")
            return
        target = self._find_participant(target_name)
        if not target:
            logger.warning("send_to target not found: %s", target_name)
            await self._log_drop(msg, "target_not_found", "send_to")
            return

        if not self._record_comm(msg.name, target_name):
            await self._log_drop(msg, "budget_exhausted", "send_to")
            return

        delivered = self._next_hop(msg)
        try:
            await target.observe(delivered)
            if self._message_logger:
                await self._message_logger.log_send(msg.name, target.name, delivered)
        except Exception as e:
            logger.warning("send_to %s failed: %s", target.name, e)

    async def publish(self, msg: Message, topic: Optional[str] = None) -> None:
        """Publish *msg* to all subscribers that match the topic pattern."""
        resolved_topic = topic or msg.topic
        if not resolved_topic:
            logger.warning("publish called without topic (id=%s)", msg.id)
            await self._log_drop(msg, "missing_topic", "publish")
            return
        accepted, reason = self._accept(msg)
        if not accepted:
            await self._log_drop(msg, reason, "publish")
            return

        recipients: List[BaseAgent] = []
        for pattern, subs in self._topic_subscriptions.items():
            if self._topic_matches(pattern, resolved_topic):
                for agent in subs:
                    if agent.name == msg.name:
                        continue
                    if agent not in recipients:
                        recipients.append(agent)

        for agent in recipients:
            if not self._record_comm(msg.name, agent.name):
                await self._log_drop(msg, "budget_exhausted", "publish")
                continue
            delivered = self._next_hop(msg)
            try:
                await agent.observe(delivered)
                if self._message_logger:
                    await self._message_logger.log_send(msg.name, agent.name, delivered)
            except Exception as e:
                logger.warning("publish(%s) -> %s failed: %s", resolved_topic, agent.name, e)

    # ── Broadcast ───────────────────────────────────────────

    async def broadcast(self, msg: Message) -> None:
        """Send *msg* to all participants except the sender itself."""
        accepted, reason = self._accept(msg)
        if not accepted:
            await self._log_drop(msg, reason, "broadcast")
            return
        for agent in self.participants:
            if agent.name == msg.name:
                continue
            if not self._record_comm(msg.name, agent.name):
                continue
            try:
                await agent.observe(self._next_hop(msg))
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
        self._topic_subscriptions.clear()

    # ── Introspection ───────────────────────────────────────

    def topology(self) -> Dict[str, List[str]]:
        """Return a dict of publisher_name → [subscriber_names]."""
        return {
            pub: [s.name for s in subs]
            for pub, subs in self._subscriptions.items()
        }

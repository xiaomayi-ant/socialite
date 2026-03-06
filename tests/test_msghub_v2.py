# -*- coding: utf-8 -*-
"""Tests for MsgHub v2 routing guards: direct send, topic publish, dedup, ttl, hops."""

from datetime import datetime, timedelta

import pytest

from core.base_agent import BaseAgent
from core.message import Message
from core.msghub import MsgHub


class CollectorAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.observed: list[Message] = []

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        self.observed.append(msg)

    async def reply(self, msg: Message) -> Message:
        return Message(name=self.name, role="assistant", content="ack")


class FakeHubLogger:
    def __init__(self) -> None:
        self.drops: list[dict] = []

    async def log_send(self, sender: str, receiver: str, msg: Message) -> None:
        return None

    async def log_broadcast(self, sender: str, msg: Message) -> None:
        return None

    async def log_hub_drop(self, msg: Message, reason: str, route: str = "") -> None:
        self.drops.append({"id": msg.id, "reason": reason, "route": route})


@pytest.mark.asyncio
async def test_send_to_routes_to_specific_agent():
    a = CollectorAgent("a")
    b = CollectorAgent("b")
    c = CollectorAgent("c")
    hub = MsgHub(participants=[a, b, c])

    msg = Message(name="x", role="system", content="direct")
    await hub.send_to("b", msg)

    assert len(a.observed) == 0
    assert len(c.observed) == 0
    assert len(b.observed) == 1
    assert b.observed[0].content == "direct"


@pytest.mark.asyncio
async def test_publish_routes_to_topic_subscribers_with_wildcard():
    analysis = CollectorAgent("analysis")
    comment = CollectorAgent("comment")
    upvote = CollectorAgent("upvote")
    hub = MsgHub(participants=[analysis, comment, upvote])
    hub.subscribe_topic(comment, "feed.*")
    hub.subscribe_topic(upvote, "feed.analyzed")

    msg = Message(
        name="analysis",
        role="assistant",
        content="analysis_result",
        topic="feed.analyzed",
    )
    await hub.publish(msg)

    assert len(comment.observed) == 1
    assert len(upvote.observed) == 1
    assert len(analysis.observed) == 0


@pytest.mark.asyncio
async def test_duplicate_message_id_is_dropped():
    a = CollectorAgent("a")
    hub = MsgHub(participants=[a])

    msg = Message(name="sys", role="system", content="x", id="dup-id")
    await hub.broadcast(msg)
    await hub.broadcast(msg)

    assert len(a.observed) == 1


@pytest.mark.asyncio
async def test_expired_message_is_dropped():
    a = CollectorAgent("a")
    hub = MsgHub(participants=[a])

    msg = Message(
        name="sys",
        role="system",
        content="stale",
        ttl_seconds=5,
        timestamp=datetime.now() - timedelta(seconds=30),
    )
    await hub.broadcast(msg)

    assert len(a.observed) == 0


@pytest.mark.asyncio
async def test_max_hops_guard_and_hop_increment():
    a = CollectorAgent("a")
    hub = MsgHub(participants=[a])

    blocked = Message(name="sys", role="system", content="blocked", hops=2, max_hops=2)
    await hub.broadcast(blocked)
    assert len(a.observed) == 0

    ok = Message(name="sys", role="system", content="ok", hops=1, max_hops=3)
    await hub.broadcast(ok)
    assert len(a.observed) == 1
    assert a.observed[0].hops == 2


@pytest.mark.asyncio
async def test_hub_drop_events_are_logged_for_duplicate_and_missing_topic():
    a = CollectorAgent("a")
    logger = FakeHubLogger()
    hub = MsgHub(participants=[a], message_logger=logger)

    dup = Message(name="sys", role="system", content="x", id="dup-id")
    await hub.broadcast(dup)
    await hub.broadcast(dup)

    no_topic = Message(name="analysis", role="assistant", content="oops")
    await hub.publish(no_topic)

    reasons = [d["reason"] for d in logger.drops]
    routes = [d["route"] for d in logger.drops]
    assert "duplicate" in reasons
    assert "missing_topic" in reasons
    assert "broadcast" in routes
    assert "publish" in routes


@pytest.mark.asyncio
async def test_broadcast_and_publish_skip_sender_self_delivery():
    sensor = CollectorAgent("sensor")
    analysis = CollectorAgent("analysis")
    hub = MsgHub(participants=[sensor, analysis])
    hub.subscribe_topic(sensor, "feed.*")
    hub.subscribe_topic(analysis, "feed.*")

    # broadcast from sensor should not deliver back to sensor
    await hub.broadcast(Message(name="sensor", role="assistant", content="raw"))
    assert len(sensor.observed) == 0
    assert len(analysis.observed) == 1

    # topic publish from analysis should not deliver back to analysis
    await hub.publish(Message(name="analysis", role="assistant", content="x", topic="feed.analyzed"))
    assert len(analysis.observed) == 1
    assert len(sensor.observed) == 1

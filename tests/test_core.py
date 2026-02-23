# -*- coding: utf-8 -*-
"""Tests for core framework: Message, BaseAgent, MsgHub, ABSelector, Proposal."""

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.message import Message
from core.base_agent import BaseAgent
from core.msghub import MsgHub
from core.proposal import Proposal
from core.ab_strategy import ABSelector


# ── Helpers ──────────────────────────────────────────────────


class EchoAgent(BaseAgent):
    """Simple agent that echoes back with a prefix."""

    async def reply(self, msg: Message) -> Message:
        return Message(
            name=self.name,
            role="assistant",
            content=f"[{self.name}] {msg.content}",
            metadata=msg.metadata,
        )


class CollectorAgent(BaseAgent):
    """Agent that collects observed messages for assertions."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.observed: list[Message] = []

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        self.observed.append(msg)

    async def reply(self, msg: Message) -> Message:
        return Message(name=self.name, role="assistant", content="ack")


# ── Message tests ────────────────────────────────────────────


class TestMessage:
    def test_create_and_fields(self):
        msg = Message(name="test", role="user", content="hello")
        assert msg.name == "test"
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.metadata == {}
        assert len(msg.id) == 32  # uuid4 hex

    def test_to_dict_and_from_dict(self):
        msg = Message(
            name="agent", role="assistant", content="hi",
            metadata={"key": "val"},
        )
        d = msg.to_dict()
        assert d["name"] == "agent"
        assert d["metadata"]["key"] == "val"
        assert "timestamp" in d

        restored = Message.from_dict(d)
        assert restored.name == msg.name
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata

    def test_metadata_isolation(self):
        msg1 = Message(name="a", role="user", content="x")
        msg2 = Message(name="b", role="user", content="y")
        msg1.metadata["foo"] = "bar"
        assert "foo" not in msg2.metadata


# ── BaseAgent tests ──────────────────────────────────────────


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_reply_raises_if_not_overridden(self):
        agent = BaseAgent(name="bare")
        with pytest.raises(NotImplementedError):
            await agent.reply(Message(name="x", role="user", content="hi"))

    @pytest.mark.asyncio
    async def test_echo_agent_reply(self):
        agent = EchoAgent(name="echo")
        msg = Message(name="user", role="user", content="ping")
        result = await agent.reply(msg)
        assert "[echo] ping" in result.content

    @pytest.mark.asyncio
    async def test_call_fans_out_to_subscribers(self):
        publisher = EchoAgent(name="pub")
        sub1 = CollectorAgent(name="sub1")
        sub2 = CollectorAgent(name="sub2")
        publisher.add_subscriber(sub1)
        publisher.add_subscriber(sub2)

        msg = Message(name="user", role="user", content="data")
        result = await publisher(msg)

        assert result.content == "[pub] data"
        assert len(sub1.observed) == 1
        assert len(sub2.observed) == 1
        assert sub1.observed[0].content == "[pub] data"

    @pytest.mark.asyncio
    async def test_observe_appends_to_memory(self):
        agent = CollectorAgent(name="col")
        msg = Message(name="sender", role="assistant", content="info")
        await agent.observe(msg)
        assert len(agent.get_memory()) == 1
        assert agent.get_memory()[0].content == "info"


# ── MsgHub tests ─────────────────────────────────────────────


class TestMsgHub:
    @pytest.mark.asyncio
    async def test_selective_subscribe(self):
        """A subscribes to B, C does NOT receive B's messages."""
        agent_a = CollectorAgent(name="A")
        agent_b = EchoAgent(name="B")
        agent_c = CollectorAgent(name="C")

        hub = MsgHub(participants=[agent_a, agent_b, agent_c])
        hub.subscribe(agent_a, agent_b)  # A observes B

        async with hub:
            msg = Message(name="user", role="user", content="test")
            await agent_b(msg)  # B processes → fans out to subscribers

        assert len(agent_a.observed) == 1
        assert agent_a.observed[0].content == "[B] test"
        assert len(agent_c.observed) == 0  # C not subscribed to B

    @pytest.mark.asyncio
    async def test_connect_all(self):
        """Full-connect: everyone observes everyone else."""
        agents = [CollectorAgent(name=n) for n in ("X", "Y", "Z")]
        hub = MsgHub(participants=agents)
        hub.connect_all()

        async with hub:
            # Verify topology has all pairs
            topo = hub.topology()
            for agent in agents:
                subs = topo.get(agent.name, [])
                for other in agents:
                    if other.name != agent.name:
                        assert other.name in subs

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Broadcast reaches all participants."""
        agents = [CollectorAgent(name=n) for n in ("P", "Q")]
        hub = MsgHub(participants=agents)

        msg = Message(name="system", role="system", content="alert")
        await hub.broadcast(msg)

        for agent in agents:
            assert len(agent.observed) == 1
            assert agent.observed[0].content == "alert"

    @pytest.mark.asyncio
    async def test_cleanup_on_exit(self):
        """Subscriptions are cleared after exiting context manager."""
        pub = EchoAgent(name="pub")
        sub = CollectorAgent(name="sub")
        hub = MsgHub(participants=[pub, sub])
        hub.subscribe(sub, pub)

        async with hub:
            pass

        assert hub.topology() == {}
        assert sub not in pub._subscribers


# ── Proposal tests ───────────────────────────────────────────


class TestProposal:
    def test_create_and_serialize(self):
        p = Proposal(
            agent_name="comment",
            action="comment",
            target_id="post_123",
            priority=0.8,
            reasoning="high engagement",
            strategy="A",
            metadata={"text": "Nice analysis!"},
        )
        d = p.to_dict()
        assert d["action"] == "comment"
        assert d["priority"] == 0.8
        assert d["strategy"] == "A"

        restored = Proposal.from_dict(d)
        assert restored.agent_name == "comment"
        assert restored.target_id == "post_123"


# ── ABSelector tests ─────────────────────────────────────────


class TestABSelector:
    def test_alternate_mode(self):
        sel = ABSelector(mode="alternate")
        assert sel.select(0) == "A"
        assert sel.select(1) == "B"
        assert sel.select(2) == "A"
        assert sel.select(3) == "B"
        assert sel.select(100) == "A"
        assert sel.select(101) == "B"

    def test_probability_mode_returns_valid(self):
        sel = ABSelector(mode="probability", p_a=0.5)
        results = {sel.select(i) for i in range(100)}
        # With 100 samples at p=0.5, both should appear
        assert "A" in results
        assert "B" in results

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ABSelector(mode="invalid")

# -*- coding: utf-8 -*-
"""Tests for agent decision protocol (memory_action + output_action)."""

import pytest

from core.base_agent import BaseAgent
from core.message import Message


class StubLogger:
    def __init__(self) -> None:
        self.decision_logs = []
        self.send_logs = []

    async def log_decision(self, agent_name: str, msg: Message, decision: dict) -> None:
        self.decision_logs.append((agent_name, msg.id, decision))

    async def log_send(self, sender: str, receiver: str, msg: Message) -> None:
        self.send_logs.append((sender, receiver, msg.id))


class SilentAgent(BaseAgent):
    async def decide(self, msg: Message):
        return {
            "memory_action": "cache",
            "output_action": "none",
            "reason": "not_relevant",
            "confidence": 0.9,
        }

    async def reply(self, msg: Message) -> Message:
        raise AssertionError("reply should not be called for output_action=none")


class ForwardAgent(BaseAgent):
    async def decide(self, msg: Message):
        return {
            "memory_action": "cache",
            "output_action": "forward",
            "forward_target": "analysis",
            "forward_topic": "feed.analyzed",
            "reason": "specialist_needed",
            "confidence": 0.8,
        }

    async def reply(self, msg: Message) -> Message:
        raise AssertionError("reply should not be called for output_action=forward")


class EchoAgent(BaseAgent):
    async def reply(self, msg: Message) -> Message:
        return Message(name=self.name, role="assistant", content=f"echo:{msg.content}")


class CollectorAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.observed = []

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        self.observed.append(msg)

    async def reply(self, msg: Message) -> Message:
        return Message(name=self.name, role="assistant", content="ack")


@pytest.mark.asyncio
async def test_output_none_returns_control_message_and_caches_input():
    logger = StubLogger()
    BaseAgent.message_logger = logger
    agent = SilentAgent(name="silent")

    incoming = Message(name="runner", role="user", content="ping")
    out = await agent(incoming)

    assert out.metadata.get("type") == "decision_no_output"
    assert out.metadata.get("decision", {}).get("output_action") == "none"
    assert len(agent.get_memory()) == 1
    assert len(logger.decision_logs) == 1


@pytest.mark.asyncio
async def test_output_forward_returns_forward_instruction():
    logger = StubLogger()
    BaseAgent.message_logger = logger
    agent = ForwardAgent(name="router")

    incoming = Message(name="runner", role="user", content="route me")
    out = await agent(incoming)

    assert out.metadata.get("type") == "decision_forward"
    assert out.metadata.get("forward_target") == "analysis"
    assert out.metadata.get("forward_topic") == "feed.analyzed"
    assert len(logger.decision_logs) == 1


@pytest.mark.asyncio
async def test_default_reply_behavior_unchanged_with_subscriber_fanout():
    logger = StubLogger()
    BaseAgent.message_logger = logger

    pub = EchoAgent(name="pub")
    sub = CollectorAgent(name="sub")
    pub.add_subscriber(sub)

    incoming = Message(name="runner", role="user", content="hello")
    out = await pub(incoming)

    assert out.content == "echo:hello"
    assert len(sub.observed) == 1
    assert sub.observed[0].content == "echo:hello"
    assert len(logger.decision_logs) == 1
    assert len(logger.send_logs) == 1

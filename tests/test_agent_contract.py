# -*- coding: utf-8 -*-
"""Tests for agent I/O contract normalization and prompt output parsing."""

import pytest

from core.agent_contract import (
    DEFAULT_INPUT_TYPE,
    parse_contract_json,
    validate_required_fields,
)
from core.base_agent import BaseAgent
from core.message import Message


class ContractProbeAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.seen_input = None

    async def decide(self, msg: Message):
        self.seen_input = msg
        return {
            "memory_action": "cache",
            "output_action": "reply",
            "reason": "contract_probe",
            "confidence": 0.9,
        }

    async def reply(self, msg: Message) -> Message:
        return Message(
            name=self.name,
            role="assistant",
            content="probe_result",
            metadata={"result": "ok"},
        )


@pytest.mark.asyncio
async def test_base_agent_normalizes_input_and_output_contract():
    agent = ContractProbeAgent("contract_probe")
    incoming = Message(name="runner", role="system", content="run probe", metadata={})

    out = await agent(incoming)

    assert agent.seen_input is not None
    assert agent.seen_input.metadata.get("type") == DEFAULT_INPUT_TYPE
    assert agent.seen_input.metadata.get("contract", {}).get("kind") == "input"

    assert out.metadata.get("type") == "contract_probe.agent_reply"
    assert out.metadata.get("contract", {}).get("kind") == "output"
    assert out.metadata.get("contract", {}).get("producer") == "contract_probe"
    assert out.metadata.get("source_agent") == "contract_probe"
    assert out.metadata.get("input_message_id") == incoming.id
    assert out.trace_id == incoming.trace_id
    assert out.causation_id == incoming.id
    assert out.target == "runner"
    assert out.topic.startswith("agent.contract_probe.")


def test_parse_contract_json_supports_fenced_or_mixed_text():
    fenced = """```json
{"memory_action":"cache","output_action":"reply","reason":"ok","confidence":0.8}
```"""
    parsed = parse_contract_json(fenced)
    assert parsed["memory_action"] == "cache"
    assert parsed["output_action"] == "reply"

    mixed = (
        "Decision summary:\n"
        '{"memory_action":"drop","output_action":"none","reason":"noise","confidence":0.7}\n'
        "done."
    )
    parsed2 = parse_contract_json(mixed)
    assert parsed2["memory_action"] == "drop"
    assert parsed2["output_action"] == "none"


def test_validate_required_fields_reports_missing_keys():
    data = {"memory_action": "cache"}
    ok, missing = validate_required_fields(data, ["memory_action", "output_action", "reason"])
    assert ok is False
    assert missing == ["output_action", "reason"]

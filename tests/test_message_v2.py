# -*- coding: utf-8 -*-
"""Tests for Message v2 envelope fields with backwards compatibility."""

from datetime import datetime, timedelta

from core.message import Message


def test_from_dict_legacy_payload_has_compatible_defaults():
    legacy = {
        "id": "abc123",
        "name": "sensor",
        "role": "assistant",
        "content": "legacy message",
        "metadata": {"type": "raw_feed"},
        "timestamp": datetime.now().isoformat(),
    }

    msg = Message.from_dict(legacy)

    assert msg.id == "abc123"
    assert msg.trace_id == "abc123"
    assert msg.topic == ""
    assert msg.target is None
    assert msg.ttl_seconds is None
    assert msg.hops == 0
    assert msg.max_hops == 8


def test_ttl_expiration_check():
    msg = Message(
        name="analysis",
        role="assistant",
        content="analysis_result",
        ttl_seconds=10,
        timestamp=datetime.now() - timedelta(seconds=12),
    )
    assert msg.is_expired() is True


def test_ttl_none_never_expires():
    msg = Message(
        name="analysis",
        role="assistant",
        content="analysis_result",
        ttl_seconds=None,
        timestamp=datetime.now() - timedelta(days=3),
    )
    assert msg.is_expired() is False


def test_message_v2_fields_roundtrip():
    msg = Message(
        name="comment",
        role="assistant",
        content="proposal",
        topic="proposal.submitted.comment",
        target="coordinator",
        trace_id="trace-1",
        causation_id="cause-1",
        ttl_seconds=120,
        hops=2,
        max_hops=5,
        metadata={"priority": 0.8},
    )

    data = msg.to_dict()
    restored = Message.from_dict(data)

    assert restored.topic == "proposal.submitted.comment"
    assert restored.target == "coordinator"
    assert restored.trace_id == "trace-1"
    assert restored.causation_id == "cause-1"
    assert restored.ttl_seconds == 120
    assert restored.hops == 2
    assert restored.max_hops == 5

# -*- coding: utf-8 -*-
"""Tests for replay loading and decision drift comparison."""

import pytest

from core.replay import (
    DecisionSnapshot,
    build_cycle_replay_report,
    compare_cycles,
    compare_decision_snapshots,
    load_decision_snapshots,
)


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    async def get_agent_messages(self, cycle_count=None, sender=None, message_type=None, limit=1000):
        rows = [r for r in self.rows if "_kind" not in r]
        if cycle_count is not None:
            rows = [r for r in rows if r.get("cycle_count") == cycle_count]
        if sender is not None:
            rows = [r for r in rows if r.get("sender") == sender]
        if message_type is not None:
            rows = [r for r in rows if r.get("message_type") == message_type]
        return rows[:limit]

    async def get_rate_limit_events(self, cycle_count=None, limit=1000):
        events = [r for r in self.rows if r.get("_kind") == "rate_limit"]
        if cycle_count is not None:
            events = [e for e in events if e.get("details", {}).get("cycle_count") == cycle_count]
        return events[:limit]

    async def get_prompt_injection_events(self, cycle_count=None, limit=1000):
        events = [r for r in self.rows if r.get("_kind") == "prompt_injection"]
        if cycle_count is not None:
            events = [e for e in events if e.get("cycle_count") == cycle_count]
        return events[:limit]


@pytest.mark.asyncio
async def test_load_decision_snapshots_filters_and_normalizes():
    rows = [
        {
            "message_id": "m2",
            "sender": "comment",
            "direction": "decision",
            "message_type": "agent_decision",
            "metadata": {"decision": {"memory_action": "cache", "output_action": "reply"}},
        },
        {
            "message_id": "m1",
            "sender": "comment",
            "direction": "send",
            "message_type": "analysis_result",
            "metadata": {},
        },
        {
            "message_id": "m3",
            "sender": "post",
            "direction": "decision",
            "message_type": "agent_decision",
            "metadata": {
                "decision": {
                    "memory_action": "drop",
                    "output_action": "none",
                    "reason": "noise",
                    "confidence": 0.7,
                }
            },
        },
    ]
    store = FakeStore(rows)
    snaps = await load_decision_snapshots(store)

    # Decision rows only, reversed to chronological order.
    assert [s.message_id for s in snaps] == ["m3", "m2"]
    assert snaps[0].output_action == "none"
    assert snaps[1].output_action == "reply"


def test_compare_decision_snapshots_reports_drift():
    baseline = [
        DecisionSnapshot(
            message_id="m1",
            agent_name="comment",
            memory_action="cache",
            output_action="reply",
            reason="default",
        ),
        DecisionSnapshot(
            message_id="m2",
            agent_name="comment",
            memory_action="cache",
            output_action="observe",
            reason="trend_wait",
        ),
    ]
    candidate = [
        DecisionSnapshot(
            message_id="m1",
            agent_name="comment",
            memory_action="cache",
            output_action="reply",
            reason="default",
        ),
        DecisionSnapshot(
            message_id="m2",
            agent_name="comment",
            memory_action="drop",
            output_action="none",
            reason="not_relevant",
        ),
    ]

    report = compare_decision_snapshots(baseline, candidate)

    assert report["compared_count"] == 2
    assert report["changed_count"] == 1
    assert report["drift_rate"] == 0.5
    assert report["output_action_drift"]["observe->none"] == 1
    assert report["memory_action_drift"]["cache->drop"] == 1
    assert len(report["changes"]) == 1


@pytest.mark.asyncio
async def test_compare_cycles_uses_store_queries_and_returns_cycle_tags():
    rows = [
        {
            "message_id": "m1",
            "cycle_count": 10,
            "sender": "comment",
            "direction": "decision",
            "message_type": "agent_decision",
            "metadata": {"decision": {"memory_action": "cache", "output_action": "reply"}},
        },
        {
            "message_id": "m1",
            "cycle_count": 11,
            "sender": "comment",
            "direction": "decision",
            "message_type": "agent_decision",
            "metadata": {"decision": {"memory_action": "cache", "output_action": "none"}},
        },
    ]
    store = FakeStore(rows)
    report = await compare_cycles(store, baseline_cycle=10, candidate_cycle=11, sender="comment")

    assert report["baseline_cycle"] == 10
    assert report["candidate_cycle"] == 11
    assert report["sender"] == "comment"
    assert report["changed_count"] == 1


@pytest.mark.asyncio
async def test_build_cycle_replay_report_aggregates_mode_decision_and_rate_limits():
    rows = [
        {
            "message_id": "mode-1",
            "cycle_count": 12,
            "sender": "runner",
            "direction": "broadcast",
            "message_type": "cycle_mode_snapshot",
            "metadata": {"mode_flags": {"comment": True, "post": False, "upvote": True, "follow": False}},
        },
        {
            "message_id": "m1",
            "cycle_count": 12,
            "sender": "comment",
            "receiver": "sensor",
            "direction": "send",
            "message_type": "comment_generated",
            "metadata": {"topic": "action.comment.generated"},
        },
        {
            "message_id": "m2",
            "cycle_count": 12,
            "sender": "runner",
            "receiver": None,
            "direction": "broadcast",
            "message_type": "action_completed",
            "metadata": {"topic": "action.completed.comment"},
        },
        {
            "message_id": "drop-1",
            "cycle_count": 12,
            "sender": "msghub",
            "receiver": None,
            "direction": "hub_drop",
            "message_type": "hub_drop.duplicate",
            "metadata": {"drop_reason": "duplicate", "topic": "action.comment.generated"},
        },
        {
            "message_id": "d1",
            "cycle_count": 12,
            "sender": "comment",
            "direction": "decision",
            "message_type": "agent_decision",
            "metadata": {"decision": {"memory_action": "cache", "output_action": "reply"}},
        },
        {
            "message_id": "d2",
            "cycle_count": 12,
            "sender": "post",
            "direction": "decision",
            "message_type": "agent_decision",
            "metadata": {"decision": {"memory_action": "drop", "output_action": "none"}},
        },
        {
            "_kind": "rate_limit",
            "action_type": "comment",
            "allowed": False,
            "reason": "daily_comment_limit",
            "details": {"cycle_count": 12},
        },
        {
            "_kind": "prompt_injection",
            "cycle_count": 12,
            "agent_name": "comment",
            "verdict": "suspected",
            "action_taken": "block",
        },
    ]
    store = FakeStore(rows)
    report = await build_cycle_replay_report(store, cycle_count=12)

    assert report["mode_flags"]["comment"] is True
    assert report["mode_flags"]["post"] is False
    assert report["decision_summary"]["total"] == 2
    assert report["decision_summary"]["output_actions"]["reply"] == 1
    assert report["decision_summary"]["output_actions"]["none"] == 1
    assert report["message_summary"]["total"] == 6
    assert report["message_summary"]["directions"]["send"] == 1
    assert report["message_summary"]["directions"]["broadcast"] == 2
    assert report["message_summary"]["directions"]["hub_drop"] == 1
    assert report["message_summary"]["topics"]["action.comment.generated"] == 2
    assert report["guard_drop_summary"]["total"] == 1
    assert report["guard_drop_summary"]["by_reason"]["duplicate"] == 1
    assert report["rate_limit_summary"]["denied"] == 1
    assert report["rate_limit_summary"]["denied_by_action"]["comment"] == 1
    assert report["prompt_injection_summary"]["total_events"] == 1
    assert report["prompt_injection_summary"]["blocked"] == 1
    assert report["prompt_injection_summary"]["by_verdict"]["suspected"] == 1
    assert report["prompt_injection_summary"]["by_agent"]["comment"] == 1

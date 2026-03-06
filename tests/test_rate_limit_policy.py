# -*- coding: utf-8 -*-
"""Tests for rate limit policy used by execution gateway."""

from datetime import datetime, timedelta

import pytest

from core.rate_limit import RateLimitPolicy


class DummyStore:
    def __init__(self, daily_counts=None):
        self.daily_counts = daily_counts or {}
        self.events = []

    async def get_daily_action_count(self, action_type: str, date: str | None = None) -> int:
        return int(self.daily_counts.get(action_type, 0))

    async def record_rate_limit_event(self, data):
        self.events.append(data)
        return True


@pytest.mark.asyncio
async def test_post_cooldown_blocks_then_allows():
    now = datetime(2026, 3, 3, 12, 0, 0)
    store = DummyStore()
    policy = RateLimitPolicy(
        structured_store=store,
        post_cooldown_minutes=30,
        comment_cooldown_seconds=20,
        max_comments_per_day=50,
        now_fn=lambda: now,
    )
    state = {"last_post": (now - timedelta(minutes=10)).isoformat()}

    denied = await policy.check("post", state=state)
    assert denied["allowed"] is False
    assert denied["reason"] == "post_cooldown"

    state = {"last_post": (now - timedelta(minutes=45)).isoformat()}
    allowed = await policy.check("post", state=state)
    assert allowed["allowed"] is True


@pytest.mark.asyncio
async def test_comment_cooldown_and_daily_limit():
    now = datetime(2026, 3, 3, 12, 0, 0)
    store = DummyStore(daily_counts={"comment": 50})
    policy = RateLimitPolicy(
        structured_store=store,
        post_cooldown_minutes=30,
        comment_cooldown_seconds=20,
        max_comments_per_day=50,
        now_fn=lambda: now,
    )
    state = {"last_comment": (now - timedelta(seconds=10)).isoformat()}

    denied_cooldown = await policy.check("comment", state=state)
    assert denied_cooldown["allowed"] is False
    assert denied_cooldown["reason"] == "comment_cooldown"

    state = {"last_comment": (now - timedelta(minutes=5)).isoformat()}
    denied_daily = await policy.check("comment", state=state)
    assert denied_daily["allowed"] is False
    assert denied_daily["reason"] == "daily_comment_limit"


@pytest.mark.asyncio
async def test_denied_event_can_be_recorded():
    now = datetime(2026, 3, 3, 12, 0, 0)
    store = DummyStore()
    policy = RateLimitPolicy(
        structured_store=store,
        post_cooldown_minutes=30,
        comment_cooldown_seconds=20,
        max_comments_per_day=1,
        now_fn=lambda: now,
    )
    result = {"allowed": False, "reason": "daily_comment_limit"}
    await policy.record_denied("comment", target_id="post_1", details=result)

    assert len(store.events) == 1
    assert store.events[0]["action_type"] == "comment"
    assert store.events[0]["reason"] == "daily_comment_limit"

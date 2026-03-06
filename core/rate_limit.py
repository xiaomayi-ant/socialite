# -*- coding: utf-8 -*-
"""Rate limit policy for execution gateway actions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional


class RateLimitPolicy:
    """Applies cooldown and daily cap checks before external actions."""

    def __init__(
        self,
        structured_store: Optional[Any] = None,
        post_cooldown_minutes: int = 30,
        comment_cooldown_seconds: int = 20,
        max_comments_per_day: int = 50,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.structured_store = structured_store
        self.post_cooldown_minutes = max(post_cooldown_minutes, 0)
        self.comment_cooldown_seconds = max(comment_cooldown_seconds, 0)
        self.max_comments_per_day = max(max_comments_per_day, 0)
        self._now_fn = now_fn or datetime.now

    def _now(self) -> datetime:
        return self._now_fn()

    @staticmethod
    def _parse_iso(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    async def _daily_count(self, action_type: str) -> int:
        if not self.structured_store:
            return 0
        if not hasattr(self.structured_store, "get_daily_action_count"):
            return 0
        try:
            return int(await self.structured_store.get_daily_action_count(action_type))
        except Exception:
            return 0

    async def check(self, action_type: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return {'allowed': bool, 'reason': str} for action_type."""
        state = state or {}
        now = self._now()

        if action_type == "post":
            last_post = self._parse_iso(state.get("last_post"))
            if last_post:
                elapsed_m = (now - last_post).total_seconds() / 60
                if elapsed_m < self.post_cooldown_minutes:
                    return {
                        "allowed": False,
                        "reason": "post_cooldown",
                        "remaining_seconds": int((self.post_cooldown_minutes - elapsed_m) * 60),
                    }
            return {"allowed": True, "reason": "ok"}

        if action_type == "comment":
            last_comment = self._parse_iso(state.get("last_comment"))
            if last_comment:
                elapsed_s = (now - last_comment).total_seconds()
                if elapsed_s < self.comment_cooldown_seconds:
                    return {
                        "allowed": False,
                        "reason": "comment_cooldown",
                        "remaining_seconds": int(self.comment_cooldown_seconds - elapsed_s),
                    }
            daily_comments = await self._daily_count("comment")
            if self.max_comments_per_day > 0 and daily_comments >= self.max_comments_per_day:
                return {
                    "allowed": False,
                    "reason": "daily_comment_limit",
                    "limit": self.max_comments_per_day,
                    "current": daily_comments,
                }
            return {"allowed": True, "reason": "ok"}

        # For other action types keep policy permissive by default.
        return {"allowed": True, "reason": "ok"}

    async def record_denied(
        self, action_type: str, target_id: str = "", details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist denied decisions when store supports it."""
        if not self.structured_store:
            return
        if not hasattr(self.structured_store, "record_rate_limit_event"):
            return
        payload = details or {}
        try:
            await self.structured_store.record_rate_limit_event({
                "action_type": action_type,
                "target_id": target_id,
                "allowed": False,
                "reason": payload.get("reason", "denied"),
                "details": payload,
            })
        except Exception:
            return

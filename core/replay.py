# -*- coding: utf-8 -*-
"""Replay and regression helpers for agent decision observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DecisionSnapshot:
    """Normalized decision record extracted from persisted agent messages."""

    message_id: str
    agent_name: str
    memory_action: str
    output_action: str
    reason: str = ""
    confidence: float = 0.0
    raw_metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_agent_message_row(cls, row: Dict[str, Any]) -> Optional["DecisionSnapshot"]:
        """Build a snapshot from ``agent_messages`` row.

        Returns None when the row is not a decision log entry.
        """
        if row.get("direction") != "decision":
            return None
        meta = row.get("metadata") or {}
        decision = meta.get("decision") or {}
        if not decision:
            return None
        return cls(
            message_id=row.get("message_id", ""),
            agent_name=row.get("sender", ""),
            memory_action=decision.get("memory_action", "cache"),
            output_action=decision.get("output_action", "reply"),
            reason=decision.get("reason", ""),
            confidence=float(decision.get("confidence", 0.0)),
            raw_metadata=meta,
        )

    def key(self) -> str:
        return f"{self.agent_name}:{self.message_id}"


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _count_values(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


async def load_decision_snapshots(
    structured_store: Any,
    cycle_count: Optional[int] = None,
    sender: Optional[str] = None,
    limit: int = 1000,
) -> List[DecisionSnapshot]:
    """Load decision snapshots from storage by querying agent messages."""
    rows = await structured_store.get_agent_messages(
        cycle_count=cycle_count,
        sender=sender,
        message_type="agent_decision",
        limit=limit,
    )
    snapshots: List[DecisionSnapshot] = []
    for row in rows:
        snap = DecisionSnapshot.from_agent_message_row(row)
        if snap:
            snapshots.append(snap)
    # get_agent_messages returns desc by id; reverse for natural replay order.
    snapshots.reverse()
    return snapshots


def compare_decision_snapshots(
    baseline: List[DecisionSnapshot],
    candidate: List[DecisionSnapshot],
) -> Dict[str, Any]:
    """Compare two decision sets and return a drift summary report."""
    base_map = {s.key(): s for s in baseline}
    cand_map = {s.key(): s for s in candidate}
    common_keys = sorted(set(base_map) & set(cand_map))

    changed = 0
    changes: List[Dict[str, Any]] = []
    output_drift: Dict[str, int] = {}
    memory_drift: Dict[str, int] = {}

    for key in common_keys:
        b = base_map[key]
        c = cand_map[key]
        output_changed = b.output_action != c.output_action
        memory_changed = b.memory_action != c.memory_action
        reason_changed = b.reason != c.reason
        if output_changed or memory_changed or reason_changed:
            changed += 1
            if output_changed:
                output_key = f"{b.output_action}->{c.output_action}"
                output_drift[output_key] = output_drift.get(output_key, 0) + 1
            if memory_changed:
                memory_key = f"{b.memory_action}->{c.memory_action}"
                memory_drift[memory_key] = memory_drift.get(memory_key, 0) + 1
            changes.append({
                "key": key,
                "baseline": {
                    "memory_action": b.memory_action,
                    "output_action": b.output_action,
                    "reason": b.reason,
                },
                "candidate": {
                    "memory_action": c.memory_action,
                    "output_action": c.output_action,
                    "reason": c.reason,
                },
            })

    report = {
        "baseline_count": len(baseline),
        "candidate_count": len(candidate),
        "compared_count": len(common_keys),
        "missing_in_candidate": len(set(base_map) - set(cand_map)),
        "new_in_candidate": len(set(cand_map) - set(base_map)),
        "changed_count": changed,
        "drift_rate": _safe_rate(changed, len(common_keys)),
        "output_action_drift": output_drift,
        "memory_action_drift": memory_drift,
        "changes": changes,
    }
    return report


async def compare_cycles(
    structured_store: Any,
    baseline_cycle: int,
    candidate_cycle: int,
    sender: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Compare decision drift across two cycles loaded from storage."""
    baseline = await load_decision_snapshots(
        structured_store=structured_store,
        cycle_count=baseline_cycle,
        sender=sender,
        limit=limit,
    )
    candidate = await load_decision_snapshots(
        structured_store=structured_store,
        cycle_count=candidate_cycle,
        sender=sender,
        limit=limit,
    )
    report = compare_decision_snapshots(baseline, candidate)
    report["baseline_cycle"] = baseline_cycle
    report["candidate_cycle"] = candidate_cycle
    report["sender"] = sender
    return report


async def build_cycle_replay_report(
    structured_store: Any,
    cycle_count: int,
    sender: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Build an aggregated cycle report for mode flags, decisions, and rate-limit hits."""
    decisions = await load_decision_snapshots(
        structured_store=structured_store,
        cycle_count=cycle_count,
        sender=sender,
        limit=limit,
    )
    output_actions = _count_values([d.output_action for d in decisions])
    memory_actions = _count_values([d.memory_action for d in decisions])
    by_agent: Dict[str, int] = {}
    for d in decisions:
        by_agent[d.agent_name] = by_agent.get(d.agent_name, 0) + 1

    mode_flags: Dict[str, Any] = {}
    mode_rows = await structured_store.get_agent_messages(
        cycle_count=cycle_count,
        sender="runner",
        message_type="cycle_mode_snapshot",
        limit=5,
    )
    if mode_rows:
        latest = mode_rows[0]
        meta = latest.get("metadata") or {}
        mode_flags = meta.get("mode_flags", {})

    rate_limit_events: List[Dict[str, Any]] = []
    if hasattr(structured_store, "get_rate_limit_events"):
        rate_limit_events = await structured_store.get_rate_limit_events(
            cycle_count=cycle_count, limit=limit
        )
    denied = [e for e in rate_limit_events if not e.get("allowed", False)]
    denied_by_action = _count_values([e.get("action_type", "") for e in denied])
    denied_by_reason = _count_values([e.get("reason", "") for e in denied])

    # Message-level observability (throughput + guard drops)
    message_rows = await structured_store.get_agent_messages(
        cycle_count=cycle_count,
        sender=sender,
        message_type=None,
        limit=limit,
    )
    directions = _count_values([r.get("direction", "") for r in message_rows])
    message_types = _count_values([r.get("message_type", "") for r in message_rows])
    by_sender_rows = _count_values([r.get("sender", "") for r in message_rows])
    topics: Dict[str, int] = {}
    drop_reasons: List[str] = []
    drop_routes: List[str] = []
    for row in message_rows:
        meta = row.get("metadata") or {}
        topic = meta.get("topic")
        if topic:
            topics[topic] = topics.get(topic, 0) + 1
        if row.get("direction") == "hub_drop":
            reason = meta.get("drop_reason")
            if not reason:
                msg_type = row.get("message_type", "")
                if msg_type.startswith("hub_drop."):
                    reason = msg_type.split(".", 1)[1]
            if reason:
                drop_reasons.append(reason)
            route = meta.get("route")
            if route:
                drop_routes.append(route)

    sql_view_summary: Dict[str, Any] = {}
    if hasattr(structured_store, "get_cycle_observability_summary"):
        try:
            sql_view_summary = await structured_store.get_cycle_observability_summary(cycle_count)
        except Exception:
            sql_view_summary = {}

    injection_events: List[Dict[str, Any]] = []
    if hasattr(structured_store, "get_prompt_injection_events"):
        try:
            injection_events = await structured_store.get_prompt_injection_events(
                cycle_count=cycle_count,
                limit=limit,
            )
        except Exception:
            injection_events = []
    verdicts = _count_values([e.get("verdict", "") for e in injection_events])
    injection_agents = _count_values([e.get("agent_name", "") for e in injection_events])
    blocked_count = sum(1 for e in injection_events if e.get("action_taken") == "block")

    return {
        "cycle_count": cycle_count,
        "sender_filter": sender,
        "mode_flags": mode_flags,
        "decision_summary": {
            "total": len(decisions),
            "output_actions": output_actions,
            "memory_actions": memory_actions,
            "by_agent": by_agent,
        },
        "message_summary": {
            "total": len(message_rows),
            "directions": directions,
            "message_types": message_types,
            "by_sender": by_sender_rows,
            "topics": topics,
        },
        "guard_drop_summary": {
            "total": len(drop_reasons),
            "by_reason": _count_values(drop_reasons),
            "by_route": _count_values(drop_routes),
        },
        "rate_limit_summary": {
            "total_events": len(rate_limit_events),
            "denied": len(denied),
            "denied_by_action": denied_by_action,
            "denied_by_reason": denied_by_reason,
        },
        "prompt_injection_summary": {
            "total_events": len(injection_events),
            "blocked": blocked_count,
            "by_verdict": verdicts,
            "by_agent": injection_agents,
        },
        "sql_view_summary": sql_view_summary,
    }

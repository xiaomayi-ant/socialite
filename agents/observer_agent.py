# -*- coding: utf-8 -*-
"""ObserverAgent — behaviour audit and A/B reporting.

Subscribes to action_completed → tracks A/B strategy performance →
detects anomalies → generates periodic reports.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.message import Message

logger = logging.getLogger(__name__)


class ObserverAgent(BaseAgent):
    """Observes all action outcomes and generates A/B strategy reports."""

    def __init__(
        self,
        name: str = "observer",
        structured_store=None,
    ) -> None:
        super().__init__(name)
        self.structured_store = structured_store
        self._action_log: List[Dict[str, Any]] = []
        self._strategy_stats: Dict[str, Dict[str, Any]] = {
            "A": {"count": 0, "successes": 0, "total_priority": 0.0},
            "B": {"count": 0, "successes": 0, "total_priority": 0.0},
        }

    async def observe(self, msg: Message) -> None:
        await super().observe(msg)
        if msg.metadata.get("type") == "action_completed":
            action_data = msg.metadata
            self._action_log.append(action_data)
            strategy = action_data.get("strategy", "A")
            success = action_data.get("success", False)
            priority = action_data.get("priority", 0.5)

            if strategy in self._strategy_stats:
                stats = self._strategy_stats[strategy]
                stats["count"] += 1
                if success:
                    stats["successes"] += 1
                stats["total_priority"] += priority

    async def generate_report(self, cycle_count: int = 0) -> Dict[str, Any]:
        """Generate a summary report of A/B strategy performance."""
        report = {
            "type": "observer_report",
            "cycle_count": cycle_count,
            "timestamp": datetime.now().isoformat(),
            "total_actions": len(self._action_log),
            "strategy_comparison": {},
            "action_breakdown": {},
            "anomalies": [],
        }

        # Strategy comparison
        for strategy, stats in self._strategy_stats.items():
            count = stats["count"]
            report["strategy_comparison"][strategy] = {
                "count": count,
                "success_rate": stats["successes"] / count if count > 0 else 0,
                "avg_priority": stats["total_priority"] / count if count > 0 else 0,
            }

        # Action type breakdown
        action_counts: Dict[str, int] = defaultdict(int)
        for action in self._action_log:
            action_counts[action.get("action", "unknown")] += 1
        report["action_breakdown"] = dict(action_counts)

        # Anomaly detection
        for strategy, stats in self._strategy_stats.items():
            count = stats["count"]
            if count >= 5:
                success_rate = stats["successes"] / count
                if success_rate < 0.1:
                    report["anomalies"].append(
                        f"Strategy {strategy} has very low success rate: {success_rate:.0%}"
                    )

        # Persist report
        if self.structured_store:
            try:
                await self.structured_store.record_observer_report(report)
            except Exception as e:
                logger.debug("Record observer report failed: %s", e)

        logger.info(
            "Observer report: %d actions, A=%d/%d, B=%d/%d",
            len(self._action_log),
            self._strategy_stats["A"]["successes"],
            self._strategy_stats["A"]["count"],
            self._strategy_stats["B"]["successes"],
            self._strategy_stats["B"]["count"],
        )

        return report

    def reset_stats(self) -> None:
        """Reset counters (e.g., at start of new reporting period)."""
        self._action_log.clear()
        for stats in self._strategy_stats.values():
            stats["count"] = 0
            stats["successes"] = 0
            stats["total_priority"] = 0.0

    async def reply(self, msg: Message) -> Message:
        return Message(
            name=self.name, role="assistant",
            content="ObserverAgent uses generate_report() interface",
        )

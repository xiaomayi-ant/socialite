# -*- coding: utf-8 -*-
"""Q-learning module for Socialite LearnerAgent.

Tabular Q-learning where:
  State  — discretized snapshot of community context + agent performance
  Action — comment/post style variants, upvote, follow
  Reward — weighted combination of upvotes, comments, follower gain

Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── Action space ──────────────────────────────────────────────────────────────

ACTIONS: List[str] = [
    "comment_technical",  # 技术性/分析性评论
    "comment_casual",     # 随性/观点性评论
    "post_insight",       # 洞察类发帖
    "post_question",      # 提问类发帖
    "upvote",
    "follow",
]

# ── State ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SocialState:
    """Discretized state for Q-table lookup.

    Kept intentionally small (4 dims × 3-4 values each = ~324 cells)
    so the table converges without thousands of cycles.
    """
    topic_category: str   # "ai_safety" | "agent_commerce" | "ai_tech" | "general"
    novelty_level: str    # "high" | "medium" | "low"
    evolution_stage: str  # "initial" | "exploration" | "optimization" | "innovation"
    recent_success: str   # "high" | "medium" | "low"

    def to_key(self) -> str:
        return f"{self.topic_category}|{self.novelty_level}|{self.evolution_stage}|{self.recent_success}"


def extract_state(
    analysis_data: Dict[str, Any],
    evolution_stage: str,
    recent_success_rate: float,
) -> SocialState:
    """Build a SocialState from AnalysisAgent result and evolution context."""
    trending: List[str] = analysis_data.get("trending_topics", [])
    topic_str = " ".join(trending[:3]).lower()

    if any(kw in topic_str for kw in ("safety", "align", "risk", "ethics", "govern")):
        topic = "ai_safety"
    elif any(kw in topic_str for kw in ("commerce", "market", "economy", "monetiz")):
        topic = "agent_commerce"
    elif any(kw in topic_str for kw in ("model", "llm", "gpt", "claude", "train", "rag")):
        topic = "ai_tech"
    else:
        topic = "general"

    novelty_scores = list(analysis_data.get("novelty_scores", {}).values())
    avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5
    novelty = "high" if avg_novelty >= 0.7 else ("low" if avg_novelty < 0.4 else "medium")

    success = "high" if recent_success_rate >= 0.6 else ("low" if recent_success_rate < 0.3 else "medium")

    return SocialState(
        topic_category=topic,
        novelty_level=novelty,
        evolution_stage=evolution_stage,
        recent_success=success,
    )


# ── Q-Table ───────────────────────────────────────────────────────────────────

class QTable:
    """Tabular Q-learning with ε-greedy exploration.

    epsilon is synced from LearnerAgent's exploration_rate so the existing
    Cosine Annealing / ReduceLROnPlateau schedule drives exploration decay.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.8,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # state_key → {action → Q-value}
        self._table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTIONS}
        )
        self.update_count: int = 0

    # ── Core methods ──────────────────────────────────────────

    def select_action(self, state: SocialState) -> str:
        """ε-greedy: explore randomly or exploit best known action."""
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_vals = self._table[state.to_key()]
        return max(q_vals, key=q_vals.__getitem__)

    def update(
        self,
        state: SocialState,
        action: str,
        reward: float,
        next_state: SocialState,
    ) -> None:
        """TD update: Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',·) − Q(s,a)]"""
        s = state.to_key()
        s_next = next_state.to_key()
        q_cur = self._table[s][action]
        best_next = max(self._table[s_next].values())
        td_error = (reward + self.gamma * best_next) - q_cur
        self._table[s][action] = round(q_cur + self.alpha * td_error, 6)
        self.update_count += 1

    # ── Query helpers ─────────────────────────────────────────

    def get_q_values(self, state: SocialState) -> Dict[str, float]:
        return dict(self._table[state.to_key()])

    def best_action_for_type(self, state: SocialState, action_type: str) -> Optional[str]:
        """Return highest-Q action whose name starts with action_type."""
        relevant = {
            a: v for a, v in self._table[state.to_key()].items()
            if a.startswith(action_type)
        }
        return max(relevant, key=relevant.__getitem__) if relevant else None

    def priority_boost(self, state: SocialState, action_type: str) -> float:
        """Map best Q-value for action_type to a [0, 0.25] priority boost.

        Uses a sigmoid so negative Q-values yield small boosts and
        large positive Q-values asymptotically approach 0.25.
        """
        relevant = {
            a: v for a, v in self._table[state.to_key()].items()
            if a.startswith(action_type)
        }
        if not relevant:
            return 0.0
        best_q = max(relevant.values())
        return round(0.25 / (1.0 + math.exp(-best_q)), 3)

    # ── Persistence ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "table": {k: dict(v) for k, v in self._table.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QTable":
        qt = cls(
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.9),
            epsilon=data.get("epsilon", 0.8),
        )
        qt.update_count = data.get("update_count", 0)
        for state_key, actions in data.get("table", {}).items():
            qt._table[state_key] = {a: actions.get(a, 0.0) for a in ACTIONS}
        return qt


# ── Reward helpers ────────────────────────────────────────────────────────────

def compute_immediate_reward(action_result: Dict[str, Any]) -> float:
    """Reward from action_completed metadata (immediate signal).

    Failed actions yield a small negative reward to discourage repetition.
    Successful actions yield a base reward scaled by proposal priority.
    """
    if not action_result.get("success", False):
        return -0.05

    action = action_result.get("action", "")
    priority = float(action_result.get("priority", 0.5))
    base = 0.1 + priority * 0.1

    if action == "comment":
        base += 0.05
    elif action == "post":
        base += 0.08

    return round(base, 4)


def compute_delayed_reward(feedback: Dict[str, Any]) -> float:
    """Reward from delayed feedback poll (upvotes_delta, comment_count, etc.)."""
    upvotes_delta = float(feedback.get("upvotes_delta", 0))
    comment_count = float(feedback.get("comment_count", 0))
    follower_gain = float(feedback.get("follower_gain", 0))
    return round(upvotes_delta * 0.5 + comment_count * 0.3 + follower_gain * 0.2, 4)

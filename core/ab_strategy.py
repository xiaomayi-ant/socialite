# -*- coding: utf-8 -*-
"""ABSelector — A/B strategy selector for action agents."""

import logging
import random

logger = logging.getLogger(__name__)


class ABSelector:
    """Select between strategy A and B.

    Modes:
      - ``"alternate"`` — strict A/B alternation per cycle.
      - ``"probability"`` — random selection with configurable probability.

    Usage::

        selector = ABSelector(mode="alternate")
        strategy = selector.select(cycle_count=0)   # → "A"
        strategy = selector.select(cycle_count=1)   # → "B"
    """

    def __init__(self, mode: str = "alternate", p_a: float = 0.5) -> None:
        """
        Args:
            mode: "alternate" or "probability".
            p_a: Probability of selecting "A" when mode is "probability".
        """
        if mode not in ("alternate", "probability"):
            raise ValueError(f"Unknown ABSelector mode: {mode!r}")
        self.mode = mode
        self.p_a = p_a

    def select(self, cycle_count: int = 0) -> str:
        """Return ``"A"`` or ``"B"`` for the given cycle."""
        if self.mode == "alternate":
            return "A" if cycle_count % 2 == 0 else "B"
        # probability mode
        return "A" if random.random() < self.p_a else "B"

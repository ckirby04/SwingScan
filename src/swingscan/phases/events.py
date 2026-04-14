"""The 8 canonical swing events.

Matches the GolfDB paper's event taxonomy. Enum order is load-bearing:
downstream code assumes events appear in this temporal order when
iterated, and per-swing label files list frame indices in the same
order.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["SwingEvent"]


class SwingEvent(Enum):
    ADDRESS = "address"
    TOE_UP = "toe_up"
    MID_BACKSWING = "mid_backswing"
    TOP = "top"
    MID_DOWNSWING = "mid_downswing"
    IMPACT = "impact"
    MID_FOLLOW_THROUGH = "mid_follow_through"
    FINISH = "finish"

    @property
    def index(self) -> int:
        """0-based event index in canonical temporal order."""
        return list(SwingEvent).index(self)

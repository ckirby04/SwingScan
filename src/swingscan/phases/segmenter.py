"""Swing-phase segmentation backends.

Two backends live here:

* :class:`HeuristicSegmenter` — pose-based, runs with no external
  weights. Uses wrist-midpoint position and velocity to estimate the
  8 canonical :class:`~swingscan.phases.events.SwingEvent` frames.
* (The SwingNet backend lives in :mod:`swingscan.phases.swingnet` and
  requires downloaded weights.)

Both implement the :class:`PhaseSegmenter` protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from swingscan.phases.events import SwingEvent
from swingscan.pose.base import PoseSequence
from swingscan.pose.keypoints import Joint

__all__ = ["PhaseMap", "PhaseSegmenter", "HeuristicSegmenter"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseMap:
    """Map of :class:`SwingEvent` → frame index.

    Stored as a tuple of (event, frame_index) pairs in canonical event
    order so iteration is deterministic.
    """

    events: tuple[tuple[SwingEvent, int], ...]

    def as_dict(self) -> dict[str, int]:
        return {ev.name: idx for ev, idx in self.events}

    def frame_for(self, event: SwingEvent) -> int:
        for ev, idx in self.events:
            if ev is event:
                return idx
        raise KeyError(f"Event {event!r} not in phase map.")

    def is_monotonic(self) -> bool:
        prev = -1
        for _, idx in self.events:
            if idx <= prev:
                return False
            prev = idx
        return True


@runtime_checkable
class PhaseSegmenter(Protocol):
    def segment(self, pose: PoseSequence) -> PhaseMap:
        ...


class HeuristicSegmenter:
    """Pose-based fallback segmenter (wrist velocity heuristic).

    Args:
        min_visibility: Visibility threshold used to pick the "address"
            frame and "finish" frame. Defaults to 0.3.
    """

    def __init__(self, min_visibility: float = 0.3) -> None:
        self._min_vis = min_visibility

    def segment(self, pose: PoseSequence) -> PhaseMap:
        n = len(pose)
        if n == 0:
            raise ValueError("Cannot segment an empty PoseSequence.")

        wrist_y = np.zeros(n, dtype=np.float32)
        wrist_x = np.zeros(n, dtype=np.float32)
        confidence = np.zeros(n, dtype=np.float32)
        for i, frame in enumerate(pose.frames):
            lw = frame.image_keypoints[Joint.LEFT_WRIST.value]
            rw = frame.image_keypoints[Joint.RIGHT_WRIST.value]
            wrist_x[i] = 0.5 * (lw[0] + rw[0])
            wrist_y[i] = 0.5 * (lw[1] + rw[1])
            confidence[i] = min(float(lw[3]), float(rw[3]))

        confident = confidence >= self._min_vis

        # Address: first confident frame, else frame 0.
        address_candidates = np.where(confident)[0]
        address = int(address_candidates[0]) if address_candidates.size else 0

        # Finish: last confident frame, else last frame.
        finish = int(address_candidates[-1]) if address_candidates.size else n - 1
        if finish <= address:
            finish = n - 1

        # Top of backswing: smallest wrist y (hands highest) among
        # confident frames, preferring the first half of the clip.
        if address_candidates.size:
            top_search = wrist_y.copy()
            top_search[~confident] = np.inf
            top = int(np.argmin(top_search))
        else:
            top = address + (finish - address) // 2

        top = max(top, address + 1)
        top = min(top, finish - 1)

        # Impact: max wrist speed between top and finish.
        speed = np.zeros(n, dtype=np.float32)
        if n > 1:
            dx = np.diff(wrist_x)
            dy = np.diff(wrist_y)
            speed[1:] = np.hypot(dx, dy)
        impact_window = speed.copy()
        impact_window[: top + 1] = -np.inf
        impact_window[finish:] = -np.inf
        impact = int(np.argmax(impact_window))
        if impact <= top:
            impact = top + 1
        impact = min(impact, finish - 1)

        # Fill intermediates with midpoints.
        toe_up = (address + top) // 2
        mid_backswing = (toe_up + top) // 2
        mid_downswing = (top + impact) // 2
        mid_follow_through = (impact + finish) // 2

        raw = [
            (SwingEvent.ADDRESS, address),
            (SwingEvent.TOE_UP, toe_up),
            (SwingEvent.MID_BACKSWING, mid_backswing),
            (SwingEvent.TOP, top),
            (SwingEvent.MID_DOWNSWING, mid_downswing),
            (SwingEvent.IMPACT, impact),
            (SwingEvent.MID_FOLLOW_THROUGH, mid_follow_through),
            (SwingEvent.FINISH, finish),
        ]

        # Enforce strict monotonicity by bumping any event that collides
        # with its predecessor.
        fixed: list[tuple[SwingEvent, int]] = []
        prev = -1
        for ev, idx in raw:
            if idx <= prev:
                idx = prev + 1
            if idx >= n:
                idx = n - 1
            fixed.append((ev, idx))
            prev = idx

        return PhaseMap(events=tuple(fixed))

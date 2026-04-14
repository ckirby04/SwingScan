"""Temporal smoothing and gap-fill for club-head detections.

Takes a list of raw :class:`~swingscan.club.detector.ClubDetection` —
one per frame, some potentially missing — and returns a :class:`ClubTrack`
with exponential smoothing applied and short gaps (≤ ``max_gap_frames``)
linearly interpolated.

This module is a pure-numpy transform with no external deps beyond the
ones already pulled by `swingscan.club.detector`. It holds the contract
that ``len(track) == len(raw)`` on the way out, so downstream code can
assume one entry per frame.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from swingscan.club.detector import ClubDetection

__all__ = ["ClubTrack", "ClubTracker"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClubTrack:
    """Smoothed, gap-filled sequence of club-head detections."""

    detections: tuple[ClubDetection, ...]

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[ClubDetection]:
        return iter(self.detections)

    @property
    def missing_ratio(self) -> float:
        if not self.detections:
            return 0.0
        return sum(1 for d in self.detections if d.source == "missing") / len(self.detections)

    @property
    def interpolated_ratio(self) -> float:
        if not self.detections:
            return 0.0
        return sum(1 for d in self.detections if d.source == "interpolated") / len(
            self.detections
        )


class ClubTracker:
    """Smooths detections and interpolates short missing runs.

    Args:
        alpha: Exponential-smoothing factor applied to ``x`` and ``y``.
            Higher = more responsive, lower = more smoothing. Default
            ``0.4`` tuned for ~30 fps consumer video.
        max_gap_frames: Longest run of missing detections the tracker
            will linearly interpolate across. Longer runs are left as
            ``source="missing"`` and a warning is logged.
    """

    def __init__(self, alpha: float = 0.4, max_gap_frames: int = 3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}.")
        if max_gap_frames < 0:
            raise ValueError(f"max_gap_frames must be ≥ 0; got {max_gap_frames}.")
        self._alpha = alpha
        self._max_gap = max_gap_frames

    def track(self, raw: Sequence[ClubDetection]) -> ClubTrack:
        """Apply gap-fill + exponential smoothing; return a new :class:`ClubTrack`."""
        if not raw:
            return ClubTrack(detections=())

        filled = self._gap_fill(list(raw))
        smoothed = self._smooth(filled)

        track = ClubTrack(detections=tuple(smoothed))
        if track.missing_ratio > 0:
            _log.warning(
                "Club tracker: %.1f%% of frames remain missing after gap-fill (max_gap=%d).",
                track.missing_ratio * 100,
                self._max_gap,
            )
        return track

    # ---------- internals ---------------------------------------------

    def _gap_fill(self, detections: list[ClubDetection]) -> list[ClubDetection]:
        """Linearly interpolate across short runs of missing detections."""
        n = len(detections)
        i = 0
        while i < n:
            if detections[i].source != "missing":
                i += 1
                continue

            # Find the end of this missing run.
            j = i
            while j < n and detections[j].source == "missing":
                j += 1

            gap_len = j - i
            left_idx = i - 1
            right_idx = j

            can_fill = (
                left_idx >= 0
                and right_idx < n
                and gap_len <= self._max_gap
                and detections[left_idx].source != "missing"
                and detections[right_idx].source != "missing"
            )

            if can_fill:
                left = detections[left_idx]
                right = detections[right_idx]
                for k in range(gap_len):
                    t = (k + 1) / (gap_len + 1)
                    det = detections[i + k]
                    detections[i + k] = ClubDetection(
                        frame_index=det.frame_index,
                        x=left.x + t * (right.x - left.x),
                        y=left.y + t * (right.y - left.y),
                        confidence=min(left.confidence, right.confidence) * 0.5,
                        source="interpolated",
                    )

            i = j

        return detections

    def _smooth(self, detections: list[ClubDetection]) -> list[ClubDetection]:
        """One-pass causal exponential smoothing over ``x``, ``y``."""
        out: list[ClubDetection] = []
        last_x: float | None = None
        last_y: float | None = None
        for det in detections:
            if det.source == "missing":
                out.append(det)
                last_x = None
                last_y = None
                continue

            if last_x is None or last_y is None:
                out.append(det)
                last_x = det.x
                last_y = det.y
                continue

            sx = self._alpha * det.x + (1.0 - self._alpha) * last_x
            sy = self._alpha * det.y + (1.0 - self._alpha) * last_y
            out.append(
                ClubDetection(
                    frame_index=det.frame_index,
                    x=sx,
                    y=sy,
                    confidence=det.confidence,
                    source=det.source,
                )
            )
            last_x = sx
            last_y = sy

        return out

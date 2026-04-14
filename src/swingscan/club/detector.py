"""Club-head detection backends.

Two backends are provided:

* :class:`YoloClubDetector` — thin wrapper around an ultralytics YOLO
  model fine-tuned for club-head detection. Requires a weights file at
  ``models/club_yolo.pt`` (or a caller-provided path).
* :class:`HeuristicClubDetector` — geometric fallback that extrapolates
  a club-head position from wrist positions in a :class:`PoseFrame`.
  Requires no weights. This is the Stage 2 "no external downloads"
  backend mandated by ``CLAUDE.md`` §5.2.

Both implement the :class:`ClubDetector` protocol and return a
:class:`ClubDetection` per frame. See
``docs/decisions/002-club-detection-fallback.md`` for the tradeoffs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from swingscan.pose.base import PoseFrame
from swingscan.pose.keypoints import Joint

__all__ = [
    "ClubDetection",
    "ClubDetector",
    "HeuristicClubDetector",
    "YoloClubDetector",
]

_log = logging.getLogger(__name__)

DetectionSource = Literal["yolo", "heuristic", "interpolated", "missing"]


@dataclass(frozen=True, slots=True)
class ClubDetection:
    """Single-frame club-head detection.

    Attributes:
        frame_index: 0-based index in the source video.
        x: Image-normalized x coordinate in ``[0, 1]``. ``0`` when missing.
        y: Image-normalized y coordinate in ``[0, 1]``. ``0`` when missing.
        confidence: Per-frame confidence in ``[0, 1]``. ``0`` when missing.
        source: Which backend produced the detection. ``"interpolated"``
            and ``"missing"`` are set by :class:`swingscan.club.tracker.ClubTracker`.
    """

    frame_index: int
    x: float
    y: float
    confidence: float
    source: DetectionSource

    @property
    def is_present(self) -> bool:
        return self.source not in ("missing",) and self.confidence > 0.0


@runtime_checkable
class ClubDetector(Protocol):
    """Structural interface every club-head backend must satisfy."""

    def detect_from_pose(self, pose: PoseFrame) -> ClubDetection:
        """Detect from a pose frame alone (pose-based backends only)."""
        ...

    def detect_from_frame(
        self,
        frame_bgr: NDArray[np.uint8],
        frame_index: int,
    ) -> ClubDetection:
        """Detect from a raw BGR image (pixel-based backends only)."""
        ...


# -------------------------------------------------------------------------
# Heuristic (pose-based) backend
# -------------------------------------------------------------------------


class HeuristicClubDetector:
    """Pose-based fallback: estimate the club-head position from the hands.

    Algorithm:

    1. Read both wrist positions in image-normalized space.
    2. Take their midpoint ``M`` as the anchor.
    3. Estimate the forearm direction from shoulder-midpoint to ``M``.
    4. Walk ``M`` a fraction of the shoulder width along that direction
       to approximate the club-head location. The fraction is configurable
       via :attr:`club_length_shoulder_ratio` (default 2.0 — i.e. the club
       head sits roughly two shoulder-widths past the hands, which is a
       rough average for a driver addressed face-on).
    5. Confidence is the product of the two wrist visibilities. When below
       :attr:`min_wrist_visibility`, emit a missing detection.

    This is explicitly a proxy, not a true detector. It is accurate
    enough to drive a Stage 4 heuristic phase segmenter but is not a
    substitute for a learned detector — see ADR 002.
    """

    def __init__(
        self,
        club_length_shoulder_ratio: float = 2.0,
        min_wrist_visibility: float = 0.3,
    ) -> None:
        self.club_length_shoulder_ratio = club_length_shoulder_ratio
        self.min_wrist_visibility = min_wrist_visibility

    def detect_from_pose(self, pose: PoseFrame) -> ClubDetection:
        kp = pose.image_keypoints
        lw = kp[Joint.LEFT_WRIST.value]
        rw = kp[Joint.RIGHT_WRIST.value]
        ls = kp[Joint.LEFT_SHOULDER.value]
        rs = kp[Joint.RIGHT_SHOULDER.value]

        lw_vis = float(lw[3])
        rw_vis = float(rw[3])
        visibility = lw_vis * rw_vis
        if lw_vis < self.min_wrist_visibility or rw_vis < self.min_wrist_visibility:
            return ClubDetection(
                frame_index=pose.frame_index,
                x=0.0,
                y=0.0,
                confidence=0.0,
                source="missing",
            )

        mx = 0.5 * (lw[0] + rw[0])
        my = 0.5 * (lw[1] + rw[1])

        sx = 0.5 * (ls[0] + rs[0])
        sy = 0.5 * (ls[1] + rs[1])

        dx = mx - sx
        dy = my - sy
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            # Hands directly under shoulders — common at address. Fall
            # back to straight-down direction (positive y in image space).
            dir_x, dir_y = 0.0, 1.0
        else:
            dir_x, dir_y = dx / norm, dy / norm

        # Shoulder width as a crude scale in image-normalized space.
        shoulder_width = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
        if shoulder_width < 1e-6:
            shoulder_width = 0.1

        club_x = float(mx + dir_x * self.club_length_shoulder_ratio * shoulder_width)
        club_y = float(my + dir_y * self.club_length_shoulder_ratio * shoulder_width)

        # Clamp into image bounds — downstream code assumes [0,1].
        club_x = max(0.0, min(1.0, club_x))
        club_y = max(0.0, min(1.0, club_y))

        return ClubDetection(
            frame_index=pose.frame_index,
            x=club_x,
            y=club_y,
            confidence=visibility,
            source="heuristic",
        )

    def detect_from_frame(
        self,
        frame_bgr: NDArray[np.uint8],
        frame_index: int,
    ) -> ClubDetection:
        # The heuristic backend does not consume pixel data.
        raise NotImplementedError(
            "HeuristicClubDetector operates on pose data. Use detect_from_pose(pose)."
        )


# -------------------------------------------------------------------------
# YOLO (pixel-based) backend
# -------------------------------------------------------------------------


class YoloClubDetector:
    """Ultralytics YOLOv8 wrapper for club-head detection.

    Lazy-imports :mod:`ultralytics` so the heavy torch stack is only paid
    when the user actually has a trained weights file.
    """

    def __init__(
        self,
        weights_path: Path | str = "models/club_yolo.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self._weights_path = Path(weights_path).expanduser().resolve()
        if not self._weights_path.is_file():
            raise FileNotFoundError(
                f"Club YOLO weights not found at {self._weights_path}. "
                "Train a detector or see models/README.md for download "
                "instructions when available. Falling back to "
                "HeuristicClubDetector is the intended alternative."
            )
        self._confidence_threshold = confidence_threshold

        from ultralytics import YOLO

        self._model = YOLO(str(self._weights_path))

    def detect_from_pose(self, pose: PoseFrame) -> ClubDetection:
        raise NotImplementedError(
            "YoloClubDetector operates on pixel data. Use detect_from_frame(frame)."
        )

    def detect_from_frame(
        self,
        frame_bgr: NDArray[np.uint8],
        frame_index: int,
    ) -> ClubDetection:
        # Ultralytics accepts numpy BGR arrays directly.
        results = self._model.predict(
            source=frame_bgr,
            conf=self._confidence_threshold,
            verbose=False,
        )
        if not results:
            return ClubDetection(
                frame_index=frame_index, x=0.0, y=0.0, confidence=0.0, source="missing"
            )

        best: tuple[float, float, float] | None = None
        img_h, img_w = frame_bgr.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xywh = boxes.xywh.cpu().numpy()  # (N, 4): cx, cy, w, h
            conf = boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                c = float(conf[i])
                if best is None or c > best[2]:
                    cx = float(xywh[i, 0]) / max(1, img_w)
                    cy = float(xywh[i, 1]) / max(1, img_h)
                    best = (cx, cy, c)

        if best is None:
            return ClubDetection(
                frame_index=frame_index, x=0.0, y=0.0, confidence=0.0, source="missing"
            )

        return ClubDetection(
            frame_index=frame_index,
            x=best[0],
            y=best[1],
            confidence=best[2],
            source="yolo",
        )

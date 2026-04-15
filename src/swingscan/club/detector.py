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
    """Pose-based fallback: estimate the club-head position from the grip.

    Algorithm:

    1. Read both wrist positions in image-normalized space; require
       each to clear :attr:`min_wrist_visibility` or emit a missing
       detection.
    2. Take the wrist midpoint ``G`` — the grip location where both
       hands meet on the club.
    3. Compute the **shaft direction** from the hand finger landmarks:
       average the visibility-weighted
       ``wrist -> (index + pinky + thumb)``
       vectors across both hands. MediaPipe's Pose model returns
       these 6 hand landmarks alongside the 27 body landmarks, so we
       get actual hand orientation at zero extra cost.
    4. Walk ``G`` along that unit direction by
       :attr:`club_forearm_ratio` times the average forearm length
       (elbow to wrist). Forearm length is proportional to the
       golfer's body scale, so this automatically adapts to camera
       distance.
    5. When the finger landmarks are too low-confidence to establish
       a direction (for example during a fast pass through the top of
       the backswing where MediaPipe sometimes loses the fingers), we
       fall back to the previous shoulder-to-wrist extrapolation with
       shoulder width as the scale — the old algorithm is still a
       reasonable last resort.
    6. Confidence is the product of the two wrist visibilities.

    This is still a geometric proxy, not a true learned detector —
    see ADR 002 — but on face-on driver clips it tracks the shaft
    orientation instead of always pointing "away from the shoulders,"
    which matters most at the top of the backswing and through impact.
    """

    def __init__(
        self,
        club_forearm_ratio: float = 3.8,
        min_wrist_visibility: float = 0.3,
        min_finger_visibility: float = 0.2,
    ) -> None:
        self.club_forearm_ratio = club_forearm_ratio
        self.min_wrist_visibility = min_wrist_visibility
        self.min_finger_visibility = min_finger_visibility

    def detect_from_pose(self, pose: PoseFrame) -> ClubDetection:
        kp = pose.image_keypoints
        lw = kp[Joint.LEFT_WRIST.value]
        rw = kp[Joint.RIGHT_WRIST.value]

        lw_vis = float(lw[3])
        rw_vis = float(rw[3])
        if lw_vis < self.min_wrist_visibility or rw_vis < self.min_wrist_visibility:
            return ClubDetection(
                frame_index=pose.frame_index,
                x=0.0,
                y=0.0,
                confidence=0.0,
                source="missing",
            )

        # Grip: midpoint of the two wrists.
        gx = 0.5 * (lw[0] + rw[0])
        gy = 0.5 * (lw[1] + rw[1])

        # Shaft direction from visibility-weighted hand-finger vectors.
        dir_x, dir_y, weight_sum = self._shaft_direction(kp)

        # Fall back to shoulder→wrist extrapolation when fingers aren't
        # trustworthy. This reproduces the old algorithm's behavior.
        if weight_sum < 1e-6:
            ls = kp[Joint.LEFT_SHOULDER.value]
            rs = kp[Joint.RIGHT_SHOULDER.value]
            sx = 0.5 * (ls[0] + rs[0])
            sy = 0.5 * (ls[1] + rs[1])
            dir_x = gx - sx
            dir_y = gy - sy

        norm = math.hypot(dir_x, dir_y)
        if norm < 1e-6:
            # Hands directly under shoulders and no finger signal —
            # common at a crisp address posture. Point straight down.
            dir_ux, dir_uy = 0.0, 1.0
        else:
            dir_ux = dir_x / norm
            dir_uy = dir_y / norm

        # Scale the club length by average forearm length (elbow→wrist).
        # Fall back to shoulder width when both elbows are missing.
        forearm_scale = self._forearm_scale(kp)
        if forearm_scale < 1e-6:
            ls = kp[Joint.LEFT_SHOULDER.value]
            rs = kp[Joint.RIGHT_SHOULDER.value]
            forearm_scale = 0.5 * math.hypot(ls[0] - rs[0], ls[1] - rs[1])
        if forearm_scale < 1e-6:
            forearm_scale = 0.1

        club_length = self.club_forearm_ratio * forearm_scale
        club_x = float(gx + dir_ux * club_length)
        club_y = float(gy + dir_uy * club_length)

        # Clamp into image bounds — downstream code assumes [0, 1].
        club_x = max(0.0, min(1.0, club_x))
        club_y = max(0.0, min(1.0, club_y))

        return ClubDetection(
            frame_index=pose.frame_index,
            x=club_x,
            y=club_y,
            confidence=lw_vis * rw_vis,
            source="heuristic",
        )

    def _shaft_direction(
        self, kp: NDArray[np.float32]
    ) -> tuple[float, float, float]:
        """Visibility-weighted wrist->finger vector for both hands."""
        dir_x = 0.0
        dir_y = 0.0
        weight_sum = 0.0
        pairs: tuple[tuple[Joint, Joint], ...] = (
            (Joint.LEFT_WRIST, Joint.LEFT_INDEX),
            (Joint.LEFT_WRIST, Joint.LEFT_PINKY),
            (Joint.LEFT_WRIST, Joint.LEFT_THUMB),
            (Joint.RIGHT_WRIST, Joint.RIGHT_INDEX),
            (Joint.RIGHT_WRIST, Joint.RIGHT_PINKY),
            (Joint.RIGHT_WRIST, Joint.RIGHT_THUMB),
        )
        for wrist, finger in pairs:
            fj = kp[finger.value]
            vis = float(fj[3])
            if vis < self.min_finger_visibility:
                continue
            wj = kp[wrist.value]
            dir_x += (float(fj[0]) - float(wj[0])) * vis
            dir_y += (float(fj[1]) - float(wj[1])) * vis
            weight_sum += vis
        return dir_x, dir_y, weight_sum

    def _forearm_scale(self, kp: NDArray[np.float32]) -> float:
        """Average forearm length from elbow to wrist across both sides."""
        pairs = (
            (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
            (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
        )
        lengths: list[float] = []
        for elbow, wrist in pairs:
            e = kp[elbow.value]
            w = kp[wrist.value]
            if float(e[3]) < self.min_wrist_visibility:
                continue
            lengths.append(math.hypot(float(w[0]) - float(e[0]), float(w[1]) - float(e[1])))
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

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

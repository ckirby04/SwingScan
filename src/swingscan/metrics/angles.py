"""Joint-angle and vector metrics computed from a :class:`PoseFrame`.

All angles are in degrees unless the function name says otherwise. Input
positions are read from the image-normalized keypoints (``[0, 1]``, y
axis pointing down) by default, with optional switch to world
coordinates where noted.

None of these functions assume MediaPipe-specific internals — they read
joints by :class:`~swingscan.pose.keypoints.Joint` enum value so the
same code works for any backend that implements the
:class:`~swingscan.pose.base.PoseEstimator` protocol.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from swingscan.pose.base import PoseFrame
from swingscan.pose.keypoints import Joint

__all__ = [
    "View",
    "signed_angle_deg",
    "hip_rotation_deg",
    "shoulder_rotation_deg",
    "x_factor_deg",
    "spine_lean_deg",
    "lead_arm_straightness_deg",
    "wrist_hinge_deg",
    "knee_flex_deg",
    "head_movement_px",
    "shoulder_width",
    "torso_length",
]

View = Literal["face_on", "down_the_line"]


def _pt(frame: PoseFrame, joint: Joint) -> NDArray[Any]:
    """Return a 2-vector (x, y) in image-normalized space."""
    return frame.image_keypoints[joint.value, :2]


def signed_angle_deg(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Signed angle from vector ``a`` to vector ``b`` in degrees.

    Positive values indicate counter-clockwise rotation in screen space
    (noting that image y axis points down).
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))


def shoulder_width(frame: PoseFrame) -> float:
    """Distance between left and right shoulder in image-normalized units."""
    diff = _pt(frame, Joint.LEFT_SHOULDER) - _pt(frame, Joint.RIGHT_SHOULDER)
    return float(np.linalg.norm(diff))


def torso_length(frame: PoseFrame) -> float:
    """Distance from shoulder midpoint to hip midpoint."""
    sh = 0.5 * (_pt(frame, Joint.LEFT_SHOULDER) + _pt(frame, Joint.RIGHT_SHOULDER))
    hp = 0.5 * (_pt(frame, Joint.LEFT_HIP) + _pt(frame, Joint.RIGHT_HIP))
    return float(np.linalg.norm(sh - hp))


def hip_rotation_deg(frame: PoseFrame, reference: PoseFrame) -> float:
    """Change in hip-line orientation relative to ``reference`` (typically address)."""

    def hip_vec(f: PoseFrame) -> NDArray[Any]:
        v = _pt(f, Joint.RIGHT_HIP) - _pt(f, Joint.LEFT_HIP)
        return np.asarray(v)

    return signed_angle_deg(hip_vec(reference), hip_vec(frame))


def shoulder_rotation_deg(frame: PoseFrame, reference: PoseFrame) -> float:
    """Change in shoulder-line orientation relative to ``reference``."""

    def sh_vec(f: PoseFrame) -> NDArray[Any]:
        v = _pt(f, Joint.RIGHT_SHOULDER) - _pt(f, Joint.LEFT_SHOULDER)
        return np.asarray(v)

    return signed_angle_deg(sh_vec(reference), sh_vec(frame))


def x_factor_deg(frame: PoseFrame, reference: PoseFrame) -> float:
    """Shoulder rotation minus hip rotation at the given frame."""
    return shoulder_rotation_deg(frame, reference) - hip_rotation_deg(frame, reference)


def spine_lean_deg(frame: PoseFrame) -> float:
    """Spine tilt relative to vertical (image y-axis).

    Computed as the angle between the shoulder-midpoint → hip-midpoint
    vector and the vertical axis.
    """
    sh = 0.5 * (_pt(frame, Joint.LEFT_SHOULDER) + _pt(frame, Joint.RIGHT_SHOULDER))
    hp = 0.5 * (_pt(frame, Joint.LEFT_HIP) + _pt(frame, Joint.RIGHT_HIP))
    v = sh - hp
    vertical = np.array([0.0, -1.0], dtype=np.float32)
    return signed_angle_deg(vertical, v)


def _arm_angle(frame: PoseFrame, shoulder: Joint, elbow: Joint, wrist: Joint) -> float:
    s = _pt(frame, shoulder)
    e = _pt(frame, elbow)
    w = _pt(frame, wrist)
    v1 = s - e
    v2 = w - e
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def lead_arm_straightness_deg(frame: PoseFrame, handedness: str = "right") -> float:
    """Interior angle at the lead elbow. 180° means fully extended.

    ``handedness="right"`` uses the golfer's **left** arm as the lead arm
    (common right-handed golfer convention).
    """
    if handedness == "right":
        return _arm_angle(frame, Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST)
    return _arm_angle(frame, Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST)


def wrist_hinge_deg(frame: PoseFrame) -> float:
    """Angle at the wrist between the forearm and the line to the index knuckle.

    Uses the lead (left) wrist by default. Returns 180° when the
    hand is in line with the forearm.
    """
    elbow = _pt(frame, Joint.LEFT_ELBOW)
    wrist = _pt(frame, Joint.LEFT_WRIST)
    index = _pt(frame, Joint.LEFT_INDEX)
    v1 = elbow - wrist
    v2 = index - wrist
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def knee_flex_deg(frame: PoseFrame, side: Literal["left", "right"] = "left") -> float:
    """Interior angle at the knee. 180° means fully extended."""
    if side == "left":
        return _arm_angle(frame, Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
    return _arm_angle(frame, Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE)


def head_movement_px(frame: PoseFrame, reference: PoseFrame) -> float:
    """Euclidean distance the nose has traveled relative to ``reference``.

    Image-normalized units (not pixels despite the function name; we
    keep the legacy "_px" suffix because consumers often think in
    pixels).
    """
    return float(np.linalg.norm(_pt(frame, Joint.NOSE) - _pt(reference, Joint.NOSE)))

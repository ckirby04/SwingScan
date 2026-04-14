"""Scale and orientation normalization for pose frames.

Produces a normalized copy of a :class:`PoseFrame` suitable for inter-
swing comparison. Two operations:

1. **Scale** by shoulder width (face-on) or torso length (down-the-line)
   so different camera distances yield comparable keypoint coordinates.
2. **Recenter** on the hip midpoint so positional bias (golfer standing
   left vs right in the frame) disappears.

A third operation, **handedness flip**, is provided separately: it
mirrors x coordinates so a left-handed golfer looks like a right-handed
one for all downstream metric code.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from swingscan.metrics.angles import shoulder_width, torso_length
from swingscan.pose.base import NUM_JOINTS, PoseFrame, empty_pose_array
from swingscan.pose.keypoints import Joint

__all__ = ["normalize_frame", "flip_handedness"]


def _hip_midpoint(frame: PoseFrame) -> NDArray[Any]:
    kp = frame.image_keypoints
    return 0.5 * (kp[Joint.LEFT_HIP.value, :2] + kp[Joint.RIGHT_HIP.value, :2])


def normalize_frame(frame: PoseFrame, view: str = "face_on") -> PoseFrame:
    """Return a centered, scale-normalized copy of ``frame``.

    Visibility and world-space keypoints are passed through unchanged.
    """
    scale = shoulder_width(frame) if view == "face_on" else torso_length(frame)
    if scale < 1e-6:
        scale = 1.0

    hip = _hip_midpoint(frame)

    out = empty_pose_array()
    src = frame.image_keypoints
    for i in range(NUM_JOINTS):
        out[i, 0] = (src[i, 0] - hip[0]) / scale
        out[i, 1] = (src[i, 1] - hip[1]) / scale
        out[i, 2] = src[i, 2] / scale
        out[i, 3] = src[i, 3]

    return PoseFrame(
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        image_keypoints=out,
        world_keypoints=frame.world_keypoints.copy(),
        is_low_confidence=frame.is_low_confidence,
    )


def flip_handedness(frame: PoseFrame) -> PoseFrame:
    """Mirror the frame horizontally — a left-handed swing becomes right-handed."""
    out = frame.image_keypoints.copy()
    out[:, 0] = 1.0 - out[:, 0]
    # Also swap the left/right joint pairs so "LEFT_WRIST" still denotes
    # the lead side after the flip.
    left_right_pairs = [
        (Joint.LEFT_EYE, Joint.RIGHT_EYE),
        (Joint.LEFT_EYE_INNER, Joint.RIGHT_EYE_INNER),
        (Joint.LEFT_EYE_OUTER, Joint.RIGHT_EYE_OUTER),
        (Joint.LEFT_EAR, Joint.RIGHT_EAR),
        (Joint.MOUTH_LEFT, Joint.MOUTH_RIGHT),
        (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
        (Joint.LEFT_ELBOW, Joint.RIGHT_ELBOW),
        (Joint.LEFT_WRIST, Joint.RIGHT_WRIST),
        (Joint.LEFT_PINKY, Joint.RIGHT_PINKY),
        (Joint.LEFT_INDEX, Joint.RIGHT_INDEX),
        (Joint.LEFT_THUMB, Joint.RIGHT_THUMB),
        (Joint.LEFT_HIP, Joint.RIGHT_HIP),
        (Joint.LEFT_KNEE, Joint.RIGHT_KNEE),
        (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE),
        (Joint.LEFT_HEEL, Joint.RIGHT_HEEL),
        (Joint.LEFT_FOOT_INDEX, Joint.RIGHT_FOOT_INDEX),
    ]
    for a, b in left_right_pairs:
        tmp = out[a.value].copy()
        out[a.value] = out[b.value]
        out[b.value] = tmp
    return PoseFrame(
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        image_keypoints=out,
        world_keypoints=frame.world_keypoints.copy(),
        is_low_confidence=frame.is_low_confidence,
    )

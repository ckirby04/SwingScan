"""BlazePose 33-joint landmark index enum and semantic joint groupings.

MediaPipe BlazePose returns 33 landmarks per frame. This module gives them
names and organizes them into groups used by the downstream pipeline
(pose overlays, biomech metrics, feedback rules).

Landmark order matches :mod:`mediapipe.solutions.pose.PoseLandmark` so
``Joint.LEFT_SHOULDER.value`` indexes a MediaPipe result list directly.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

__all__ = [
    "Joint",
    "HEAD",
    "SHOULDERS",
    "ELBOWS",
    "WRISTS",
    "HIPS",
    "KNEES",
    "ANKLES",
    "FEET",
    "LEFT_SIDE",
    "RIGHT_SIDE",
    "CORE_JOINTS",
]


class Joint(IntEnum):
    """BlazePose 33-landmark index.

    Values mirror :class:`mediapipe.solutions.pose.PoseLandmark` so an
    enum member can index directly into a MediaPipe result list.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# --- Semantic groupings --------------------------------------------------
# Each tuple is ordered for deterministic iteration and overlay drawing.

HEAD: Final[tuple[Joint, ...]] = (
    Joint.NOSE,
    Joint.LEFT_EYE_INNER,
    Joint.LEFT_EYE,
    Joint.LEFT_EYE_OUTER,
    Joint.RIGHT_EYE_INNER,
    Joint.RIGHT_EYE,
    Joint.RIGHT_EYE_OUTER,
    Joint.LEFT_EAR,
    Joint.RIGHT_EAR,
    Joint.MOUTH_LEFT,
    Joint.MOUTH_RIGHT,
)

SHOULDERS: Final[tuple[Joint, ...]] = (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
ELBOWS: Final[tuple[Joint, ...]] = (Joint.LEFT_ELBOW, Joint.RIGHT_ELBOW)
WRISTS: Final[tuple[Joint, ...]] = (Joint.LEFT_WRIST, Joint.RIGHT_WRIST)
HIPS: Final[tuple[Joint, ...]] = (Joint.LEFT_HIP, Joint.RIGHT_HIP)
KNEES: Final[tuple[Joint, ...]] = (Joint.LEFT_KNEE, Joint.RIGHT_KNEE)
ANKLES: Final[tuple[Joint, ...]] = (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)
FEET: Final[tuple[Joint, ...]] = (
    Joint.LEFT_HEEL,
    Joint.RIGHT_HEEL,
    Joint.LEFT_FOOT_INDEX,
    Joint.RIGHT_FOOT_INDEX,
)

# Per-side convenience groupings — useful for lead-side vs trail-side
# metrics that need to flip under handedness.
LEFT_SIDE: Final[tuple[Joint, ...]] = (
    Joint.LEFT_SHOULDER,
    Joint.LEFT_ELBOW,
    Joint.LEFT_WRIST,
    Joint.LEFT_HIP,
    Joint.LEFT_KNEE,
    Joint.LEFT_ANKLE,
)
RIGHT_SIDE: Final[tuple[Joint, ...]] = (
    Joint.RIGHT_SHOULDER,
    Joint.RIGHT_ELBOW,
    Joint.RIGHT_WRIST,
    Joint.RIGHT_HIP,
    Joint.RIGHT_KNEE,
    Joint.RIGHT_ANKLE,
)

# The "core" joints used by most biomech metrics — head, fingers,
# and toes are excluded. Exposed so downstream metric code can
# iterate a stable subset.
CORE_JOINTS: Final[tuple[Joint, ...]] = (
    *SHOULDERS,
    *ELBOWS,
    *WRISTS,
    *HIPS,
    *KNEES,
    *ANKLES,
)


def joint_by_name(name: str) -> Joint:
    """Resolve a joint by its enum name (case-insensitive).

    Args:
        name: Enum member name, e.g. ``"left_shoulder"`` or
            ``"LEFT_SHOULDER"``.

    Returns:
        The matching :class:`Joint`.

    Raises:
        KeyError: If no joint has that name.
    """
    return Joint[name.upper()]

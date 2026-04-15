"""Tests for :class:`swingscan.club.detector.HeuristicClubDetector`.

The current algorithm uses:
    1. MediaPipe hand finger landmarks (INDEX, PINKY, THUMB) to
       establish shaft direction when visible;
    2. Forearm length (elbow -> wrist) as the club-length scale;
    3. A shoulder -> wrist fallback direction when fingers aren't
       visible enough to trust.

Tests cover both the primary path and the fallback path.
"""

from __future__ import annotations

import math

from swingscan.club.detector import HeuristicClubDetector
from swingscan.pose.base import PoseFrame, empty_pose_array
from swingscan.pose.keypoints import Joint


def _pose_with(
    joints: dict[Joint, tuple[float, float]],
    visibility: float = 0.95,
) -> PoseFrame:
    kp = empty_pose_array()
    for joint, (x, y) in joints.items():
        kp[joint.value, 0] = x
        kp[joint.value, 1] = y
        kp[joint.value, 3] = visibility
    return PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_keypoints=kp,
        world_keypoints=empty_pose_array(),
    )


def _with_full_hands(
    shoulders: tuple[tuple[float, float], tuple[float, float]],
    elbows: tuple[tuple[float, float], tuple[float, float]],
    wrists: tuple[tuple[float, float], tuple[float, float]],
    fingers: tuple[tuple[float, float], tuple[float, float]],
) -> PoseFrame:
    ls, rs = shoulders
    le, re = elbows
    lw, rw = wrists
    lf, rf = fingers
    return _pose_with(
        {
            Joint.LEFT_SHOULDER: ls,
            Joint.RIGHT_SHOULDER: rs,
            Joint.LEFT_ELBOW: le,
            Joint.RIGHT_ELBOW: re,
            Joint.LEFT_WRIST: lw,
            Joint.RIGHT_WRIST: rw,
            Joint.LEFT_INDEX: lf,
            Joint.LEFT_PINKY: lf,
            Joint.LEFT_THUMB: lf,
            Joint.RIGHT_INDEX: rf,
            Joint.RIGHT_PINKY: rf,
            Joint.RIGHT_THUMB: rf,
        }
    )


def test_heuristic_missing_on_low_wrist_visibility() -> None:
    pose = _pose_with(
        {
            Joint.LEFT_SHOULDER: (0.40, 0.30),
            Joint.RIGHT_SHOULDER: (0.60, 0.30),
            Joint.LEFT_WRIST: (0.48, 0.55),
            Joint.RIGHT_WRIST: (0.52, 0.55),
        },
        visibility=0.05,
    )
    det = HeuristicClubDetector().detect_from_pose(pose)
    assert det.source == "missing"
    assert det.confidence == 0.0
    assert not det.is_present


def test_hand_landmark_path_follows_finger_direction() -> None:
    # Address posture: shoulders up, elbows bent, wrists at the grip,
    # fingers pointing down-and-slightly-forward (positive y, larger
    # than the wrist y). This is the primary hand-landmark path.
    pose = _with_full_hands(
        shoulders=((0.40, 0.30), (0.60, 0.30)),
        elbows=((0.42, 0.45), (0.58, 0.45)),
        wrists=((0.48, 0.55), (0.52, 0.55)),
        fingers=((0.48, 0.60), (0.52, 0.60)),
    )
    det = HeuristicClubDetector().detect_from_pose(pose)

    assert det.source == "heuristic"
    assert det.confidence > 0.9
    # Direction from wrist -> finger was +y, so the club head should
    # land below the wrist midpoint.
    assert det.y > 0.55
    # Grip x is 0.50 and fingers had the same x as wrists, so the
    # detection should stay horizontally centered.
    assert abs(det.x - 0.50) < 0.02


def test_hand_landmark_path_tracks_angled_shaft() -> None:
    # Top of backswing proxy: wrists up-and-right, fingers pointing
    # further up-and-right. The detection should walk past the wrists
    # in the same direction the fingers point.
    pose = _with_full_hands(
        shoulders=((0.40, 0.40), (0.55, 0.40)),
        elbows=((0.50, 0.35), (0.60, 0.35)),
        wrists=((0.60, 0.25), (0.62, 0.25)),
        fingers=((0.66, 0.18), (0.68, 0.18)),
    )
    det = HeuristicClubDetector().detect_from_pose(pose)

    assert det.source == "heuristic"
    # Detection should be up-and-right of the grip midpoint (0.61, 0.25).
    grip_x = 0.61
    grip_y = 0.25
    assert det.x > grip_x
    assert det.y < grip_y
    # And along the finger-pointing direction (up-right).
    expected_angle = math.atan2(0.18 - 0.25, 0.67 - 0.61)
    actual_angle = math.atan2(det.y - grip_y, det.x - grip_x)
    assert abs(expected_angle - actual_angle) < 0.2


def test_falls_back_to_shoulder_extrapolation_without_fingers() -> None:
    # No fingers -> the weight_sum is zero and the detector should
    # reach for the shoulder -> wrist fallback direction.
    pose = _pose_with(
        {
            Joint.LEFT_SHOULDER: (0.40, 0.20),
            Joint.RIGHT_SHOULDER: (0.60, 0.20),
            Joint.LEFT_ELBOW: (0.44, 0.40),
            Joint.RIGHT_ELBOW: (0.56, 0.40),
            Joint.LEFT_WRIST: (0.48, 0.55),
            Joint.RIGHT_WRIST: (0.52, 0.55),
        }
    )
    det = HeuristicClubDetector().detect_from_pose(pose)

    assert det.source == "heuristic"
    # Club should sit below the grip (positive y from shoulder -> wrist).
    assert det.y > 0.55
    assert 0.0 <= det.x <= 1.0
    assert 0.0 <= det.y <= 1.0


def test_club_length_scales_with_forearm_length() -> None:
    # Two poses identical except for forearm length: the one with the
    # longer forearm should produce a club head farther from the grip.
    short = _with_full_hands(
        shoulders=((0.40, 0.20), (0.60, 0.20)),
        elbows=((0.48, 0.40), (0.52, 0.40)),
        wrists=((0.48, 0.45), (0.52, 0.45)),  # short forearm (0.05)
        fingers=((0.48, 0.48), (0.52, 0.48)),
    )
    long = _with_full_hands(
        shoulders=((0.40, 0.20), (0.60, 0.20)),
        elbows=((0.48, 0.30), (0.52, 0.30)),
        wrists=((0.48, 0.45), (0.52, 0.45)),  # long forearm (0.15)
        fingers=((0.48, 0.48), (0.52, 0.48)),
    )
    short_det = HeuristicClubDetector().detect_from_pose(short)
    long_det = HeuristicClubDetector().detect_from_pose(long)

    # Both point the same direction (straight down), so the y-delta
    # from the grip is what differs. Longer forearm -> larger offset.
    assert long_det.y > short_det.y

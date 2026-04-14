"""Tests for :class:`swingscan.club.detector.HeuristicClubDetector`."""

from __future__ import annotations

import math

from swingscan.club.detector import HeuristicClubDetector
from swingscan.pose.base import PoseFrame, empty_pose_array
from swingscan.pose.keypoints import Joint


def _make_pose(
    left_shoulder: tuple[float, float],
    right_shoulder: tuple[float, float],
    left_wrist: tuple[float, float],
    right_wrist: tuple[float, float],
    visibility: float = 0.95,
) -> PoseFrame:
    kp = empty_pose_array()

    for joint, (x, y) in (
        (Joint.LEFT_SHOULDER, left_shoulder),
        (Joint.RIGHT_SHOULDER, right_shoulder),
        (Joint.LEFT_WRIST, left_wrist),
        (Joint.RIGHT_WRIST, right_wrist),
    ):
        kp[joint.value, 0] = x
        kp[joint.value, 1] = y
        kp[joint.value, 3] = visibility

    world = empty_pose_array()
    return PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_keypoints=kp,
        world_keypoints=world,
    )


def test_heuristic_places_club_below_hands_in_address_posture() -> None:
    # Address: shoulders high, wrists centered directly below them.
    pose = _make_pose(
        left_shoulder=(0.40, 0.30),
        right_shoulder=(0.60, 0.30),
        left_wrist=(0.48, 0.55),
        right_wrist=(0.52, 0.55),
    )
    det = HeuristicClubDetector(club_length_shoulder_ratio=2.0).detect_from_pose(pose)

    assert det.source == "heuristic"
    assert det.confidence > 0.9
    # Club head should be lower than the wrists (larger y in image space).
    wrist_midy = 0.55
    assert det.y > wrist_midy
    # And still within frame bounds.
    assert 0.0 <= det.x <= 1.0
    assert 0.0 <= det.y <= 1.0


def test_heuristic_missing_on_low_visibility() -> None:
    pose = _make_pose(
        left_shoulder=(0.40, 0.30),
        right_shoulder=(0.60, 0.30),
        left_wrist=(0.48, 0.55),
        right_wrist=(0.52, 0.55),
        visibility=0.05,
    )
    det = HeuristicClubDetector().detect_from_pose(pose)
    assert det.source == "missing"
    assert det.confidence == 0.0
    assert not det.is_present


def test_heuristic_extrapolation_direction_matches_forearm() -> None:
    # Rotate: shoulders top-left, wrists middle — club should go
    # bottom-right relative to the wrist midpoint.
    pose = _make_pose(
        left_shoulder=(0.20, 0.20),
        right_shoulder=(0.30, 0.20),
        left_wrist=(0.48, 0.48),
        right_wrist=(0.52, 0.48),
    )
    det = HeuristicClubDetector().detect_from_pose(pose)

    # Expected direction: from shoulder midpoint (0.25, 0.20) to wrist
    # midpoint (0.50, 0.48), pointing down-right.
    dx = det.x - 0.50
    dy = det.y - 0.48
    assert dx > 0
    assert dy > 0
    # The ratio of dx/dy should roughly match the forearm direction.
    expected = math.atan2(0.48 - 0.20, 0.50 - 0.25)
    actual = math.atan2(dy, dx)
    assert abs(expected - actual) < 0.1

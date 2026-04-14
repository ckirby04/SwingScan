"""Tests for :mod:`swingscan.metrics.angles`."""

from __future__ import annotations

import pytest

from swingscan.metrics import angles
from swingscan.pose.base import PoseFrame, empty_pose_array
from swingscan.pose.keypoints import Joint


def _frame_with(**joints: tuple[float, float]) -> PoseFrame:
    kp = empty_pose_array()
    for name, (x, y) in joints.items():
        j = Joint[name.upper()]
        kp[j.value, 0] = x
        kp[j.value, 1] = y
        kp[j.value, 3] = 1.0
    return PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_keypoints=kp,
        world_keypoints=empty_pose_array(),
    )


def test_signed_angle_orthogonal_is_90() -> None:
    import numpy as np

    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert angles.signed_angle_deg(a, b) == pytest.approx(90.0, abs=0.001)


def test_shoulder_width() -> None:
    f = _frame_with(
        left_shoulder=(0.4, 0.3),
        right_shoulder=(0.6, 0.3),
    )
    assert angles.shoulder_width(f) == pytest.approx(0.2, abs=1e-6)


def test_hip_rotation_is_zero_relative_to_self() -> None:
    f = _frame_with(
        left_hip=(0.45, 0.55),
        right_hip=(0.55, 0.55),
    )
    assert angles.hip_rotation_deg(f, f) == pytest.approx(0.0, abs=1e-3)


def test_hip_rotation_90_degrees() -> None:
    address = _frame_with(
        left_hip=(0.40, 0.50),
        right_hip=(0.60, 0.50),
    )
    # Rotate the hip line 90 degrees (LEFT and RIGHT swap y-axis roles).
    top = _frame_with(
        left_hip=(0.50, 0.40),
        right_hip=(0.50, 0.60),
    )
    rot = angles.hip_rotation_deg(top, address)
    assert abs(abs(rot) - 90.0) < 1.0


def test_lead_arm_straight_when_collinear() -> None:
    f = _frame_with(
        left_shoulder=(0.30, 0.30),
        left_elbow=(0.40, 0.50),
        left_wrist=(0.50, 0.70),
    )
    assert angles.lead_arm_straightness_deg(f) == pytest.approx(180.0, abs=1.0)


def test_lead_arm_bent_90() -> None:
    f = _frame_with(
        left_shoulder=(0.30, 0.30),
        left_elbow=(0.30, 0.50),
        left_wrist=(0.50, 0.50),
    )
    # Elbow at 90°: vectors (0, -0.2) and (0.2, 0) are perpendicular.
    assert angles.lead_arm_straightness_deg(f) == pytest.approx(90.0, abs=1.0)


def test_spine_lean_when_upright_is_near_zero() -> None:
    f = _frame_with(
        left_shoulder=(0.45, 0.30),
        right_shoulder=(0.55, 0.30),
        left_hip=(0.45, 0.70),
        right_hip=(0.55, 0.70),
    )
    assert abs(angles.spine_lean_deg(f)) < 1.0


def test_head_movement_zero_when_same_frame() -> None:
    f = _frame_with(nose=(0.5, 0.4))
    assert angles.head_movement_px(f, f) == pytest.approx(0.0)


def test_head_movement_euclidean() -> None:
    a = _frame_with(nose=(0.5, 0.4))
    b = _frame_with(nose=(0.5 + 0.03, 0.4 + 0.04))
    # Pythagoras: sqrt(0.03^2 + 0.04^2) = 0.05
    assert angles.head_movement_px(b, a) == pytest.approx(0.05, abs=1e-6)

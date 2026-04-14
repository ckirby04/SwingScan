"""Tests for :mod:`swingscan.pose.keypoints`."""

from __future__ import annotations

import pytest

from swingscan.pose.keypoints import (
    ANKLES,
    CORE_JOINTS,
    ELBOWS,
    FEET,
    HEAD,
    HIPS,
    KNEES,
    LEFT_SIDE,
    RIGHT_SIDE,
    SHOULDERS,
    WRISTS,
    Joint,
    joint_by_name,
)


def test_joint_enum_has_33_members() -> None:
    assert len(Joint) == 33


def test_joint_values_are_contiguous() -> None:
    assert [j.value for j in Joint] == list(range(33))


def test_groupings_cover_expected_joint_counts() -> None:
    assert len(HEAD) == 11
    assert len(SHOULDERS) == 2
    assert len(ELBOWS) == 2
    assert len(WRISTS) == 2
    assert len(HIPS) == 2
    assert len(KNEES) == 2
    assert len(ANKLES) == 2
    assert len(FEET) == 4


def test_left_and_right_sides_are_symmetric() -> None:
    # Shoulders → ankles: six joints per side.
    assert len(LEFT_SIDE) == 6
    assert len(RIGHT_SIDE) == 6
    for lj, rj in zip(LEFT_SIDE, RIGHT_SIDE, strict=True):
        assert lj.name.replace("LEFT_", "") == rj.name.replace("RIGHT_", "")


def test_core_joints_spans_major_limb_joints() -> None:
    assert set(CORE_JOINTS) == (
        set(SHOULDERS) | set(ELBOWS) | set(WRISTS) | set(HIPS) | set(KNEES) | set(ANKLES)
    )
    assert len(CORE_JOINTS) == 12


def test_joint_by_name_case_insensitive() -> None:
    assert joint_by_name("LEFT_WRIST") is Joint.LEFT_WRIST
    assert joint_by_name("left_wrist") is Joint.LEFT_WRIST


def test_joint_by_name_raises_on_unknown() -> None:
    with pytest.raises(KeyError):
        joint_by_name("NOT_A_JOINT")

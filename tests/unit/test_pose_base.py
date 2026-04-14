"""Tests for :mod:`swingscan.pose.base`."""

from __future__ import annotations

import numpy as np
import pytest

from swingscan.pose.base import (
    KEYPOINT_COLUMNS,
    NUM_JOINTS,
    PoseEstimator,
    PoseFrame,
    PoseSequence,
    empty_pose_array,
)
from swingscan.pose.keypoints import Joint


def _sample_frame(frame_index: int = 0, visibility: float = 0.9) -> PoseFrame:
    kp = empty_pose_array()
    kp[:, 0] = 0.5
    kp[:, 1] = 0.5
    kp[:, 3] = visibility
    return PoseFrame(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        image_keypoints=kp.copy(),
        world_keypoints=kp.copy(),
    )


def test_empty_pose_array_shape_and_dtype() -> None:
    arr = empty_pose_array()
    assert arr.shape == (NUM_JOINTS, len(KEYPOINT_COLUMNS))
    assert arr.dtype == np.float32


def test_pose_frame_rejects_wrong_shape() -> None:
    bad = np.zeros((10, 4), dtype=np.float32)
    good = empty_pose_array()
    with pytest.raises(ValueError, match="shape"):
        PoseFrame(
            frame_index=0,
            timestamp_s=0.0,
            image_keypoints=bad,
            world_keypoints=good,
        )


def test_pose_frame_rejects_wrong_dtype() -> None:
    wrong_dtype = np.zeros((NUM_JOINTS, 4), dtype=np.float64)
    good = empty_pose_array()
    with pytest.raises(ValueError, match="float32"):
        PoseFrame(
            frame_index=0,
            timestamp_s=0.0,
            image_keypoints=wrong_dtype,
            world_keypoints=good,
        )


def test_pose_frame_visibility_lookup() -> None:
    frame = _sample_frame(visibility=0.75)
    assert frame.visibility(Joint.LEFT_WRIST) == pytest.approx(0.75)
    assert frame.median_core_visibility() == pytest.approx(0.75)


def test_pose_sequence_iteration_and_length() -> None:
    frames = tuple(_sample_frame(i) for i in range(5))
    seq = PoseSequence(frames=frames, fps=30.0, width=160, height=160, duration_s=5 / 30)
    assert len(seq) == 5
    assert seq.frame_count == 5
    assert next(iter(seq)).frame_index == 0
    assert seq.low_confidence_ratio() == 0.0


def test_pose_sequence_low_confidence_ratio() -> None:
    frames = (
        _sample_frame(0, visibility=0.9),
        PoseFrame(
            frame_index=1,
            timestamp_s=1 / 30,
            image_keypoints=empty_pose_array(),
            world_keypoints=empty_pose_array(),
            is_low_confidence=True,
        ),
    )
    seq = PoseSequence(frames=frames, fps=30.0, width=160, height=160, duration_s=2 / 30)
    assert seq.low_confidence_ratio() == pytest.approx(0.5)


def test_pose_sequence_empty_low_confidence_ratio_is_zero() -> None:
    seq = PoseSequence(frames=(), fps=30.0, width=0, height=0, duration_s=0.0)
    assert seq.low_confidence_ratio() == 0.0


class _FakeBackend:
    def estimate(self, frame, frame_index, timestamp_s):  # type: ignore[no-untyped-def]
        return _sample_frame(frame_index)

    def estimate_video(self, video):  # type: ignore[no-untyped-def]
        return PoseSequence(frames=(), fps=30.0, width=0, height=0, duration_s=0.0)


def test_fake_backend_satisfies_protocol() -> None:
    # runtime_checkable protocols only check attribute presence, which is
    # exactly what we want for a structural duck-typing contract.
    assert isinstance(_FakeBackend(), PoseEstimator)

"""Round-trip tests for :mod:`swingscan.io.serialize`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swingscan.io.serialize import (
    load_pose_sequence,
    load_pose_sequence_json,
    load_pose_sequence_parquet,
    save_pose_sequence,
    save_pose_sequence_json,
    save_pose_sequence_parquet,
)
from swingscan.pose.base import NUM_JOINTS, PoseFrame, PoseSequence, empty_pose_array
from swingscan.pose.keypoints import Joint


def _make_sequence(n_frames: int = 4) -> PoseSequence:
    rng = np.random.default_rng(seed=17)
    frames: list[PoseFrame] = []
    for i in range(n_frames):
        img = empty_pose_array()
        world = empty_pose_array()
        img[:] = rng.random((NUM_JOINTS, 4), dtype=np.float32)
        world[:] = rng.random((NUM_JOINTS, 4), dtype=np.float32)
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp_s=i / 30.0,
                image_keypoints=img,
                world_keypoints=world,
                is_low_confidence=(i % 2 == 0),
            )
        )
    return PoseSequence(
        frames=tuple(frames),
        fps=30.0,
        width=160,
        height=160,
        duration_s=n_frames / 30.0,
        source_path="dummy.mp4",
    )


def _assert_sequences_equal(a: PoseSequence, b: PoseSequence) -> None:
    assert len(a) == len(b)
    assert a.fps == pytest.approx(b.fps)
    assert a.width == b.width
    assert a.height == b.height
    assert a.duration_s == pytest.approx(b.duration_s)
    assert a.source_path == b.source_path
    for fa, fb in zip(a, b, strict=True):
        assert fa.frame_index == fb.frame_index
        assert fa.timestamp_s == pytest.approx(fb.timestamp_s)
        assert fa.is_low_confidence == fb.is_low_confidence
        np.testing.assert_allclose(fa.image_keypoints, fb.image_keypoints, atol=1e-6)
        np.testing.assert_allclose(fa.world_keypoints, fb.world_keypoints, atol=1e-6)


def test_parquet_round_trip(tmp_path: Path) -> None:
    seq = _make_sequence()
    out = save_pose_sequence_parquet(seq, tmp_path / "seq.parquet")
    assert out.is_file()
    reloaded = load_pose_sequence_parquet(out)
    _assert_sequences_equal(seq, reloaded)


def test_json_round_trip(tmp_path: Path) -> None:
    seq = _make_sequence()
    out = save_pose_sequence_json(seq, tmp_path / "seq.json")
    assert out.is_file()
    reloaded = load_pose_sequence_json(out)
    _assert_sequences_equal(seq, reloaded)


def test_save_dispatches_on_extension(tmp_path: Path) -> None:
    seq = _make_sequence()
    parquet_path = save_pose_sequence(seq, tmp_path / "seq.parquet")
    json_path = save_pose_sequence(seq, tmp_path / "seq.json")
    assert parquet_path.is_file()
    assert json_path.is_file()

    _assert_sequences_equal(seq, load_pose_sequence(parquet_path))
    _assert_sequences_equal(seq, load_pose_sequence(json_path))


def test_unknown_extension_raises(tmp_path: Path) -> None:
    seq = _make_sequence()
    with pytest.raises(ValueError, match="extension"):
        save_pose_sequence(seq, tmp_path / "seq.bogus")


def test_parquet_preserves_individual_joint_columns(tmp_path: Path) -> None:
    seq = _make_sequence(n_frames=1)
    out = save_pose_sequence_parquet(seq, tmp_path / "seq.parquet")
    reloaded = load_pose_sequence_parquet(out)
    # Specifically check one joint to guard against column-ordering bugs.
    expected = seq.frames[0].image_keypoints[Joint.LEFT_WRIST.value]
    actual = reloaded.frames[0].image_keypoints[Joint.LEFT_WRIST.value]
    np.testing.assert_allclose(actual, expected, atol=1e-6)

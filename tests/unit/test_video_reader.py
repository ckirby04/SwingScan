"""Tests for :class:`swingscan.io.video.VideoReader`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swingscan.io.video import VideoReader


def test_open_and_read_all_frames(sample_swing_path: Path) -> None:
    with VideoReader(sample_swing_path) as video:
        frames = list(video)

    # The generator emits 60 frames (2s @ 30 fps), but codec settling
    # may trim one or two; allow a small tolerance.
    assert 58 <= len(frames) <= 60
    assert video.fps == pytest.approx(30.0, abs=0.5)
    assert video.width == 160
    assert video.height == 160
    assert video.rotation_deg == 0
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (160, 160, 3)
        assert frame.dtype == np.uint8


def test_frame_count_matches_metadata(sample_swing_path: Path) -> None:
    with VideoReader(sample_swing_path) as video:
        declared = video.frame_count
        actual = sum(1 for _ in video)
    assert declared == pytest.approx(actual, abs=2)


def test_reiterate_within_context(sample_swing_path: Path) -> None:
    with VideoReader(sample_swing_path) as video:
        first = sum(1 for _ in video)
        second = sum(1 for _ in video)
    assert first == second


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "nope.mp4")

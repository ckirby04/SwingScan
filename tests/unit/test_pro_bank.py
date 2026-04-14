"""Tests for :mod:`swingscan.compare.pro_bank`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swingscan.compare.pro_bank import ProBank, ProSwing
from swingscan.phases.events import SwingEvent
from swingscan.pose.base import PoseFrame, empty_pose_array


def _pose(frame_index: int, base: float) -> PoseFrame:
    img = empty_pose_array()
    world = empty_pose_array()
    img[:, 0] = base
    img[:, 1] = base
    img[:, 3] = 0.9
    return PoseFrame(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        image_keypoints=img,
        world_keypoints=world,
    )


def _swing(swing_id: str, view: str, hand: str, club: str, base: float) -> ProSwing:
    events = tuple(e.name for e in SwingEvent)
    frames = tuple(range(8))
    poses = tuple(_pose(i, base + i * 0.01) for i in range(8))
    return ProSwing(
        swing_id=swing_id,
        view=view,
        handedness=hand,
        club=club,
        event_names=events,
        event_frames=frames,
        event_poses=poses,
    )


def _mini_bank() -> ProBank:
    return ProBank.from_rows(
        [
            _swing("rory_1", "face_on", "right", "driver", base=0.10),
            _swing("rory_2", "down_the_line", "right", "driver", base=0.20),
            _swing("phil_1", "face_on", "left", "iron7", base=0.30),
        ]
    )


def test_bank_length_and_iteration() -> None:
    bank = _mini_bank()
    assert len(bank) == 3
    assert [s.swing_id for s in bank] == ["rory_1", "rory_2", "phil_1"]


def test_filter_by_view() -> None:
    bank = _mini_bank().filter(view="face_on")
    assert len(bank) == 2
    assert all(s.view == "face_on" for s in bank)


def test_filter_by_handedness_and_club() -> None:
    bank = _mini_bank().filter(handedness="right", club="driver")
    assert len(bank) == 2
    assert {s.swing_id for s in bank} == {"rory_1", "rory_2"}


def test_filter_chain_yields_empty_when_overconstrained() -> None:
    bank = _mini_bank().filter(view="face_on", handedness="left", club="driver")
    assert len(bank) == 0


def test_event_poses_returns_stacked_arrays() -> None:
    bank = _mini_bank()
    arrays = bank.event_poses(SwingEvent.IMPACT.name)
    assert len(arrays) == 3
    for arr in arrays:
        assert arr.shape == (33, 4)


def test_parquet_round_trip(tmp_path: Path) -> None:
    bank = _mini_bank()
    out = bank.save(tmp_path / "bank.parquet")
    assert out.is_file()

    reloaded = ProBank.load(out)
    assert len(reloaded) == 3
    # Row-level sanity: matching first swing has the expected base offset.
    first = next(s for s in reloaded if s.swing_id == "rory_1")
    # base=0.10 → first event pose's x column is 0.10.
    np.testing.assert_allclose(first.event_poses[0].image_keypoints[:, 0], 0.10, atol=1e-5)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProBank.load(tmp_path / "nope.parquet")

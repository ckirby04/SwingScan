"""Tests for :class:`swingscan.club.tracker.ClubTracker`."""

from __future__ import annotations

import pytest

from swingscan.club.detector import ClubDetection
from swingscan.club.tracker import ClubTracker


def _det(frame_index: int, x: float, y: float, source: str = "heuristic") -> ClubDetection:
    # `source` is a Literal in the dataclass; cast through the type for the fixture.
    return ClubDetection(
        frame_index=frame_index,
        x=x,
        y=y,
        confidence=1.0 if source != "missing" else 0.0,
        source=source,  # type: ignore[arg-type]
    )


def test_short_gap_is_interpolated() -> None:
    raw = [
        _det(0, 0.1, 0.1),
        _det(1, 0.0, 0.0, source="missing"),
        _det(2, 0.0, 0.0, source="missing"),
        _det(3, 0.4, 0.4),
    ]
    track = ClubTracker(alpha=1.0, max_gap_frames=3).track(raw)

    assert len(track) == 4
    assert track.detections[1].source == "interpolated"
    assert track.detections[2].source == "interpolated"
    # Linear interpolation: frames 1 and 2 lie on the line between 0 and 3.
    assert track.detections[1].x == pytest.approx(0.1 + (0.4 - 0.1) * 1 / 3, abs=1e-6)
    assert track.detections[2].x == pytest.approx(0.1 + (0.4 - 0.1) * 2 / 3, abs=1e-6)


def test_long_gap_stays_missing() -> None:
    raw = [
        _det(0, 0.0, 0.0),
        *(_det(i, 0.0, 0.0, source="missing") for i in range(1, 6)),
        _det(6, 1.0, 1.0),
    ]
    track = ClubTracker(alpha=1.0, max_gap_frames=3).track(raw)

    assert len(track) == 7
    missing_count = sum(1 for d in track.detections if d.source == "missing")
    assert missing_count == 5
    assert track.missing_ratio == pytest.approx(5 / 7)


def test_exponential_smoothing_pulls_towards_new_data() -> None:
    raw = [_det(i, 0.5, 0.5) for i in range(3)]
    raw.append(_det(3, 0.9, 0.9))
    track = ClubTracker(alpha=0.5, max_gap_frames=3).track(raw)

    # At alpha=0.5, the new point should shift halfway toward 0.9.
    assert track.detections[3].x == pytest.approx((0.5 + 0.9) / 2, abs=1e-6)
    assert track.detections[3].y == pytest.approx((0.5 + 0.9) / 2, abs=1e-6)


def test_gap_at_start_is_left_missing() -> None:
    raw = [
        _det(0, 0.0, 0.0, source="missing"),
        _det(1, 0.0, 0.0, source="missing"),
        _det(2, 0.4, 0.4),
    ]
    track = ClubTracker().track(raw)
    # No left neighbor → cannot interpolate.
    assert track.detections[0].source == "missing"
    assert track.detections[1].source == "missing"


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        ClubTracker(alpha=0.0)
    with pytest.raises(ValueError):
        ClubTracker(alpha=1.5)


def test_invalid_max_gap_rejected() -> None:
    with pytest.raises(ValueError):
        ClubTracker(max_gap_frames=-1)


def test_empty_track_passes_through() -> None:
    track = ClubTracker().track([])
    assert len(track) == 0
    assert track.missing_ratio == 0.0
    assert track.interpolated_ratio == 0.0

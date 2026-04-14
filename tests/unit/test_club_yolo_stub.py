"""Test that YoloClubDetector surfaces a clear error when weights are missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from swingscan.club.detector import YoloClubDetector


def test_yolo_detector_raises_when_weights_missing(tmp_path: Path) -> None:
    missing = tmp_path / "club_yolo.pt"
    with pytest.raises(FileNotFoundError, match="Club YOLO weights"):
        YoloClubDetector(weights_path=missing)

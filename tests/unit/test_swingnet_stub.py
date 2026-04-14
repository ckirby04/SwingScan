"""Test that SwingNetSegmenter surfaces a clear error when weights are missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from swingscan.phases.swingnet import SwingNetSegmenter, load_swingnet


def test_load_swingnet_raises_without_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SwingNet weights"):
        load_swingnet(tmp_path / "swingnet.pt")


def test_segmenter_init_raises_without_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SwingNetSegmenter(weights_path=tmp_path / "missing.pt")

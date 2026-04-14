"""Tests for :mod:`swingscan.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swingscan.config import SwingScanConfig, load_config


def test_default_config_is_valid() -> None:
    cfg = SwingScanConfig()
    assert cfg.pose.backend == "mediapipe"
    assert cfg.seed == 17
    assert cfg.logging.level == "INFO"


def test_load_config_from_yaml(tmp_path: Path) -> None:
    payload = {
        "seed": 99,
        "pose": {"backend": "mediapipe", "min_detection_confidence": 0.7},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(path)

    assert cfg.seed == 99
    assert cfg.pose.min_detection_confidence == pytest.approx(0.7)


def test_load_config_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = {"seed": 1, "bogus": True}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_load_config_none_returns_default_instance() -> None:
    # Whether or not configs/default.yaml exists on disk at test time, the
    # returned value is always a validated SwingScanConfig.
    cfg = load_config(None)
    assert isinstance(cfg, SwingScanConfig)

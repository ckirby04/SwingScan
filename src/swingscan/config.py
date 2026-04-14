"""Typed configuration models loaded from YAML under ``configs/``.

All runtime configuration flows through :class:`SwingScanConfig`. Later
stages extend the nested models (pose backends, phase segmenter, feedback
rules, demo settings) — Stage 0 only ships the top-level skeleton so the
smoke test and type checker stay green.

Load flow::

    from swingscan.config import load_config
    cfg = load_config()                       # default.yaml (if present)
    cfg = load_config("configs/custom.yaml")  # explicit path

All models set ``extra="forbid"`` so typos in YAML raise
:class:`pydantic.ValidationError` instead of silently dropping fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

from swingscan.utils.paths import configs_dir

_DEFAULT_CONFIG_FILENAME: Final[str] = "default.yaml"


class LoggingConfig(BaseModel):
    """Controls how SwingScan configures stdlib logging at startup."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", description="Root log level.")


class PoseConfig(BaseModel):
    """Pose-backend selection and confidence thresholds.

    Stage 0 only declares the schema. Stage 1 wires these values into the
    MediaPipe BlazePose backend.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(
        default="mediapipe",
        description="Name of the pose backend. Only 'mediapipe' is supported in V1.",
    )
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SwingScanConfig(BaseModel):
    """Top-level SwingScan configuration."""

    model_config = ConfigDict(extra="forbid")

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    seed: int = Field(default=17, description="Seed applied by seed_everything().")


def load_config(path: Path | str | None = None) -> SwingScanConfig:
    """Load a :class:`SwingScanConfig` from a YAML file.

    Args:
        path: Path to a YAML config. When ``None``, loads
            ``configs/default.yaml`` from the repo root; if that file is
            absent, returns a default-constructed config so that Stage 0
            callers don't need to ship a config file.

    Returns:
        A validated :class:`SwingScanConfig` instance.

    Raises:
        FileNotFoundError: If an explicit ``path`` was provided but does
            not exist on disk.
        ValueError: If the YAML file's top-level node is not a mapping.
        pydantic.ValidationError: If the YAML contents fail schema
            validation.
    """
    if path is None:
        default_path = configs_dir() / _DEFAULT_CONFIG_FILENAME
        if not default_path.exists():
            return SwingScanConfig()
        path = default_path

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config at {resolved} must be a YAML mapping at the top level; "
            f"got {type(raw).__name__}."
        )
    return SwingScanConfig.model_validate(raw)

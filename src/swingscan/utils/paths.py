"""Repository-relative path helpers.

SwingScan is a source-layout package (``src/swingscan/...``), which means
the repo root is three parents above this file::

    <repo_root>/src/swingscan/utils/paths.py

All of these helpers return absolute paths so callers don't need to worry
about the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


def data_dir() -> Path:
    """Absolute path to ``data/`` under the repo root. Gitignored at runtime."""
    return _REPO_ROOT / "data"


def models_dir() -> Path:
    """Absolute path to ``models/`` under the repo root. Gitignored at runtime."""
    return _REPO_ROOT / "models"


def configs_dir() -> Path:
    """Absolute path to ``configs/`` under the repo root."""
    return _REPO_ROOT / "configs"

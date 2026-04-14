"""Tests for :mod:`swingscan.utils.paths`."""

from __future__ import annotations

from swingscan.utils.paths import configs_dir, data_dir, models_dir, repo_root


def test_repo_root_contains_pyproject() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()


def test_dir_helpers_are_anchored_at_repo_root() -> None:
    root = repo_root()
    assert data_dir() == root / "data"
    assert models_dir() == root / "models"
    assert configs_dir() == root / "configs"

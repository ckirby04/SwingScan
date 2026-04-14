"""Pytest configuration and shared fixtures for the SwingScan test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from gen_sample_swing import generate_sample_swing


@pytest.fixture(scope="session")
def sample_swing_path() -> Path:
    """Absolute path to a synthetic swing clip.

    The clip is generated lazily on first use and cached for the remainder
    of the session. Regenerating is cheap (<1s) and keeps us from shipping
    an mp4 that has to be re-downloaded on every clone.
    """
    fixture_dir = Path(__file__).parent / "fixtures"
    target = fixture_dir / "sample_swing.mp4"
    if not target.is_file():
        generate_sample_swing(target)
    return target

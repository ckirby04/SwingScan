"""Tests for :mod:`swingscan.utils.ffmpeg`."""

from __future__ import annotations

import pytest

from swingscan.utils.ffmpeg import ensure_ffmpeg_available, probe_ffmpeg


def test_probe_ffmpeg_returns_info() -> None:
    info = probe_ffmpeg()
    # This test runs in an environment where ffmpeg was installed as a
    # hard prerequisite for Stage 1 (see docs/plans/stage1-status.md),
    # so we expect it to be found.
    assert info.available is True
    assert info.path is not None


def test_ensure_ffmpeg_available_returns_info_when_present() -> None:
    info = ensure_ffmpeg_available()
    assert info.available is True


def test_ensure_ffmpeg_available_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import swingscan.utils.ffmpeg as mod

    monkeypatch.setattr(mod, "probe_ffmpeg", lambda: mod.FfmpegInfo(False, None, None))
    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        mod.ensure_ffmpeg_available()

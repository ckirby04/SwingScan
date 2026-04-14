"""ffmpeg availability probing.

``CLAUDE.md`` §3 requires that we never assume ffmpeg is installed: we must
check and report when it is missing rather than letting OpenCV fail with an
opaque error deep in the video pipeline. This module provides the check
helper used by :mod:`swingscan.io.video` at :class:`VideoReader` open time.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["FfmpegInfo", "probe_ffmpeg", "ensure_ffmpeg_available"]


@dataclass(frozen=True, slots=True)
class FfmpegInfo:
    """Outcome of probing the environment for an ffmpeg install."""

    available: bool
    path: str | None
    version: str | None


def probe_ffmpeg() -> FfmpegInfo:
    """Detect ffmpeg on ``PATH`` and capture its version string.

    Returns:
        A :class:`FfmpegInfo`. When ``available`` is False, both ``path``
        and ``version`` are ``None``.
    """
    resolved = shutil.which("ffmpeg")
    if resolved is None:
        return FfmpegInfo(available=False, path=None, version=None)

    try:
        completed = subprocess.run(
            [resolved, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return FfmpegInfo(available=True, path=resolved, version=None)

    first_line = completed.stdout.splitlines()[0] if completed.stdout else None
    return FfmpegInfo(available=True, path=resolved, version=first_line)


def ensure_ffmpeg_available() -> FfmpegInfo:
    """Return :class:`FfmpegInfo` or raise if ffmpeg is missing.

    Raises:
        RuntimeError: If no ``ffmpeg`` binary is reachable via ``PATH``.
            The error message points at install instructions.
    """
    info = probe_ffmpeg()
    if not info.available:
        raise RuntimeError(
            "ffmpeg was not found on PATH. SwingScan's video pipeline needs "
            "ffmpeg for robust decoding and rotation metadata.\n"
            "Install options:\n"
            "  * Windows: winget install Gyan.FFmpeg\n"
            "  * macOS:   brew install ffmpeg\n"
            "  * Linux:   apt install ffmpeg  (or your distro equivalent)"
        )
    return info

"""Video reading with rotation handling.

Wraps OpenCV's ``VideoCapture`` with a small, typed interface that downstream
pipeline stages can consume without touching cv2 directly. The reader:

* Exposes ``fps``, ``width``, ``height``, ``frame_count``, ``duration_s``.
* Iterates frames in BGR ``uint8`` (OpenCV's native layout).
* Honors container-level rotation metadata via ffprobe when available,
  falling back to ``cv2.CAP_PROP_ORIENTATION_META`` on OpenCV ≥ 4.6 and
  finally to zero rotation with a warning log.
* Raises a clear error at open time if ffmpeg is missing, per
  ``CLAUDE.md`` §3.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from swingscan.utils.ffmpeg import ensure_ffmpeg_available

__all__ = ["VideoReader", "VideoMetadata"]

_log = logging.getLogger(__name__)

# OpenCV rotation codes for the four legal values.
_ROTATE_CODES: Final[dict[int, int]] = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class VideoMetadata:
    """Lightweight bundle of read-only video metadata.

    Exposed as a plain class (not a dataclass) so the attribute names
    remain stable for parquet/JSON serialization in
    :mod:`swingscan.io.serialize`.
    """

    __slots__ = ("fps", "width", "height", "frame_count", "duration_s", "rotation_deg")

    def __init__(
        self,
        fps: float,
        width: int,
        height: int,
        frame_count: int,
        duration_s: float,
        rotation_deg: int,
    ) -> None:
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.duration_s = duration_s
        self.rotation_deg = rotation_deg


def _ffprobe_rotation(path: Path) -> int:
    """Return the rotation stored in the container in degrees (0 / 90 / 180 / 270).

    Best-effort: ffprobe is invoked on the first video stream. Errors are
    logged and treated as zero rotation.
    """
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=rotation:side_data=rotation",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("ffprobe rotation lookup failed for %s: %s", path, exc)
        return 0

    if completed.returncode != 0 or not completed.stdout.strip():
        return 0

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return 0

    streams = payload.get("streams") or []
    if not streams:
        return 0

    stream = streams[0]
    # Modern ffprobe puts rotation under side_data_list; older versions
    # expose a 'rotation' tag on the stream itself.
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                continue
    raw_rot = stream.get("rotation")
    if raw_rot is not None:
        try:
            return int(round(float(raw_rot))) % 360
        except (TypeError, ValueError):
            return 0
    return 0


class VideoReader(AbstractContextManager["VideoReader"]):
    """Context-managed iterator over the frames of a single video file.

    Usage::

        with VideoReader("swing.mp4") as video:
            for idx, frame_bgr in enumerate(video):
                ...

    The reader rewinds on :meth:`__iter__`, so it can be iterated
    multiple times within a single ``with`` block.
    """

    def __init__(self, path: Path | str) -> None:
        ensure_ffmpeg_available()

        self._path = Path(path).expanduser().resolve()
        if not self._path.is_file():
            raise FileNotFoundError(f"Video not found: {self._path}")

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            cap.release()
            raise OSError(f"OpenCV could not open video: {self._path}")

        self._cap: cv2.VideoCapture | None = cap

        # Populate metadata. cv2 reports fps as a double; guard against
        # zero/NaN that some phone codecs emit.
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0 or np.isnan(fps):
            _log.warning("cv2 reported non-positive fps for %s; defaulting to 30.", self._path)
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        rotation = _ffprobe_rotation(self._path)
        if rotation == 0:
            # OpenCV ≥ 4.6 exposes CAP_PROP_ORIENTATION_META.
            try:
                meta_rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
            except (AttributeError, cv2.error):
                meta_rot = 0
            rotation = meta_rot % 360 if meta_rot > 0 else 0

        if rotation not in (0, 90, 180, 270):
            _log.warning(
                "Unexpected rotation %s° on %s; ignoring.", rotation, self._path
            )
            rotation = 0

        # After rotation, width/height may swap.
        effective_width, effective_height = (
            (height, width) if rotation in (90, 270) else (width, height)
        )

        self._metadata = VideoMetadata(
            fps=fps,
            width=effective_width,
            height=effective_height,
            frame_count=frame_count,
            duration_s=frame_count / fps if fps > 0 else 0.0,
            rotation_deg=rotation,
        )

    # ---------- properties ------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    @property
    def fps(self) -> float:
        return self._metadata.fps

    @property
    def width(self) -> int:
        return self._metadata.width

    @property
    def height(self) -> int:
        return self._metadata.height

    @property
    def frame_count(self) -> int:
        return self._metadata.frame_count

    @property
    def duration_s(self) -> float:
        return self._metadata.duration_s

    @property
    def rotation_deg(self) -> int:
        return self._metadata.rotation_deg

    # ---------- iteration -------------------------------------------------

    def __iter__(self) -> Iterator[NDArray[np.uint8]]:
        if self._cap is None:
            raise RuntimeError("VideoReader is closed; reopen before iterating.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self._iter_frames()

    def _iter_frames(self) -> Iterator[NDArray[np.uint8]]:
        assert self._cap is not None  # narrowed by __iter__
        rotate_code = _ROTATE_CODES.get(self._metadata.rotation_deg)
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            # cv2 returns ndarray[uint8] at runtime; narrow the type for callers.
            yield np.asarray(frame, dtype=np.uint8)

    # ---------- lifecycle -------------------------------------------------

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VideoReader(path={self._path.name!r}, "
            f"{self._metadata.width}x{self._metadata.height} @ "
            f"{self._metadata.fps:.1f}fps, "
            f"{self._metadata.frame_count} frames, "
            f"rot={self._metadata.rotation_deg}°)"
        )

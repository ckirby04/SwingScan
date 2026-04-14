"""Annotated video rendering.

Draws pose skeletons, club trails, a phase banner, and a metrics HUD on
each frame of the source video. Writes an mp4 using OpenCV's
``VideoWriter`` with a preferred-codec chain (``mp4v`` first, then
``avc1``, then ``MJPG`` as a last resort).

Stage 7 goal: given a `PipelineResult` and the original source video,
produce a watchable annotated output in under 2x real-time on CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from swingscan.club.tracker import ClubTrack
from swingscan.io.video import VideoReader
from swingscan.phases.segmenter import PhaseMap
from swingscan.pose.base import PoseFrame, PoseSequence
from swingscan.pose.keypoints import Joint

__all__ = [
    "draw_skeleton",
    "draw_club_trail",
    "draw_phase_banner",
    "render_annotated_video",
]

_log = logging.getLogger(__name__)

_SKELETON_LINKS: tuple[tuple[Joint, Joint], ...] = (
    # Torso box.
    (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_HIP),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    # Arms.
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    # Legs.
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
)

_LINE_COLOR: tuple[int, int, int] = (80, 220, 80)
_POINT_COLOR: tuple[int, int, int] = (0, 180, 255)
_CLUB_COLOR: tuple[int, int, int] = (0, 150, 255)
_BANNER_BG: tuple[int, int, int] = (30, 30, 30)
_BANNER_FG: tuple[int, int, int] = (240, 240, 240)


def _ip(frame_shape: tuple[int, int], nx: float, ny: float) -> tuple[int, int]:
    h, w = frame_shape
    return int(round(nx * w)), int(round(ny * h))


def draw_skeleton(
    canvas: NDArray[np.uint8],
    pose: PoseFrame,
    min_visibility: float = 0.3,
) -> None:
    """Draw the torso/arm/leg skeleton from a :class:`PoseFrame` in place."""
    h, w = canvas.shape[:2]

    for a, b in _SKELETON_LINKS:
        va = float(pose.image_keypoints[a.value, 3])
        vb = float(pose.image_keypoints[b.value, 3])
        if va < min_visibility or vb < min_visibility:
            continue
        pa = _ip((h, w), float(pose.image_keypoints[a.value, 0]),
                 float(pose.image_keypoints[a.value, 1]))
        pb = _ip((h, w), float(pose.image_keypoints[b.value, 0]),
                 float(pose.image_keypoints[b.value, 1]))
        cv2.line(canvas, pa, pb, _LINE_COLOR, 2)

    for joint in (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER,
                  Joint.LEFT_ELBOW, Joint.RIGHT_ELBOW,
                  Joint.LEFT_WRIST, Joint.RIGHT_WRIST,
                  Joint.LEFT_HIP, Joint.RIGHT_HIP,
                  Joint.LEFT_KNEE, Joint.RIGHT_KNEE,
                  Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE):
        v = float(pose.image_keypoints[joint.value, 3])
        if v < min_visibility:
            continue
        p = _ip((h, w), float(pose.image_keypoints[joint.value, 0]),
                float(pose.image_keypoints[joint.value, 1]))
        cv2.circle(canvas, p, 3, _POINT_COLOR, -1)


def draw_club_trail(
    canvas: NDArray[np.uint8],
    club: ClubTrack,
    up_to_frame: int,
    trail_length: int = 15,
) -> None:
    """Draw the last ``trail_length`` club positions up to the given frame."""
    h, w = canvas.shape[:2]
    start = max(0, up_to_frame - trail_length + 1)
    end = min(len(club), up_to_frame + 1)
    points: list[tuple[int, int]] = []
    for i in range(start, end):
        d = club.detections[i]
        if d.source == "missing" or d.confidence <= 0.0:
            continue
        points.append(_ip((h, w), d.x, d.y))

    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], _CLUB_COLOR, 2)
    if points:
        cv2.circle(canvas, points[-1], 4, _CLUB_COLOR, -1)


def _active_event(phases: PhaseMap | None, frame_index: int) -> str:
    if phases is None:
        return "—"
    current = "ADDRESS"
    for ev, idx in phases.events:
        if idx <= frame_index:
            current = ev.name
        else:
            break
    return current


def draw_phase_banner(
    canvas: NDArray[np.uint8],
    phases: PhaseMap | None,
    frame_index: int,
) -> None:
    """Draw a small top-left phase label."""
    text = f"Phase: {_active_event(phases, frame_index)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 4
    x0, y0 = 6, 6
    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + tw + pad * 2, y0 + th + pad * 2),
        _BANNER_BG,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        (x0 + pad, y0 + th + pad - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _BANNER_FG,
        1,
        cv2.LINE_AA,
    )


def _open_writer(
    out_path: Path,
    fps: float,
    size: tuple[int, int],
) -> Any:
    """Try codecs in order and return the first writer that opens."""
    for fourcc_tag in ("mp4v", "avc1", "MJPG"):
        fourcc = cv2.VideoWriter.fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if writer.isOpened():
            _log.debug("VideoWriter opened with codec %s", fourcc_tag)
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open any VideoWriter codec for {out_path}. Tried: mp4v, avc1, MJPG."
    )


def render_annotated_video(
    source_video: Path | str,
    pose: PoseSequence,
    club: ClubTrack,
    phases: PhaseMap | None,
    out_path: Path | str,
) -> Path:
    """Render an annotated output video. Returns the written path.

    Each source frame is decorated with the pose skeleton, the trailing
    club-head arc, and a phase banner.
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with VideoReader(source_video) as video:
        size = (video.width, video.height)
        writer = _open_writer(out, float(video.fps), size)
        try:
            for i, bgr in enumerate(video):
                canvas = bgr.copy()
                if i < len(pose):
                    draw_skeleton(canvas, pose.frames[i])
                draw_club_trail(canvas, club, i)
                draw_phase_banner(canvas, phases, i)
                writer.write(canvas)
        finally:
            writer.release()

    _log.info("Wrote annotated video to %s", out)
    return out

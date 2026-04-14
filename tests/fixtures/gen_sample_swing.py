"""Generate a tiny synthetic swing-shaped video for the test suite.

Run this once to (re)create ``tests/fixtures/sample_swing.mp4``. The
resulting clip is a 160x160 blank canvas with a moving colored circle
(intended as a "club head" proxy) and an overlaid bouncing rectangle
("torso" proxy). It is deterministic — seeded — so regenerating on any
machine produces the same bytes (modulo codec quirks).

This file intentionally does NOT depict a real golfer. Per ``CLAUDE.md``
§6, test fixtures must be synthetic or permissively licensed to avoid
likeness/rights concerns.

Usage::

    python tests/fixtures/gen_sample_swing.py
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

_DEFAULT_OUTPUT = Path(__file__).with_name("sample_swing.mp4")


def generate_sample_swing(
    out_path: Path | str | None = None,
    width: int = 160,
    height: int = 160,
    fps: int = 30,
    duration_s: float = 2.0,
) -> Path:
    """Write a synthetic clip to ``out_path``; return the resolved path.

    Args:
        out_path: Destination mp4 file. Defaults to
            ``tests/fixtures/sample_swing.mp4``.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second of the output container.
        duration_s: Clip duration in seconds.
    """
    out = Path(out_path if out_path is not None else _DEFAULT_OUTPUT).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(round(fps * duration_s))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out}.")

    try:
        cx_torso = width // 2
        cy_torso = height // 2

        for i in range(total_frames):
            # Gray background.
            frame = np.full((height, width, 3), 40, dtype=np.uint8)

            # Torso: a vertical rectangle that bobs slightly.
            torso_h = 60
            torso_w = 20
            dy = int(4 * math.sin(2 * math.pi * i / total_frames))
            top_left = (cx_torso - torso_w // 2, cy_torso - torso_h // 2 + dy)
            bot_right = (cx_torso + torso_w // 2, cy_torso + torso_h // 2 + dy)
            cv2.rectangle(frame, top_left, bot_right, (200, 200, 200), thickness=-1)

            # "Club head": a circle that sweeps an arc around the torso.
            arc_angle = math.pi * 1.8 * (i / max(1, total_frames - 1)) - math.pi * 0.9
            radius = 50
            club_x = int(cx_torso + radius * math.sin(arc_angle))
            club_y = int(cy_torso - radius * math.cos(arc_angle))
            cv2.circle(frame, (club_x, club_y), 5, (0, 180, 255), thickness=-1)

            writer.write(frame)
    finally:
        writer.release()

    return out


if __name__ == "__main__":
    path = generate_sample_swing()
    print(f"Wrote synthetic sample swing to {path}")

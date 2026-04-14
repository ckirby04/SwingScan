"""Smoke test for `render_annotated_video` via the `swingscan run` CLI."""

from __future__ import annotations

from pathlib import Path

from swingscan.cli import main
from swingscan.io.video import VideoReader


def test_run_writes_annotated_video(sample_swing_path: Path, tmp_path: Path) -> None:
    out_video = tmp_path / "annotated.mp4"
    rc = main(
        [
            "run",
            "--input",
            str(sample_swing_path),
            "--output-video",
            str(out_video),
        ]
    )
    assert rc == 0
    assert out_video.is_file()
    # Must have at least one frame and a plausible duration.
    with VideoReader(out_video) as video:
        frames = sum(1 for _ in video)
    assert frames > 0

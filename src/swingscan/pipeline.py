"""Top-level pipeline orchestration.

Stage 2 introduces this module to stitch together pose extraction and
club-head tracking. Later stages append phase segmentation, metrics,
comparison, and feedback — each added behind its own flag so earlier
stages can be exercised in isolation.

The CLI's ``swingscan run`` subcommand dispatches here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from swingscan.club.detector import (
    ClubDetection,
    ClubDetector,
    HeuristicClubDetector,
    YoloClubDetector,
)
from swingscan.club.tracker import ClubTrack, ClubTracker
from swingscan.config import SwingScanConfig, load_config
from swingscan.io.video import VideoReader
from swingscan.pose.base import PoseSequence
from swingscan.pose.mediapipe_backend import MediaPipePoseEstimator

__all__ = ["PipelineResult", "run_pipeline", "save_pipeline_result"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Combined output of the Stage 2 pipeline: pose + club trajectory."""

    pose: PoseSequence
    club: ClubTrack

    @property
    def frame_count(self) -> int:
        return len(self.pose)


def _build_club_detector(
    club_weights: Path | str | None,
) -> tuple[ClubDetector, str]:
    """Pick the club backend. Returns (detector, label)."""
    if club_weights is not None:
        try:
            det = YoloClubDetector(weights_path=club_weights)
        except FileNotFoundError as exc:
            _log.warning(
                "YOLO club weights unavailable (%s). Falling back to heuristic.",
                exc,
            )
        else:
            _log.info("Using YoloClubDetector with weights %s", club_weights)
            return det, "yolo"

    _log.info("Using HeuristicClubDetector (pose-derived fallback).")
    return HeuristicClubDetector(), "heuristic"


def run_pipeline(
    video_path: Path | str,
    config: SwingScanConfig | None = None,
    club_weights: Path | str | None = None,
) -> PipelineResult:
    """Run Stage 1 + Stage 2 on a single video and return the result.

    Args:
        video_path: Path to the source swing video.
        config: Optional :class:`SwingScanConfig`. When ``None``, the
            default config is loaded via :func:`load_config`.
        club_weights: Optional path to a trained YOLO weights file. When
            absent or the path is missing, the heuristic backend is used.

    Returns:
        A :class:`PipelineResult` with pose + club outputs. Frame counts
        match: ``len(result.pose) == len(result.club)``.
    """
    cfg = config or load_config()

    with (
        VideoReader(video_path) as video,
        MediaPipePoseEstimator(cfg.pose) as pose_est,
    ):
        pose_seq = pose_est.estimate_video(video)

    club_detector, backend_label = _build_club_detector(club_weights)

    raw: list[ClubDetection] = []
    if backend_label == "yolo":
        # Pixel-based backend: re-open the video and step through frames.
        with VideoReader(video_path) as video:
            for idx, frame_bgr in enumerate(video):
                raw.append(club_detector.detect_from_frame(frame_bgr, idx))
    else:
        # Pose-based backend: iterate over the already-computed PoseSequence.
        for frame in pose_seq:
            raw.append(club_detector.detect_from_pose(frame))

    tracker = ClubTracker()
    track = tracker.track(raw)

    _log.info(
        "Pipeline complete: %d pose frames, %d club entries "
        "(%.1f%% missing, %.1f%% interpolated).",
        len(pose_seq),
        len(track),
        track.missing_ratio * 100,
        track.interpolated_ratio * 100,
    )

    return PipelineResult(pose=pose_seq, club=track)


def save_pipeline_result(result: PipelineResult, path: Path | str) -> Path:
    """Write a pipeline result to a JSON report.

    The report contains pose metadata (not the full per-frame keypoints —
    those belong in the parquet saved by ``swingscan pose``), the full
    club trajectory, and a tiny summary block. It is intended for Stage 2
    smoke inspection; later stages write richer artifacts.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "swingscan_pipeline_v1",
        "pose_summary": {
            "fps": result.pose.fps,
            "width": result.pose.width,
            "height": result.pose.height,
            "duration_s": result.pose.duration_s,
            "source_path": result.pose.source_path,
            "frame_count": len(result.pose),
            "low_confidence_ratio": result.pose.low_confidence_ratio(),
        },
        "club_track": [
            {
                "frame_index": d.frame_index,
                "x": d.x,
                "y": d.y,
                "confidence": d.confidence,
                "source": d.source,
            }
            for d in result.club
        ],
        "summary": {
            "frame_count": result.frame_count,
            "club_missing_ratio": result.club.missing_ratio,
            "club_interpolated_ratio": result.club.interpolated_ratio,
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out

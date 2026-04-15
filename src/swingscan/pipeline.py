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
from swingscan.compare.diff import SwingDiff, compare_against_bank
from swingscan.compare.pro_bank import ProBank
from swingscan.config import SwingScanConfig, load_config
from swingscan.feedback.render import render_text
from swingscan.feedback.rules import FeedbackItem, RuleEngine
from swingscan.io.video import VideoReader
from swingscan.phases.segmenter import HeuristicSegmenter, PhaseMap, PhaseSegmenter
from swingscan.pose.base import PoseSequence
from swingscan.pose.mediapipe_backend import MediaPipePoseEstimator

__all__ = ["PipelineResult", "run_pipeline", "save_pipeline_result"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Combined output of the pipeline.

    Populated incrementally by each stage — Stage 2 fills pose + club;
    Stage 4 fills phases; Stage 5 fills the diff; Stage 6 fills the
    feedback list. Later stages may leave fields ``None`` when the
    required inputs are unavailable (e.g. no pro bank on disk).
    """

    pose: PoseSequence
    club: ClubTrack
    phases: PhaseMap | None = None
    diff: SwingDiff | None = None
    feedback: tuple[FeedbackItem, ...] = ()

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


def _build_segmenter(swingnet_weights: Path | str | None) -> tuple[PhaseSegmenter, str]:
    """Pick the phase segmenter. Returns (segmenter, label)."""
    if swingnet_weights is not None:
        from swingscan.phases.swingnet import SwingNetSegmenter

        try:
            segmenter = SwingNetSegmenter(weights_path=swingnet_weights)
        except FileNotFoundError as exc:
            _log.warning(
                "SwingNet weights unavailable (%s). Falling back to HeuristicSegmenter.",
                exc,
            )
        else:
            _log.info("Using SwingNetSegmenter with weights %s", swingnet_weights)
            return segmenter, "swingnet"

    _log.info("Using HeuristicSegmenter (no SwingNet weights).")
    return HeuristicSegmenter(), "heuristic"


def run_pipeline(
    video_path: Path | str,
    config: SwingScanConfig | None = None,
    club_weights: Path | str | None = None,
    pro_bank_path: Path | str | None = None,
    rules_path: Path | str | None = None,
    swingnet_weights: Path | str | None = None,
    handedness: str = "right",
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
        "Pose + club complete: %d frames, %d club entries "
        "(%.1f%% missing, %.1f%% interpolated).",
        len(pose_seq),
        len(track),
        track.missing_ratio * 100,
        track.interpolated_ratio * 100,
    )

    # Stage 4: phase segmentation. Prefer SwingNet when weights are
    # supplied; fall back to the wrist-velocity heuristic otherwise.
    phase_map: PhaseMap | None = None
    if len(pose_seq) > 0:
        segmenter, _ = _build_segmenter(swingnet_weights)
        phase_map = segmenter.segment(pose_seq)

    # Stage 5 + 6: comparison against a bank + feedback.
    diff: SwingDiff | None = None
    feedback: tuple[FeedbackItem, ...] = ()
    if phase_map is not None and pro_bank_path is not None:
        try:
            bank = ProBank.load(pro_bank_path)
        except FileNotFoundError:
            _log.warning("Pro bank not found at %s; skipping comparison.", pro_bank_path)
        else:
            diff = compare_against_bank(pose_seq, phase_map, bank, handedness=handedness)
            try:
                engine = RuleEngine.from_yaml(rules_path)
            except FileNotFoundError as exc:
                _log.warning("Feedback rules not found: %s", exc)
            else:
                feedback = tuple(engine.evaluate(diff))

    return PipelineResult(
        pose=pose_seq,
        club=track,
        phases=phase_map,
        diff=diff,
        feedback=feedback,
    )


def render_report(result: PipelineResult) -> str:
    """Human-readable summary of the pipeline result."""
    lines = [
        f"SwingScan report: {result.frame_count} frames",
        f"  club: {result.club.missing_ratio:.0%} missing, "
        f"{result.club.interpolated_ratio:.0%} interpolated",
    ]
    if result.phases is not None:
        lines.append(
            "  phases: " + ", ".join(f"{ev.name}={idx}" for ev, idx in result.phases.events)
        )
    if result.diff is not None:
        lines.append(f"  cohort: {result.diff.cohort_size} pro swings")
    lines.append("")
    lines.append(render_text(list(result.feedback)))
    return "\n".join(lines)


def save_pipeline_result(result: PipelineResult, path: Path | str) -> Path:
    """Write a pipeline result to a JSON report.

    The report contains pose metadata (not the full per-frame keypoints —
    those belong in the parquet saved by ``swingscan pose``), the full
    club trajectory, and a tiny summary block. It is intended for Stage 2
    smoke inspection; later stages write richer artifacts.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
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
    if result.phases is not None:
        payload["phases"] = result.phases.as_dict()
    if result.feedback:
        payload["feedback"] = [item.as_dict() for item in result.feedback]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out

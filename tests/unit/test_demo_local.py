"""Unit tests for scripts/demo_local.py.

These tests exercise the non-Gradio parts of the demo (metrics table
formatting, argparse) without actually launching the Gradio server.
"""

from __future__ import annotations

from demo_local import (  # type: ignore[import-not-found]
    _build_parser,
    _format_feedback,
    _format_metrics_table,
)
from swingscan.club.tracker import ClubTrack
from swingscan.feedback.rules import FeedbackItem
from swingscan.phases.segmenter import HeuristicSegmenter
from swingscan.pipeline import PipelineResult
from swingscan.pose.base import PoseFrame, PoseSequence, empty_pose_array


def _empty_pose_seq(n: int = 8) -> PoseSequence:
    frames = tuple(
        PoseFrame(
            frame_index=i,
            timestamp_s=i / 30,
            image_keypoints=empty_pose_array(),
            world_keypoints=empty_pose_array(),
        )
        for i in range(n)
    )
    return PoseSequence(frames=frames, fps=30.0, width=160, height=160, duration_s=n / 30)


def _empty_result() -> PipelineResult:
    pose = _empty_pose_seq()
    phases = HeuristicSegmenter().segment(pose)
    return PipelineResult(
        pose=pose,
        club=ClubTrack(detections=()),
        phases=phases,
    )


def test_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.max_mb == 100.0
    assert args.max_duration_s == 30.0


def test_format_metrics_table_without_bank() -> None:
    rows = _format_metrics_table(_empty_result())
    # 8 events → 8 rows, each with 6 columns.
    assert len(rows) == 8
    assert all(len(r) == 6 for r in rows)


def test_format_feedback_empty() -> None:
    text = _format_feedback(_empty_result())
    assert "No feedback cues" in text


def test_format_feedback_with_items() -> None:
    result = PipelineResult(
        pose=_empty_pose_seq(),
        club=ClubTrack(detections=()),
        phases=HeuristicSegmenter().segment(_empty_pose_seq()),
        feedback=(
            FeedbackItem(
                rule="test",
                phase="TOP",
                metric="hip_rotation_deg",
                severity=3,
                message="hip stall",
                amateur=5.0,
                delta=-10.0,
                z_score=-2.0,
            ),
        ),
    )
    text = _format_feedback(result)
    assert "hip stall" in text
    assert "!!!" in text

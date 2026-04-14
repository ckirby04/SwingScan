"""Command-line interface entry point for SwingScan.

Subcommands:

    swingscan version                    Print the installed package version.
    swingscan pose   --input X --output Y  Run pose extraction on a video.
    swingscan run    --input X           End-to-end pipeline stub (Stage 6+).

Later stages add ``phases``, ``compare``, and ``demo`` subcommands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from swingscan import __version__
from swingscan.utils.logging import configure_logging

log = logging.getLogger("swingscan.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingscan",
        description="SwingScan — golf swing analysis pipeline.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=(
            "Override log level (DEBUG, INFO, WARNING, ERROR). Defaults to "
            "INFO or the SWINGSCAN_LOG_LEVEL environment variable."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    subparsers.add_parser(
        "version",
        help="Print the installed SwingScan version and exit.",
    )

    pose_p = subparsers.add_parser(
        "pose",
        help="Extract per-frame pose keypoints from a swing video.",
    )
    pose_p.add_argument("--input", required=True, help="Path to a swing video.")
    pose_p.add_argument(
        "--output",
        required=True,
        help="Output path for the pose sequence. Format chosen by extension (.parquet or .json).",
    )
    pose_p.add_argument(
        "--config",
        default=None,
        help="Optional path to a SwingScan YAML config. Defaults to configs/default.yaml if present.",
    )

    phases_p = subparsers.add_parser(
        "phases",
        help="Segment a swing video into the 8 canonical events.",
    )
    phases_p.add_argument("--input", required=True, help="Path to a swing video.")
    phases_p.add_argument("--output", required=True, help="Output JSON path for the phase map.")
    phases_p.add_argument("--config", default=None)

    run_p = subparsers.add_parser(
        "run",
        help="Run the pipeline on a single swing video (Stage 2: pose + club).",
    )
    run_p.add_argument("--input", required=True, help="Path to a swing video.")
    run_p.add_argument(
        "--output",
        default=None,
        help="Optional path to write a pipeline JSON report.",
    )
    run_p.add_argument(
        "--club-weights",
        default=None,
        help="Optional path to a YOLO club-head weights file. When absent, uses heuristic fallback.",
    )
    run_p.add_argument(
        "--output-video",
        default=None,
        help="Optional path to write an annotated output video (Stage 7+, no-op today).",
    )
    run_p.add_argument(
        "--config",
        default=None,
        help="Optional path to a SwingScan YAML config.",
    )

    return parser


def _cmd_version() -> int:
    sys.stdout.write(f"swingscan {__version__}\n")
    return 0


def _cmd_phases(args: argparse.Namespace) -> int:
    import json as _json

    from swingscan.config import load_config
    from swingscan.io.video import VideoReader
    from swingscan.phases.segmenter import HeuristicSegmenter
    from swingscan.pose.mediapipe_backend import MediaPipePoseEstimator

    cfg = load_config(args.config)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    log.info("Segmenting phases: %s -> %s", input_path, output_path)
    with (
        VideoReader(input_path) as video,
        MediaPipePoseEstimator(cfg.pose) as pose_est,
    ):
        pose_seq = pose_est.estimate_video(video)

    segmenter = HeuristicSegmenter()
    phase_map = segmenter.segment(pose_seq)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "swingscan_phases_v1",
        "source": str(input_path),
        "frame_count": len(pose_seq),
        "events": phase_map.as_dict(),
    }
    output_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    sys.stdout.write(f"Wrote phase map to {output_path}\n")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    # Deferred imports so `swingscan version` stays fast.
    from swingscan.config import load_config
    from swingscan.pipeline import run_pipeline, save_pipeline_result

    cfg = load_config(args.config)
    input_path = Path(args.input).expanduser().resolve()

    log.info("Running pipeline: %s", input_path)
    result = run_pipeline(
        video_path=input_path,
        config=cfg,
        club_weights=args.club_weights,
    )

    summary = (
        f"frames={result.frame_count} "
        f"pose_low_conf={result.pose.low_confidence_ratio():.1%} "
        f"club_missing={result.club.missing_ratio:.1%} "
        f"club_interp={result.club.interpolated_ratio:.1%}"
    )
    sys.stdout.write(f"Pipeline complete: {summary}\n")

    if args.output is not None:
        out_path = Path(args.output).expanduser().resolve()
        save_pipeline_result(result, out_path)
        sys.stdout.write(f"Wrote pipeline report to {out_path}\n")

    if args.output_video is not None:
        log.warning(
            "--output-video requested but Stage 7 (overlays) is not yet implemented; "
            "skipping."
        )

    return 0


def _cmd_pose(args: argparse.Namespace) -> int:
    # Imports are local so `swingscan version` stays fast and doesn't
    # eagerly load MediaPipe / OpenCV.
    from swingscan.config import load_config
    from swingscan.io.serialize import save_pose_sequence
    from swingscan.io.video import VideoReader
    from swingscan.pose.mediapipe_backend import MediaPipePoseEstimator

    cfg = load_config(args.config)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    log.info("Running pose extraction: %s -> %s", input_path, output_path)
    with VideoReader(input_path) as video:
        log.info(
            "Opened %s (%dx%d @ %.1f fps, %d frames, rot=%d°)",
            input_path.name,
            video.width,
            video.height,
            video.fps,
            video.frame_count,
            video.rotation_deg,
        )
        with MediaPipePoseEstimator(cfg.pose) as estimator:
            sequence = estimator.estimate_video(video)

    save_pose_sequence(sequence, output_path)
    low_pct = sequence.low_confidence_ratio() * 100
    log.info(
        "Wrote %d frames to %s (%.1f%% low-confidence).",
        len(sequence),
        output_path,
        low_pct,
    )
    sys.stdout.write(f"Wrote {len(sequence)} pose frames to {output_path}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "version":
        return _cmd_version()
    if args.command == "pose":
        return _cmd_pose(args)
    if args.command == "phases":
        return _cmd_phases(args)
    if args.command == "run":
        return _cmd_run(args)

    parser.error(f"Unknown command: {args.command}")
    return 2  # pragma: no cover — parser.error exits before returning.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

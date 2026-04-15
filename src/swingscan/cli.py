"""Command-line interface entry point for SwingScan.

Subcommands:

    swingscan version                    Print the installed package version.
    swingscan pose   --input X --output Y  Run pose extraction on a video.
    swingscan run    --input X           End-to-end pipeline stub (Stage 6+).

Later stages add ``phases``, ``compare``, and ``demo`` subcommands.
"""

from __future__ import annotations

import argparse
import contextlib
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
        "--pro-bank",
        default=None,
        help="Optional path to a pro bank parquet. Enables Stage 5/6 comparison + feedback.",
    )
    run_p.add_argument(
        "--rules",
        default=None,
        help="Optional path to a feedback rules YAML. Defaults to configs/feedback_rules.yaml.",
    )
    run_p.add_argument(
        "--swingnet-weights",
        default=None,
        help=(
            "Optional path to SwingNet weights (e.g. models/swingnet_1800.pth.tar). "
            "Auto-discovered if present. Without weights, the heuristic segmenter "
            "is used as the fallback."
        ),
    )
    run_p.add_argument(
        "--handedness",
        default="right",
        choices=["right", "left"],
        help="Golfer handedness. Defaults to right.",
    )
    run_p.add_argument(
        "--output-video",
        default=None,
        help="Optional path to write an annotated output video.",
    )
    run_p.add_argument(
        "--draw-club",
        action="store_true",
        help=(
            "Overlay the pose-derived club-head trail on the annotated video. "
            "Off by default because the current heuristic tracker is visually "
            "noisy; will become the default once a learned club detector ships."
        ),
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
    from swingscan.pipeline import render_report, run_pipeline, save_pipeline_result

    cfg = load_config(args.config)
    input_path = Path(args.input).expanduser().resolve()

    # Auto-discover SwingNet weights if the user didn't pass a path.
    swingnet_weights = args.swingnet_weights
    if swingnet_weights is None:
        from swingscan.phases.swingnet import default_swingnet_path

        candidate = default_swingnet_path()
        if candidate.is_file():
            swingnet_weights = str(candidate)
            log.info("Auto-discovered SwingNet weights at %s", candidate)

    log.info("Running pipeline: %s", input_path)
    result = run_pipeline(
        video_path=input_path,
        config=cfg,
        club_weights=args.club_weights,
        pro_bank_path=args.pro_bank,
        rules_path=args.rules,
        swingnet_weights=swingnet_weights,
        handedness=args.handedness,
    )

    sys.stdout.write(render_report(result))

    if args.output is not None:
        out_path = Path(args.output).expanduser().resolve()
        save_pipeline_result(result, out_path)
        sys.stdout.write(f"Wrote pipeline report to {out_path}\n")

    if args.output_video is not None:
        from swingscan.viz.overlay import render_annotated_video

        video_out = Path(args.output_video).expanduser().resolve()
        render_annotated_video(
            source_video=input_path,
            pose=result.pose,
            club=result.club,
            phases=result.phases,
            out_path=video_out,
            draw_club=args.draw_club,
        )
        sys.stdout.write(f"Wrote annotated video to {video_out}\n")

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


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 so non-ASCII feedback renders.

    On Windows PowerShell the default console encoding is cp1252, which
    cannot encode characters used in our feedback templates (em dash,
    middle dot, degree symbol). Reconfiguring to UTF-8 fixes the
    rendering without requiring the user to run ``chcp 65001``.

    Python 3.7+ exposes ``sys.stdout.reconfigure``. The attribute is
    absent on a handful of embedded Python builds, so we defensively
    swallow that case.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    _force_utf8_stdio()
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

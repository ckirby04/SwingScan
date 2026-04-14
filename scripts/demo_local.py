"""Gradio demo — local-only, no persistence, no analytics.

Implements ``CLAUDE.md`` §5.8. Drop in a swing video, receive back an
annotated video, a metrics table, and a list of heuristic coaching cues.

**Scope guardrails, enforced here:**

* Upload size capped (default 100 MB) and duration capped (default 30s).
* Rejected uploads return a clear error string — no partial processing.
* No authentication and no uploads leave the local machine. The app
  binds to ``127.0.0.1`` unless explicitly overridden with ``--host``.
* No analytics. The Gradio ``analytics_enabled`` flag is disabled.
* Uploads are not persisted beyond the single request — processed
  files live under ``/tmp`` (or the system temp dir).

Usage::

    python scripts/demo_local.py                              # defaults
    python scripts/demo_local.py --pro-bank data/pro_bank/bank.parquet
    python scripts/demo_local.py --host 0.0.0.0 --port 7860   # bind all
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from swingscan.pipeline import PipelineResult, run_pipeline
from swingscan.utils.logging import configure_logging

_log = logging.getLogger(__name__)

_MAX_MB: float = 100.0
_MAX_DURATION_S: float = 30.0


def _format_metrics_table(result: PipelineResult) -> list[list[Any]]:
    """Produce a Gradio `gr.Dataframe`-compatible 2D list."""
    rows: list[list[Any]] = []
    if result.diff is None:
        # No pro bank supplied → show raw amateur metrics without cohort
        # deltas. We reconstruct them from the SwingMetrics for the
        # amateur directly.
        from swingscan.metrics.biomech import compute_swing_metrics

        if result.phases is not None:
            metrics = compute_swing_metrics(result.pose, result.phases)
            for phase in metrics.phases:
                rows.append(
                    [
                        phase.event,
                        f"{phase.hip_rotation_deg:+.1f}°",
                        f"{phase.shoulder_rotation_deg:+.1f}°",
                        f"{phase.x_factor_deg:+.1f}°",
                        f"{phase.lead_arm_angle_deg:.1f}°",
                        f"{phase.spine_lean_deg:+.1f}°",
                    ]
                )
        return rows

    for phase in result.diff.phases:
        row = [phase.event]
        for metric_name in (
            "hip_rotation_deg",
            "shoulder_rotation_deg",
            "x_factor_deg",
            "lead_arm_angle_deg",
            "spine_lean_deg",
        ):
            md = phase.by_metric(metric_name)
            if md is None or md.delta is None:
                row.append(f"{md.amateur:+.1f}" if md is not None else "-")
            else:
                row.append(f"{md.amateur:+.1f} (Δ{md.delta:+.1f})")
        rows.append(row)
    return rows


def _format_feedback(result: PipelineResult) -> str:
    if not result.feedback:
        return (
            "No feedback cues fired. Supply `--pro-bank` to enable "
            "cohort comparison; without a pro bank, the app can only "
            "show raw metrics."
        )
    lines: list[str] = []
    for item in result.feedback[:5]:
        marker = "!" * item.severity
        lines.append(f"[{marker}] {item.phase}: {item.message}")
    lines.append("")
    lines.append("Heuristic coaching cues only — not medical or clinical advice.")
    return "\n".join(lines)


def _build_app(
    pro_bank_path: Path | None,
    rules_path: Path | None,
    max_mb: float,
    max_duration_s: float,
) -> Any:
    import gradio as gr

    def process(video_path: str | None) -> tuple[str | None, list[list[Any]], str]:
        if video_path is None:
            return None, [], "Please upload a swing video."

        src = Path(video_path)
        size_mb = src.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            return None, [], f"Upload too large ({size_mb:.1f} MB > {max_mb:.0f} MB cap)."

        # Quick duration probe via cv2, before we spin up MediaPipe.
        import cv2

        cap = cv2.VideoCapture(str(src))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frame_count / fps if fps > 0 else 0.0
        if duration > max_duration_s:
            return None, [], (
                f"Upload too long ({duration:.1f}s > {max_duration_s:.0f}s cap)."
            )

        with tempfile.TemporaryDirectory() as workdir:
            annotated = Path(workdir) / "annotated.mp4"
            result = run_pipeline(
                video_path=src,
                pro_bank_path=pro_bank_path,
                rules_path=rules_path,
            )

            from swingscan.viz.overlay import render_annotated_video

            render_annotated_video(
                source_video=src,
                pose=result.pose,
                club=result.club,
                phases=result.phases,
                out_path=annotated,
            )

            # Gradio needs a persistent file path outside the tempdir
            # (otherwise the caller can't read it once we return).
            persistent = Path(tempfile.gettempdir()) / f"swingscan_{src.stem}_annotated.mp4"
            persistent.write_bytes(annotated.read_bytes())

        return (
            str(persistent),
            _format_metrics_table(result),
            _format_feedback(result),
        )

    with gr.Blocks(analytics_enabled=False, title="SwingScan local demo") as app:
        gr.Markdown("# SwingScan")
        gr.Markdown(
            f"Drop a golf swing video (≤ {int(max_mb)} MB, "
            f"≤ {int(max_duration_s)} s) to see pose keypoints, phase "
            "segmentation, and heuristic coaching cues. Local-only; "
            "uploads are not persisted."
        )
        with gr.Row():
            with gr.Column():
                upload = gr.Video(label="Swing video", sources=["upload"])
                submit = gr.Button("Analyze", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Annotated output")
                metrics = gr.Dataframe(
                    headers=[
                        "Phase",
                        "Hip rot",
                        "Shoulder rot",
                        "X-factor",
                        "Lead arm",
                        "Spine lean",
                    ],
                    label="Per-phase metrics",
                )
                feedback = gr.Textbox(label="Coaching cues", lines=8)

        submit.click(process, inputs=upload, outputs=[output_video, metrics, feedback])

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro-bank", default=None, help="Path to pro bank parquet.")
    parser.add_argument("--rules", default=None, help="Path to feedback rules YAML.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-mb", type=float, default=_MAX_MB)
    parser.add_argument("--max-duration-s", type=float, default=_MAX_DURATION_S)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    pro_bank_path = Path(args.pro_bank).expanduser().resolve() if args.pro_bank else None
    rules_path = Path(args.rules).expanduser().resolve() if args.rules else None

    app = _build_app(
        pro_bank_path=pro_bank_path,
        rules_path=rules_path,
        max_mb=args.max_mb,
        max_duration_s=args.max_duration_s,
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_api=False,
        quiet=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

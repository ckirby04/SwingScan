"""Evaluation harness — measure pipeline quality on a labeled test set.

Computes three metrics over a labels JSON (same format as
:mod:`scripts.build_pro_bank`):

1. **Percentage of Correctly detected Events (PCE)** — the GolfDB
   metric. For each labeled swing, how many of the 8 event frames did
   the heuristic segmenter place within a tolerance (default 5 frames)?
2. **Pose completeness** — fraction of frames in the test set where the
   median core-joint visibility is above the config threshold.
3. **Pro bank coverage** — fraction of labels for which the pro bank
   contains at least one matching (view, handedness, club) swing.

Output is a JSON report. On an empty labels set (or a dataset with no
reachable videos) the script still succeeds and writes a report with
zero-count fields and a clear "no_data" marker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from swingscan.compare.pro_bank import ProBank, SwingLabel
from swingscan.config import load_config
from swingscan.io.video import VideoReader
from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import HeuristicSegmenter, PhaseSegmenter
from swingscan.pose.mediapipe_backend import MediaPipePoseEstimator
from swingscan.utils.logging import configure_logging
from swingscan.utils.paths import data_dir

_log = logging.getLogger(__name__)


@dataclass
class PerSwingResult:
    swing_id: str
    total_frames: int
    events_correct: int
    pose_complete_ratio: float
    bank_match: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=None, help="Path to labels JSON.")
    parser.add_argument("--bank", default=None, help="Path to pro bank parquet.")
    parser.add_argument(
        "--output",
        default=None,
        help="Report path. Defaults to data/processed/evaluation.json.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Frames of tolerance for PCE scoring.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap number of swings evaluated."
    )
    parser.add_argument(
        "--segmenter",
        choices=["heuristic", "swingnet", "auto"],
        default="auto",
        help=(
            "Which phase segmenter to evaluate. 'auto' uses SwingNet if "
            "models/swingnet_1800.pth.tar is present, otherwise heuristic."
        ),
    )
    parser.add_argument(
        "--swingnet-weights",
        default=None,
        help="Override path to SwingNet weights.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _build_segmenter(
    choice: str, swingnet_weights: str | None
) -> tuple[PhaseSegmenter, str]:
    """Pick the segmenter per --segmenter / auto-discovery."""
    if choice == "heuristic":
        return HeuristicSegmenter(), "heuristic"

    if choice in ("swingnet", "auto"):
        from swingscan.phases.swingnet import SwingNetSegmenter, default_swingnet_path

        path = Path(swingnet_weights).expanduser().resolve() if swingnet_weights else default_swingnet_path()
        if path.is_file():
            return SwingNetSegmenter(weights_path=path), "swingnet"
        if choice == "swingnet":
            raise FileNotFoundError(f"SwingNet weights not found: {path}")
        _log.warning("SwingNet weights not found at %s; falling back to heuristic.", path)

    return HeuristicSegmenter(), "heuristic"


def _score_swing(
    label: SwingLabel,
    bank: ProBank | None,
    tolerance: int,
    segmenter: PhaseSegmenter,
) -> PerSwingResult | None:
    video_path = Path(label.video_path).expanduser().resolve()
    if not video_path.is_file():
        _log.warning("Missing video for %s: %s", label.swing_id, video_path)
        return None

    cfg = load_config()
    with VideoReader(video_path) as video, MediaPipePoseEstimator(cfg.pose) as est:
        pose_seq = est.estimate_video(video)

    if len(pose_seq) == 0:
        return None

    predicted = segmenter.segment(pose_seq)
    predicted_by_name = predicted.as_dict()

    correct = 0
    events = list(SwingEvent)
    for ev, truth in zip(events, label.event_frames, strict=False):
        pred = predicted_by_name.get(ev.name)
        if pred is None:
            continue
        if abs(pred - int(truth)) <= tolerance:
            correct += 1

    pose_complete = sum(1 for f in pose_seq if not f.is_low_confidence) / len(pose_seq)

    bank_match = False
    if bank is not None:
        matches = bank.filter(
            view=label.view,
            handedness=label.handedness,
            club=label.club,
        )
        bank_match = len(matches) > 0

    return PerSwingResult(
        swing_id=label.swing_id,
        total_frames=len(pose_seq),
        events_correct=correct,
        pose_complete_ratio=pose_complete,
        bank_match=bank_match,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    labels_path = (
        Path(args.labels).expanduser().resolve()
        if args.labels
        else data_dir() / "raw" / "labels.json"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else data_dir() / "processed" / "evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bank: ProBank | None = None
    if args.bank is not None:
        bank_path = Path(args.bank).expanduser().resolve()
        try:
            bank = ProBank.load(bank_path)
        except FileNotFoundError:
            _log.warning("Pro bank not found at %s; coverage will be 0.", bank_path)

    if not labels_path.is_file():
        output_path.write_text(
            json.dumps(
                {
                    "format": "swingscan_evaluation_v1",
                    "labels_path": str(labels_path),
                    "marker": "no_data",
                    "reason": "labels file not found",
                    "pce": None,
                    "pose_complete_ratio": None,
                    "bank_coverage": None,
                    "swings_total": 0,
                    "swings_scored": 0,
                    "tolerance": args.tolerance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote empty evaluation report to {output_path}")
        return 0

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    raw_labels = payload.get("swings", [])
    if args.limit is not None:
        raw_labels = raw_labels[: args.limit]
    labels = [SwingLabel.model_validate(row) for row in raw_labels]

    segmenter, segmenter_label = _build_segmenter(args.segmenter, args.swingnet_weights)
    _log.info("Using %s segmenter", segmenter_label)

    results: list[PerSwingResult] = []
    for label in labels:
        r = _score_swing(label, bank, args.tolerance, segmenter)
        if r is not None:
            results.append(r)

    total_events = len(results) * 8
    correct_events = sum(r.events_correct for r in results)
    pce = correct_events / total_events if total_events else 0.0
    pose_complete_avg = (
        sum(r.pose_complete_ratio for r in results) / len(results) if results else 0.0
    )
    bank_coverage = (
        sum(1 for r in results if r.bank_match) / len(results) if results else 0.0
    )

    report = {
        "format": "swingscan_evaluation_v1",
        "labels_path": str(labels_path),
        "segmenter": segmenter_label,
        "tolerance": args.tolerance,
        "swings_total": len(labels),
        "swings_scored": len(results),
        "pce": pce,
        "pose_complete_ratio": pose_complete_avg,
        "bank_coverage": bank_coverage,
        "per_swing": [
            {
                "swing_id": r.swing_id,
                "total_frames": r.total_frames,
                "events_correct": r.events_correct,
                "pose_complete_ratio": r.pose_complete_ratio,
                "bank_match": r.bank_match,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Wrote evaluation report to {output_path} "
        f"(pce={pce:.2%}, pose_complete={pose_complete_avg:.2%})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

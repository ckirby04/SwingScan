"""Calibrate feedback rule thresholds against a real pro bank.

Reads a ``ProBank`` parquet, computes per-event SwingMetrics for every
pro swing, measures the distribution (median + stdev, circular-aware
for angular metrics) of each metric at each event, and either prints a
summary table or overwrites the ``threshold`` field on each matching
rule in ``configs/feedback_rules.yaml``.

This is intended as a one-off calibration tool, run whenever the pro
bank changes meaningfully. The updated YAML is still hand-reviewed
before commit — the script prints a diff of proposed threshold
changes and requires ``--apply`` to write them to disk.

Usage::

    python scripts/calibrate_thresholds.py            # dry-run, prints table
    python scripts/calibrate_thresholds.py --apply    # rewrites YAML in place

``--sigma`` (default 1.5) controls how many standard deviations from
the cohort median a metric must be before a rule fires. Smaller values
= more rules fire = noisier feedback. 1.5 is a reasonable starting
point; tune after watching real feedback on your own swings.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import yaml

from swingscan.compare.diff import _circular_mean_deg, _circular_stdev_deg
from swingscan.compare.pro_bank import ProBank
from swingscan.metrics.biomech import compute_swing_metrics
from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import PhaseMap
from swingscan.pose.base import PoseSequence
from swingscan.utils.logging import configure_logging
from swingscan.utils.paths import configs_dir, data_dir

_log = logging.getLogger(__name__)

_ANGULAR = {
    "hip_rotation_deg",
    "shoulder_rotation_deg",
    "x_factor_deg",
    "spine_lean_deg",
    "lead_arm_angle_deg",
    "wrist_hinge_deg",
    "lead_knee_flex_deg",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        default=None,
        help="Path to pro bank parquet. Defaults to data/pro_bank/bank.parquet.",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Path to feedback rules YAML. Defaults to configs/feedback_rules.yaml.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Threshold = sigma * stdev. Default 1.5.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the rules YAML with the calibrated thresholds. "
        "Without this flag the script is a dry-run that only prints.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _distribution(
    bank: ProBank,
    event: SwingEvent,
    metric: str,
) -> tuple[float, float, int] | None:
    """Return (center, stdev, n) for a metric at a given event. None if empty."""
    values: list[float] = []
    for swing in bank:
        events = [(SwingEvent[name], i) for i, name in enumerate(swing.event_names)]
        pm = PhaseMap(events=tuple(events))
        pseudo = PoseSequence(
            frames=swing.event_poses,
            fps=30.0,
            width=1,
            height=1,
            duration_s=len(swing.event_poses) / 30.0,
            source_path=f"probank:{swing.swing_id}",
        )
        metrics = compute_swing_metrics(pseudo, pm, handedness=swing.handedness)
        phase = metrics.by_event(event)
        if phase is None:
            continue
        val = getattr(phase, metric)
        if isinstance(val, int | float) and not math.isnan(float(val)):
            values.append(float(val))

    if len(values) < 2:
        return None

    if metric in _ANGULAR:
        center = _circular_mean_deg(values)
        stdev = _circular_stdev_deg(values, center)
    else:
        center = statistics.median(values)
        stdev = statistics.stdev(values)

    return center, stdev, len(values)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    bank_path = (
        Path(args.bank).expanduser().resolve()
        if args.bank
        else data_dir() / "pro_bank" / "bank.parquet"
    )
    rules_path = (
        Path(args.rules).expanduser().resolve()
        if args.rules
        else configs_dir() / "feedback_rules.yaml"
    )

    if not bank_path.is_file():
        _log.error("Pro bank not found: %s", bank_path)
        return 1
    if not rules_path.is_file():
        _log.error("Feedback rules YAML not found: %s", rules_path)
        return 1

    bank = ProBank.load(bank_path)
    _log.info("Loaded %d swings from %s", len(bank), bank_path)

    raw = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    rules = raw.get("rules") or []

    header = f"{'rule':35} {'event':20} {'metric':24} {'center':>9} {'stdev':>7} {'thr_old':>9} {'thr_new':>9}"
    print(header)
    print("-" * len(header))

    changes = 0
    for rule in rules:
        try:
            event = SwingEvent[rule["phase"]]
        except KeyError:
            continue
        metric = rule["metric"]
        direction = rule["direction"]

        dist = _distribution(bank, event, metric)
        if dist is None:
            continue
        center, stdev, _n = dist

        new_threshold = args.sigma * stdev if direction == "above" else -args.sigma * stdev
        old_threshold = float(rule.get("threshold", 0.0))
        if abs(new_threshold - old_threshold) > 1e-3:
            changes += 1
        rule["threshold"] = round(new_threshold, 2)

        print(
            f"{rule['name']:35} {event.name:20} {metric:24} "
            f"{center:>9.2f} {stdev:>7.2f} {old_threshold:>9.2f} {new_threshold:>9.2f}"
        )

    print()
    print(f"Rules with changed thresholds: {changes}")

    if args.apply:
        rules_path.write_text(
            yaml.safe_dump({"rules": rules}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"Wrote calibrated thresholds to {rules_path}")
    else:
        print("Dry-run only. Pass --apply to write the rules YAML.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

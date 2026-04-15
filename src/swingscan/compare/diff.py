"""Per-phase swing comparison against a :class:`ProBank` cohort.

Consumes a :class:`SwingMetrics` for the amateur's swing plus a
:class:`ProBank` (filtered by view/handedness/club) and produces a
:class:`SwingDiff` that the feedback rule engine reads.

For each phase and each metric, the diff records:

* the measured amateur value,
* the cohort median,
* the delta (amateur - cohort median),
* a z-score against the cohort distribution (clamped to +/-10 for
  numerical stability).

When the cohort is empty or has fewer than ``MIN_COHORT`` samples, the
z-score is ``None``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from swingscan.compare.pro_bank import ProBank, ProSwing
from swingscan.metrics.biomech import SwingMetrics, compute_swing_metrics
from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import HeuristicSegmenter, PhaseMap
from swingscan.pose.base import PoseSequence

__all__ = ["MetricDelta", "PhaseDiff", "SwingDiff", "compare_against_bank"]

MIN_COHORT = 2
_METRIC_NAMES: tuple[str, ...] = (
    "hip_rotation_deg",
    "shoulder_rotation_deg",
    "x_factor_deg",
    "spine_lean_deg",
    "lead_arm_angle_deg",
    "wrist_hinge_deg",
    "lead_knee_flex_deg",
    "head_movement",
)

# Metrics whose values are angles in degrees and therefore live on a
# circle. Comparing them naively breaks at the +/-180 boundary: two
# swings at +175 and -175 are structurally identical but their
# arithmetic mean is 0. For these metrics we use a circular mean for
# cohort central tendency and wrap deltas to [-180, 180].
_ANGULAR_METRICS: frozenset[str] = frozenset(
    {
        "hip_rotation_deg",
        "shoulder_rotation_deg",
        "x_factor_deg",
        "spine_lean_deg",
        "lead_arm_angle_deg",
        "wrist_hinge_deg",
        "lead_knee_flex_deg",
    }
)


def _circular_mean_deg(values: list[float]) -> float:
    """Circular mean of angles in degrees. Output is in [-180, 180]."""
    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(sin_sum, cos_sum))


def _wrap_signed_deg(x: float) -> float:
    """Wrap an angle delta to [-180, 180]."""
    return ((x + 180.0) % 360.0) - 180.0


def _circular_stdev_deg(values: list[float], center_deg: float) -> float:
    """Population-style stdev of angular samples around a given center.

    Computed after wrapping each deviation into [-180, 180] so samples
    straddling the +/-180 boundary no longer blow the estimate up.
    """
    deviations = [_wrap_signed_deg(v - center_deg) for v in values]
    if len(deviations) < 2:
        return 0.0
    return statistics.stdev(deviations)


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric: str
    amateur: float
    cohort_median: float | None
    delta: float | None
    z_score: float | None


@dataclass(frozen=True, slots=True)
class PhaseDiff:
    event: str
    metrics: tuple[MetricDelta, ...]

    def by_metric(self, name: str) -> MetricDelta | None:
        for m in self.metrics:
            if m.metric == name:
                return m
        return None


@dataclass(frozen=True, slots=True)
class SwingDiff:
    phases: tuple[PhaseDiff, ...]
    cohort_size: int
    metadata: dict[str, str] = field(default_factory=dict)

    def by_event(self, event: SwingEvent) -> PhaseDiff | None:
        for p in self.phases:
            if p.event == event.name:
                return p
        return None

    def significant_items(self, z_threshold: float = 1.5) -> list[tuple[str, MetricDelta]]:
        """Return a list of (event_name, MetricDelta) for all metrics whose
        absolute z-score exceeds ``z_threshold``.
        """
        out: list[tuple[str, MetricDelta]] = []
        for phase in self.phases:
            for m in phase.metrics:
                if m.z_score is not None and abs(m.z_score) >= z_threshold:
                    out.append((phase.event, m))
        return out


def _cohort_metric_values(
    swings: list[ProSwing],
    event: SwingEvent,
    metric: str,
    reference_metrics_by_swing: dict[str, SwingMetrics],
) -> list[float]:
    values: list[float] = []
    for swing in swings:
        sm = reference_metrics_by_swing.get(swing.swing_id)
        if sm is None:
            continue
        phase = sm.by_event(event)
        if phase is None:
            continue
        val = getattr(phase, metric)
        if isinstance(val, int | float) and not math.isnan(val):
            values.append(float(val))
    return values


def _compute_probank_metrics(bank: ProBank) -> dict[str, SwingMetrics]:
    """Compute :class:`SwingMetrics` for each swing in a bank.

    The bank stores per-event snapshots (not full sequences), so we
    reconstruct a pseudo-:class:`PoseSequence` containing just those
    snapshots and a :class:`PhaseMap` whose frame indices match.
    """
    out: dict[str, SwingMetrics] = {}
    for swing in bank:
        events = [(SwingEvent[name], i) for i, name in enumerate(swing.event_names)]
        pm = PhaseMap(events=tuple(events))
        # Minimal PoseSequence: 8 frames, fps=30, dimensions from
        # whatever metadata we have (kept abstract because these are
        # bank snapshots, not real videos).
        pseudo = PoseSequence(
            frames=swing.event_poses,
            fps=30.0,
            width=1,
            height=1,
            duration_s=len(swing.event_poses) / 30.0,
            source_path=f"probank:{swing.swing_id}",
        )
        out[swing.swing_id] = compute_swing_metrics(pseudo, pm, handedness=swing.handedness)
    return out


def compare_against_bank(
    amateur_pose: PoseSequence,
    amateur_phases: PhaseMap | None,
    bank: ProBank,
    handedness: str = "right",
) -> SwingDiff:
    """Compute a :class:`SwingDiff` for ``amateur_pose`` against ``bank``.

    If ``amateur_phases`` is ``None``, the default
    :class:`HeuristicSegmenter` is used.
    """
    pmap = amateur_phases or HeuristicSegmenter().segment(amateur_pose)
    amateur_metrics = compute_swing_metrics(amateur_pose, pmap, handedness=handedness)

    bank_metrics = _compute_probank_metrics(bank)
    swings = list(bank)

    phases: list[PhaseDiff] = []
    for event, _ in pmap.events:
        amateur_phase = amateur_metrics.by_event(event)
        if amateur_phase is None:
            continue

        metric_deltas: list[MetricDelta] = []
        for metric in _METRIC_NAMES:
            amateur_val = float(getattr(amateur_phase, metric))
            cohort = _cohort_metric_values(swings, event, metric, bank_metrics)
            if len(cohort) < MIN_COHORT:
                metric_deltas.append(
                    MetricDelta(
                        metric=metric,
                        amateur=amateur_val,
                        cohort_median=None,
                        delta=None,
                        z_score=None,
                    )
                )
                continue
            if metric in _ANGULAR_METRICS:
                # Circular mean is the stable "central tendency" for an
                # angular distribution. Wrap the delta into [-180, 180]
                # so wraparound noise (e.g. cohort at +175 vs amateur at
                # -175) never appears as a spurious 350 degree gap.
                median = _circular_mean_deg(cohort)
                delta = _wrap_signed_deg(amateur_val - median)
                stdev = _circular_stdev_deg(cohort, median)
            else:
                median = statistics.median(cohort)
                try:
                    stdev = statistics.stdev(cohort)
                except statistics.StatisticsError:
                    stdev = 0.0
                delta = amateur_val - median
            z = 0.0 if stdev < 1e-6 else delta / stdev
            z = max(-10.0, min(10.0, z))
            metric_deltas.append(
                MetricDelta(
                    metric=metric,
                    amateur=amateur_val,
                    cohort_median=median,
                    delta=delta,
                    z_score=z,
                )
            )
        phases.append(PhaseDiff(event=event.name, metrics=tuple(metric_deltas)))

    return SwingDiff(
        phases=tuple(phases),
        cohort_size=len(swings),
        metadata={"handedness": handedness},
    )

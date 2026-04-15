"""Composite biomechanical metrics computed across phases.

These are the per-swing numbers that the feedback rule engine and
the evaluation harness read. :func:`compute_swing_metrics` takes a
``PoseSequence`` plus a ``PhaseMap`` and returns :class:`SwingMetrics`
— per-event :class:`PhaseMetrics` plus a couple of whole-swing
summaries (backswing tempo, total head drift).

Everything here works in image-normalized 2D keypoints. A proper
3D-lifted biomech module is V2 roadmap work.
"""

from __future__ import annotations

from dataclasses import dataclass

from swingscan.metrics import angles
from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import PhaseMap
from swingscan.pose.base import PoseFrame, PoseSequence

__all__ = ["PhaseMetrics", "SwingMetrics", "compute_swing_metrics"]


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    """Per-event metrics for a single swing."""

    event: str
    hip_rotation_deg: float
    shoulder_rotation_deg: float
    x_factor_deg: float
    spine_lean_deg: float
    lead_arm_angle_deg: float
    wrist_hinge_deg: float
    lead_knee_flex_deg: float
    head_movement: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "event": self.event,
            "hip_rotation_deg": self.hip_rotation_deg,
            "shoulder_rotation_deg": self.shoulder_rotation_deg,
            "x_factor_deg": self.x_factor_deg,
            "spine_lean_deg": self.spine_lean_deg,
            "lead_arm_angle_deg": self.lead_arm_angle_deg,
            "wrist_hinge_deg": self.wrist_hinge_deg,
            "lead_knee_flex_deg": self.lead_knee_flex_deg,
            "head_movement": self.head_movement,
        }


@dataclass(frozen=True, slots=True)
class SwingMetrics:
    """Full-swing summary: per-event metrics + a few whole-swing numbers."""

    phases: tuple[PhaseMetrics, ...]
    backswing_tempo: float
    total_head_drift: float

    def by_event(self, event: SwingEvent) -> PhaseMetrics | None:
        for p in self.phases:
            if p.event == event.name:
                return p
        return None


def _metrics_for_frame(
    frame: PoseFrame,
    reference: PoseFrame,
    event_name: str,
    handedness: str,
) -> PhaseMetrics:
    return PhaseMetrics(
        event=event_name,
        hip_rotation_deg=angles.hip_rotation_deg(frame, reference),
        shoulder_rotation_deg=angles.shoulder_rotation_deg(frame, reference),
        x_factor_deg=angles.x_factor_deg(frame, reference),
        spine_lean_deg=angles.spine_lean_deg(frame),
        lead_arm_angle_deg=angles.lead_arm_straightness_deg(frame, handedness),
        wrist_hinge_deg=angles.wrist_hinge_deg(frame),
        lead_knee_flex_deg=angles.knee_flex_deg(
            frame, "left" if handedness == "right" else "right"
        ),
        head_movement=angles.head_movement_px(frame, reference),
    )


def compute_swing_metrics(
    pose: PoseSequence,
    phase_map: PhaseMap,
    handedness: str = "right",
) -> SwingMetrics:
    """Compute :class:`SwingMetrics` for a single swing.

    Uses the pose frame at the ``ADDRESS`` event as the reference for
    rotation and head-drift deltas.
    """
    address_idx = phase_map.frame_for(SwingEvent.ADDRESS)
    reference = pose.frames[address_idx]

    phases: list[PhaseMetrics] = []
    for event, frame_idx in phase_map.events:
        if frame_idx >= len(pose):
            continue
        frame = pose.frames[frame_idx]
        phases.append(_metrics_for_frame(frame, reference, event.name, handedness))

    # Composite: backswing tempo = backswing frames / downswing frames.
    top = phase_map.frame_for(SwingEvent.TOP)
    impact = phase_map.frame_for(SwingEvent.IMPACT)
    back_frames = max(1, top - address_idx)
    down_frames = max(1, impact - top)
    tempo = back_frames / down_frames

    # Total head drift: max head displacement from address across all phases.
    total_head_drift = max((p.head_movement for p in phases), default=0.0)

    return SwingMetrics(
        phases=tuple(phases),
        backswing_tempo=tempo,
        total_head_drift=total_head_drift,
    )

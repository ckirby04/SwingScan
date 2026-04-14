"""Tests for metrics.biomech and compare.diff."""

from __future__ import annotations

import math

from swingscan.compare.diff import compare_against_bank
from swingscan.compare.pro_bank import ProBank, ProSwing
from swingscan.metrics.biomech import compute_swing_metrics
from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import HeuristicSegmenter
from swingscan.pose.base import PoseFrame, PoseSequence, empty_pose_array
from swingscan.pose.keypoints import Joint


def _synthetic_pose_sequence(n_frames: int = 30) -> PoseSequence:
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        kp = empty_pose_array()
        # Shoulders, hips, wrists with plausible positions.
        kp[Joint.LEFT_SHOULDER.value] = [0.40, 0.30, 0, 1.0]
        kp[Joint.RIGHT_SHOULDER.value] = [0.60, 0.30, 0, 1.0]
        kp[Joint.LEFT_HIP.value] = [0.45, 0.55, 0, 1.0]
        kp[Joint.RIGHT_HIP.value] = [0.55, 0.55, 0, 1.0]
        y = 0.45 + 0.25 * math.cos(2 * math.pi * t)
        kp[Joint.LEFT_WRIST.value] = [0.48, y, 0, 1.0]
        kp[Joint.RIGHT_WRIST.value] = [0.52, y, 0, 1.0]
        kp[Joint.LEFT_ELBOW.value] = [0.44, (y + 0.30) / 2, 0, 1.0]
        kp[Joint.RIGHT_ELBOW.value] = [0.56, (y + 0.30) / 2, 0, 1.0]
        kp[Joint.LEFT_KNEE.value] = [0.46, 0.75, 0, 1.0]
        kp[Joint.RIGHT_KNEE.value] = [0.54, 0.75, 0, 1.0]
        kp[Joint.LEFT_ANKLE.value] = [0.46, 0.95, 0, 1.0]
        kp[Joint.RIGHT_ANKLE.value] = [0.54, 0.95, 0, 1.0]
        kp[Joint.NOSE.value] = [0.50, 0.20, 0, 1.0]
        kp[Joint.LEFT_INDEX.value] = [0.47, y + 0.02, 0, 1.0]
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp_s=i / 30.0,
                image_keypoints=kp,
                world_keypoints=empty_pose_array(),
            )
        )
    return PoseSequence(
        frames=tuple(frames), fps=30.0, width=160, height=160, duration_s=n_frames / 30
    )


def test_compute_swing_metrics_returns_eight_phases() -> None:
    pose = _synthetic_pose_sequence()
    pmap = HeuristicSegmenter().segment(pose)
    metrics = compute_swing_metrics(pose, pmap)

    assert len(metrics.phases) == 8
    events = {p.event for p in metrics.phases}
    assert events == {e.name for e in SwingEvent}
    # Backswing tempo is a positive float.
    assert metrics.backswing_tempo > 0


def test_swing_diff_against_empty_bank_has_null_deltas() -> None:
    pose = _synthetic_pose_sequence()
    empty_bank = ProBank.from_rows([])
    diff = compare_against_bank(pose, None, empty_bank)
    assert diff.cohort_size == 0
    for phase in diff.phases:
        for metric in phase.metrics:
            assert metric.cohort_median is None
            assert metric.z_score is None


def _pro_swing_from(pose: PoseSequence, swing_id: str) -> ProSwing:
    pmap = HeuristicSegmenter().segment(pose)
    events = tuple(e.name for e in SwingEvent)
    frames = tuple(pmap.frame_for(e) for e in SwingEvent)
    poses = tuple(pose.frames[f] for f in frames)
    return ProSwing(
        swing_id=swing_id,
        view="face_on",
        handedness="right",
        club="driver",
        event_names=events,
        event_frames=frames,
        event_poses=poses,
    )


def test_swing_diff_against_populated_bank_produces_z_scores() -> None:
    base_pose = _synthetic_pose_sequence()
    bank = ProBank.from_rows(
        [_pro_swing_from(base_pose, f"swing_{i}") for i in range(3)]
    )
    diff = compare_against_bank(base_pose, None, bank)
    assert diff.cohort_size == 3

    # Since the "amateur" is identical to every pro in the bank, deltas
    # should be zero (or very close) and z-scores should be zero.
    for phase in diff.phases:
        for m in phase.metrics:
            if m.z_score is not None:
                assert abs(m.z_score) < 1e-6

"""Tests for :class:`swingscan.phases.segmenter.HeuristicSegmenter`."""

from __future__ import annotations

import math

import numpy as np

from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import HeuristicSegmenter
from swingscan.pose.base import PoseFrame, PoseSequence, empty_pose_array
from swingscan.pose.keypoints import Joint


def _synthetic_swing(n_frames: int = 60) -> PoseSequence:
    """Build a synthetic 60-frame swing with a plausible wrist trajectory.

    Wrists travel from address (middle-low) → top (middle-high) → back
    down through impact (peak speed) → finish. The trajectory is a
    cosine curve in y-space with an added spike in x-velocity near the
    impact frame.
    """
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        # y: 0.70 at address, 0.20 at top (mid), 0.70 at finish.
        y = 0.45 + 0.25 * math.cos(2 * math.pi * t)
        x = 0.5 + 0.05 * math.sin(4 * math.pi * t)

        kp = empty_pose_array()
        for joint in (Joint.LEFT_WRIST, Joint.RIGHT_WRIST):
            kp[joint.value, 0] = x
            kp[joint.value, 1] = y
            kp[joint.value, 3] = 0.95
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


def test_segmenter_returns_eight_events_in_order() -> None:
    pose = _synthetic_swing()
    pmap = HeuristicSegmenter().segment(pose)

    assert len(pmap.events) == 8
    events = [ev for ev, _ in pmap.events]
    assert events == list(SwingEvent)
    assert pmap.is_monotonic()


def test_segmenter_finds_top_near_y_min() -> None:
    pose = _synthetic_swing()
    pmap = HeuristicSegmenter().segment(pose)
    top_idx = pmap.frame_for(SwingEvent.TOP)

    # Actual argmin of the cosine curve is around the mid-frame.
    ys = np.array([f.image_keypoints[Joint.LEFT_WRIST.value, 1] for f in pose.frames])
    expected = int(np.argmin(ys))
    assert abs(top_idx - expected) <= 2


def test_segmenter_handles_zero_visibility_fallback() -> None:
    frames = []
    for i in range(20):
        kp = empty_pose_array()  # all zero visibility
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp_s=i / 30.0,
                image_keypoints=kp,
                world_keypoints=empty_pose_array(),
            )
        )
    pose = PoseSequence(frames=tuple(frames), fps=30.0, width=160, height=160, duration_s=20 / 30)
    pmap = HeuristicSegmenter().segment(pose)
    # All 8 events present, in strict temporal order, covering the clip.
    assert pmap.is_monotonic()
    assert pmap.frame_for(SwingEvent.ADDRESS) == 0
    assert pmap.frame_for(SwingEvent.FINISH) == 19

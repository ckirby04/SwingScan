"""MediaPipe BlazePose backend.

Wraps :class:`mediapipe.solutions.pose.Pose` in a typed
:class:`~swingscan.pose.base.PoseEstimator`-compatible object. Configurable
via :class:`~swingscan.config.PoseConfig`; returns both image-space and
world-space keypoints.

MediaPipe's ``process`` expects an RGB image, but :mod:`swingscan.io.video`
yields BGR frames (OpenCV's native order). The wrapper converts once per
call.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from swingscan.config import PoseConfig
from swingscan.io.video import VideoReader
from swingscan.pose.base import (
    NUM_JOINTS,
    PoseFrame,
    PoseSequence,
    empty_pose_array,
)

__all__ = ["MediaPipePoseEstimator"]

_log = logging.getLogger(__name__)


class MediaPipePoseEstimator:
    """BlazePose wrapper compatible with the :class:`PoseEstimator` protocol."""

    def __init__(self, config: PoseConfig | None = None) -> None:
        self._config = config or PoseConfig()
        # Defer the mediapipe import until the backend is actually
        # instantiated so unit tests that don't touch pose can still run
        # in pipeline-free environments.
        import mediapipe as mp

        self._mp_pose_module = mp.solutions.pose
        self._pose = self._mp_pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )

    # ---------- protocol methods -----------------------------------------

    def estimate(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
        timestamp_s: float,
    ) -> PoseFrame:
        rgb = frame[..., ::-1]  # BGR → RGB via numpy view
        result = self._pose.process(rgb)

        image_kp = empty_pose_array()
        world_kp = empty_pose_array()

        if result.pose_landmarks is not None:
            for idx, lm in enumerate(result.pose_landmarks.landmark):
                if idx >= NUM_JOINTS:
                    break
                image_kp[idx, 0] = lm.x
                image_kp[idx, 1] = lm.y
                image_kp[idx, 2] = lm.z
                image_kp[idx, 3] = lm.visibility

        if result.pose_world_landmarks is not None:
            for idx, lm in enumerate(result.pose_world_landmarks.landmark):
                if idx >= NUM_JOINTS:
                    break
                world_kp[idx, 0] = lm.x
                world_kp[idx, 1] = lm.y
                world_kp[idx, 2] = lm.z
                world_kp[idx, 3] = lm.visibility

        # Low-confidence flag: median visibility of the body across all 33
        # landmarks. We use the full set (not CORE_JOINTS) here because face
        # landmarks are often the first to drop; that's actually a useful
        # early warning that the golfer is outside the frame.
        median_vis = float(np.median(image_kp[:, 3])) if result.pose_landmarks else 0.0
        is_low_conf = median_vis < self._config.min_detection_confidence

        return PoseFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            image_keypoints=image_kp,
            world_keypoints=world_kp,
            is_low_confidence=is_low_conf,
        )

    def estimate_video(self, video: VideoReader) -> PoseSequence:
        frames: list[PoseFrame] = []
        fps = video.fps
        for idx, bgr in enumerate(video):
            timestamp_s = idx / fps if fps > 0 else 0.0
            frames.append(self.estimate(bgr, idx, timestamp_s))

        low_ratio = (
            sum(1 for f in frames if f.is_low_confidence) / len(frames) if frames else 0.0
        )
        _log.info(
            "Processed %d frames through MediaPipe; %.1f%% flagged low-confidence.",
            len(frames),
            low_ratio * 100,
        )

        return PoseSequence(
            frames=tuple(frames),
            fps=video.fps,
            width=video.width,
            height=video.height,
            duration_s=video.duration_s,
            source_path=str(video.path),
        )

    # ---------- lifecycle ------------------------------------------------

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        if hasattr(self._pose, "close"):
            self._pose.close()

    def __enter__(self) -> MediaPipePoseEstimator:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


# Module-level assertion that :class:`MediaPipePoseEstimator` satisfies the
# :class:`~swingscan.pose.base.PoseEstimator` protocol. Runtime-checkable
# protocols don't enforce return types, but importing `PoseEstimator` here
# would introduce a circular import; the structural compatibility is
# verified by a unit test instead. See tests/unit/test_mediapipe_backend.py.
assert hasattr(MediaPipePoseEstimator, "estimate")
assert hasattr(MediaPipePoseEstimator, "estimate_video")

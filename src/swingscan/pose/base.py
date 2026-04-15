"""Core pose data structures and the ``PoseEstimator`` protocol.

This module is backend-agnostic: it does not import MediaPipe, OpenCV, or
torch. Concrete estimators (e.g. :mod:`swingscan.pose.mediapipe_backend`)
produce instances of the dataclasses defined here.

The dataclasses are frozen and slotted so downstream code can rely on
equality, hashing, and structural sharing. NumPy arrays are stored as
2D ``float32`` tensors with shape ``(33, 4)`` — one row per
:class:`~swingscan.pose.keypoints.Joint`, columns ``[x, y, z, visibility]``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from swingscan.pose.keypoints import CORE_JOINTS, Joint

if TYPE_CHECKING:
    from swingscan.io.video import VideoReader

__all__ = [
    "NUM_JOINTS",
    "KEYPOINT_COLUMNS",
    "PoseFrame",
    "PoseSequence",
    "PoseEstimator",
    "empty_pose_array",
]

NUM_JOINTS: int = 33
KEYPOINT_COLUMNS: tuple[str, ...] = ("x", "y", "z", "visibility")


def empty_pose_array() -> NDArray[np.float32]:
    """Return a zero-filled ``(NUM_JOINTS, 4)`` pose array."""
    return np.zeros((NUM_JOINTS, len(KEYPOINT_COLUMNS)), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Per-frame pose estimate.

    Attributes:
        frame_index: 0-based index in the source video.
        timestamp_s: Wall-clock time of the frame in seconds, measured from
            the start of the video.
        image_keypoints: ``(NUM_JOINTS, 4)`` float32 array in image-normalized
            coordinates. ``x`` and ``y`` are in ``[0, 1]``, ``z`` is in
            image-normalized depth (MediaPipe's convention), and the last
            column is visibility in ``[0, 1]``.
        world_keypoints: ``(NUM_JOINTS, 4)`` float32 array in world
            coordinates (meters, MediaPipe origin at the body center).
            Populated when the backend provides it; otherwise all zeros.
        is_low_confidence: True when the median visibility across
            :data:`~swingscan.pose.keypoints.CORE_JOINTS` is below the
            backend's configured threshold. Downstream metric code can use
            this to decide whether to skip or interpolate the frame.
    """

    frame_index: int
    timestamp_s: float
    image_keypoints: NDArray[np.float32]
    world_keypoints: NDArray[np.float32]
    is_low_confidence: bool = False

    def __post_init__(self) -> None:
        for name, arr in (
            ("image_keypoints", self.image_keypoints),
            ("world_keypoints", self.world_keypoints),
        ):
            if arr.shape != (NUM_JOINTS, len(KEYPOINT_COLUMNS)):
                raise ValueError(
                    f"{name} must have shape ({NUM_JOINTS}, {len(KEYPOINT_COLUMNS)}); "
                    f"got {arr.shape}."
                )
            if arr.dtype != np.float32:
                raise ValueError(f"{name} must be float32; got {arr.dtype}.")

    def visibility(self, joint: Joint) -> float:
        """Visibility in ``[0, 1]`` for a given joint (image space)."""
        return float(self.image_keypoints[joint.value, 3])

    def median_core_visibility(self) -> float:
        """Median visibility across :data:`CORE_JOINTS`."""
        indices = np.array([j.value for j in CORE_JOINTS], dtype=np.int64)
        return float(np.median(self.image_keypoints[indices, 3]))


@dataclass(frozen=True, slots=True)
class PoseSequence:
    """Ordered sequence of :class:`PoseFrame` plus source-video metadata.

    Attributes:
        frames: Tuple of :class:`PoseFrame`, one per decoded video frame.
        fps: Frames-per-second of the source video.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_s: Total duration of the source video in seconds.
        source_path: Absolute path string of the source video. Stored as a
            string (not :class:`pathlib.Path`) so dataclass equality and
            parquet metadata serialization stay trivial.
    """

    frames: tuple[PoseFrame, ...]
    fps: float
    width: int
    height: int
    duration_s: float
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[PoseFrame]:
        return iter(self.frames)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def low_confidence_ratio(self) -> float:
        """Fraction of frames flagged as low-confidence."""
        if not self.frames:
            return 0.0
        return sum(1 for f in self.frames if f.is_low_confidence) / len(self.frames)


@runtime_checkable
class PoseEstimator(Protocol):
    """Structural interface every pose backend must satisfy.

    Concrete backends (e.g. MediaPipe) are standalone classes that
    implement both methods. Code that orchestrates the pipeline accepts
    any object matching this protocol, which keeps backend swaps cheap
    (important for the V2 roadmap in ``CLAUDE.md`` — MMPose, 3D lifters,
    etc.).
    """

    def estimate(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
        timestamp_s: float,
    ) -> PoseFrame:
        """Run the backend on a single BGR frame."""
        ...

    def estimate_video(self, video: VideoReader) -> PoseSequence:
        """Run the backend over every frame of a video.

        Implementations may assume ``video`` has already been opened. They
        should consume the iterator exactly once and close the reader on
        completion.
        """
        ...

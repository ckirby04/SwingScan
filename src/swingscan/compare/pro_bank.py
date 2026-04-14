"""Professional swing reference bank.

A :class:`ProBank` stores a small number of pose snapshots per swing —
one at each of the 8 canonical events — along with metadata (swing id,
view, handedness, club). This is deliberately compact: full per-frame
trajectories live in parquet files next to each video and are only
loaded on demand by Stage 5 comparison code.

Schema (parquet):

    swing_id        string
    view            string   # "face_on" | "down_the_line"
    handedness      string   # "right" | "left"
    club            string   # "driver" | "iron7" | ...
    event           string   # one of SwingEvent names
    frame_index     int32
    <joint>_img_{x|y|z|visibility}     float32
    <joint>_world_{x|y|z|visibility}   float32

One row per (swing, event). A full 100-swing bank is therefore 800 rows,
roughly 250 KB on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from swingscan.pose.base import PoseFrame, empty_pose_array

__all__ = [
    "SwingView",
    "Handedness",
    "SwingLabel",
    "ProSwing",
    "ProBank",
    "build_pro_bank",
]

_log = logging.getLogger(__name__)

SwingView = str  # "face_on" | "down_the_line"
Handedness = str  # "right" | "left"


class SwingLabel(BaseModel):
    """One swing's metadata + 8 event-frame indices."""

    model_config = ConfigDict(extra="forbid")

    swing_id: str
    video_path: str
    view: str = Field(default="face_on")
    handedness: str = Field(default="right")
    club: str = Field(default="driver")
    # Ordered list of frame indices, one per SwingEvent in enum order.
    event_frames: list[int]


@dataclass(frozen=True, slots=True)
class ProSwing:
    """One professional swing's per-event pose snapshots.

    ``event_frames`` maps event name → frame index within the source
    video. ``event_poses`` holds the corresponding :class:`PoseFrame`
    at that index, in the same order as the event names.
    """

    swing_id: str
    view: str
    handedness: str
    club: str
    event_names: tuple[str, ...]
    event_frames: tuple[int, ...]
    event_poses: tuple[PoseFrame, ...]


# --- ProBank -----------------------------------------------------------


class ProBank:
    """Queryable store of :class:`ProSwing` rows."""

    def __init__(self, swings: tuple[ProSwing, ...] = ()) -> None:
        self._swings = swings

    # ---------- factories ---------------------------------------------

    @classmethod
    def from_rows(cls, swings: list[ProSwing]) -> ProBank:
        return cls(tuple(swings))

    @classmethod
    def load(cls, path: Path | str) -> ProBank:
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Pro bank parquet not found: {src}")

        table = pq.read_table(src)
        rows: list[dict[str, Any]] = table.to_pylist()

        # Group rows by swing_id, preserving the event order in which
        # they appear in the file.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["swing_id"]), []).append(row)

        swings: list[ProSwing] = []
        for swing_id, rows_for_swing in grouped.items():
            if not rows_for_swing:
                continue
            head = rows_for_swing[0]
            event_names: list[str] = []
            event_frames: list[int] = []
            event_poses: list[PoseFrame] = []
            for r in rows_for_swing:
                event_names.append(str(r["event"]))
                event_frames.append(int(r["frame_index"]))
                img = empty_pose_array()
                world = empty_pose_array()
                from swingscan.pose.keypoints import Joint

                for joint in Joint:
                    jn = joint.name.lower()
                    img[joint.value, 0] = float(r[f"{jn}_img_x"])
                    img[joint.value, 1] = float(r[f"{jn}_img_y"])
                    img[joint.value, 2] = float(r[f"{jn}_img_z"])
                    img[joint.value, 3] = float(r[f"{jn}_img_visibility"])
                    world[joint.value, 0] = float(r[f"{jn}_world_x"])
                    world[joint.value, 1] = float(r[f"{jn}_world_y"])
                    world[joint.value, 2] = float(r[f"{jn}_world_z"])
                    world[joint.value, 3] = float(r[f"{jn}_world_visibility"])
                event_poses.append(
                    PoseFrame(
                        frame_index=int(r["frame_index"]),
                        timestamp_s=0.0,
                        image_keypoints=img,
                        world_keypoints=world,
                    )
                )
            swings.append(
                ProSwing(
                    swing_id=swing_id,
                    view=str(head["view"]),
                    handedness=str(head["handedness"]),
                    club=str(head["club"]),
                    event_names=tuple(event_names),
                    event_frames=tuple(event_frames),
                    event_poses=tuple(event_poses),
                )
            )
        return cls(tuple(swings))

    # ---------- query -------------------------------------------------

    def __len__(self) -> int:
        return len(self._swings)

    def __iter__(self) -> Iterator[ProSwing]:
        return iter(self._swings)

    @property
    def swings(self) -> tuple[ProSwing, ...]:
        return self._swings

    def filter(
        self,
        view: str | None = None,
        handedness: str | None = None,
        club: str | None = None,
    ) -> ProBank:
        def matches(s: ProSwing) -> bool:
            if view is not None and s.view != view:
                return False
            if handedness is not None and s.handedness != handedness:
                return False
            return not (club is not None and s.club != club)

        return ProBank(tuple(s for s in self._swings if matches(s)))

    def event_poses(self, event: str) -> list[NDArray[np.float32]]:
        """Return stacked image-keypoint arrays for the given event.

        Returns an empty list when no swing in the bank has that event.
        """
        out: list[NDArray[np.float32]] = []
        for swing in self._swings:
            for name, pose in zip(swing.event_names, swing.event_poses, strict=False):
                if name == event:
                    out.append(pose.image_keypoints)
        return out

    # ---------- save --------------------------------------------------

    def save(self, path: Path | str) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        from swingscan.pose.keypoints import Joint

        columns: dict[str, list[object]] = {
            "swing_id": [],
            "view": [],
            "handedness": [],
            "club": [],
            "event": [],
            "frame_index": [],
        }
        for joint in Joint:
            jn = joint.name.lower()
            for space in ("img", "world"):
                for comp in ("x", "y", "z", "visibility"):
                    columns[f"{jn}_{space}_{comp}"] = []

        for swing in self._swings:
            for event_name, frame_idx, pose in zip(
                swing.event_names, swing.event_frames, swing.event_poses, strict=True
            ):
                columns["swing_id"].append(swing.swing_id)
                columns["view"].append(swing.view)
                columns["handedness"].append(swing.handedness)
                columns["club"].append(swing.club)
                columns["event"].append(event_name)
                columns["frame_index"].append(int(frame_idx))
                for joint in Joint:
                    jn = joint.name.lower()
                    img_row = pose.image_keypoints[joint.value]
                    w_row = pose.world_keypoints[joint.value]
                    columns[f"{jn}_img_x"].append(float(img_row[0]))
                    columns[f"{jn}_img_y"].append(float(img_row[1]))
                    columns[f"{jn}_img_z"].append(float(img_row[2]))
                    columns[f"{jn}_img_visibility"].append(float(img_row[3]))
                    columns[f"{jn}_world_x"].append(float(w_row[0]))
                    columns[f"{jn}_world_y"].append(float(w_row[1]))
                    columns[f"{jn}_world_z"].append(float(w_row[2]))
                    columns[f"{jn}_world_visibility"].append(float(w_row[3]))

        table = pa.table(columns)
        table = table.replace_schema_metadata(
            {b"swingscan_format": b"pro_bank_v1", b"swing_count": str(len(self)).encode()}
        )
        pq.write_table(table, out)
        _log.info("Wrote pro bank (%d swings) to %s", len(self), out)
        return out


# --- Bank construction -------------------------------------------------


def build_pro_bank(
    labels: list[SwingLabel],
    pose_backend_factory: Callable[[], Any] | None = None,
) -> ProBank:
    """Run pose on every labeled swing and emit a :class:`ProBank`.

    Args:
        labels: List of :class:`SwingLabel`. Each entry names a video
            file on disk plus the 8 event frame indices.
        pose_backend_factory: Zero-argument callable returning a pose
            backend with an ``estimate_video(video)`` method. When
            ``None``, a default :class:`MediaPipePoseEstimator` is
            instantiated. Tests inject fakes through this hook.

    Returns:
        A populated :class:`ProBank`. Swings whose video files cannot be
        opened are skipped with a warning; the bank may be empty.
    """
    from swingscan.io.video import VideoReader
    from swingscan.phases.events import SwingEvent

    if pose_backend_factory is None:
        from swingscan.pose.mediapipe_backend import MediaPipePoseEstimator

        def _default_factory() -> Any:
            return MediaPipePoseEstimator()

        pose_backend_factory = _default_factory

    event_names = tuple(e.name for e in SwingEvent)
    pro_swings: list[ProSwing] = []

    for label in labels:
        vp = Path(label.video_path).expanduser().resolve()
        if not vp.is_file():
            _log.warning("Skipping %s: video not found at %s", label.swing_id, vp)
            continue

        try:
            with VideoReader(vp) as video:
                backend = pose_backend_factory()
                pose_seq = backend.estimate_video(video)
                if hasattr(backend, "close"):
                    backend.close()
        except Exception as exc:
            _log.warning("Skipping %s: %s", label.swing_id, exc)
            continue

        if len(label.event_frames) != len(event_names):
            _log.warning(
                "Skipping %s: expected %d event frames, got %d.",
                label.swing_id,
                len(event_names),
                len(label.event_frames),
            )
            continue

        event_poses: list[PoseFrame] = []
        valid = True
        for ename, frame_idx in zip(event_names, label.event_frames, strict=True):
            if frame_idx < 0 or frame_idx >= len(pose_seq):
                _log.warning(
                    "Skipping %s: %s frame index %d outside pose sequence of length %d.",
                    label.swing_id,
                    ename,
                    frame_idx,
                    len(pose_seq),
                )
                valid = False
                break
            event_poses.append(pose_seq.frames[frame_idx])
        if not valid:
            continue

        pro_swings.append(
            ProSwing(
                swing_id=label.swing_id,
                view=label.view,
                handedness=label.handedness,
                club=label.club,
                event_names=event_names,
                event_frames=tuple(label.event_frames),
                event_poses=tuple(event_poses),
            )
        )

    return ProBank.from_rows(pro_swings)

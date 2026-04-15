"""Serialization of :class:`~swingscan.pose.base.PoseSequence`.

Two formats are supported:

* **parquet** (primary) — one row per frame, flat column layout. Fast to
  load with pandas/pyarrow and trivially indexable. Video metadata (fps,
  width, height, duration, source path, rotation) is stored in the file's
  key/value metadata dict, not as columns.
* **JSON** (debug) — a dict with ``metadata`` and ``frames``. Slower and
  bulkier; use only when hand-inspecting is needed.

Both paths round-trip losslessly. The parquet schema is documented in
``docs/pipeline.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from swingscan.pose.base import (
    KEYPOINT_COLUMNS,
    NUM_JOINTS,
    PoseFrame,
    PoseSequence,
)
from swingscan.pose.keypoints import Joint

__all__ = [
    "save_pose_sequence",
    "load_pose_sequence",
    "save_pose_sequence_parquet",
    "load_pose_sequence_parquet",
    "save_pose_sequence_json",
    "load_pose_sequence_json",
]

_log = logging.getLogger(__name__)

_META_KEYS: Final[tuple[str, ...]] = ("fps", "width", "height", "duration_s", "source_path")


def _column_names() -> list[str]:
    """Flat column list: `frame_index`, `timestamp_s`, `is_low_confidence`,
    then ``<joint>_{img|world}_{x|y|z|visibility}``."""
    cols: list[str] = ["frame_index", "timestamp_s", "is_low_confidence"]
    for joint in Joint:
        for space in ("img", "world"):
            for col in KEYPOINT_COLUMNS:
                cols.append(f"{joint.name.lower()}_{space}_{col}")
    return cols


_COLUMNS: Final[list[str]] = _column_names()


def _sequence_to_arrays(seq: PoseSequence) -> dict[str, NDArray[Any]]:
    """Flatten a :class:`PoseSequence` into a dict of 1-D numpy arrays."""
    n = len(seq.frames)
    out: dict[str, NDArray[Any]] = {
        "frame_index": np.empty(n, dtype=np.int32),
        "timestamp_s": np.empty(n, dtype=np.float64),
        "is_low_confidence": np.empty(n, dtype=bool),
    }
    for joint in Joint:
        for space in ("img", "world"):
            for col in KEYPOINT_COLUMNS:
                out[f"{joint.name.lower()}_{space}_{col}"] = np.empty(n, dtype=np.float32)

    for row, frame in enumerate(seq.frames):
        out["frame_index"][row] = frame.frame_index
        out["timestamp_s"][row] = frame.timestamp_s
        out["is_low_confidence"][row] = frame.is_low_confidence
        for joint in Joint:
            img_row = frame.image_keypoints[joint.value]
            world_row = frame.world_keypoints[joint.value]
            jn = joint.name.lower()
            out[f"{jn}_img_x"][row] = img_row[0]
            out[f"{jn}_img_y"][row] = img_row[1]
            out[f"{jn}_img_z"][row] = img_row[2]
            out[f"{jn}_img_visibility"][row] = img_row[3]
            out[f"{jn}_world_x"][row] = world_row[0]
            out[f"{jn}_world_y"][row] = world_row[1]
            out[f"{jn}_world_z"][row] = world_row[2]
            out[f"{jn}_world_visibility"][row] = world_row[3]

    return out


def _arrays_to_frames(arrays: dict[str, NDArray[Any]]) -> tuple[PoseFrame, ...]:
    n = len(arrays["frame_index"])
    frames: list[PoseFrame] = []
    for row in range(n):
        image_kp = np.zeros((NUM_JOINTS, len(KEYPOINT_COLUMNS)), dtype=np.float32)
        world_kp = np.zeros((NUM_JOINTS, len(KEYPOINT_COLUMNS)), dtype=np.float32)
        for joint in Joint:
            jn = joint.name.lower()
            image_kp[joint.value, 0] = arrays[f"{jn}_img_x"][row]
            image_kp[joint.value, 1] = arrays[f"{jn}_img_y"][row]
            image_kp[joint.value, 2] = arrays[f"{jn}_img_z"][row]
            image_kp[joint.value, 3] = arrays[f"{jn}_img_visibility"][row]
            world_kp[joint.value, 0] = arrays[f"{jn}_world_x"][row]
            world_kp[joint.value, 1] = arrays[f"{jn}_world_y"][row]
            world_kp[joint.value, 2] = arrays[f"{jn}_world_z"][row]
            world_kp[joint.value, 3] = arrays[f"{jn}_world_visibility"][row]
        frames.append(
            PoseFrame(
                frame_index=int(arrays["frame_index"][row]),
                timestamp_s=float(arrays["timestamp_s"][row]),
                image_keypoints=image_kp,
                world_keypoints=world_kp,
                is_low_confidence=bool(arrays["is_low_confidence"][row]),
            )
        )
    return tuple(frames)


# --- Parquet -------------------------------------------------------------


def save_pose_sequence_parquet(seq: PoseSequence, path: Path | str) -> Path:
    """Write ``seq`` to a parquet file. Returns the resolved absolute path."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays = _sequence_to_arrays(seq)
    table = pa.table(arrays)
    file_metadata = {
        "swingscan_format": "pose_sequence_v1",
        "fps": repr(float(seq.fps)),
        "width": repr(int(seq.width)),
        "height": repr(int(seq.height)),
        "duration_s": repr(float(seq.duration_s)),
        "source_path": seq.source_path,
    }
    # Merge our metadata into the schema; pyarrow stores them as bytes.
    encoded = {k.encode("utf-8"): v.encode("utf-8") for k, v in file_metadata.items()}
    table = table.replace_schema_metadata(encoded)

    pq.write_table(table, out)
    _log.debug("Wrote %d-frame PoseSequence to %s", len(seq.frames), out)
    return out


def load_pose_sequence_parquet(path: Path | str) -> PoseSequence:
    """Load a :class:`PoseSequence` from a parquet file."""
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Pose parquet not found: {src}")

    table = pq.read_table(src)
    schema_meta = table.schema.metadata or {}
    meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in schema_meta.items()}

    if meta.get("swingscan_format") != "pose_sequence_v1":
        raise ValueError(f"{src}: missing or unrecognized swingscan_format in parquet metadata.")

    arrays = {
        name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names
    }
    frames = _arrays_to_frames(arrays)

    return PoseSequence(
        frames=frames,
        fps=float(meta.get("fps", "0.0")),
        width=int(float(meta.get("width", "0"))),
        height=int(float(meta.get("height", "0"))),
        duration_s=float(meta.get("duration_s", "0.0")),
        source_path=meta.get("source_path", ""),
    )


# --- JSON ---------------------------------------------------------------


def save_pose_sequence_json(seq: PoseSequence, path: Path | str) -> Path:
    """Write ``seq`` to a JSON file for human-readable debugging."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "pose_sequence_v1",
        "metadata": {
            "fps": seq.fps,
            "width": seq.width,
            "height": seq.height,
            "duration_s": seq.duration_s,
            "source_path": seq.source_path,
        },
        "frames": [
            {
                "frame_index": f.frame_index,
                "timestamp_s": f.timestamp_s,
                "is_low_confidence": f.is_low_confidence,
                "image_keypoints": f.image_keypoints.tolist(),
                "world_keypoints": f.world_keypoints.tolist(),
            }
            for f in seq.frames
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_pose_sequence_json(path: Path | str) -> PoseSequence:
    """Load a :class:`PoseSequence` from a JSON file."""
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Pose JSON not found: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    if payload.get("format") != "pose_sequence_v1":
        raise ValueError(f"{src}: missing or unrecognized 'format' field.")

    meta = payload["metadata"]
    frames_raw = payload["frames"]
    frames = tuple(
        PoseFrame(
            frame_index=int(f["frame_index"]),
            timestamp_s=float(f["timestamp_s"]),
            image_keypoints=np.asarray(f["image_keypoints"], dtype=np.float32),
            world_keypoints=np.asarray(f["world_keypoints"], dtype=np.float32),
            is_low_confidence=bool(f["is_low_confidence"]),
        )
        for f in frames_raw
    )
    return PoseSequence(
        frames=frames,
        fps=float(meta["fps"]),
        width=int(meta["width"]),
        height=int(meta["height"]),
        duration_s=float(meta["duration_s"]),
        source_path=str(meta.get("source_path", "")),
    )


# --- Format dispatch ----------------------------------------------------


def _format_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return "parquet"
    if ext == ".json":
        return "json"
    raise ValueError(f"Unsupported extension for pose sequence: {ext!r}. Use .parquet or .json.")


def save_pose_sequence(seq: PoseSequence, path: Path | str) -> Path:
    """Save a pose sequence; format is chosen by the file extension."""
    p = Path(path)
    fmt = _format_for(p)
    if fmt == "parquet":
        return save_pose_sequence_parquet(seq, p)
    return save_pose_sequence_json(seq, p)


def load_pose_sequence(path: Path | str) -> PoseSequence:
    """Load a pose sequence; format is inferred from the file extension."""
    p = Path(path)
    fmt = _format_for(p)
    if fmt == "parquet":
        return load_pose_sequence_parquet(p)
    return load_pose_sequence_json(p)

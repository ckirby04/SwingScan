"""SwingNet phase-segmentation wrapper.

Wires the vendored upstream architecture (see
:mod:`swingscan.phases._swingnet_model`) to SwingScan's
:class:`~swingscan.phases.segmenter.PhaseSegmenter` protocol. The
segmenter takes a :class:`~swingscan.pose.base.PoseSequence`, reads the
source video back from ``pose.source_path``, runs SwingNet inference on
the raw frames, and returns a :class:`PhaseMap`.

SwingNet was trained on 160x160 inputs (letterboxed, ImageNet
normalized). Inputs are preprocessed to match at inference time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from swingscan.phases.events import SwingEvent
from swingscan.phases.segmenter import PhaseMap
from swingscan.pose.base import PoseSequence

__all__ = ["SwingNetSegmenter", "load_swingnet", "default_swingnet_path"]

_log = logging.getLogger(__name__)

# ImageNet normalization statistics SwingNet was trained against.
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE = 160
_DEFAULT_SEQ_LENGTH = 64


def default_swingnet_path() -> Path:
    from swingscan.utils.paths import models_dir

    return models_dir() / "swingnet_1800.pth.tar"


def load_swingnet(weights_path: Path | str | None = None) -> SwingNetSegmenter:
    """Convenience factory. Raises FileNotFoundError if the weights are absent."""
    return SwingNetSegmenter(weights_path=weights_path)


def _preprocess_frame(bgr: NDArray[Any]) -> NDArray[np.float32]:
    """Resize + letterbox + RGB + ImageNet normalize a single BGR frame."""
    h, w = bgr.shape[:2]
    ratio = _INPUT_SIZE / max(h, w)
    new_size = (int(w * ratio), int(h * ratio))
    resized = cv2.resize(bgr, new_size)
    delta_w = _INPUT_SIZE - new_size[0]
    delta_h = _INPUT_SIZE - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        # ImageNet BGR means scaled to 0..255, matching upstream.
        value=(0.406 * 255, 0.456 * 255, 0.485 * 255),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - _IMG_MEAN) / _IMG_STD
    # CHW layout.
    return np.transpose(normed, (2, 0, 1))


class SwingNetSegmenter:
    """Real SwingNet-backed phase segmenter.

    Usage::

        segmenter = SwingNetSegmenter()  # loads models/swingnet_1800.pth.tar
        phase_map = segmenter.segment(pose_sequence)

    The ``pose`` argument is consumed only for ``pose.source_path`` and
    ``len(pose)``. SwingNet runs on the raw video, not on the pose
    data. Passing a pseudo-sequence (e.g. one reconstructed from a
    :class:`~swingscan.compare.pro_bank.ProBank`) will raise because
    there is no backing video file to read.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        seq_length: int = _DEFAULT_SEQ_LENGTH,
        device: str | torch.device | None = None,
    ) -> None:
        resolved = Path(weights_path or default_swingnet_path()).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"SwingNet weights not found at {resolved}. Download "
                "`swingnet_1800.pth.tar` from the upstream GolfDB release "
                "(see models/README.md) or use HeuristicSegmenter as a "
                "no-download fallback."
            )

        from swingscan.phases._swingnet_model import load_swingnet_checkpoint

        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = load_swingnet_checkpoint(str(resolved), self._device)
        self._seq_length = seq_length
        _log.info("Loaded SwingNet weights from %s (device=%s)", resolved, self._device)

    @torch.no_grad()
    def segment(self, pose: PoseSequence) -> PhaseMap:
        video_path = pose.source_path
        if not video_path:
            raise ValueError(
                "SwingNetSegmenter requires a PoseSequence with source_path "
                "set. Pro-bank pseudo-sequences are not supported."
            )

        src = Path(video_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source video not found: {src}")

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise OSError(f"Could not open video for SwingNet inference: {src}")

        frames: list[NDArray[np.float32]] = []
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            frames.append(_preprocess_frame(bgr))
        cap.release()

        if not frames:
            raise ValueError(f"No frames read from {src}")

        tensor = torch.from_numpy(np.stack(frames)).float()  # (T, C, H, W)
        tensor = tensor.unsqueeze(0).to(self._device)  # (1, T, C, H, W)

        total_frames = tensor.shape[1]
        probs_parts: list[NDArray[Any]] = []
        batch = 0
        while batch * self._seq_length < total_frames:
            start = batch * self._seq_length
            end = min((batch + 1) * self._seq_length, total_frames)
            logits = self._model(tensor[:, start:end, :, :, :])
            softmax = F.softmax(logits, dim=1).cpu().numpy()
            probs_parts.append(softmax)
            batch += 1

        probs = np.concatenate(probs_parts, axis=0)  # (T, 9)
        assert probs.shape == (total_frames, 9)

        # For each of the 8 canonical events, pick the frame with max
        # probability. Class 8 is "no event" and is dropped.
        event_frames = np.argmax(probs[:, :8], axis=0)

        pairs: list[tuple[SwingEvent, int]] = []
        prev = -1
        for ev, idx in zip(list(SwingEvent), event_frames, strict=True):
            # Enforce monotonicity in the rare case SwingNet picks the
            # same or an earlier frame for a later event.
            clamped = int(idx)
            if clamped <= prev:
                clamped = min(prev + 1, total_frames - 1)
            pairs.append((ev, clamped))
            prev = clamped

        return PhaseMap(events=tuple(pairs))

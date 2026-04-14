"""SwingNet phase-segmentation wrapper.

Thin shim around the SwingNet CNN from the GolfDB repository. Requires a
downloaded weights file at ``models/swingnet.pt`` (configurable).

The module is intentionally minimal: it imports ``torch`` lazily so the
absence of weights is the most common failure mode, and we want that
failure to be fast and clearly attributed. Users who do not have the
weights fall back to :class:`~swingscan.phases.segmenter.HeuristicSegmenter`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from swingscan.phases.segmenter import PhaseMap
from swingscan.pose.base import PoseSequence

__all__ = ["SwingNetSegmenter", "load_swingnet"]

_log = logging.getLogger(__name__)


def load_swingnet(weights_path: Path | str = "models/swingnet.pt") -> Any:
    """Load the SwingNet model from a weights file.

    Raises:
        FileNotFoundError: If the weights file is not present. Points at
            ``models/README.md`` for download instructions.
    """
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"SwingNet weights not found at {path}. See models/README.md "
            "for download instructions, or use HeuristicSegmenter as a "
            "no-download fallback."
        )

    import torch

    model = torch.load(str(path), map_location="cpu")
    _log.info("Loaded SwingNet weights from %s", path)
    return model


class SwingNetSegmenter:
    """SwingNet-backed phase segmenter.

    Stage 4 implements construction + the missing-weights error path.
    Actual inference against a live SwingNet checkpoint is out of scope
    for this stage — the heuristic fallback is the V1 default. This
    class exists so Stage 5+ code can branch on backend availability
    without special-casing.
    """

    def __init__(self, weights_path: Path | str = "models/swingnet.pt") -> None:
        self._weights_path = Path(weights_path).expanduser().resolve()
        self._model = load_swingnet(self._weights_path)

    def segment(self, pose: PoseSequence) -> PhaseMap:  # pragma: no cover
        raise NotImplementedError(
            "SwingNetSegmenter.segment() is not implemented in V1. Use "
            "HeuristicSegmenter for the no-weights path. A full SwingNet "
            "inference path is planned for V2."
        )

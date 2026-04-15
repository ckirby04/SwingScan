"""Deterministic seeding for reproducible runs.

Seeds Python's ``random`` and ``numpy.random``. If ``torch`` is
importable (it is when the ``pipeline`` extra is installed), also
seeds torch's CPU and CUDA RNGs. The torch branch is guarded so the
helper still runs in environments that only have the core extras.

Every training or evaluation script should call
:func:`seed_everything` before doing any random work.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Final

import numpy as np

_log = logging.getLogger(__name__)

_DEFAULT_SEED: Final[int] = 17


def seed_everything(seed: int = _DEFAULT_SEED) -> int:
    """Seed Python, NumPy, and (if present) Torch RNGs.

    Args:
        seed: Integer seed applied across all RNGs. Defaults to 17.

    Returns:
        The seed that was applied. Useful when the caller passes a randomly
        drawn seed and wants to record exactly which value took effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        _log.debug("torch not installed; skipping torch seeding")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed

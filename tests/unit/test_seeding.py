"""Tests for :mod:`swingscan.utils.seeding`."""

from __future__ import annotations

import random

import numpy as np

from swingscan.utils.seeding import seed_everything


def test_seed_everything_is_deterministic() -> None:
    seed_everything(42)
    a_py = [random.random() for _ in range(3)]
    a_np = np.random.rand(3).tolist()

    seed_everything(42)
    b_py = [random.random() for _ in range(3)]
    b_np = np.random.rand(3).tolist()

    assert a_py == b_py
    assert a_np == b_np


def test_seed_everything_returns_applied_seed() -> None:
    assert seed_everything(1234) == 1234


def test_different_seeds_produce_different_streams() -> None:
    seed_everything(1)
    stream_1 = [random.random() for _ in range(5)]
    seed_everything(2)
    stream_2 = [random.random() for _ in range(5)]
    assert stream_1 != stream_2

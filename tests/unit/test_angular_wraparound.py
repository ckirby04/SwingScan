"""Regression tests for the angular-metric wraparound fix in compare.diff.

Before the fix, comparing an amateur rotation of +179 deg against a
cohort sitting at -179 deg produced a ~358 deg delta even though the
two swings are structurally 2 deg apart. This file pins the fix in
place so a future "just use the arithmetic mean" refactor can't silently
regress.
"""

from __future__ import annotations

from swingscan.compare.diff import (
    _circular_mean_deg,
    _circular_stdev_deg,
    _wrap_signed_deg,
)


def test_wrap_signed_handles_large_positive() -> None:
    assert _wrap_signed_deg(350.0) == -10.0


def test_wrap_signed_handles_large_negative() -> None:
    assert _wrap_signed_deg(-350.0) == 10.0


def test_wrap_signed_is_identity_in_range() -> None:
    assert _wrap_signed_deg(45.0) == 45.0
    assert _wrap_signed_deg(-45.0) == -45.0


def test_circular_mean_averages_across_boundary() -> None:
    # 175 and -175 are structurally identical rotations; their circular
    # mean should be +/-180, NOT 0 (which is what the arithmetic mean
    # returns).
    result = _circular_mean_deg([175.0, -175.0])
    assert abs(abs(result) - 180.0) < 1.0


def test_circular_mean_matches_arithmetic_mean_in_safe_range() -> None:
    # When all samples are well away from the boundary, the circular
    # mean should track the arithmetic mean closely.
    result = _circular_mean_deg([80.0, 85.0, 90.0])
    assert abs(result - 85.0) < 0.1


def test_circular_stdev_absorbs_boundary_noise() -> None:
    # 179 and -179 are 2 deg apart in circular geometry. The circular
    # stdev should reflect that, not the 358 deg naive gap.
    center = _circular_mean_deg([179.0, -179.0])
    result = _circular_stdev_deg([179.0, -179.0], center)
    assert result < 5.0

"""Integration test for `swingscan phases` on the synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path

from swingscan.cli import main
from swingscan.phases.events import SwingEvent


def test_phases_cli_produces_monotonic_events(
    sample_swing_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "phases.json"
    rc = main(["phases", "--input", str(sample_swing_path), "--output", str(out)])
    assert rc == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "swingscan_phases_v1"
    assert set(payload["events"].keys()) == {e.name for e in SwingEvent}

    frame_indices = [payload["events"][e.name] for e in SwingEvent]
    assert frame_indices == sorted(set(frame_indices))

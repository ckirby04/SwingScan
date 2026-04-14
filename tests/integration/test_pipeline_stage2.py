"""End-to-end pipeline test (Stage 2): pose + club trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swingscan.cli import main
from swingscan.pipeline import run_pipeline


def test_run_pipeline_produces_club_track_per_frame(sample_swing_path: Path) -> None:
    result = run_pipeline(sample_swing_path)
    assert result.frame_count > 0
    # One club entry per pose frame, even when the heuristic can't recover
    # a position (entries are tagged "missing").
    assert len(result.club) == len(result.pose)


def test_cli_run_writes_pipeline_report(
    sample_swing_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "report.json"
    rc = main(["run", "--input", str(sample_swing_path), "--output", str(out)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "SwingScan report" in captured.out
    assert out.is_file()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "swingscan_pipeline_v1"
    assert "pose_summary" in payload
    assert "club_track" in payload
    # Club trajectory matches pose frame count.
    assert len(payload["club_track"]) == payload["pose_summary"]["frame_count"]


def test_cli_run_nonexistent_weights_falls_back_gracefully(
    sample_swing_path: Path,
    tmp_path: Path,
) -> None:
    rc = main(
        [
            "run",
            "--input",
            str(sample_swing_path),
            "--club-weights",
            str(tmp_path / "not_real.pt"),
        ]
    )
    # Missing weights should log a warning and fall back to heuristic,
    # not crash.
    assert rc == 0

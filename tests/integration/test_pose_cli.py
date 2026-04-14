"""Integration test for the ``swingscan pose`` CLI subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from swingscan.cli import main
from swingscan.io.serialize import load_pose_sequence


def test_pose_subcommand_writes_parquet(
    sample_swing_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "pose.parquet"
    rc = main(["pose", "--input", str(sample_swing_path), "--output", str(out)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out.is_file()
    assert "Wrote" in captured.out
    assert "pose frames" in captured.out

    reloaded = load_pose_sequence(out)
    assert len(reloaded) > 0
    # Source path round-trips through parquet metadata.
    assert Path(reloaded.source_path) == sample_swing_path


def test_pose_subcommand_writes_json(
    sample_swing_path: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "pose.json"
    rc = main(["pose", "--input", str(sample_swing_path), "--output", str(out)])
    assert rc == 0
    assert out.is_file()

    reloaded = load_pose_sequence(out)
    assert len(reloaded) > 0

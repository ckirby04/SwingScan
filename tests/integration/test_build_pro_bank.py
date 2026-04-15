"""End-to-end integration test for Stage 3 bank construction.

Uses the synthetic sample swing fixture + a hand-crafted label to
exercise `build_pro_bank` without needing any real GolfDB download.
The resulting bank will contain one swing whose poses are all
zero-confidence (the fixture has no human in frame) — that's fine for
a plumbing test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_pro_bank import main as build_main  # type: ignore[import-not-found]
from swingscan.compare.pro_bank import ProBank


@pytest.fixture()
def fake_labels_file(tmp_path: Path, sample_swing_path: Path) -> Path:
    payload = {
        "swings": [
            {
                "swing_id": "synthetic_1",
                "video_path": str(sample_swing_path),
                "view": "face_on",
                "handedness": "right",
                "club": "driver",
                # 8 event frames distributed across the 60-frame fixture.
                "event_frames": [0, 8, 16, 24, 32, 40, 48, 56],
            }
        ]
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_pro_bank_cli_round_trip(fake_labels_file: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "pro_bank"
    rc = build_main(
        [
            "--labels",
            str(fake_labels_file),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    bank_path = out_dir / "bank.parquet"
    manifest_path = out_dir / "manifest.json"
    assert bank_path.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["requested"] == 1
    assert manifest["succeeded"] == 1
    assert manifest["skipped"] == 0

    bank = ProBank.load(bank_path)
    assert len(bank) == 1
    swing = next(iter(bank))
    assert swing.swing_id == "synthetic_1"
    assert len(swing.event_poses) == 8


def test_build_pro_bank_handles_empty_labels(tmp_path: Path) -> None:
    labels = tmp_path / "empty.json"
    labels.write_text(json.dumps({"swings": []}), encoding="utf-8")
    out_dir = tmp_path / "pro_bank"
    rc = build_main(
        [
            "--labels",
            str(labels),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    bank = ProBank.load(out_dir / "bank.parquet")
    assert len(bank) == 0

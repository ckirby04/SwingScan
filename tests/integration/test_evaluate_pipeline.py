"""Integration test for scripts/evaluate_pipeline.py."""

from __future__ import annotations

import json
from pathlib import Path

from evaluate_pipeline import main as eval_main  # type: ignore[import-not-found]


def test_evaluate_with_empty_labels_writes_no_data_report(tmp_path: Path) -> None:
    missing_labels = tmp_path / "labels.json"  # not created
    report = tmp_path / "report.json"
    rc = eval_main(
        [
            "--labels",
            str(missing_labels),
            "--output",
            str(report),
        ]
    )
    assert rc == 0
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["format"] == "swingscan_evaluation_v1"
    assert payload["marker"] == "no_data"
    assert payload["swings_total"] == 0


def test_evaluate_on_synthetic_fixture(sample_swing_path: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "swings": [
                    {
                        "swing_id": "synth",
                        "video_path": str(sample_swing_path),
                        "view": "face_on",
                        "handedness": "right",
                        "club": "driver",
                        # Hand-picked "truth" event frames at even spacing.
                        "event_frames": [0, 8, 16, 24, 32, 40, 48, 56],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    rc = eval_main(
        [
            "--labels",
            str(labels),
            "--output",
            str(report),
            "--tolerance",
            "5",
        ]
    )
    assert rc == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["swings_total"] == 1
    assert payload["swings_scored"] == 1
    # PCE is reported as a float in [0, 1]; value depends on the heuristic
    # against zero-visibility synthetic pose — we only assert that the
    # script produced a number, not a specific quality target.
    assert isinstance(payload["pce"], float)
    assert 0.0 <= payload["pce"] <= 1.0

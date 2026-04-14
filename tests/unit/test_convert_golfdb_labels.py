"""Tests for scripts/convert_golfdb_labels.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from convert_golfdb_labels import main as convert_main  # type: ignore[import-not-found]


def _fake_pickle(tmp_path: Path) -> Path:
    rows = [
        {
            "id": 0,
            "youtube_id": "abc",
            "player": "RORY",
            "sex": "m",
            "club": "driver",
            "view": "face-on",
            "slow": 0,
            "events": [100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
            "bbox": [0, 0, 1, 1],
            "split": 1,
        },
        {
            "id": 1,
            "youtube_id": "def",
            "player": "PHIL",
            "sex": "m",
            "club": "iron",
            "view": "down-the-line",
            "slow": 1,
            "events": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            "bbox": [0, 0, 1, 1],
            "split": 1,
        },
        {
            "id": 2,
            "youtube_id": "ghi",
            "player": "BRYSON",
            "sex": "m",
            "club": "driver",
            "view": "face-on",
            "slow": 0,
            "events": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            "bbox": [0, 0, 1, 1],
            "split": 1,
        },
    ]
    df = pd.DataFrame(rows)
    pkl = tmp_path / "golfDB.pkl"
    df.to_pickle(pkl)
    return pkl


def _fake_clip(path: Path) -> None:
    # Create a zero-byte placeholder — the converter only checks `is_file`.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_converter_filters_and_rebases_events(tmp_path: Path) -> None:
    pkl = _fake_pickle(tmp_path)
    clips = tmp_path / "clips"
    # Provide clips for rows 0 and 2 (both face-on driver, non-slow-mo).
    _fake_clip(clips / "0.mp4")
    _fake_clip(clips / "2.mp4")
    out = tmp_path / "labels.json"

    rc = convert_main(
        [
            "--pickle",
            str(pkl),
            "--clips",
            str(clips),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.is_file()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["swings"]) == 2
    ids = {s["swing_id"] for s in payload["swings"]}
    assert ids == {"golfdb_0", "golfdb_2"}

    first = payload["swings"][0]
    assert first["view"] == "face_on"
    assert first["club"] == "driver"
    assert first["handedness"] == "right"
    # Row 0: events[0]=100 → start; canonical = events[1..8] - 100
    assert first["event_frames"] == [10, 20, 30, 40, 50, 60, 70, 80]


def test_converter_skips_rows_with_missing_clips(tmp_path: Path) -> None:
    pkl = _fake_pickle(tmp_path)
    clips = tmp_path / "clips"
    clips.mkdir()
    # No clip files provided at all.
    out = tmp_path / "labels.json"

    rc = convert_main(
        [
            "--pickle",
            str(pkl),
            "--clips",
            str(clips),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["swings"] == []


def test_converter_no_filter_includes_all_matched_rows(tmp_path: Path) -> None:
    pkl = _fake_pickle(tmp_path)
    clips = tmp_path / "clips"
    _fake_clip(clips / "0.mp4")
    _fake_clip(clips / "1.mp4")
    _fake_clip(clips / "2.mp4")
    out = tmp_path / "labels.json"

    rc = convert_main(
        [
            "--pickle",
            str(pkl),
            "--clips",
            str(clips),
            "--output",
            str(out),
            "--no-filter",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["swings"]) == 3


def test_converter_missing_pickle_returns_error(tmp_path: Path) -> None:
    rc = convert_main(
        [
            "--pickle",
            str(tmp_path / "nope.pkl"),
            "--clips",
            str(tmp_path),
            "--output",
            str(tmp_path / "labels.json"),
        ]
    )
    assert rc == 1

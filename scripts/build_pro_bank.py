"""Build a professional reference swing bank from labeled clips.

Reads a labels JSON file (list of :class:`SwingLabel` records), runs the
pose pipeline on each referenced video, and writes a queryable
``bank.parquet`` plus ``manifest.json`` under ``data/pro_bank/``.

Labels JSON format (matches :class:`SwingLabel`)::

    {
      "swings": [
        {
          "swing_id": "abc",
          "video_path": "data/raw/golfdb/clips/abc.mp4",
          "view": "face_on",
          "handedness": "right",
          "club": "driver",
          "event_frames": [0, 12, 24, 36, 48, 60, 72, 84]
        },
        ...
      ]
    }

This script never touches YouTube and does not depend on ``yt-dlp``. The
user supplies the video paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from swingscan.compare.pro_bank import SwingLabel, build_pro_bank
from swingscan.utils.logging import configure_logging
from swingscan.utils.paths import data_dir

_log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to a labels JSON. Defaults to data/raw/labels.json if present.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/pro_bank/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N labels (useful for smoke tests).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    labels_path = (
        Path(args.labels).expanduser().resolve()
        if args.labels
        else data_dir() / "raw" / "labels.json"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir() / "pro_bank"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "bank.parquet"
    manifest_path = output_dir / "manifest.json"

    if not labels_path.is_file():
        _log.warning(
            "No labels file at %s. Writing an empty bank. To populate, run "
            "scripts/download_golfdb.py and produce a labels file.",
            labels_path,
        )
        payload = {"labels_path": str(labels_path), "swings": []}
    else:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))

    raw_swings = payload.get("swings", [])
    if args.limit is not None:
        raw_swings = raw_swings[: args.limit]

    labels: list[SwingLabel] = [SwingLabel.model_validate(row) for row in raw_swings]
    _log.info("Processing %d labels from %s", len(labels), labels_path)

    bank = build_pro_bank(labels)
    bank.save(bank_path)

    manifest = {
        "source_labels": str(labels_path),
        "bank_parquet": str(bank_path),
        "requested": len(labels),
        "succeeded": len(bank),
        "skipped": len(labels) - len(bank),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sys.stdout.write(
        f"Wrote {len(bank)}-swing bank to {bank_path}\n" f"Manifest: {manifest_path}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

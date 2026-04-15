"""Convert the GolfDB annotation pickle into a SwingScan labels.json.

The upstream GolfDB pickle (``data/raw/golfdb/golfDB.pkl``) has columns::

    id youtube_id player sex club view slow events bbox split

Our ``SwingLabel`` expects::

    swing_id video_path view handedness club event_frames

This script reconciles the two: it loads the pickle, filters rows, maps
GolfDB's 10-entry ``events`` array to the 8 canonical SwingScan events
(indices 1-8), normalizes the view strings, and writes a ``labels.json``
pointing at clip files already on disk. Clips whose files are missing
are skipped with a warning - the resulting labels.json only references
swings the downstream pipeline can actually process.

Usage::

    # Default: face-on driver, non-slow-mo, clips at data/raw/golfdb/clips/<id>.mp4
    python scripts/convert_golfdb_labels.py \\
        --pickle data/raw/golfdb/golfDB.pkl \\
        --clips  data/raw/golfdb/clips \\
        --output data/raw/labels.json

    # Include all views, all clubs, all swings — unfiltered:
    python scripts/convert_golfdb_labels.py --no-filter

The GolfDB 160x160 preprocessed clips live in the upstream project's
Google Drive release. See
https://github.com/wmcnally/golfdb for the download link. This script
does not auto-fetch them.
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

from swingscan.utils.logging import configure_logging
from swingscan.utils.paths import data_dir

_log = logging.getLogger(__name__)


_VIEW_MAP = {
    "face-on": "face_on",
    "down-the-line": "down_the_line",
    "other": "other",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle",
        default=None,
        help="Path to golfDB.pkl. Defaults to data/raw/golfdb/golfDB.pkl.",
    )
    parser.add_argument(
        "--clips",
        default=None,
        help="Directory containing <id>.mp4 files. Defaults to data/raw/golfdb/clips.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output labels.json path. Defaults to data/raw/labels.json.",
    )
    parser.add_argument(
        "--clip-template",
        default="{id}.mp4",
        help="Filename template for clip lookup. Default: {id}.mp4",
    )
    parser.add_argument(
        "--view",
        default="face-on",
        help="Filter by GolfDB view. Default: face-on. Use --no-filter to skip.",
    )
    parser.add_argument(
        "--club",
        default="driver",
        help="Filter by GolfDB club. Default: driver. Use --no-filter to skip.",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=0,
        help="Filter by GolfDB slow-mo flag (0=normal, 1=slow-mo). Default: 0.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip all view/club/slow filters. Emits every row with a matching clip.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Emit at most N rows (after filtering and clip-presence check).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    # pandas import is deferred so --help doesn't pay the pandas cost.
    import pandas as pd

    pickle_path = (
        Path(args.pickle).expanduser().resolve()
        if args.pickle
        else data_dir() / "raw" / "golfdb" / "golfDB.pkl"
    )
    clips_dir = (
        Path(args.clips).expanduser().resolve()
        if args.clips
        else data_dir() / "raw" / "golfdb" / "clips"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else data_dir() / "raw" / "labels.json"
    )

    if not pickle_path.is_file():
        _log.error("Annotation pickle not found: %s", pickle_path)
        _log.error("Run `python scripts/download_golfdb.py` first.")
        return 1

    df = pd.read_pickle(pickle_path)
    _log.info("Loaded %d rows from %s", len(df), pickle_path)

    original_count = len(df)
    if not args.no_filter:
        if args.view:
            df = df[df["view"] == args.view]
        if args.club:
            df = df[df["club"] == args.club]
        df = df[df["slow"] == args.slow]
        _log.info(
            "After filters (view=%s, club=%s, slow=%d): %d rows (from %d).",
            args.view,
            args.club,
            args.slow,
            len(df),
            original_count,
        )

    if not clips_dir.is_dir():
        _log.warning(
            "Clip directory does not exist: %s. Every label will be skipped. "
            "Download the GolfDB 160x160 clip release first.",
            clips_dir,
        )

    swings: list[dict[str, object]] = []
    missing = 0
    for _, row in df.iterrows():
        swing_id = str(row["id"])
        clip_name = args.clip_template.format(id=swing_id)
        clip_path = clips_dir / clip_name
        if not clip_path.is_file():
            missing += 1
            continue

        events = list(row["events"])
        if len(events) != 10:
            _log.warning("Skipping swing %s: expected 10 events, got %d.", swing_id, len(events))
            continue

        # GolfDB events: [clip_start, ADDRESS, TOE_UP, MID_BACKSWING, TOP,
        #                 MID_DOWNSWING, IMPACT, MID_FOLLOW_THROUGH, FINISH,
        #                 clip_end]
        # We need indices 1..8 and we rebase them so clip_start → frame 0.
        clip_start = int(events[0])
        canonical = [int(events[i]) - clip_start for i in range(1, 9)]
        if any(e < 0 for e in canonical):
            _log.warning("Skipping swing %s: event frames go negative.", swing_id)
            continue

        view_norm = _VIEW_MAP.get(str(row["view"]), str(row["view"]))
        label = {
            "swing_id": f"golfdb_{swing_id}",
            "video_path": str(clip_path),
            "view": view_norm,
            "handedness": "right",  # GolfDB pickle has no handedness field.
            "club": str(row["club"]),
            "event_frames": canonical,
        }
        swings.append(label)

        if args.limit is not None and len(swings) >= args.limit:
            break

    _log.info(
        "Wrote %d labels (%d skipped because clips were missing).",
        len(swings),
        missing,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"swings": swings}, indent=2),
        encoding="utf-8",
    )
    sys.stdout.write(
        f"Wrote {len(swings)} labels to {output_path} "
        f"(skipped {missing} due to missing clips).\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

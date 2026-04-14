"""Fetch GolfDB annotation artifacts from the upstream GitHub repo.

This script does NOT download videos. Per ``CLAUDE.md`` §5.3 and §6, we
do not redistribute or auto-fetch GolfDB video content. Users who want
source videos must run ``yt-dlp`` or similar themselves, subject to
YouTube's terms of service.

Usage::

    python scripts/download_golfdb.py                    # fetch annotations
    python scripts/download_golfdb.py --dry-run          # print what would happen
    python scripts/download_golfdb.py --output DIR       # pick a destination

The annotation pickle lives at
``https://raw.githubusercontent.com/wmcnally/golfdb/master/data/golfDB.pkl``
(verified against the upstream repo at time of writing). If that URL
ever moves, update ``_ANNOTATIONS_URL`` below.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

from swingscan.utils.logging import configure_logging
from swingscan.utils.paths import data_dir

_log = logging.getLogger(__name__)

_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/wmcnally/golfdb/master/data/golfDB.pkl"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Target directory. Defaults to <repo>/data/raw/golfdb.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the URLs and destination without downloading anything.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    target_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else data_dir() / "raw" / "golfdb"
    )
    target_file = target_dir / "golfDB.pkl"

    sys.stdout.write(
        f"GolfDB annotation source: {_ANNOTATIONS_URL}\n"
        f"Destination:              {target_file}\n"
    )
    sys.stdout.write(
        "Videos are NOT downloaded by this script. Use yt-dlp yourself, "
        "subject to YouTube TOS, if you want source clips.\n"
    )

    if args.dry_run:
        sys.stdout.write("[dry-run] Nothing written.\n")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_ANNOTATIONS_URL, target_file)
    except OSError as exc:
        _log.error("Download failed: %s", exc)
        sys.stdout.write(
            "Download failed. Check your network, or download the file "
            "manually and place it at the destination above.\n"
        )
        return 1

    size_kb = target_file.stat().st_size / 1024
    sys.stdout.write(f"Wrote {target_file} ({size_kb:.1f} KB).\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

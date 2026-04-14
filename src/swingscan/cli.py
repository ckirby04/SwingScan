"""Command-line interface entry point for SwingScan.

Stage 0 ships two subcommands:

    swingscan version             Print the installed package version.
    swingscan run --input PATH    Stub; real pipeline lands in Stage 6.

Later stages add ``pose``, ``phases``, ``compare``, and ``demo`` subcommands
as the corresponding modules come online.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from swingscan import __version__
from swingscan.utils.logging import configure_logging

log = logging.getLogger("swingscan.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingscan",
        description="SwingScan — golf swing analysis pipeline.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=(
            "Override log level (DEBUG, INFO, WARNING, ERROR). Defaults to "
            "INFO or the SWINGSCAN_LOG_LEVEL environment variable."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    subparsers.add_parser(
        "version",
        help="Print the installed SwingScan version and exit.",
    )

    run_p = subparsers.add_parser(
        "run",
        help="Run the end-to-end pipeline on a single swing video (stub in Stage 0).",
    )
    run_p.add_argument("--input", required=True, help="Path to a swing video.")
    run_p.add_argument(
        "--output-video",
        default=None,
        help="Optional path to write an annotated output video.",
    )

    return parser


def _cmd_version() -> int:
    sys.stdout.write(f"swingscan {__version__}\n")
    return 0


def _cmd_run(_args: argparse.Namespace) -> int:
    log.warning(
        "`swingscan run` is a Stage 0 stub. The end-to-end pipeline lands in Stage 6; "
        "see CLAUDE.md §5 for progress."
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "version":
        return _cmd_version()
    if args.command == "run":
        return _cmd_run(args)

    parser.error(f"Unknown command: {args.command}")
    return 2  # pragma: no cover — parser.error exits before returning.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Logging configuration for SwingScan.

Call :func:`configure_logging` once at program start (the CLI does this) to
install a stdlib ``logging`` handler with a consistent format. Every module
in the package should obtain its logger via ``logging.getLogger(__name__)``
— never ``print`` for errors or progress, per ``CLAUDE.md`` §1.5.

The effective level resolves in this order:

1. Explicit argument to :func:`configure_logging`.
2. The ``SWINGSCAN_LOG_LEVEL`` environment variable.
3. ``INFO``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s %(levelname)-8s %(name)s :: %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%dT%H:%M:%S"
_ENV_VAR: Final[str] = "SWINGSCAN_LOG_LEVEL"


def _resolve_level(explicit: str | int | None) -> int:
    """Resolve a human-friendly level to the numeric value stdlib expects."""
    if explicit is not None:
        if isinstance(explicit, int):
            return explicit
        mapping = logging.getLevelNamesMapping()
        return mapping.get(explicit.upper(), logging.INFO)
    env = os.environ.get(_ENV_VAR)
    if env:
        mapping = logging.getLevelNamesMapping()
        return mapping.get(env.upper(), logging.INFO)
    return logging.INFO


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger.

    This function is idempotent: calling it multiple times replaces any
    previously-installed SwingScan handler so the effective level always
    reflects the most recent call.

    Args:
        level: Optional explicit level, either a name (``"DEBUG"``) or an
            integer constant. When ``None``, the ``SWINGSCAN_LOG_LEVEL``
            environment variable is consulted before falling back to
            ``INFO``.
    """
    resolved = _resolve_level(level)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(resolved)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Thin alias for :func:`logging.getLogger`."""
    return logging.getLogger(name)

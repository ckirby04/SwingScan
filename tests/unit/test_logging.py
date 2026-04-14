"""Tests for :mod:`swingscan.utils.logging`."""

from __future__ import annotations

import logging

import pytest

from swingscan.utils.logging import configure_logging, get_logger


def test_configure_logging_installs_single_handler() -> None:
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("WARNING")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING


def test_configure_logging_accepts_numeric_level() -> None:
    configure_logging(logging.ERROR)
    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_honors_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWINGSCAN_LOG_LEVEL", "DEBUG")
    configure_logging(None)
    assert logging.getLogger().level == logging.DEBUG


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("swingscan.test.logger")
    assert logger.name == "swingscan.test.logger"

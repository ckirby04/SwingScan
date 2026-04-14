"""Stage 0 smoke test: package imports and reports a version."""

from __future__ import annotations

import pytest


def test_package_importable() -> None:
    import swingscan

    assert hasattr(swingscan, "__version__")
    assert isinstance(swingscan.__version__, str)
    assert swingscan.__version__, "__version__ must be non-empty"


def test_cli_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    from swingscan.cli import main

    rc = main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("swingscan ")
    assert "0.0.1" in captured.out


def test_cli_run_stub_is_stub() -> None:
    from swingscan.cli import main

    # Stage 0 `run` is a no-op warning that returns 2 so callers can detect
    # that the real pipeline isn't wired up yet.
    rc = main(["run", "--input", "nonexistent.mp4"])
    assert rc == 2

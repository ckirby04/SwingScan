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
    from swingscan import __version__

    assert __version__ in captured.out


def test_cli_run_errors_on_missing_input() -> None:
    # `run` dispatches into the real pipeline, so passing a path that
    # doesn't exist should surface a FileNotFoundError rather than
    # silently returning zero.
    from swingscan.cli import main

    with pytest.raises(FileNotFoundError):
        main(["run", "--input", "definitely-not-a-real-file.mp4"])

# Stage 0 — Bootstrap: status

**Branch:** `stage-0-bootstrap`
**Owner:** lead engineer (per `CLAUDE.md`)
**Last updated:** 2026-04-14

## Goal

Repo exists, runs, passes lint, passes a trivial test. Exit criterion:
`make install && make lint && make typecheck && make test` all pass on a
fresh checkout and `swingscan version` prints the package version.

## Environment notes

- Platform: Windows 11, `bash` (Git Bash / MSYS).
- Python 3.11.9 was not available on the host at the start of the session
  and was installed via `winget install Python.Python.3.11`.
- GNU Make was not available and was installed via
  `winget install GnuWin32.Make` (GNU Make 3.81) so the Makefile targets
  can be exercised. The install target chooses the right venv layout based
  on `$OS`.
- `ffmpeg` is **not** installed. That is tolerable for Stage 0 (no video
  decoding happens yet); Stage 1 will surface a clear error when it needs
  ffmpeg and it is absent.
- `uv` is not installed. `CLAUDE.md` §3 permits `pip + venv` as a fallback
  and that is what the `Makefile` uses.

## What was done

Scaffold:

- Full directory tree from `CLAUDE.md` §4, with `.gitkeep` files where the
  directory would otherwise be empty.
- `pyproject.toml` with Python 3.11 pin, core deps (`numpy`, `pandas`,
  `pydantic`, `pyyaml`, `pyarrow`), a `pipeline` extra holding the heavy
  ML stack (`opencv-python`, `scipy`, `scikit-learn`, `mediapipe`, `torch`,
  `torchvision`, `ultralytics`, `gradio`), and a `dev` extra (`pytest`,
  `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `types-PyYAML`). All versions
  pinned. `[project.scripts] swingscan = "swingscan.cli:main"`.
- `ruff` and `mypy` configured in `pyproject.toml`. `mypy` runs strict on
  `src/swingscan` and lenient on `tests/` and `scripts/`. `pydantic.mypy`
  plugin enabled. Missing-import overrides for cv2/mediapipe/ultralytics/
  gradio/torch/torchvision/scipy/sklearn so Stage 0 can pass typecheck
  without the heavy extras installed.
- `pytest` configured with `--cov=swingscan --cov-report=term-missing`.
- `Makefile` with targets `install`, `lint`, `format`, `typecheck`, `test`,
  `test-cov`, `demo` (Stage 8 placeholder), `version`, `clean`. Cross-
  platform venv layout via `$OS`.
- `.gitignore` covering caches, venvs, `data/`, `models/`, media files
  (except `tests/fixtures/`), and common secret files.
- `.python-version` pinned to `3.11.9`.
- `.pre-commit-config.yaml` with `ruff`, `ruff-format`, and the standard
  `pre-commit-hooks` baseline.
- `README.md` with pitch, install instructions, stage status table, and
  explicit scope guardrails per `CLAUDE.md` §2.
- Placeholder `docs/architecture.md` and `docs/pipeline.md` (filled in
  later stages).
- `docs/decisions/001-dependency-extras.md` — ADR explaining why heavy
  pipeline deps live in an optional extra.

Package skeleton under `src/swingscan/`:

- `__init__.py` exposing `__version__ = "0.0.1"`.
- `cli.py` — argparse CLI with `version` and a `run` Stage 0 stub.
- `config.py` — pydantic v2 `SwingScanConfig`, `LoggingConfig`, `PoseConfig`;
  `load_config(path)` supports explicit path, default-file discovery, and
  default-constructed fallback.
- `utils/logging.py` — idempotent root-logger configurator with env-var
  fallback.
- `utils/seeding.py` — `seed_everything()` seeds Python, NumPy, and (if
  present) torch CPU/CUDA.
- `utils/paths.py` — repo-root anchored path helpers.
- Empty subpackages for `io/`, `pose/`, `club/`, `phases/`, `metrics/`,
  `compare/`, `feedback/`, `viz/`, each with a module docstring explaining
  which stage will populate it.

Configs: `default.yaml` with matching schema, plus placeholder
`pose_mediapipe.yaml`, `phases_swingnet.yaml`, `feedback_rules.yaml`.

Tests under `tests/unit/`:

- `test_smoke.py` — package importable, CLI `version` prints, `run` stub
  returns sentinel exit code.
- `test_config.py` — default construction, YAML loading, missing path,
  unknown field rejection, non-mapping root rejection, `None` path
  behavior.
- `test_logging.py` — handler install, idempotency, numeric level, env
  override, named loggers.
- `test_seeding.py` — determinism across double-seed, return value, seed
  independence.
- `test_paths.py` — repo-root anchored and pointing at `pyproject.toml`.

## Exit criterion verification

Run on a fresh venv:

```
make install
make lint
make typecheck
make test
```

All four must succeed. `swingscan version` must print `swingscan 0.0.1`.
The verification command output is captured in the corresponding session
transcript; subsequent stages should re-run these targets as their own
smoke checks.

## Known issues / carry-overs

- **ffmpeg missing.** Tolerable for Stage 0; Stage 1 must explicitly check
  for ffmpeg at startup and raise a clear error.
- **No fixture swing video.** `tests/fixtures/.gitkeep` is the placeholder.
  Stage 1 must either commit a tiny permissively-licensed clip or generate
  a synthetic moving-shape video programmatically before pose tests land.
- **No CI configured.** Deliberately out of Stage 0 scope; revisit after
  Stage 1 so CI can exercise the real pipeline.

## Next stage

Stage 1 — Video I/O and Pose Extraction (MediaPipe baseline). Before
writing code, produce `docs/plans/stage1-plan.md` per `CLAUDE.md` §1.3.
